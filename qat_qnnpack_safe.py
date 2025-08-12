# qat_qnnpack_safe.py
# QAT + static quant convert + 10-epoch checkpoint + resume + TorchScript export
# - GPU 장치 고정 (코드 상단 GPU_ID로 제어)
# - datasets.py의 build_dataloaders(seed=...) 사용
# - (secret, cover) -> Haar DWT -> 채널 concat -> 모델 입력
# - calibrate / train / eval 진행 로그
# - 10 epoch마다 checkpoint(.pth) 저장 및 --resume로 이어서 학습
# - 최종 int8 eager 모델(.pt)과 TorchScript(.pt) 저장 (변환은 반드시 CPU에서)

import os
import math
import copy
import argparse
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq

import config as c
from datasets import build_dataloaders
from hinet import Hinet  # 프로젝트의 HiNet 모델

# ----------------------------
# 0) GPU 고정 (명령행 인자 없이)
# ----------------------------
GPU_ID = 0  # <<<< 사용할 GPU 번호를 여기서 지정하세요 (예: 0)

# ----------------------------
# 1) 유틸
# ----------------------------
def now() -> str:
    return datetime.now().strftime("%y-%m-%d %H:%M:%S")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def logit(msg: str, logger=None):
    print(f"{now()} - INFO: {msg}")
    if logger is not None:
        logger.write(f"{now()} - INFO: {msg}\n")
        logger.flush()

def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((x - y) ** 2).item()
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

# ----------------------------
# 2) Haar DWT (채널당 4배, H/2, W/2)
# ----------------------------
@torch.no_grad()
def haar_dwt(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W]  ->  [B, 4C, H/2, W/2]
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H, W must be even for Haar DWT"
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)

def make_input(secret: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
    # secret, cover: [B,3,H,W] in [0,1]
    sd = haar_dwt(secret)  # [B,12,H/2,W/2]
    cd = haar_dwt(cover)   # [B,12,H/2,W/2]
    x = torch.cat([sd, cd], dim=1)  # [B,24,H/2,W/2]
    return x

# ----------------------------
# 3) 모델 래퍼(QAT/양자화 경계 포함)
# ----------------------------
class QATWrapper(nn.Module):
    """
    - 입력은 (secret, cover) -> DWT -> concat 이후 들어오도록 외부에서 준비.
    - 래퍼는 모델 앞뒤에 QuantStub/DeQuantStub를 둠.
    """
    def __init__(self, core: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.core = core
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        xq = self.quant(x)
        yq = self.core(xq, rev=rev)
        y = self.dequant(yq)
        return y

# ----------------------------
# 4) 학습/캘리브/평가 루프
# ----------------------------
def train_one_epoch(model: nn.Module,
                    device: torch.device,
                    optim: optim.Optimizer,
                    trainloader,
                    epoch: int,
                    total_epochs: int,
                    show_progress: bool = False) -> float:
    model.train()
    running = 0.0
    total = len(trainloader)
    logit(f"TRAIN: start epoch {epoch}/{total_epochs}")

    for i, (secret, cover) in enumerate(trainloader, 1):
        if show_progress and (i % max(1, total // 10) == 0 or i == total):
            pct = int(i * 100 / total)
            logit(f"TRAIN: {pct}% ({i}/{total})")

        secret = secret.to(device, non_blocking=True)
        cover  = cover.to(device, non_blocking=True)
        x = make_input(secret, cover)

        # 단순 L1 복원 손실 (QAT 안정화 목적)
        y = model(x)
        loss = torch.nn.functional.l1_loss(y, x)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        running += loss.item()

    avg = running / max(1, total)
    logit(f"TRAIN: done | loss {avg:.4f}")
    return avg

@torch.no_grad()
def calibrate(model: nn.Module, device: torch.device, trainloader, steps: int = 64, show_progress: bool = False):
    model.eval()
    logit(f"calibrating observers with {steps} batches ...")
    seen = 0
    total = steps
    for i, (secret, cover) in enumerate(trainloader, 1):
        secret = secret.to(device, non_blocking=True)
        cover  = cover.to(device, non_blocking=True)
        x = make_input(secret, cover)
        _ = model(x, rev=False)
        seen += 1
        if show_progress and (seen % max(1, total // 5) == 0 or seen == total):
            pct = int(seen * 100 / total)
            logit(f"CALIB: {pct}% ({seen}/{total})")
        if seen >= steps:
            break
    logit("calibrate: done")

@torch.no_grad()
def evaluate(model: nn.Module, device: torch.device, valloader, show_progress: bool = False) -> Tuple[float, float]:
    model.eval()
    total_batches = len(valloader)
    logit(f"EVAL: start (batches={total_batches})")
    psnr_c_list, psnr_s_list = [], []

    for i, (secret, cover) in enumerate(valloader, 1):
        if show_progress and (i % max(1, total_batches // 5) == 0 or i == total_batches):
            pct = int(i * 100 / total_batches)
            logit(f"EVAL: {pct}% ({i}/{total_batches})")

        secret = secret.to(device, non_blocking=True)
        cover  = cover.to(device, non_blocking=True)
        x = make_input(secret, cover)
        y = model(x, rev=False)

        y_c, y_s = torch.chunk(y, 2, dim=1)
        x_c, x_s = torch.chunk(x, 2, dim=1)
        psnr_c_list.append(psnr(y_c.clamp(0, 1).cpu(), x_c.clamp(0, 1).cpu()))
        psnr_s_list.append(psnr(y_s.clamp(0, 1).cpu(), x_s.clamp(0, 1).cpu()))

    psnr_c = sum(psnr_c_list) / max(1, len(psnr_c_list))
    psnr_s = sum(psnr_s_list) / max(1, len(psnr_s_list))
    logit("EVAL: done")
    logit(f"EVAL | PSNR_C {psnr_c:.3f} dB | PSNR_S {psnr_s:.3f} dB")
    return psnr_c, psnr_s

# ----------------------------
# 5) 체크포인트 I/O (10 epoch마다 저장)
# ----------------------------
def save_checkpoint(save_dir: str, epoch: int, model: nn.Module, optimizer: optim.Optimizer, best: dict):
    ensure_dir(save_dir)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best": best,
    }
    path = os.path.join(save_dir, f"checkpoint_ep{epoch:04d}.pth")
    torch.save(ckpt, path)
    torch.save(ckpt, os.path.join(save_dir, "last.pth"))
    logit(f"checkpoint saved: {path}")

def load_checkpoint(resume_path: str, model: nn.Module, optimizer: optim.Optimizer):
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best = ckpt.get("best", {"psnr_c": -1, "psnr_s": -1})
    logit(f"checkpoint loaded: {resume_path} (start from epoch {start_epoch})")
    return start_epoch, best

# ----------------------------
# 6) QAT 준비/변환
# ----------------------------
def choose_backend() -> str:
    # 서버(x86)라면 fbgemm 우선, ARM 계열이면 qnnpack
    engines = torch.backends.quantized.supported_engines
    if "fbgemm" in engines:
        return "fbgemm"
    if "qnnpack" in engines:
        return "qnnpack"
    return engines[0] if engines else "fbgemm"

def prepare_qat(model: nn.Module, backend: str):
    torch.backends.quantized.engine = backend
    model.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(model, inplace=True)
    # 디버깅용 한두 개만 로깅
    for name, mod in model.named_modules():
        if hasattr(mod, "activation_post_process"):
            ap = mod.activation_post_process
            logit(f"INIT_QCONFIG | {name}.activation_post_process | dtype={getattr(ap, 'dtype', None)} "
                  f"qscheme={getattr(ap, 'qscheme', None)} qmin={getattr(ap, 'quant_min', None)} qmax={getattr(ap, 'quant_max', None)}")
            break
    for name, mod in model.named_modules():
        if hasattr(mod, "weight_fake_quant"):
            wfq = mod.weight_fake_quant
            logit(f"INIT_QCONFIG | {name}.weight_fake_quant | dtype={getattr(wfq, 'dtype', None)} "
                  f"qscheme={getattr(wfq, 'qscheme', None)} qmin={getattr(wfq, 'quant_min', None)} qmax={getattr(wfq, 'quant_max', None)}")
            break

def convert_to_int8(model: nn.Module, backend: str) -> nn.Module:
    """
    - 항상 CPU에서 convert 수행
    - 학습 모델을 건드리지 않도록 deepcopy 후 .cpu().eval()
    """
    torch.backends.quantized.engine = backend
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    int8_model = tq.convert(model_cpu, inplace=False)
    return int8_model

# ----------------------------
# 7) 메인
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default=None, help="FP32 pretrained .pt (state_dict or full)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--calib", type=int, default=64)
    parser.add_argument("--export-script", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--save-every", type=int, default=10, help="save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="resume checkpoint path (.pth)")
    args = parser.parse_args()

    # 로그 디렉토리
    tag = f"qat_qbackend_safe_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{args.epochs}_calib{args.calib}"
    save_dir = os.path.join("logging", tag)
    ensure_dir(save_dir)
    log_path = os.path.join(save_dir, "train.log")
    logger = open(log_path, "a", buffering=1)
    logit(f"log file: {log_path}", logger)

    # 디바이스 (코드 고정 GPU_ID)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        torch.cuda.set_device(GPU_ID)
        logit(f"Using GPU {GPU_ID}: {torch.cuda.get_device_name(GPU_ID)}", logger)
    else:
        device = torch.device("cpu")
        logit("Using CPU", logger)

    # 백엔드 선택
    backend = choose_backend()
    torch.backends.quantized.engine = backend
    logit(f"quant backend engine: {backend}", logger)

    # 데이터 로더
    trainloader, valloader = build_dataloaders(seed=getattr(c, "seed", 1234))

    # 모델 생성 및 래핑
    core = Hinet()
    model = QATWrapper(core).to(device)

    # 사전학습 로드(있으면)
    if args.pretrained and os.path.isfile(args.pretrained):
        try:
            sd = torch.load(args.pretrained, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logit(f"loaded pretrained FP32: {args.pretrained}", logger)
            if isinstance(missing, list) and len(missing) > 0:
                logit(f"  missing keys: {len(missing)}", logger)
            if isinstance(unexpected, list) and len(unexpected) > 0:
                logit(f"  unexpected keys: {len(unexpected)}", logger)
        except Exception as e:
            logit(f"pretrained load failed: {e}", logger)

    # QAT 준비
    prepare_qat(model, backend=backend)

    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # resume
    start_epoch = 1
    best = {"psnr_c": -1.0, "psnr_s": -1.0}
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best = load_checkpoint(args.resume, model, optimizer)

    # 캘리브 (trainloader에서 정확히 steps 만큼)
    calibrate(model, device, trainloader, steps=args.calib, show_progress=args.progress)

    # 학습 루프
    for epoch in range(start_epoch, args.epochs + 1):
        _ = train_one_epoch(model, device, optimizer, trainloader, epoch, args.epochs, show_progress=args.progress)
        psnr_c, psnr_s = evaluate(model, device, valloader, show_progress=True)

        best["psnr_c"] = max(best["psnr_c"], psnr_c)
        best["psnr_s"] = max(best["psnr_s"], psnr_s)

        # 체크포인트 저장 (매 N epoch)
        if (epoch % max(1, args.save_every) == 0) or (epoch == args.epochs):
            save_checkpoint(save_dir, epoch, model, optimizer, best)

    # ===== 학습 완료 후에만 INT8 변환/스크립트 =====
    int8_model = convert_to_int8(model, backend=backend).eval()
    psnr_c, psnr_s = evaluate(int8_model, torch.device("cpu"), valloader, show_progress=True)
    logit(f"EVAL | PSNR_C {psnr_c:.3f} dB | PSNR_S {psnr_s:.3f} dB", logger)

    # 저장 (eager)
    eager_path = os.path.join(save_dir, "hinet_qat_int8_full.pt")
    torch.save(int8_model, eager_path)
    logit(f"saved full int8 model: {eager_path}", logger)

    # TorchScript export (옵션)
    if args.export_script:
        try:
            scripted = torch.jit.script(int8_model)
            ts_path = os.path.join(save_dir, "hinet_qat_int8_scripted.pt")
            scripted.save(ts_path)
            logit(f"saved TorchScript int8: {ts_path}", logger)
        except Exception as e:
            logit(f"TorchScript export failed: {e}", logger)

    logger.close()

if __name__ == "__main__":
    main()
