#!/usr/bin/env python3
import os
import platform
import argparse
import logging
import time
from datetime import datetime
from typing import Tuple, List, Dict, Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq

from hinet import Hinet
import modules.Unet_common as common
import datasets
import config as c

# =============== 선택 tqdm ===============
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

gpu_id = 3

def progress_iter(it: Iterable, desc: str, total: Optional[int], enable: bool):
    """
    tqdm이 있으면 진행바, 없으면 10% 단위 로그만.
    """
    if enable and _HAS_TQDM:
        return tqdm(it, total=total, desc=desc)
    # fallback: 10% 단위 로깅
    if total is None or total <= 0:
        for x in it:
            yield x
        return
    step = 0
    tenth = max(1, total // 10)
    logging.info(f"{desc}: start (total={total})")
    for x in it:
        yield x
        step += 1
        if step % tenth == 0:
            pct = int(100 * step / total)
            logging.info(f"{desc}: {pct}% ({step}/{total})")
    logging.info(f"{desc}: done")


# ===================== 전역/엔진 =====================
def select_engine() -> str:
    engines = getattr(torch.backends.quantized, "supported_engines", [])
    mach = platform.machine().lower()
    if ("x86_64" in mach or "amd64" in mach) and "fbgemm" in engines:
        return "fbgemm"
    if ("arm" in mach or "aarch64" in mach) and "qnnpack" in engines:
        return "qnnpack"
    for cand in ("fbgemm", "qnnpack", "onednn", "x86"):
        if cand in engines:
            return cand
    return "none"

try:
    torch.backends.quantized.engine = select_engine()
except Exception:
    pass


# ===================== 유틸/로그 =====================
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    logging.info(f"log@{log_path}")
    return log_path


def list_quantized_modules(model: nn.Module) -> List[str]:
    return [m.__class__.__name__ for m in model.modules()
            if 'quantized' in m.__class__.__module__.lower()
            or 'Quantized' in m.__class__.__name__]


# ===================== 모델 래퍼 =====================
class QATWrapper(nn.Module):
    """
    모델 경계에만 QuantStub/DeQuantStub 배치. 내부는 Conv만 QAT 대상.
    """
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        x = self.quant(x)
        y = self.core(x, rev=rev)
        y = self.dequant(y)
        if y.is_quantized:
            y = y.dequantize()
        return y


# ===================== 패치/표식 =====================
def strip_internal_qstubs(module: nn.Module):
    if getattr(module, "__quant_protect__", False):
        return
    for name, child in list(module.named_children()):
        if isinstance(child, (tq.QuantStub, tq.DeQuantStub)):
            setattr(module, name, nn.Identity())
        else:
            strip_internal_qstubs(child)


def replace_leakyrelu_with_relu(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LeakyReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_leakyrelu_with_relu(child)


def mark_qconfig_conv_and_stubs(module: nn.Module, qconfig: tq.QConfig):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, tq.QuantStub)):
            m.qconfig = qconfig


def build_qconfig(backend: str) -> tq.QConfig:
    try:
        return tq.qconfig.get_default_qat_qconfig(backend)
    except AttributeError:
        return tq.get_default_qat_qconfig(backend)


# ===================== 학습/캘리브/평가 =====================
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2)
    return float("inf") if mse.item() == 0 else 10 * torch.log10(1.0 / mse).item()


def evaluate(model: nn.Module, device: torch.device, show_progress: bool = False) -> Tuple[float, float]:
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    model.eval()
    c_list, s_list = [], []
    loader = datasets.testloader
    it = progress_iter(loader, "EVAL", total=len(loader), enable=show_progress)
    with torch.no_grad():
        for secret, cover in it:
            secret = secret.to(device)
            cover = cover.to(device)
            cover_in = dwt(cover)
            secret_in = dwt(secret)
            x = torch.cat((cover_in, secret_in), 1)
            y = model(x)  # rev=False
            ch = 4 * c.channels_in
            steg = iwt(y.narrow(1, 0, ch))
            z = torch.randn_like(y.narrow(1, ch, y.size(1) - ch))
            rev_in = torch.cat((y.narrow(1, 0, ch), z), 1)
            back = model(rev_in, rev=True)
            sec = iwt(back.narrow(1, ch, back.size(1) - ch))
            c_list.append(psnr(steg, cover))
            s_list.append(psnr(sec, secret))
    mc = float(np.mean(c_list)) if c_list else 0.0
    ms = float(np.mean(s_list)) if s_list else 0.0
    logging.info(f"EVAL | PSNR_C {mc:.3f} dB | PSNR_S {ms:.3f} dB")
    return mc, ms


def train_one_epoch(model: nn.Module, device: torch.device,
                    optim: torch.optim.Optimizer, show_progress: bool = False) -> float:
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    model.train()
    total = 0.0
    loader = datasets.trainloader
    it = progress_iter(loader, "TRAIN", total=len(loader), enable=show_progress)
    for secret, cover in it:
        secret = secret.to(device)
        cover = cover.to(device)
        cover_in = dwt(cover)
        secret_in = dwt(secret)
        x = torch.cat((cover_in, secret_in), 1)

        y = model(x)  # forward
        ch = 4 * c.channels_in
        steg = iwt(y.narrow(1, 0, ch))

        z = torch.randn_like(y.narrow(1, ch, y.size(1) - ch))
        rev_in = torch.cat((y.narrow(1, 0, ch), z), 1)
        back = model(rev_in, rev=True)
        secret_rev = iwt(back.narrow(1, ch, back.size(1) - ch))

        g_loss = F.mse_loss(steg, cover, reduction="sum")
        r_loss = F.mse_loss(secret_rev, secret, reduction="sum")
        steg_low = y.narrow(1, 0, ch).narrow(1, 0, c.channels_in)
        cover_low = cover_in.narrow(1, 0, c.channels_in)
        l_loss = F.mse_loss(steg_low, cover_low, reduction="sum")

        loss = (c.lamda_reconstruction * r_loss
                + c.lamda_guide * g_loss
                + c.lamda_low_frequency * l_loss)

        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item()
    return total / max(1, len(loader))


def calibrate(model: nn.Module, device: torch.device, steps: int = 64, show_progress: bool = False):
    dwt = common.DWT().to(device)
    model.eval()
    it_data = iter(datasets.trainloader)
    rng = range(steps)
    it = progress_iter(rng, "CALIB", total=steps, enable=show_progress)
    with torch.no_grad():
        for _ in it:
            try:
                secret, cover = next(it_data)
            except StopIteration:
                it_data = iter(datasets.trainloader)
                secret, cover = next(it_data)
            secret = secret.to(device)
            cover = cover.to(device)
            x = torch.cat((dwt(cover), dwt(secret)), 1)
            y = model(x)  # fwd
            ch = 4 * c.channels_in
            z = torch.randn_like(y.narrow(1, ch, y.size(1) - ch))
            model(torch.cat((y.narrow(1, 0, ch), z), 1), rev=True)  # rev


# ===================== 검증/리포트 =====================
def quantization_report(model: nn.Module) -> Dict[str, Any]:
    """
    모델이 진짜 int8 경로를 쓰는지 가벼운 정적/동적 체크를 수행하고
    통계를 dict로 반환.
    """
    report: Dict[str, Any] = {}
    engine = getattr(torch.backends.quantized, "engine", "unknown")
    report["engine"] = engine

    # 모듈 인벤토리
    qmods = []
    fconvs = []
    qconvs = []
    qstubs = []
    dqstubs = []
    qz = []
    dqz = []

    for m in model.modules():
        name = m.__class__.__name__
        modpath = m.__class__.__module__.lower()

        if isinstance(m, nn.Conv2d):
            fconvs.append(name)
        if 'quantized' in modpath or 'quantized' in name.lower():
            qmods.append(name)
        if name == "QuantizedConv2d" or name.endswith("Conv2d") and 'quantized' in modpath:
            qconvs.append(m)
        if name == "Quantize":
            qz.append(m)
        if name == "DeQuantize":
            dqz.append(m)
        if isinstance(m, tq.QuantStub):
            qstubs.append(m)
        if isinstance(m, tq.DeQuantStub):
            dqstubs.append(m)

    report["num_float_conv2d"] = len(fconvs)
    report["num_quantized_modules"] = len(qmods)
    report["num_quantized_conv2d"] = len(qconvs)
    report["num_quantize_modules"] = len(qz)
    report["num_dequantize_modules"] = len(dqz)
    report["num_quantstub_left"] = len(qstubs)
    report["num_dequantstub_left"] = len(dqstubs)

    # 예시 qparams (앞에서 몇 개만)
    qparams = []
    for m in qconvs[:10]:
        try:
            w = m.weight()
            qs = str(w.qscheme())
            scales = w.q_per_channel_scales().tolist()[:4] if hasattr(w, "q_per_channel_scales") else [float(w.q_scale())]
            zps = w.q_per_channel_zero_points().tolist()[:4] if hasattr(w, "q_per_channel_zero_points") else [int(w.q_zero_point())]
            qparams.append({"module": m.__class__.__name__, "qscheme": qs, "scales_head": scales, "zps_head": zps})
        except Exception:
            pass
    report["sampled_qparams"] = qparams
    return report


def assert_int8_cpu(model: nn.Module):
    for p in model.parameters(recurse=True):
        if p.is_cuda:
            raise RuntimeError("INT8 모델이 CUDA에 올라가 있음. convert 후에는 CPU 고정 필요")

    qmods = list_quantized_modules(model)
    if not qmods:
        raise RuntimeError("quantized 모듈이 없음. prepare/convert 흐름 점검 필요")

    cin = int(getattr(c, "channels_in", 3))
    ch_in = 8 * cin
    H = W = 64

    try:
        with torch.inference_mode():
            x = torch.randn(1, ch_in, H, W)
            y = model(x)  # fwd
            ch4 = 4 * cin
            if y.dim() == 4 and y.size(1) >= ch4:
                z = torch.randn(y.size(0), y.size(1) - ch4, y.size(2), y.size(3))
                rev_in = torch.cat((y.narrow(1, 0, ch4), z), 1)
                model(rev_in, rev=True)
    except Exception as e:
        raise RuntimeError(f"INT8 추론 스모크 테스트 실패: {e}")


def verify_and_log(model: nn.Module, save_dir: str, show_progress: bool = True):
    """
    단계별(6단계) 검증 진행상황을 로그/진행바로 출력하고, 리포트를 파일로 저장.
    """
    steps = [
        "환경/엔진 점검",
        "정적 인벤토리 수집",
        "더미 전/역추론 스모크",
        "경계 Quantize/DeQuantize 배치 확인",
        "간이 성능 타이밍",
        "리포트 저장"
    ]
    if show_progress and _HAS_TQDM:
        bar = tqdm(total=len(steps), desc="VERIFY", leave=False)
        def bump(): bar.update(1)
    else:
        def bump(): pass

    # 1) 환경/엔진
    logging.info(f"[1/6] {steps[0]}")
    engine = getattr(torch.backends.quantized, "engine", "unknown")
    logging.info(f" - quantized.engine={engine}")
    bump()

    # 2) 인벤토리
    logging.info(f"[2/6] {steps[1]}")
    report = quantization_report(model)
    logging.info(f" - float conv2d: {report['num_float_conv2d']}")
    logging.info(f" - quantized conv2d: {report['num_quantized_conv2d']}")
    logging.info(f" - quantized modules (all): {report['num_quantized_modules']}")
    logging.info(f" - boundary Quantize/DeQuantize: {report['num_quantize_modules']}/{report['num_dequantize_modules']}")
    logging.info(f" - leftover QuantStub/DeQuantStub: {report['num_quantstub_left']}/{report['num_dequantstub_left']}")
    bump()

    # 3) 스모크
    logging.info(f"[3/6] {steps[2]}")
    assert_int8_cpu(model)
    logging.info(" - int8 forward/reverse OK")
    bump()

    # 4) 경계 배치 확인 (권장: 상위 래퍼에만 존재)
    logging.info(f"[4/6] {steps[3]}")
    if report["num_quantstub_left"] > 0 or report["num_dequantstub_left"] > 0:
        logging.warning(" - 아직 QuantStub/DeQuantStub 잔여 있음 (내부 FP32 경로에 남아있지 않은지 확인)")
    else:
        logging.info(" - QuantStub/DeQuantStub 잔여 없음")
    bump()

    # 5) 간이 타이밍
    logging.info(f"[5/6] {steps[4]}")
    cin = int(getattr(c, "channels_in", 3))
    x = torch.randn(1, 8 * cin, 128, 128)
    with torch.inference_mode():
        t0 = time.perf_counter()
        _ = model(x)
        t1 = time.perf_counter()
    logging.info(f" - 1x int8 forward(128x128) {1000*(t1-t0):.2f} ms")
    bump()

    # 6) 저장
    logging.info(f"[6/6] {steps[5]}")
    path = os.path.join(save_dir, "quant_report.txt")
    with open(path, "w") as f:
        f.write(f"engine: {report['engine']}\n")
        f.write(f"num_float_conv2d: {report['num_float_conv2d']}\n")
        f.write(f"num_quantized_modules: {report['num_quantized_modules']}\n")
        f.write(f"num_quantized_conv2d: {report['num_quantized_conv2d']}\n")
        f.write(f"num_quantize_modules: {report['num_quantize_modules']}\n")
        f.write(f"num_dequantize_modules: {report['num_dequantize_modules']}\n")
        f.write(f"num_quantstub_left: {report['num_quantstub_left']}\n")
        f.write(f"num_dequantstub_left: {report['num_dequantstub_left']}\n\n")
        f.write("sampled_qparams (head):\n")
        for i, qp in enumerate(report["sampled_qparams"]):
            f.write(f" {i:02d}: {qp}\n")
    logging.info(f" - saved {path}")


# ===================== 파이프라인 =====================
def run(args):
    save_dir = os.path.join("logging", f"qat_qbackend_safe_{now_tag()}_ep{args.epochs}_calib{args.calib}")
    setup_logger(save_dir)
    logging.info(f"engine={getattr(torch.backends.quantized, 'engine', 'n/a')} (tqdm={'on' if _HAS_TQDM and args.progress else 'off'})")

    # 1) FP32 모델
    base = Hinet()
    if args.strip_internal_qstubs:
        strip_internal_qstubs(base)
        logging.info("stripped internal Quant/DeQuant stubs")
    if args.replace_leakyrelu:
        replace_leakyrelu_with_relu(base)
        logging.info("replaced LeakyReLU -> ReLU")

    # 사전학습 로드
    ckpt = os.path.join(c.MODEL_PATH, c.suffix) if args.pretrained is None else args.pretrained
    if os.path.isfile(ckpt):
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict):
            for top in ("net", "state_dict", "model"):
                if top in sd and isinstance(sd[top], dict):
                    sd = sd[top]
                    break
        base.load_state_dict({k.replace("module.", "").replace("model.", ""): v
                              for k, v in sd.items()}, strict=False)
        logging.info(f"loaded pretrained: {ckpt}")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = QATWrapper(base).to(device)

    # 2) QAT 준비 (Conv + QuantStub)
    qconfig = build_qconfig(getattr(torch.backends.quantized, "engine", "fbgemm"))
    mark_qconfig_conv_and_stubs(model, qconfig)
    tq.prepare_qat(model, inplace=True)
    logging.info("prepare_qat done")

    # 3) 학습
    optim = torch.optim.Adam(model.parameters(), lr=c.lr)
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, device, optim, show_progress=args.progress)
        logging.info(f"epoch {ep:03d} | loss {loss:.4f}")
        if ep % max(1, args.eval_every) == 0:
            evaluate(model, device, show_progress=args.progress)

    # 4) 캘리브
    calibrate(model, device, steps=args.calib, show_progress=args.progress)

    # 5) INT8 변환(반드시 CPU)
    model.eval(); model.cpu()
    qmodel = tq.convert(model, inplace=False).eval()

    # 통계/안전 점검
    qmods = list_quantized_modules(qmodel)
    logging.info(f"quantized modules: {sorted(set(qmods))} (count={len(qmods)})")
    logging.info(str(qmodel))
    assert_int8_cpu(qmodel)

    # 6) CPU 평가 + 저장
    evaluate(qmodel, torch.device("cpu"), show_progress=args.progress)

    full_pt = os.path.join(save_dir, "hinet_qat_int8_full.pt")
    torch.save(qmodel, full_pt)
    logging.info(f"saved full int8 model: {full_pt}")

    # 7) (옵션) TorchScript 저장
    if args.export_script:
        scripted = torch.jit.script(qmodel)
        ts_pt = os.path.join(save_dir, "hinet_qat_int8_scripted.pt")
        scripted.save(ts_pt)
        logging.info(f"saved TorchScript int8: {ts_pt}")

    # 8) **양자화 검증 + 리포트 저장 + 진행상황 표시**
    if not args.no_verify:
        verify_and_log(qmodel, save_dir, show_progress=args.progress)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, default=None, help="FP32 체크포인트 경로(없으면 config.suffix 사용)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--calib", type=int, default=128, help="캘리브레이션 배치 수 (fwd+rev 모두 통과)")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--replace-leakyrelu", action="store_true")
    ap.add_argument("--strip-internal-qstubs", dest="strip_internal_qstubs", action="store_true")
    ap.add_argument("--export-script", action="store_true")
    ap.add_argument("--progress", action="store_true", help="tqdm 진행바/진행 로그 표시")
    ap.add_argument("--no-verify", action="store_true", help="최종 검증/리포트 건너뛰기")
    args = ap.parse_args()
    run(args)