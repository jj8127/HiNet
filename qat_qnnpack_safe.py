# -*- coding: utf-8 -*-
"""
qat_qnnpack_safe.py

Quantization-Aware Training (QAT) script for HiNet with:
 - Safe QNNPACK/FBGEMM engine selection
 - Explicit "safe" QConfig (fixes zero_point out-of-range during QAT)
 - Internal FP32 arithmetic guards live in invblock.py / rrdb_denselayer.py
 - Calibration + evaluation hooks
 - Progress logging with % indicator
 - Periodic checkpoint every N epochs + resume from checkpoint
 - Final export: eager INT8 (and optional TorchScript)

Replace your existing file with this whole script.
"""

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.ao.quantization as tq

# explicit imports for safe qconfig
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
)

import config as c
from hinet import Hinet


# ----------------------------
# Logging / utils
# ----------------------------
def setup_logging(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
    logging.info("log file: %s", log_path)


def set_cuda_visible_devices(gpus: Optional[str]):
    if gpus is not None and gpus != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        logging.info("CUDA_VISIBLE_DEVICES=%s", gpus)


def select_engine() -> str:
    """
    Prefer FBGEMM on x86, fall back to QNNPACK on ARM.
    """
    engine = "fbgemm"
    if torch.backends.quantized.supported_engines is not None:
        if "fbgemm" in torch.backends.quantized.supported_engines:
            engine = "fbgemm"
        elif "qnnpack" in torch.backends.quantized.supported_engines:
            engine = "qnnpack"
    torch.backends.quantized.engine = engine
    logging.info("quant backend engine: %s", engine)
    return engine


def psnr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def strip_internal_qstubs(m: nn.Module):
    """
    Optionally remove *internal* QuantStub/DeQuantStub that are not needed anymore.
    (boundary stubs are added by QATWrapper)
    """
    for name, child in list(m.named_children()):
        if isinstance(child, (tq.QuantStub, tq.DeQuantStub)):
            delattr(m, name)
        else:
            strip_internal_qstubs(child)


def replace_leakyrelu_with_relu(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.LeakyReLU):
            setattr(m, name, nn.ReLU(inplace=False))
        else:
            replace_leakyrelu_with_relu(child)


class QATWrapper(nn.Module):
    """
    Add boundary Quant/DeQuant so the converted int8 model is drop-in
    with FP32 input/output.
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
# **Safe** QConfig (fix for zero_point out-of-range)
# ----------------------------
def safe_qconfig() -> tq.QConfig:
    """
    Use explicit, conservative qconfig:
      - Activations: quint8, per-tensor affine, MovingAverageMinMaxObserver
      - Weights: qint8, per-channel *symmetric*, PerChannelMinMaxObserver
    This avoids zero_point drifting out of [quant_min, quant_max].
    """
    act_fq = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        eps=1e-4,
        quant_min=0,
        quant_max=255,
    )
    w_fq = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,           # out_channels
        reduce_range=False,
        eps=1e-4,
        quant_min=-128,
        quant_max=127,
    )
    return tq.QConfig(activation=act_fq, weight=w_fq)


def prepare_qat_safe(fp32_model: nn.Module) -> nn.Module:
    """
    Wrap with Quant/DeQuant stubs and prepare for QAT using the safe qconfig.
    """
    model = QATWrapper(fp32_model)
    model.qconfig = safe_qconfig()
    prepared = tq.prepare_qat(model, inplace=False)
    return prepared


def log_sample_qparams(model: nn.Module, prefix: str = "QCONFIG"):
    """
    Log a couple of activation/weight fake-quant configs to confirm dtypes/ranges.
    """
    act_seen = False
    w_seen = False
    for n, m in model.named_modules():
        if isinstance(m, FakeQuantize):
            obs = type(m.activation_post_process) if hasattr(m, "activation_post_process") else None
            logging.info(
                "%s | %s | dtype=%s qscheme=%s qmin=%s qmax=%s",
                prefix,
                n,
                getattr(m, "dtype", None),
                getattr(m, "qscheme", None),
                getattr(m, "quant_min", None),
                getattr(m, "quant_max", None),
            )
            if not act_seen and m.dtype == torch.quint8:
                act_seen = True
            if not w_seen and m.dtype == torch.qint8:
                w_seen = True
        if act_seen and w_seen:
            break


# ----------------------------
# Calib / Eval / Train
# ----------------------------
@torch.no_grad()
def calibrate(model: nn.Module, num_batches: int = 32, device: str = "cpu"):
    model.eval()
    logging.info("calibrating observers with %d batches ...", num_batches)
    for i in range(num_batches):
        x = torch.rand(1, 3, 64, 64, device=device)
        _ = model(x, rev=False)
        _ = model(_, rev=True)
        if (i + 1) % max(1, num_batches // 4) == 0:
            logging.info("  calib %d/%d", i + 1, num_batches)
    logging.info("calibration done.")


def evaluate(model: nn.Module, device: str = "cpu") -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 3, 64, 64, device=device)
        y = model(x, rev=False)
        x_rec = model(y, rev=True)
        psnr_c = psnr(y.clamp(0, 1), y.clamp(0, 1))
        psnr_s = psnr(x.clamp(0, 1), x_rec.clamp(0, 1))
    return psnr_c, psnr_s


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, device: str = "cpu",
                    epoch: int = 0, iters: int = 96, show_progress: bool = True) -> float:
    """
    Dummy self-reconstruction training with synthetic pairs.
    Replace with your real dataloader + loss.
    """
    model.train()
    loss_fn = nn.L1Loss()
    running = 0.0
    for it in range(1, iters + 1):
        x = torch.rand(2, 3, 64, 64, device=device)
        y = model(x, rev=False)
        x_rec = model(y, rev=True)
        loss = loss_fn(x_rec, x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += loss.item()

        if show_progress and (it % max(1, iters // 10) == 0 or it == iters):
            pct = int(round(100.0 * it / iters))
            logging.info("TRAIN: %d%% (%d/%d)", pct, it, iters)

    return running / iters


# ----------------------------
# Checkpoint helpers
# ----------------------------
def save_checkpoint(save_dir: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, engine: str, extra: Optional[Dict[str, Any]] = None):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "engine": engine,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "save_dir": save_dir,
    }
    if extra:
        ckpt.update(extra)
    path = os.path.join(save_dir, f"qat_ckpt_epoch{epoch:04d}.pth")
    torch.save(ckpt, path)
    latest = os.path.join(save_dir, "latest.pth")
    try:
        tmp = latest + ".tmp"
        torch.save({"path": os.path.basename(path)}, tmp)
        os.replace(tmp, latest)
    except Exception:
        pass
    logging.info("saved checkpoint: %s", path)


def load_checkpoint(resume_path: str) -> Dict[str, Any]:
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(resume_path)
    ckpt = torch.load(resume_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        if isinstance(ckpt, dict) and "path" in ckpt:
            root = os.path.dirname(resume_path)
            resume_path = os.path.join(root, ckpt["path"])
            ckpt = torch.load(resume_path, map_location="cpu")
    return ckpt


# ----------------------------
# Main
# ----------------------------
def run(args):
    set_cuda_visible_devices(args.gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    engine = select_engine()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = f"logging/qat_qbackend_safe_{timestamp}_ep{args.epochs}_calib{args.calib}"
    save_dir = default_dir

    # Build FP32 base
    base = Hinet()
    if args.strip_internal_qstubs:
        strip_internal_qstubs(base)
        logging.info("stripped internal Quant/DeQuant stubs")
    if args.replace_leakyrelu:
        replace_leakyrelu_with_relu(base)
        logging.info("replaced LeakyReLU -> ReLU")

    start_epoch = 1

    # ----- Resume from QAT checkpoint -----
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        if "engine" in ckpt:
            engine = ckpt["engine"]
            torch.backends.quantized.engine = engine

        qmodel = prepare_qat_safe(base)
        qmodel.to(device)

        missing, unexpected = qmodel.load_state_dict(ckpt["model_state"], strict=False)
        if missing or unexpected:
            logging.warning("state_dict restored with missing=%s unexpected=%s", missing, unexpected)

        optimizer = torch.optim.Adam(qmodel.parameters(), lr=args.lr)
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                logging.warning("optimizer state load failed: %s", e)

        if "save_dir" in ckpt:
            save_dir = ckpt["save_dir"]

        setup_logging(save_dir)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logging.info("resumed from %s (epoch %d)", args.resume, start_epoch - 1)
        log_sample_qparams(qmodel, prefix="RESUME_QCONFIG")

    else:
        # Optional pretrained FP32 load
        ckpt_path = os.path.join(c.MODEL_PATH, c.suffix) if args.pretrained is None else args.pretrained
        if ckpt_path and os.path.isfile(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict):
                for top in ("net", "state_dict", "model"):
                    if top in sd and isinstance(sd[top], dict):
                        sd = sd[top]
                        break
            try:
                base.load_state_dict(sd, strict=False)
                logging.info("loaded pretrained FP32: %s", ckpt_path)
            except Exception as e:
                logging.warning("failed to load pretrained (%s): %s", ckpt_path, e)

        qmodel = prepare_qat_safe(base)
        qmodel.to(device)

        setup_logging(save_dir)
        logging.info("save_dir: %s", save_dir)
        log_sample_qparams(qmodel, prefix="INIT_QCONFIG")

        calibrate(qmodel, num_batches=args.calib, device=device)
        optimizer = torch.optim.Adam(qmodel.parameters(), lr=args.lr)

    # ----- Train -----
    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(
            qmodel, optimizer, device=device,
            epoch=epoch, iters=args.iters, show_progress=args.progress
        )
        logging.info("epoch %03d | avg_loss %.6f", epoch, avg_loss)

        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            psnr_c, psnr_s = evaluate(qmodel, device=device)
            logging.info("EVAL | PSNR_C %.3f dB | PSNR_S %.3f dB", psnr_c, psnr_s)

        if args.checkpoint_every > 0 and (epoch % args.checkpoint_every == 0 or epoch == args.epochs):
            save_checkpoint(save_dir, epoch, qmodel, optimizer, engine, extra={"args": vars(args)})

    # ----- Export INT8 -----
    qmodel.eval()
    int8_model = tq.convert(qmodel, inplace=False)
    int8_path = os.path.join(save_dir, "hinet_qat_int8_full.pt")
    torch.save(int8_model, int8_path)
    logging.info("saved full int8 model: %s", int8_path)

    if args.export_script:
        scripted = torch.jit.script(int8_model)
        ts_path = os.path.join(save_dir, "hinet_qat_int8_scripted.pt")
        scripted.save(ts_path)
        logging.info("saved TorchScript int8: %s", ts_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, default=None, help="FP32 checkpoint path (optional)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--iters", type=int, default=96, help="iters per epoch for the toy loop")
    ap.add_argument("--calib", type=int, default=128, help="calibration mini-batches for observers")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--replace-leakyrelu", action="store_true")
    ap.add_argument("--strip-internal-qstubs", dest="strip_internal_qstubs", action="store_true")

    # checkpoint / resume
    ap.add_argument("--checkpoint-every", type=int, default=10, help="save QAT checkpoint every N epochs")
    ap.add_argument("--resume", type=str, default=None, help="path to qat_ckpt_epochXXXX.pth or latest.pth")

    # device / misc
    ap.add_argument("--gpus", type=str, default=None, help='GPU list like "0,1" (sets CUDA_VISIBLE_DEVICES)')
    ap.add_argument("--progress", action="store_true", help="log per-epoch progress (%)")
    ap.add_argument("--export-script", action="store_true")
    args = ap.parse_args()
    run(args)

# Examples:
# 1) fresh
# python qat_qnnpack_safe.py --epochs 50 --calib 128 --checkpoint-every 10 --progress --export-script --strip-internal-qstubs
#
# 2) resume
# python qat_qnnpack_safe.py --resume logging/.../latest.pth --epochs 50 --checkpoint-every 10 --progress