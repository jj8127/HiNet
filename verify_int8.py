#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import platform
import time
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.ao.quantization as tq

# ====== eager 저장물 언피클용 심볼 ======
class QATWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        x = self.quant(x)
        y = self.core(x, rev=rev)
        y = self.dequant(y)
        if isinstance(y, torch.Tensor) and getattr(y, "is_quantized", False):
            y = y.dequantize()
        return y

# 심볼 로딩(언피클 시 필요)
try:
    import hinet  # noqa: F401
    import invblock  # noqa: F401
    import rrdb_denselayer  # noqa: F401
    import config as c
except Exception:
    class c:  # 최소 더미
        channels_in = 3

# ====== 엔진 선택 ======
def _select_engine() -> str:
    engines = getattr(torch.backends.quantized, "supported_engines", [])
    mach = platform.machine().lower()
    if ("x86_64" in mach or "amd64" in mach) and "fbgemm" in engines:
        return "fbgemm"
    if ("arm" in mach or "aarch64" in mach) and "qnnpack" in engines:
        return "qnnpack"
    for cand in ("fbgemm", "qnnpack"):
        if cand in engines:
            return cand
    return engines[0] if engines else "none"

try:
    torch.backends.quantized.engine = _select_engine()
except Exception:
    pass

# ====== 로깅 ======
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )

# ====== 유틸 ======
def _is_zip(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == b"PK\x03\x04"
    except Exception:
        return False

# ====== Eager 모델 리포트/체크 ======
def list_quantized_modules(model: nn.Module):
    names = []
    for m in model.modules():
        mod = m.__class__.__module__.lower()
        cls = m.__class__.__name__
        if "quantized" in mod or "quantized" in cls.lower():
            names.append(cls)
    return names

def eager_quant_report(model: nn.Module) -> Dict[str, Any]:
    report: Dict[str, Any] = {"engine": getattr(torch.backends.quantized, "engine", "unknown")}
    qmods = list_quantized_modules(model)
    qconvs = []
    qz = []
    dqz = []
    fconvs = 0
    for m in model.modules():
        name = m.__class__.__name__
        modpath = m.__class__.__module__.lower()
        if isinstance(m, nn.Conv2d):
            fconvs += 1
        if name == "QuantizedConv2d" or (name.endswith("Conv2d") and "quantized" in modpath):
            qconvs.append(name)
        if name == "Quantize":
            qz.append(name)
        if name == "DeQuantize":
            dqz.append(name)
    report.update({
        "num_float_conv2d": fconvs,
        "num_quantized_modules": len(qmods),
        "num_quantized_conv2d": len(qconvs),
        "num_quantize_modules": len(qz),
        "num_dequantize_modules": len(dqz),
    })
    # qparams 샘플
    sampled = []
    for m in model.modules():
        if m.__class__.__name__ == "QuantizedConv2d":
            try:
                w = m.weight()
                qs = str(w.qscheme())
                scales = (w.q_per_channel_scales().tolist()[:4]
                          if hasattr(w, "q_per_channel_scales")
                          else [float(w.q_scale())])
                zps = (w.q_per_channel_zero_points().tolist()[:4]
                       if hasattr(w, "q_per_channel_zero_points")
                       else [int(w.q_zero_point())])
                sampled.append({"module": "QuantizedConv2d", "qscheme": qs,
                                "scales_head": scales, "zps_head": zps})
            except Exception:
                pass
        if len(sampled) >= 10:
            break
    report["sampled_qparams"] = sampled
    return report

def eager_assert_int8(model: nn.Module):
    for p in model.parameters(recurse=True):
        if p.is_cuda:
            raise RuntimeError("INT8 모델이 CUDA에 올라가 있음. convert 후에는 CPU에 있어야 합니다.")
    if not list_quantized_modules(model):
        raise RuntimeError("quantized 모듈이 없음. prepare/convert 흐름 점검 필요")

    cin = int(getattr(c, "channels_in", 3))
    ch_in = 8 * cin
    with torch.inference_mode():
        x = torch.randn(1, ch_in, 64, 64)
        y = model(x)
        ch4 = 4 * cin
        if y.dim() == 4 and y.size(1) >= ch4:
            z = torch.randn(y.size(0), y.size(1) - ch4, y.size(2), y.size(3))
            rev_in = torch.cat((y.narrow(1, 0, ch4), z), 1)
            model(rev_in, rev=True)

# ====== TorchScript 리포트/체크 (그래프 안전 접근) ======
def ts_quant_report(tsm) -> Dict[str, Any]:
    report: Dict[str, Any] = {"engine": getattr(torch.backends.quantized, "engine", "unknown")}

    # op 존재 여부는 export_opnames로, 대략적인 개수는 IR 문자열로 카운트
    try:
        opnames = set(torch.jit.export_opnames(tsm))
    except Exception:
        opnames = set()

    try:
        ir = tsm._c.dump_to_str()
    except Exception:
        ir = ""

    def c(substr: str) -> int:
        return ir.count(substr)

    present_quant = {
        "quantized::conv2d": any("quantized::conv2d" in n for n in opnames),
        "quantized::conv2d_relu": any("quantized::conv2d_relu" in n for n in opnames),
        "aten::quantize_per_tensor": any("aten::quantize_per_tensor" in n for n in opnames),
        "aten::quantize_per_channel": any("aten::quantize_per_channel" in n for n in opnames),
        "aten::dequantize": any("aten::dequantize" in n for n in opnames),
    }

    counts = {
        "quantized::conv2d": c("quantized::conv2d"),
        "quantized::conv2d_relu": c("quantized::conv2d_relu"),
        "aten::quantize_per_tensor": c("aten::quantize_per_tensor"),
        "aten::quantize_per_channel": c("aten::quantize_per_channel"),
        "aten::dequantize": c("aten::dequantize"),
    }

    report.update({
        "present": present_quant,
        "counts_from_ir": counts,
        "num_unique_ops": len(opnames),
    })
    return report

def ts_assert_int8(tsm):
    rep = ts_quant_report(tsm)
    has_q = (
        rep["present"]["quantized::conv2d"]
        or rep["present"]["quantized::conv2d_relu"]
        or rep["present"]["aten::quantize_per_tensor"]
        or rep["present"]["aten::quantize_per_channel"]
    )
    if not has_q:
        raise RuntimeError("TorchScript 모듈에서 양자화 연산을 발견하지 못했습니다.")
    # 스모크
    cin = int(getattr(c, "channels_in", 3))
    ch_in = 8 * cin
    with torch.inference_mode():
        x = torch.randn(1, ch_in, 64, 64)
        _ = tsm(x)

# ====== 메인 ======
def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str, help="hinet_qat_int8_full.pt 또는 hinet_qat_int8_scripted.pt")
    ap.add_argument("--scripted", action="store_true", help="TorchScript 포맷 강제")
    args = ap.parse_args()

    path = args.model_path
    if not os.path.isfile(path):
        raise SystemExit(f"파일이 없습니다: {path}")

    logging.info(f"loading: {path}")

    is_zip = _is_zip(path)
    model = None
    is_ts = False

    # 우선 힌트를 따름
    if args.scripted or is_zip:
        try:
            model = torch.jit.load(path, map_location="cpu").eval()
            is_ts = True
        except Exception as e:
            if args.scripted:
                raise
            logging.info(f"torchscript 로드 실패({e}); eager 로 재시도")
            model = torch.load(path, map_location="cpu")
            if isinstance(model, nn.Module):
                model = model.eval()
            is_ts = isinstance(model, torch.jit.ScriptModule)
    else:
        try:
            model = torch.load(path, map_location="cpu")
            if isinstance(model, nn.Module):
                model = model.eval()
            is_ts = isinstance(model, torch.jit.ScriptModule)
        except Exception as e:
            logging.info(f"eager 로드 실패({e}); torchscript 로 재시도")
            model = torch.jit.load(path, map_location="cpu").eval()
            is_ts = True

    # ===== 진행 로그 =====
    logging.info("step 1/4: inventory")
    if is_ts:
        rep = ts_quant_report(model)
        pres = rep["present"]
        cnts = rep["counts_from_ir"]
        logging.info(f" - engine: {rep['engine']}")
        logging.info(f" - present: "
                     f"qconv2d={pres['quantized::conv2d']}, "
                     f"qconv2d_relu={pres['quantized::conv2d_relu']}, "
                     f"q_per_tensor={pres['aten::quantize_per_tensor']}, "
                     f"q_per_channel={pres['aten::quantize_per_channel']}, "
                     f"dequant={pres['aten::dequantize']}")
        logging.info(f" - counts(IR): {cnts}")
    else:
        rep = eager_quant_report(model)
        logging.info(f" - engine: {rep['engine']}")
        logging.info(f" - float conv2d: {rep['num_float_conv2d']}")
        logging.info(f" - quantized conv2d: {rep['num_quantized_conv2d']}")
        logging.info(f" - quantized modules: {rep['num_quantized_modules']}")
        logging.info(f" - boundary Quantize/DeQuantize: {rep['num_quantize_modules']}/{rep['num_dequantize_modules']}")
        if rep["sampled_qparams"]:
            logging.info(f" - qparams sample[0]: {rep['sampled_qparams'][0]}")

    logging.info("step 2/4: smoke")
    if is_ts:
        ts_assert_int8(model)
    else:
        eager_assert_int8(model)
    logging.info(" - int8 forward OK")

    logging.info("step 3/4: timing")
    cin = int(getattr(c, "channels_in", 3))
    x = torch.randn(1, 8 * cin, 128, 128)
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = model(x)
    t1 = time.perf_counter()
    logging.info(f" - 1x forward(128x128): {1000*(t1-t0):.2f} ms")

    logging.info("step 4/4: verdict")
    if is_ts:
        pres = rep["present"]
        if any(pres.values()):
            logging.info("VERDICT: TorchScript 그래프에 양자화 연산이 존재합니다 ✅")
        else:
            logging.info("VERDICT: 양자화 연산을 찾지 못했습니다 ❌ (스크립팅/컨버트 경로 점검)")
    else:
        if rep["num_quantized_conv2d"] > 0 or rep["num_quantize_modules"] > 0:
            logging.info("VERDICT: Eager 모델에 양자화 모듈이 존재합니다 ✅")
        else:
            logging.info("VERDICT: 양자화 모듈을 찾지 못했습니다 ❌ (prepare/convert 경로 점검)")

if __name__ == "__main__":
    main()