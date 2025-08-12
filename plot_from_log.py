# plot_from_log.py
# -*- coding: utf-8 -*-
"""
HiNet QAT 로그에서 epoch별 psnr_c / psnr_r(psnr_s) / loss 를 읽어
하나의 이미지(3 서브플롯)로 저장합니다.

경로는 코드 상단의 상수로 지정합니다(명령행 인자 불필요).
- LOG_ROOT:     세션 폴더들이 모여있는 루트 (기본 ./logging)
- SESSION_NAME: 특정 세션 폴더 이름을 강제로 지정하고 싶을 때만 사용
- OUT_NAME:     저장될 이미지 파일명
"""

import os
import re
import glob
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt

# ---------- 경로 설정(코드에서 고정) ----------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_ROOT     = os.path.join(PROJECT_ROOT, "logging")        # 로그 루트
SESSION_NAME = "/root/Desktop/HiNet/logging/qat_qbackend_safe_20250812_161125_ep50_calib128"                                            # 예: "qat_qbackend_safe_20250812_161125_ep50_calib128"
OUT_NAME     = "metrics_from_log.png"                        # 저장 파일명
# -------------------------------------------


# 로그 한 줄에서 epoch / loss / psnr_c / psnr_r(or s) 추출
_EPOCH_RE = re.compile(
    r"""
    EPOCH\s+(\d+)\s*\|\s*                # EPOCH 001 |
    loss\s+([0-9]*\.?[0-9]+)\s*\|\s*     # loss 0.0123 |
    psnr_c\s+([0-9]*\.?[0-9]+)\s*\|\s*   # psnr_c 36.12 |
    psnr_(?:r|s)\s+([0-9]*\.?[0-9]+)     # psnr_r 38.90  (또는 psnr_s)
    """,
    re.IGNORECASE | re.VERBOSE,
)

def find_latest_session(log_root: str) -> str:
    """logging/ 아래에서 가장 최근에 수정된 세션 디렉토리를 찾음."""
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"log root not found: {log_root}")
    candidates = [d for d in glob.glob(os.path.join(log_root, "*")) if os.path.isdir(d)]
    if not candidates:
        raise FileNotFoundError(f"no sessions in: {log_root}")
    latest = max(candidates, key=os.path.getmtime)
    return latest

def load_log_lines(session_dir: str) -> List[str]:
    """세션 폴더에서 train.log 읽기."""
    log_path = os.path.join(session_dir, "train.log")
    if not os.path.isfile(log_path):
        # 혹시 다른 이름으로 저장되었으면 .log 검색
        logs = glob.glob(os.path.join(session_dir, "*.log"))
        if not logs:
            raise FileNotFoundError(f"train.log not found in: {session_dir}")
        log_path = max(logs, key=os.path.getmtime)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()

def parse_metrics(lines: List[str]) -> Tuple[List[int], List[float], List[float], List[float]]:
    """로그 라인들에서 epoch, loss, psnr_c, psnr_rs 배열 생성."""
    epochs, losses, psnr_c, psnr_rs = [], [], [], []
    for ln in lines:
        m = _EPOCH_RE.search(ln)
        if m:
            e   = int(m.group(1))
            lss = float(m.group(2))
            pc  = float(m.group(3))
            pr  = float(m.group(4))
            epochs.append(e)
            losses.append(lss)
            psnr_c.append(pc)
            psnr_rs.append(pr)
    return epochs, losses, psnr_c, psnr_rs

def plot_and_save(epochs: List[int],
                  losses: List[float],
                  psnr_c: List[float],
                  psnr_rs: List[float],
                  out_path: str) -> None:
    """1×3 서브플롯으로 그리고 저장."""
    if not epochs:
        raise RuntimeError("로그에서 epoch/metrics를 찾지 못했습니다. 로그 형식을 확인하세요.")

    # 정렬(혹시 섞여 있을 경우)
    zipped = sorted(zip(epochs, losses, psnr_c, psnr_rs), key=lambda x: x[0])
    ep, ls, pc, pr = map(list, zip(*zipped))

    fig = plt.figure(figsize=(22, 6))  # 가로로 넓게
    # PSNR_C
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(ep, pc, marker="o", linewidth=2)
    ax1.set_title("PSNR_C")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("dB")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # PSNR_R/S
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(ep, pr, marker="o", linewidth=2)
    ax2.set_title("PSNR_S" if "s" in _EPOCH_RE.pattern else "PSNR_R")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("dB")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Loss
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(ep, ls, marker="o", linewidth=2)
    ax3.set_title("Train Loss")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[OK] saved: {out_path}")

def main():
    # 세션 결정
    session_dir = (
        os.path.join(LOG_ROOT, SESSION_NAME)
        if SESSION_NAME
        else find_latest_session(LOG_ROOT)
    )
    out_path = os.path.join(session_dir, OUT_NAME)

    print(f"[INFO] session_dir: {session_dir}")
    lines = load_log_lines(session_dir)
    epochs, losses, psnr_c, psnr_rs = parse_metrics(lines)
    print(f"[INFO] parsed epochs: {len(epochs)}")
    plot_and_save(epochs, losses, psnr_c, psnr_rs, out_path)

if __name__ == "__main__":
    main()