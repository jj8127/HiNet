# evaluate_QAT_model.py
# - 커버 해상도 유지(풀 이미지), 시크릿만 리사이즈 + 패딩(짝수화)
# - 모델 로딩: 1) 같은 폴더의 last.pth 있으면 QAT -> INT8 재구성(CPU, fbgemm/qnnpack)
#             2) 없으면 eager .pt 그대로 사용(추가 .to()/.eval() 안 함)
# - 저장: /root/Desktop/HiNet/image/{cover|secret|steg|secret-rev}/{모델파일명}/{0000.png,...}
# - CSV/플롯: 모델 파일과 같은 디렉토리에 저장 (eval_metrics.csv, eval_metrics.png)
# - 플롯 형식: 1x5 패널 (PSNR_C, PSNR_R, SSIM_C, SSIM_R, SSIM_AVG)

import os
import math
import csv
from datetime import datetime
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config as c
from hinet import Hinet
from invblock import INV_block
from rrdb_denselayer import ResidualDenseBlock_out

# ===== 사용자 변경 구역 =====
MODEL_PATH = "/root/Desktop/HiNet/model/pretrained_QAT_scripted.pt"
BASE_OUT_DIR = "/root/Desktop/HiNet/image"
# ==========================

def logit(msg: str):
    print(f"{datetime.now():%y-%m-%d %H:%M:%S} - INFO: {msg}")

# 저장 당시 클래스 이름을 맞추기 위한 안전망
class QATWrapper(nn.Module):
    def __init__(self, core: nn.Module = None):
        super().__init__()
        self.quant = tq.QuantStub()
        self.core = core if core is not None else Hinet()
        self.dequant = tq.DeQuantStub()
    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        xq = self.quant(x)
        yq = self.core(xq, rev=rev)
        y = self.dequant(yq)
        return y

# -------------- DWT / iDWT --------------
@torch.no_grad()
def haar_dwt(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    # DWT는 짝수 해상도 필요
    if (H % 2) != 0 or (W % 2) != 0:
        pad_h = (0, (W % 2 != 0))  # dummy to avoid flake, not used
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)

@torch.no_grad()
def inv_haar_dwt(d: torch.Tensor) -> torch.Tensor:
    B, C4, H2, W2 = d.shape
    C = C4 // 4
    ll, lh, hl, hh = torch.chunk(d, 4, dim=1)
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5
    out = torch.zeros((B, C, H2 * 2, W2 * 2), device=d.device, dtype=d.dtype)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    return out

def make_input(secret: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
    sd = haar_dwt(secret)
    cd = haar_dwt(cover)
    # 학습과 동일한 채널 순서: [secret_dwt, cover_dwt]
    return torch.cat([sd, cd], dim=1)

# -------------- Metrics --------------
def psnr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

def ssim_simple(a: torch.Tensor, b: torch.Tensor, C1=0.01 ** 2, C2=0.03 ** 2) -> float:
    mu_x = a.mean().item()
    mu_y = b.mean().item()
    var_x = a.var(unbiased=False).item()
    var_y = b.var(unbiased=False).item()
    cov = ((a - mu_x) * (b - mu_y)).mean().item()
    num = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)
    return float(num / den) if den != 0 else 1.0

# -------------- IO utils --------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_uint8_image(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1)
    arr = (t[0].detach().cpu() * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)

def save_image_tensor(t: torch.Tensor, out_dir: str, fname: str):
    ensure_dir(out_dir)
    to_uint8_image(t).save(os.path.join(out_dir, fname))

# --------- Pair prep: cover full-res / secret resize+pad-even ----------
def _pad_to_even_pil(img: Image.Image) -> Image.Image:
    w, h = img.size
    pad_right = 1 if (w % 2) else 0
    pad_bottom = 1 if (h % 2) else 0
    if pad_right == 0 and pad_bottom == 0:
        return img
    new_im = Image.new("RGB", (w + pad_right, h + pad_bottom))
    new_im.paste(img, (0, 0))
    # 가장자리 복제
    if pad_right:
        strip = img.crop((w - 1, 0, w, h)).resize((1, h))
        new_im.paste(strip, (w, 0))
    if pad_bottom:
        strip = img.crop((0, h - 1, w, h)).resize((w, 1))
        new_im.paste(strip, (0, h))
    if pad_right and pad_bottom:
        pix = img.getpixel((w - 1, h - 1))
        new_im.putpixel((w, h), pix)
    return new_im

def prepare_pair_full_cover(secret_img: Image.Image, cover_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    # 커버 해상도 유지
    Wc, Hc = cover_img.size
    # 시크릿을 커버에 맞춰 리사이즈
    secret_resized = secret_img.resize((Wc, Hc), Image.BICUBIC)
    # 짝수화(둘 다 DWT 필요)
    cover_even  = _pad_to_even_pil(cover_img)
    secret_even = _pad_to_even_pil(secret_resized)
    return secret_even, cover_even

# -------------- Data --------------
def load_eval_list(secret_dir: str, cover_dir: str, fmt: str) -> List[Tuple[str, str]]:
    import glob
    from natsort import natsorted
    s_list = natsorted(glob.glob(os.path.join(secret_dir, f"*.{fmt}")))
    c_list = natsorted(glob.glob(os.path.join(cover_dir,  f"*.{fmt}")))
    n = min(len(s_list), len(c_list))
    return list(zip(s_list[:n], c_list[:n]))

# -------------- Model loading --------------
def choose_backend() -> str:
    engines = torch.backends.quantized.supported_engines
    if "fbgemm" in engines:
        return "fbgemm"
    if "qnnpack" in engines:
        return "qnnpack"
    return engines[0] if engines else "fbgemm"

def build_int8_from_checkpoint(ckpt_path: str) -> nn.Module:
    backend = choose_backend()
    torch.backends.quantized.engine = backend
    logit(f"[convert] quant backend: {backend}")

    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model_state"]

    qat_model = QATWrapper(Hinet())
    missing, unexpected = qat_model.load_state_dict(sd, strict=False)
    if missing:
        logit(f"[convert] missing keys: {len(missing)}")
    if unexpected:
        logit(f"[convert] unexpected keys: {len(unexpected)}")

    qat_model.eval()
    int8_model = tq.convert(qat_model, inplace=False)  # CPU에서 변환
    return int8_model

def load_model_any(model_path: str) -> nn.Module:
    model_dir = os.path.dirname(model_path)
    sib = os.path.join(model_dir, "last.pth")
    if os.path.isfile(sib):
        logit(f"[load] use sibling checkpoint: {sib}")
        try:
            return build_int8_from_checkpoint(sib)
        except Exception as e:
            logit(f"[load] convert-from-checkpoint failed: {e}. fallback to eager .pt")

    obj = torch.load(model_path, map_location="cpu")
    logit(f"[load] eager object loaded (CPU): {model_path}")
    return obj  # 추가 .to()/.eval() 호출 안 함

# -------------- Main --------------
def main():
    # 데이터 경로
    secret_dir = os.path.abspath(c.VAL_PATH.rstrip("/"))
    cover_dir  = os.path.abspath(c.VAL_COVER_PATH.rstrip("/"))
    fmt = getattr(c, "format_val", "png")
    pairs = load_eval_list(secret_dir, cover_dir, fmt)

    print(f"Eval data - secret_dir: {secret_dir}")
    print(f"Eval data - cover_dir : {cover_dir}")
    print(f"Eval data - format    : {fmt}")
    print(f"Total pairs used      : {len(pairs)}")

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = load_model_any(MODEL_PATH)  # CPU 추론 객체

    # 저장 경로
    model_tag = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    out_cover_dir     = os.path.join(BASE_OUT_DIR, "cover", model_tag)
    out_secret_dir    = os.path.join(BASE_OUT_DIR, "secret", model_tag)
    out_steg_dir      = os.path.join(BASE_OUT_DIR, "steg", model_tag)
    out_secretrev_dir = os.path.join(BASE_OUT_DIR, "secret-rev", model_tag)
    for d in [out_cover_dir, out_secret_dir, out_steg_dir, out_secretrev_dir]:
        ensure_dir(d)

    # 메트릭 모음
    psnr_c_all, psnr_r_all = [], []
    ssim_c_all, ssim_r_all = [], []

    to_tensor = T.ToTensor()

    with torch.no_grad():
        for idx, (sp, cp) in enumerate(pairs):
            s_img_raw = Image.open(sp).convert("RGB")
            c_img_raw = Image.open(cp).convert("RGB")

            # 커버 풀해상도 유지, 시크릿 리사이즈+짝수화(패딩)
            s_img, c_img = prepare_pair_full_cover(s_img_raw, c_img_raw)

            sb = to_tensor(s_img).unsqueeze(0)  # [1,3,H,W]
            cb = to_tensor(c_img).unsqueeze(0)

            x = make_input(sb, cb)             # [1,24,h,w]
            y = model(x, rev=False)            # 추론

            # 분리 규칙: (앞=secret_dwt, 뒤=cover_dwt) 이면?  → make_input에서 [secret, cover]로 합쳤으므로
            # 모델 출력도 동일 정렬을 가정: y = [secret_dwt', cover_dwt']
            y_s_dwt, y_c_dwt = torch.chunk(y, 2, dim=1)
            x_s_dwt, x_c_dwt = torch.chunk(x, 2, dim=1)

            # 공간영역 복원
            secret_ref = inv_haar_dwt(x_s_dwt)
            cover_ref  = inv_haar_dwt(x_c_dwt)
            secret_rev = inv_haar_dwt(y_s_dwt)
            steg_img   = inv_haar_dwt(y_c_dwt)

            # 메트릭
            psnr_c = psnr(steg_img, cover_ref)
            psnr_r = psnr(secret_rev, secret_ref)
            ssim_c = ssim_simple(steg_img, cover_ref)
            ssim_r = ssim_simple(secret_rev, secret_ref)

            psnr_c_all.append(psnr_c); psnr_r_all.append(psnr_r)
            ssim_c_all.append(ssim_c); ssim_r_all.append(ssim_r)

            # 저장(0000.png)
            name = f"{idx:04d}.png"
            save_image_tensor(cover_ref,  out_cover_dir,  name)
            save_image_tensor(secret_ref, out_secret_dir, name)
            save_image_tensor(steg_img,   out_steg_dir,   name)        # stego → steg/
            save_image_tensor(secret_rev, out_secretrev_dir, name)     # recovered secret → secret-rev/

    # 집계
    avg_psnr_c = sum(psnr_c_all) / max(1, len(psnr_c_all))
    avg_psnr_r = sum(psnr_r_all) / max(1, len(psnr_r_all))
    avg_ssim_c = sum(ssim_c_all) / max(1, len(ssim_c_all))
    avg_ssim_r = sum(ssim_r_all) / max(1, len(ssim_r_all))
    avg_ssim   = 0.5 * (avg_ssim_c + avg_ssim_r)
    logit(f"AVERAGE | PSNR_C {avg_psnr_c:.3f} | PSNR_R {avg_psnr_r:.3f} | "
          f"SSIM_C {avg_ssim_c:.6f} | SSIM_R {avg_ssim_r:.6f} | SSIM_AVG {avg_ssim:.6f}")

    # CSV/플롯: 모델 파일과 동일 폴더
    model_dir = os.path.dirname(MODEL_PATH)
    csv_path  = os.path.join(model_dir, "eval_metrics.csv")
    png_path  = os.path.join(model_dir, "eval_metrics.png")

    # CSV 저장
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "PSNR_C(dB)", "PSNR_R(dB)", "SSIM_C", "SSIM_R"])
        for i, (pc, pr, sc, sr) in enumerate(zip(psnr_c_all, psnr_r_all, ssim_c_all, ssim_r_all)):
            w.writerow([i, f"{pc:.6f}", f"{pr:.6f}", f"{sc:.6f}", f"{sr:.6f}"])
        w.writerow([])
        w.writerow(["AVERAGE", f"{avg_psnr_c:.6f}", f"{avg_psnr_r:.6f}",
                    f"{avg_ssim_c:.6f}", f"{avg_ssim_r:.6f}", "SSIM_AVG", f"{avg_ssim:.6f}"])
    logit(f"saved CSV: {csv_path}")

    # ===== 플롯: 1x5 패널 (예시 이미지 스타일) =====
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(psnr_c_all)))
    ssim_avg_all = [(a + b) * 0.5 for a, b in zip(ssim_c_all, ssim_r_all)]

    fig, axes = plt.subplots(1, 5, figsize=(24, 4), dpi=160)
    titles = ["PSNR_C", "PSNR_R", "SSIM_C", "SSIM_R", "SSIM_AVG"]
    series = [psnr_c_all, psnr_r_all, ssim_c_all, ssim_r_all, ssim_avg_all]
    ylabels = ["PSNR (dB)", "PSNR (dB)", "SSIM", "SSIM", "SSIM"]

    for ax, title, vals, yl in zip(axes, titles, series, ylabels):
        ax.plot(xs, vals, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Image Index")
        ax.set_ylabel(yl)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    logit(f"saved plot: {png_path}")

if __name__ == "__main__":
    main()