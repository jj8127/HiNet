import argparse
import glob
import os
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torchvision.utils as vutils
import numpy as np
import cv2

from hinet import Hinet
from invblock import INV_block
import modules.Unet_common as common
import datasets
import config as c


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mark_quant_layers(module: nn.Module):
    for child in module.children():
        if isinstance(child, INV_block):
            continue
        if isinstance(child, nn.Conv2d):
            child.qconfig = tq.get_default_qat_qconfig("fbgemm")
        mark_quant_layers(child)


def load_quantized(model_path: str) -> nn.Module:
    model = Hinet()
    mark_quant_layers(model)
    tq.prepare_qat(model, inplace=True)
    model = tq.convert(model.eval(), inplace=True)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def pair_batches(loader):
    buffer = []
    for batch in loader:
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        for img in batch:
            buffer.append(img)
            if len(buffer) == 2:
                yield torch.stack(buffer, 0)
                buffer = []
    if buffer:
        print("Warning: ignoring the last image without a pair")


def gauss_noise(shape):
    return torch.randn(shape, device=device)


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()


def bgr2ycbcr(img: np.ndarray, only_y: bool = True) -> np.ndarray:
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = (
            np.matmul(
                img,
                [
                    [24.966, 112.0, -18.214],
                    [128.553, -74.203, -93.786],
                    [65.481, -37.797, 112.0],
                ],
            )
            / 255.0
            + [16, 128, 128]
        )
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.mean([ssim(img1[..., i], img2[..., i]) for i in range(3)])
        elif img1.shape[2] == 1:
            return ssim(img1.squeeze(), img2.squeeze())
    else:
        raise ValueError("Wrong input image dimensions")


def run(model: nn.Module):
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    loader = pair_batches(datasets.testloader)
    psnr_cover_scores = []
    psnr_secret_scores = []
    ssim_cover_scores = []
    ssim_secret_scores = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            cover = batch[1:]
            secret = batch[:1]
            cover_in = dwt(cover)
            secret_in = dwt(secret)
            input_img = torch.cat((cover_in, secret_in), 1)
            output = model(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)
            backward_z = gauss_noise(output_z.shape)
            rev_input = torch.cat((output_steg, backward_z), 1)
            backward = model(rev_input, rev=True)
            secret_rev = iwt(backward.narrow(1, 4 * c.channels_in, backward.shape[1] - 4 * c.channels_in))

            # save images
            vutils.save_image(cover, os.path.join(c.IMAGE_PATH_cover, f"{i:05d}.png"))
            vutils.save_image(secret, os.path.join(c.IMAGE_PATH_secret, f"{i:05d}.png"))
            vutils.save_image(steg_img, os.path.join(c.IMAGE_PATH_steg, f"{i:05d}.png"))
            vutils.save_image(secret_rev, os.path.join(c.IMAGE_PATH_secret_rev, f"{i:05d}.png"))

            # metrics in Y channel
            cover_np = cover.squeeze(0).permute(1, 2, 0).cpu().numpy()
            steg_np = steg_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            secret_np = secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
            rev_np = secret_rev.squeeze(0).permute(1, 2, 0).cpu().numpy()

            cover_y = bgr2ycbcr(cover_np, only_y=True)
            steg_y = bgr2ycbcr(steg_np, only_y=True)
            secret_y = bgr2ycbcr(secret_np, only_y=True)
            rev_y = bgr2ycbcr(rev_np, only_y=True)

            psnr_cover_scores.append(psnr(steg_img, cover))
            psnr_secret_scores.append(psnr(secret_rev, secret))
            ssim_cover_scores.append(calculate_ssim(cover_y, steg_y))
            ssim_secret_scores.append(calculate_ssim(secret_y, rev_y))

    print(
        f"PSNR cover: {np.mean(psnr_cover_scores):.2f} dB, secret: {np.mean(psnr_secret_scores):.2f} dB"
    )
    print(
        f"SSIM cover: {np.mean(ssim_cover_scores):.4f}, secret: {np.mean(ssim_secret_scores):.4f}"
    )

def main(model_path: str):
    model = load_quantized(model_path)
    run(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quantized HiNet model for testing")
    parser.add_argument("--model", type=str, default=None, help="path to quantized model")
    args = parser.parse_args()

    if args.model is None:
        files = sorted(glob.glob(os.path.join("model", "model_qat_*.pt")))
        if not files:
            raise FileNotFoundError("No quantized model found in ./model")
        args.model = files[-1]
        print(f"loading latest model {args.model}")

    main(args.model)