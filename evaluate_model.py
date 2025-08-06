import argparse
import os
import csv
import glob
from typing import List

import torch
import numpy as np
from model import Model, init_model
import config as c
from torch.utils.data import DataLoader
import datasets
import modules.Unet_common as common
from calculate_PSNR_SSIM import calculate_psnr, calculate_ssim


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load model weights from checkpoint."""
    state_dicts = torch.load(ckpt_path, map_location="cpu")
    network_state_dict = {
        k: v for k, v in state_dicts["net"].items() if "tmp_var" not in k
    }
    model.load_state_dict(network_state_dict)


def verify_pairs(secret_files: List[str], cover_files: List[str]) -> None:
    """Ensure image files match by name and count."""
    if len(secret_files) != len(cover_files):
        raise ValueError(
            f"Mismatched image counts: {len(secret_files)} secret vs {len(cover_files)} cover"
        )
    for s, c_path in zip(secret_files, cover_files):
        if os.path.basename(s) != os.path.basename(c_path):
            raise ValueError(
                f"Filename mismatch: {os.path.basename(s)} vs {os.path.basename(c_path)}"
            )


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor in [0,1] to uint8 numpy array HxWxC."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img


def evaluate(model_path: str) -> str:
    device = torch.device("cpu")
    net = Model().to(device)
    init_model(net)
    load_checkpoint(net, model_path)
    net.eval()

    dwt = common.DWT()
    iwt = common.IWT()

    secret_files = sorted(glob.glob(os.path.join(c.VAL_PATH, f"*.{c.format_val}")))
    cover_files = sorted(glob.glob(os.path.join(c.VAL_COVER_PATH, f"*.{c.format_val}")))
    verify_pairs(secret_files, cover_files)

    dataset = datasets.HinetDataset(
        c.VAL_PATH, c.VAL_COVER_PATH, datasets.transform_val, c.format_val
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    psnr_c_list = []
    psnr_r_list = []
    ssim_list = []

    with torch.no_grad():
        for secret, cover in loader:
            secret = secret.to(device)
            cover = cover.to(device)

            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)

            output = net(input_img)
            output_steg = output[:, : 4 * c.channels_in]
            output_z = output[:, 4 * c.channels_in :]
            steg_img = iwt(output_steg)
            backward_z = torch.randn_like(output_z)

            output_rev = torch.cat((output_steg, backward_z), 1)
            backward_img = net(output_rev, rev=True)
            secret_rev = iwt(backward_img[:, 4 * c.channels_in :])

            cover_np = tensor_to_image(cover[0])
            steg_np = tensor_to_image(steg_img[0])
            secret_np = tensor_to_image(secret[0])
            secret_rev_np = tensor_to_image(secret_rev[0])

            psnr_c = calculate_psnr(cover_np, steg_np)
            psnr_r = calculate_psnr(secret_np, secret_rev_np)
            ssim_c = calculate_ssim(cover_np, steg_np)
            ssim_r = calculate_ssim(secret_np, secret_rev_np)
            ssim_val = (ssim_c + ssim_r) / 2

            psnr_c_list.append(psnr_c)
            psnr_r_list.append(psnr_r)
            ssim_list.append(ssim_val)

    assert len(psnr_c_list) == len(secret_files)

    avg_psnr_c = sum(psnr_c_list) / len(psnr_c_list)
    avg_psnr_r = sum(psnr_r_list) / len(psnr_r_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    csv_name = f"evaluation_{model_name}.csv"
    csv_path = os.path.join(os.getcwd(), csv_name)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "PSNR_c", "PSNR_r", "SSIM"])
        for i, (p_c, p_r, s) in enumerate(zip(psnr_c_list, psnr_r_list, ssim_list), start=1):
            writer.writerow([i, f"{p_c:.6f}", f"{p_r:.6f}", f"{s:.6f}"])
        writer.writerow([
            "Average",
            f"{avg_psnr_c:.6f}",
            f"{avg_psnr_r:.6f}",
            f"{avg_ssim:.6f}",
        ])

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate model PSNR/SSIM")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    csv_path = evaluate(args.model)
    print(csv_path)


if __name__ == "__main__":
    main()