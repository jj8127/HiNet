import os
import csv
import math
import torch
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
from calculate_PSNR_SSIM import calculate_psnr, calculate_ssim
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    state_dicts = torch.load(ckpt_path, map_location="cpu")
    # FP32 모델이 일반적으로 'net' 또는 'state_dict' 키로 저장됨
    if "net" in state_dicts:
        state = state_dicts["net"]
    elif "state_dict" in state_dicts:
        state = state_dicts["state_dict"]
    elif "model" in state_dicts and isinstance(state_dicts["model"], dict):
        state = state_dicts["model"]
    else:
        state = state_dicts
    new_state = {}
    for k, v in state.items():
        name = k
        if name.startswith("module."):
            name = name[len("module.") :]
        new_state[name] = v
    new_state = {k: v for k, v in new_state.items() if "tmp_var" not in k}
    model.load_state_dict(new_state, strict=False)
    model.to(device)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img

def evaluate_fp32_model(model_path: str):
    net = Model()
    load_checkpoint(net, model_path)
    net.eval()

    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)

    testloader = datasets.testloader
    psnr_c_list, psnr_r_list = [], []
    ssim_c_list, ssim_r_list, ssim_avg_list = [], [], []
    img_names = [f"{i+1:04d}.png" for i in range(len(testloader))]

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader), desc="Evaluating FP32"):
            data = data.to(device)
            cover = data[data.shape[0] // 2 :, :, :, :]
            secret = data[: data.shape[0] // 2, :, :, :]

            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)

            # Forward
            output = net(input_img)
            output_steg = output[:, : 4 * c.channels_in]
            output_z = output[:, 4 * c.channels_in :]
            steg_img = iwt(output_steg)
            backward_z = torch.randn_like(output_z).to(device)

            # Backward
            output_rev = torch.cat((output_steg, backward_z), 1)
            backward_img = net(output_rev, rev=True)
            secret_rev = iwt(backward_img[:, 4 * c.channels_in :])

            # Numpy 변환
            cover_np = tensor_to_image(cover[0])
            steg_np = tensor_to_image(steg_img[0])
            secret_np = tensor_to_image(secret[0])
            secret_rev_np = tensor_to_image(secret_rev[0])

            psnr_c = calculate_psnr(cover_np, steg_np)
            psnr_r = calculate_psnr(secret_np, secret_rev_np)
            ssim_c = calculate_ssim(cover_np, steg_np)
            ssim_r = calculate_ssim(secret_np, secret_rev_np)
            ssim_avg = (ssim_c + ssim_r) / 2

            psnr_c_list.append(psnr_c)
            psnr_r_list.append(psnr_r)
            ssim_c_list.append(ssim_c)
            ssim_r_list.append(ssim_r)
            ssim_avg_list.append(ssim_avg)

    # 평균 계산
    avg_psnr_c = sum(psnr_c_list) / len(psnr_c_list)
    avg_psnr_r = sum(psnr_r_list) / len(psnr_r_list)
    avg_ssim_c = sum(ssim_c_list) / len(ssim_c_list)
    avg_ssim_r = sum(ssim_r_list) / len(ssim_r_list)
    avg_ssim_avg = sum(ssim_avg_list) / len(ssim_avg_list)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    csv_name = f"{model_name}.csv"
    csv_path = os.path.join(os.getcwd(), csv_name)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["img_name", "psnr_c", "psnr_r", "ssim_c", "ssim_r", "ssim_avg"])
        for name, pc, pr, sc, sr, sa in zip(img_names, psnr_c_list, psnr_r_list, ssim_c_list, ssim_r_list, ssim_avg_list):
            writer.writerow([name, f"{pc:.6f}", f"{pr:.6f}", f"{sc:.6f}", f"{sr:.6f}", f"{sa:.6f}"])
        writer.writerow([
            "average",
            f"{avg_psnr_c:.6f}",
            f"{avg_psnr_r:.6f}",
            f"{avg_ssim_c:.6f}",
            f"{avg_ssim_r:.6f}",
            f"{avg_ssim_avg:.6f}",
        ])
    print(f"Saved evaluation results to {csv_path}")
    return csv_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="FP32 model path")
    args = parser.parse_args()
    evaluate_fp32_model(args.model)
