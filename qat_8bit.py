#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
import logging
from datetime import datetime
import pandas as pd

from hinet import Hinet
from invblock import INV_block
import modules.Unet_common as common
import datasets
import config as c

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------- 8bit Quantization utils ----------------------
def get_8bit_qconfig():
    activation = tq.default_fake_quant.with_args(quant_min=0, quant_max=255)
    weight = tq.default_per_channel_weight_fake_quant.with_args(
        quant_min=-128, quant_max=127
    )
    return tq.QConfig(activation=activation, weight=weight)

def mark_quant_layers(module):
    for child in module.children():
        if isinstance(child, INV_block):
            continue  # INV_block은 FP32 유지
        if isinstance(child, nn.Conv2d):
            child.qconfig = get_8bit_qconfig()
        mark_quant_layers(child)

def prepare_model_for_qat(model):
    model.train()
    mark_quant_layers(model)
    tq.prepare_qat(model, inplace=True)

def convert(model):
    model.cpu()
    return tq.convert(model.eval(), inplace=False)

def load_pretrained(model, path=None):
    if path is None:
        path = os.path.join(c.MODEL_PATH, c.suffix)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if "net" in state:
            state = state["net"]
        elif "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    new_state = {}
    for k, v in state.items():
        name = k
        if name.startswith("module.model."):
            name = name[len("module.model.") :]
        elif name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("model."):
            name = name[len("model.") :]
        new_state[name] = v
    model.load_state_dict(new_state, strict=False)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()

def evaluate(model, silent=False):
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    model.eval()
    scores_cover, scores_secret = [], []
    with torch.no_grad():
        for secret, cover in datasets.testloader:
            secret = secret.to(device)
            cover = cover.to(device)
            cover_in = dwt(cover)
            secret_in = dwt(secret)
            input_img = torch.cat((cover_in, secret_in), 1)
            output = model(input_img)
            steg = iwt(output.narrow(1, 0, 4 * c.channels_in))
            z = torch.randn_like(output.narrow(1, 4 * c.channels_in, output.size(1) - 4 * c.channels_in))
            rev_input = torch.cat((output.narrow(1, 0, 4 * c.channels_in), z), 1)
            backward = model(rev_input, rev=True)
            secret_rev = iwt(backward.narrow(1, 4 * c.channels_in, backward.size(1) - 4 * c.channels_in))
            scores_cover.append(psnr(steg, cover))
            scores_secret.append(psnr(secret_rev, secret))
    mean_cover = float(np.mean(scores_cover))
    mean_secret = float(np.mean(scores_secret))
    if not silent:
        logging.info(f"TEST:   PSNR_S: {mean_secret:.4f} | PSNR_C: {mean_cover:.4f} | ")
    return mean_cover, mean_secret

def train(model, epochs=1, metrics=None, checkpoint_interval=10, label=None, save_dir=None):
    optim = torch.optim.Adam(model.parameters(), lr=c.lr)
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for secret, cover in datasets.trainloader:
            secret = secret.to(device)
            cover = cover.to(device)
            cover_in = dwt(cover)
            secret_in = dwt(secret)
            input_img = torch.cat((cover_in, secret_in), 1)
            output = model(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            steg_img = iwt(output_steg)
            output_z = output.narrow(1, 4 * c.channels_in, output.size(1) - 4 * c.channels_in)
            noise = torch.randn_like(output_z)
            rev_input = torch.cat((output_steg, noise), 1)
            backward = model(rev_input, rev=True)
            secret_rev = iwt(backward.narrow(1, 4 * c.channels_in, backward.size(1) - 4 * c.channels_in))
            g_loss = F.mse_loss(steg_img, cover, reduction="sum")
            r_loss = F.mse_loss(secret_rev, secret, reduction="sum")
            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_in.narrow(1, 0, c.channels_in)
            l_loss = F.mse_loss(steg_low, cover_low, reduction="sum")
            loss = (
                c.lamda_reconstruction * r_loss
                + c.lamda_guide * g_loss
                + c.lamda_low_frequency * l_loss
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        avg = epoch_loss / max(1, len(datasets.trainloader))
        logging.info(
            f"Train epoch {epoch}:   Loss: {avg:.4f} | r_Loss: {r_loss.item():.4f} | g_Loss: {g_loss.item():.4f} | l_Loss: {l_loss.item():.4f} | "
        )
        if metrics is not None:
            metrics["loss"].append(avg)
            ps_cover, ps_secret = evaluate(model, silent=True)
            metrics["psnr_train_cover"].append(ps_cover)
            metrics["psnr_train_secret"].append(ps_secret)

        # --- checkpoint 저장 (10 에폭마다, 또는 첫 에폭) ---
        if (checkpoint_interval is not None) and (epoch % checkpoint_interval == 0 or epoch == 1):
            if label is None:
                checkpoint_label = f"epoch{epoch}"
            else:
                checkpoint_label = f"{label}_epoch{epoch}"
            ckpt_path = os.path.join(save_dir, f"checkpoint_{checkpoint_label}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Checkpoint saved at epoch {epoch} to {ckpt_path}")

def calibrate(model, steps=5, metrics=None):
    model.eval()
    dwt = common.DWT().to(device)
    loader = iter(datasets.trainloader)
    with torch.no_grad():
        for step in range(1, steps + 1):
            try:
                secret, cover = next(loader)
            except StopIteration:
                loader = iter(datasets.trainloader)
                secret, cover = next(loader)
            secret = secret.to(device)
            cover = cover.to(device)
            input_img = torch.cat((dwt(cover), dwt(secret)), 1)
            assert input_img.size(1) == 24, f"expected 24 channels, got {input_img.size(1)}"
            model(input_img)
            if metrics is not None:
                ps_cover, ps_secret = evaluate(model, silent=True)
                metrics["psnr_calib_cover"].append(ps_cover)
                metrics["psnr_calib_secret"].append(ps_secret)
            logging.info(f"Calibration step {step}/{steps} done.")

def plot_metrics(metrics, save_dir):
    np.savez(os.path.join(save_dir, f"qat_metrics.npz"), **metrics)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(range(1, len(metrics["psnr_train_cover"]) + 1), metrics["psnr_train_cover"])
    axes[0].set_title("PSNR_C")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("dB")
    axes[0].grid(True)
    axes[1].plot(range(1, len(metrics["psnr_train_secret"]) + 1), metrics["psnr_train_secret"], color="red")
    axes[1].set_title("PSNR_S")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("dB")
    axes[1].grid(True)
    axes[2].plot(range(1, len(metrics["loss"]) + 1), metrics["loss"])
    axes[2].set_title("Train Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True)
    fig.tight_layout()
    png_path = os.path.join(save_dir, f"metrics_plot.png")
    fig.savefig(png_path)
    plt.close(fig)
    logging.info(f"Saved plots to {png_path}")

    # CSV로 저장
    df = pd.DataFrame({
        "epoch": list(range(1, len(metrics["loss"]) + 1)),
        "loss": metrics["loss"],
        "psnr_train_cover": metrics["psnr_train_cover"],
        "psnr_train_secret": metrics["psnr_train_secret"],
    })
    csv_path = os.path.join(save_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved metrics csv to {csv_path}")

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    fmt = "%(asctime)s - %(levelname)s: %(message)s"
    datefmt = "%y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ],
    )
    logging.info("Logger initialized")
    return log_path

def get_save_dir(epochs, calib_steps):
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H-%M")
    label = f"qat_8bit_{date_str}_ep{epochs}_calib{calib_steps}"
    save_dir = os.path.join("logging", label)
    return save_dir, label

def main(pretrained=None, epochs=1, calib_steps=5, checkpoint_interval=10):
    save_dir, label = get_save_dir(epochs, calib_steps)
    log_file = setup_logger(save_dir)
    logging.info(f"Label: {label}")
    logging.info(f"Device: {device}")
    logging.info(f"Save directory: {save_dir}")

    metrics = {
        "loss": [],
        "psnr_train_cover": [],
        "psnr_train_secret": [],
        "psnr_calib_cover": [],
        "psnr_calib_secret": [],
    }

    model = Hinet().to(device)
    load_pretrained(model, pretrained)
    prepare_model_for_qat(model)

    train(model, epochs=epochs, metrics=metrics, checkpoint_interval=checkpoint_interval, label=label, save_dir=save_dir)
    calibrate(model, steps=calib_steps, metrics=metrics)

    qmodel = convert(model)
    evaluate(qmodel.to(device))

    final_model_path = os.path.join(save_dir, f"finetuned_model_qat.pt")
    torch.save(qmodel.state_dict(), final_model_path)
    logging.info(f"Quantized model saved to {final_model_path}")

    plot_metrics(metrics, save_dir)
    logging.info(f"Training log saved to {log_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run partial 8-bit QAT")
    parser.add_argument("--pretrained", type=str, default=None, help="path to FP32 model")
    parser.add_argument("--epochs", type=int, default=50, help="number of QAT training epochs")
    parser.add_argument("--calib-steps", type=int, default=10, help="number of calibration batches")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="number of epochs per checkpoint")
    args = parser.parse_args()
    main(
        pretrained=args.pretrained,
        epochs=args.epochs,
        calib_steps=args.calib_steps,
        checkpoint_interval=args.checkpoint_interval,
    )
