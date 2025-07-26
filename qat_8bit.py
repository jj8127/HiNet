import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
import logging

from hinet import Hinet
from invblock import INV_block
import modules.Unet_common as common
import datasets
import config as c

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")

# ---------------------------- Quantization utils ----------------------------
def mark_quant_layers(module):
    for child in module.children():
        if isinstance(child, INV_block):
            continue  # INV_block은 FP32 유지
        if isinstance(child, nn.Conv2d):
            child.qconfig = tq.get_default_qat_qconfig("fbgemm")
        mark_quant_layers(child)


def prepare_model_for_qat(model):
    model.train()
    mark_quant_layers(model)
    tq.prepare_qat(model, inplace=True)


def convert(model):
    model.cpu()
    return tq.convert(model.eval(), inplace=False)


# ---------------------------- Load FP32 weights -----------------------------
def load_pretrained(model, path=None):
    if path is None:
        path = os.path.join(c.MODEL_PATH, c.suffix)
    if not os.path.isfile(path):
        logging.warning(f"pretrained model not found at {path}")
        return
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

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        logging.warning(f"missing keys {missing}")
    if unexpected:
        logging.warning(f"unexpected keys {unexpected}")
    logging.info(f"loaded pretrained weights from {path}")


# ---------------------------- Metrics helpers -------------------------------
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


# ---------------------------- Train / Calibrate -----------------------------
def train(model, epochs=1, metrics=None):
    optim = torch.optim.Adam(model.parameters(), lr=c.lr)
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)

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


# ---------------------------- Plot / Logging -------------------------------
def plot_metrics(metrics, label):
    os.makedirs("logging", exist_ok=True)
    np.savez(os.path.join("logging", f"qat_metrics_{label}.npz"), **metrics)

    # 1장짜리 그림
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PSNR_C
    axes[0].plot(range(1, len(metrics["psnr_train_cover"]) + 1), metrics["psnr_train_cover"])
    axes[0].set_title("PSNR_C")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("dB")
    axes[0].grid(True)

    # PSNR_S
    axes[1].plot(range(1, len(metrics["psnr_train_secret"]) + 1), metrics["psnr_train_secret"], color="red")
    axes[1].set_title("PSNR_S")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("dB")
    axes[1].grid(True)

    # Train Loss
    axes[2].plot(range(1, len(metrics["loss"]) + 1), metrics["loss"])
    axes[2].set_title("Train Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True)

    fig.tight_layout()
    png_path = os.path.join("logging", f"stage1_8bit_{label}.png")
    fig.savefig(png_path)
    plt.close(fig)
    logging.info(f"saved plots to {png_path}")


def setup_logger(label):
    os.makedirs("logging", exist_ok=True)
    log_path = os.path.join("logging", f"train__8bit_{label}.log")

    # 기존 핸들러 제거(중복 방지)
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


# ---------------------------- Main -----------------------------------------
def main(pretrained=None, epochs=1, calib_steps=5):
    label = f"ep{epochs}_calib{calib_steps}"
    log_file = setup_logger(label)
    logging.info(f"Label: {label}")
    logging.info(f"Device: {device}")

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

    train(model, epochs=epochs, metrics=metrics)
    calibrate(model, steps=calib_steps, metrics=metrics)

    qmodel = convert(model)
    evaluate(qmodel.to(device))

    os.makedirs("model", exist_ok=True)
    save_path = os.path.join("model", f"model_qat_{label}.pt")
    torch.save(qmodel.state_dict(), save_path)
    logging.info(f"quantized model saved to {save_path}")

    plot_metrics(metrics, label)
    logging.info(f"training log saved to {log_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run partial INT8 QAT")
    parser.add_argument("--pretrained", type=str, default=None, help="path to FP32 model")
    parser.add_argument("--epochs", type=int, default=50, help="number of QAT training epochs")
    parser.add_argument("--calib-steps", type=int, default=10, help="number of calibration batches")
    args = parser.parse_args()

    main(pretrained=args.pretrained, epochs=args.epochs, calib_steps=args.calib_steps)
