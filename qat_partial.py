import torch
import torch.nn as nn
import torch.ao.quantization as tq
import numpy as np
import os
from datetime import datetime

from hinet import Hinet
from invblock import INV_block
import datasets
import modules.Unet_common as common
import config as c


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mark_quant_layers(module):
    for child in module.children():
        if isinstance(child, INV_block):
            continue
        if isinstance(child, nn.Conv2d):
            child.qconfig = tq.get_default_qat_qconfig("fbgemm")
        mark_quant_layers(child)


def prepare_model_for_qat(model):
    model.train()
    mark_quant_layers(model)
    tq.prepare_qat(model, inplace=True)


def train_dummy(model, steps=10):
    """Run a few QAT steps using the training loader."""
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    dwt = common.DWT().to(device)
    loader = iter(datasets.trainloader)
    for step in range(1, steps + 1):
        try:
            data = next(loader)
        except StopIteration:
            loader = iter(datasets.trainloader)
            data = next(loader)
        data = data.to(device)
        half = data.size(0) // 2
        cover = data[half:]
        secret = data[:half]
        input_img = torch.cat((dwt(cover), dwt(secret)), 1)
        assert input_img.size(1) == 24, f"expected 24 channels, got {input_img.size(1)}"
        out = model(input_img)
        loss = out.abs().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"step {step}: loss {loss.item():.6f}")


def calibrate(model, steps=5):
    """Run a short calibration on a few training batches."""
    model.eval()
    dwt = common.DWT().to(device)
    loader = iter(datasets.trainloader)
    with torch.no_grad():
        for _ in range(steps):
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(datasets.trainloader)
                data = next(loader)
            data = data.to(device)
            half = data.size(0) // 2
            cover = data[half:]
            secret = data[:half]
            input_img = torch.cat((dwt(cover), dwt(secret)), 1)
            assert input_img.size(1) == 24, f"expected 24 channels, got {input_img.size(1)}"
            model(input_img)


def convert(model):
    model.cpu()
    return tq.convert(model.eval(), inplace=False)


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()


def evaluate(model):
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        for data in datasets.testloader:
            data = data.to(device)
            cover = data[data.size(0) // 2 :]
            secret = data[: data.size(0) // 2]
            cover_in = dwt(cover)
            secret_in = dwt(secret)
            input_img = torch.cat((cover_in, secret_in), 1)
            output = model(input_img)
            steg = iwt(output.narrow(1, 0, 4 * c.channels_in))
            scores.append(psnr(steg, cover))
    mean_psnr = float(np.mean(scores))
    print(f"PSNR: {mean_psnr:.2f} dB")


def main():
    model = Hinet().to(device)
    prepare_model_for_qat(model)
    train_dummy(model, steps=2)
    calibrate(model, steps=2)
    qmodel = convert(model)
    evaluate(qmodel.to(device))
    os.makedirs("model", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join("model", f"model_qat_{timestamp}.pt")
    torch.save(qmodel.state_dict(), save_path)
    print(f"quantized model saved to {save_path}")


if __name__ == "__main__":
    main()

