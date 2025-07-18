import torch
import torch.nn as nn
import torch.ao.quantization as tq
import numpy as np

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
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = iter(datasets.trainloader)
    for _ in range(steps):
        try:
            data = next(loader)
        except StopIteration:
            loader = iter(datasets.trainloader)
            data = next(loader)
        data = data.to(device)
        out = model(data)
        loss = out.abs().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()


def calibrate(model, steps=5):
    model.eval()
    loader = iter(datasets.trainloader)
    with torch.no_grad():
        for _ in range(steps):
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(datasets.trainloader)
                data = next(loader)
            data = data.to(device)
            model(data)


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
    torch.save(qmodel.state_dict(), "hinet_qat_int8.pth")
    print("quantized model saved")


if __name__ == "__main__":
    main()

