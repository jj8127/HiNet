import argparse
import glob
import os
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torchvision.utils as vutils

from hinet import Hinet
from invblock import INV_block
import modules.Unet_common as common
import datasets
import config as c


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mark_quant_layers(module):
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


def run_once(model: nn.Module, batch):
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    batch = batch.to(device)
    half = batch.size(0) // 2
    cover = batch[half:]
    secret = batch[:half]
    cover_in = dwt(cover)
    secret_in = dwt(secret)
    input_img = torch.cat((cover_in, secret_in), 1)
    output = model(input_img)
    steg = iwt(output.narrow(1, 0, 4 * c.channels_in))
    z_noise = torch.randn_like(output.narrow(1, 4 * c.channels_in, output.size(1) - 4 * c.channels_in))
    rev_input = torch.cat((output.narrow(1, 0, 4 * c.channels_in), z_noise), 1)
    backward = model(rev_input, rev=True)
    secret_rev = iwt(backward.narrow(1, 4 * c.channels_in, backward.size(1) - 4 * c.channels_in))
    return cover, secret, steg.clamp(0, 1), secret_rev.clamp(0, 1)


def save_images(images, prefix: str, index: int):
    os.makedirs(prefix, exist_ok=True)
    for img, name in images:
        vutils.save_image(img, os.path.join(prefix, f"{name}_{index:05d}.png"))


def demo(model_path: str, num_batches: int = 1):
    model = load_quantized(model_path)
    loader = iter(datasets.testloader)
    for i in range(num_batches):
        try:
            data = next(loader)
        except StopIteration:
            break
        cover, secret, steg, secret_rev = run_once(model, data)
        save_images([
            (cover, "cover"),
            (secret, "secret"),
            (steg, "steg"),
            (secret_rev, "secret_rev"),
        ], c.IMAGE_PATH, i)
        print(f"batch {i}: images saved to {c.IMAGE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quantized HiNet model and save sample images")
    parser.add_argument("--model", type=str, default=None, help="path to quantized model")
    parser.add_argument("--batches", type=int, default=1, help="number of batches to process")
    args = parser.parse_args()

    if args.model is None:
        files = sorted(glob.glob(os.path.join("model", "model_qat_*.pt")))
        if not files:
            raise FileNotFoundError("No quantized model found in ./model")
        args.model = files[-1]
        print(f"loading latest model {args.model}")

    demo(args.model, args.batches)

