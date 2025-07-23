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


def run(model: nn.Module):
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)
    loader = pair_batches(datasets.testloader)
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
            vutils.save_image(cover, os.path.join(c.IMAGE_PATH_cover, f"{i:05d}.png"))
            vutils.save_image(secret, os.path.join(c.IMAGE_PATH_secret, f"{i:05d}.png"))
            vutils.save_image(steg_img, os.path.join(c.IMAGE_PATH_steg, f"{i:05d}.png"))
            vutils.save_image(secret_rev, os.path.join(c.IMAGE_PATH_secret_rev, f"{i:05d}.png"))

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