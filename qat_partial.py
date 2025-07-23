import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.nn.functional as F
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


def load_pretrained(model, path=None):
    """Load FP32 weights before QAT."""
    if path is None:
        path = os.path.join(c.MODEL_PATH, c.suffix)
    if not os.path.isfile(path):
        print(f"warning: pretrained model not found at {path}")
        return
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        # handle checkpoints saved with different wrappers
        if 'net' in state:
            state = state['net']
        elif 'state_dict' in state:
            state = state['state_dict']
        elif 'model' in state and isinstance(state['model'], dict):
            state = state['model']

    # strip common prefixes such as 'module.' or 'model.'
    new_state = {}
    for k, v in state.items():
        name = k
        if name.startswith('module.model.'):
            name = name[len('module.model.') :]
        elif name.startswith('module.'):
            name = name[len('module.') :]
        if name.startswith('model.'):
            name = name[len('model.') :]

        new_state[name] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"warning: missing keys {missing}")
    if unexpected:
        print(f"warning: unexpected keys {unexpected}")
    print(f"loaded pretrained weights from {path}")


def train(model, epochs=1):
    """Train the network with partial QAT for several epochs."""
    optim = torch.optim.Adam(model.parameters(), lr=c.lr)
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data in datasets.trainloader:
            data = data.to(device)
            half = data.size(0) // 2
            cover = data[half:]
            secret = data[:half]
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
        print(f"epoch {epoch + 1}: loss {avg:.6f}")


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
    scores_cover = []
    scores_secret = []

    def pair_batches(loader):
        """Yield image pairs regardless of original batch size."""
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

    with torch.no_grad():
        for batch in pair_batches(datasets.testloader):
            batch = batch.to(device)
            cover = batch[1:]
            secret = batch[:1]
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
    print(f"PSNR cover: {mean_cover:.2f} dB, secret: {mean_secret:.2f} dB")


def main(pretrained=None, epochs=1, calib_steps=5):
    model = Hinet().to(device)
    load_pretrained(model, pretrained)
    prepare_model_for_qat(model)
    train(model, epochs=epochs)
    calibrate(model, steps=calib_steps)
    qmodel = convert(model)
    evaluate(qmodel.to(device))
    os.makedirs("model", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join("model", f"model_qat_{timestamp}.pt")
    torch.save(qmodel.state_dict(), save_path)
    print(f"quantized model saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run partial INT8 QAT")
    parser.add_argument("--pretrained", type=str, default=None, help="path to FP32 model")
    parser.add_argument("--epochs", type=int, default=1, help="number of QAT training epochs")
    parser.add_argument("--calib-steps", type=int, default=5, help="number of calibration batches")
    args = parser.parse_args()

    main(pretrained=args.pretrained, epochs=args.epochs, calib_steps=args.calib_steps)
