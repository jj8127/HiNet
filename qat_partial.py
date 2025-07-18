import torch
import torch.nn as nn
import torch.ao.quantization as tq

from hinet import Hinet
from invblock import INV_block


def mark_quant_layers(module):
    for name, child in module.named_children():
        if isinstance(child, INV_block):
            # Do not quantize inside invertible blocks
            continue
        if isinstance(child, nn.Conv2d):
            child.qconfig = tq.get_default_qat_qconfig('fbgemm')
        else:
            mark_quant_layers(child)


def prepare_model_for_qat(model):
    mark_quant_layers(model)
    tq.prepare_qat(model, inplace=True)


def calibrate(model, steps=8):
    model.train()
    for _ in range(steps):
        dummy = torch.randn(1, 3, 64, 64)
        model(dummy)


def convert(model):
    model.eval()
    quantized = tq.convert(model.eval(), inplace=False)
    return quantized


def main():
    model = Hinet()
    prepare_model_for_qat(model)

    # Fake train loop
    for _ in range(2):
        data = torch.randn(1, 3, 64, 64)
        out = model(data)
        loss = out.abs().mean()
        loss.backward()

    calibrate(model)
    qmodel = convert(model)
    torch.save(qmodel.state_dict(), 'hinet_qat_int8.pth')
    print('quantized model saved')


if __name__ == '__main__':
    main()
