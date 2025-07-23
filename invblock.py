from math import exp
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import config as c
from rrdb_denselayer import ResidualDenseBlock_out


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

        # quantization stubs to keep the path through f and g quantized
        self.quant_f = tq.QuantStub()
        self.dequant_f = tq.DeQuantStub()
        self.quant_g = tq.QuantStub()
        self.dequant_g = tq.DeQuantStub()

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            x2_q = self.quant_f(x2)
            t2 = self.f(x2_q)
            t2 = self.dequant_f(t2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            y2_q = self.quant_g(y2)
            t2 = self.f(y2_q)
            t2 = self.dequant_g(t2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)

