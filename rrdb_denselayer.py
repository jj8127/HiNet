import torch
import torch.nn as nn
import torch.ao.quantization as tq
import modules.module_util as mutil


# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        mutil.initialize_weights([self.conv5], 0.)

        # stubs to preserve quantized tensor flow
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        # quantize input and keep it separate to avoid mixing tensor types
        x_q = self.quant(x)

        x1 = self.lrelu(self.conv1(x_q))
        x2 = self.lrelu(self.conv2(torch.cat((x_q, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x_q, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x_q, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x_q, x1, x2, x3, x4), 1))
        x5 = self.dequant(x5)
        return x5
