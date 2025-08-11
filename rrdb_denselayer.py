import torch
import torch.nn as nn
import torch.ao.quantization as tq
from typing import List


class ResidualDenseBlock_out(nn.Module):
    """
    RRDB 스타일 Dense block.
    - 내부 산술/concat은 FP32에서만 수행
    - 각 Conv 입력 직전에만 QuantStub, Conv 출력 직후 DeQuantStub
    - cat 앞뒤로 FP32를 확실히 강제하여 qscheme 관련 크래시/경고 방지
    """
    def __init__(self, in_channels: int, out_channels: int, growth: int = 32):
        super().__init__()
        g = growth
        c1, c2, c3, c4 = g, g, g, g
        c5_in = in_channels + c1 + c2 + c3 + c4

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, c1, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + c1, c2, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + c1 + c2, c3, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + c1 + c2 + c3, c4, 3, 1, 1)
        self.conv5 = nn.Conv2d(c5_in, out_channels, 3, 1, 1)

        # Act: LeakyReLU -> ReLU (양자화 안전)
        self.lrelu = nn.ReLU(inplace=False)

        # 각 conv 앞뒤 전용 Quant/DeQuant
        self.q1, self.dq1 = tq.QuantStub(), tq.DeQuantStub()
        self.q2, self.dq2 = tq.QuantStub(), tq.DeQuantStub()
        self.q3, self.dq3 = tq.QuantStub(), tq.DeQuantStub()
        self.q4, self.dq4 = tq.QuantStub(), tq.DeQuantStub()
        self.q5, self.dq5 = tq.QuantStub(), tq.DeQuantStub()

        # 내부 스텁 제거 방지 힌트
        self.__quant_protect__ = True

    @staticmethod
    def _dq(t: torch.Tensor) -> torch.Tensor:
        return t.dequantize() if t.is_quantized else t

    @staticmethod
    def _cat_fp32(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for t in tensors:
            outs.append(t.dequantize() if t.is_quantized else t)
        return torch.cat(outs, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 블록 입력 FP32 보장
        x = self._dq(x)

        # conv1
        x1 = self.lrelu(self.dq1(self.conv1(self.q1(x))))
        x1 = self._dq(x1)

        # conv2
        in2 = self._cat_fp32([x, x1], 1)
        x2 = self.lrelu(self.dq2(self.conv2(self.q2(in2))))
        x2 = self._dq(x2)

        # conv3
        in3 = self._cat_fp32([x, x1, x2], 1)
        x3 = self.lrelu(self.dq3(self.conv3(self.q3(in3))))
        x3 = self._dq(x3)

        # conv4
        in4 = self._cat_fp32([x, x1, x2, x3], 1)
        x4 = self.lrelu(self.dq4(self.conv4(self.q4(in4))))
        x4 = self._dq(x4)

        # conv5 (출력은 FP32 유지)
        in5 = self._cat_fp32([x, x1, x2, x3, x4], 1)
        out = self.dq5(self.conv5(self.q5(in5)))
        return self._dq(out)