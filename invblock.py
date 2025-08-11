import torch
import torch.nn as nn
import torch.ao.quantization as tq
from typing import List

import config as c
from rrdb_denselayer import ResidualDenseBlock_out


class INV_block(nn.Module):
    """
    내부 산술/concat은 FP32, Conv 경로만 양자화되도록 구성.
    - 블록 입구에서 FP32 보장(DeQuant + 런타임 가드)
    - f/r/y 서브넷 내부에서만 Quant/DeQuant 수행 (서브넷이 처리)
    - cat은 항상 FP32에서만 수행
    """
    def __init__(
        self,
        subnet_constructor=ResidualDenseBlock_out,
        clamp=c.clamp,
        harr=True,
        in_1=3,
        in_2=3,
    ):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        else:
            self.split_len1 = in_1
            self.split_len2 = in_2

        self.clamp = clamp

        # 서브넷(각 서브넷 내부에서 Conv 전후로만 Quant/DeQuant)
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        self.f = subnet_constructor(self.split_len2, self.split_len1)

        # 내부 스텁 제거 방지 플래그
        self.__quant_protect__ = True

        # 블록 입구에서 FP32 보장 (convert 후에도 올바른 op로 대체됨)
        self.deq_in = tq.DeQuantStub()

    def e(self, s: torch.Tensor) -> torch.Tensor:
        # exp(sigmoid) 기반 scale
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    @staticmethod
    def _to_fp32(t: torch.Tensor) -> torch.Tensor:
        # TorchScript 친화적
        return t.dequantize() if t.is_quantized else t

    @staticmethod
    def _cat_fp32(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for t in tensors:
            outs.append(t.dequantize() if t.is_quantized else t)
        return torch.cat(outs, dim)

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        # 입구에서 FP32 보장 (양자 텐서가 들어와도 안전)
        x = self._to_fp32(self.deq_in(x))

        # 채널 분할
        x1 = x.narrow(1, 0, self.split_len1)
        x2 = x.narrow(1, self.split_len1, self.split_len2)

        if not rev:
            # y1 = x1 + f(x2)
            t2 = self.f(x2)              # f 내부에서만 양자화
            t2 = self._to_fp32(t2)       # 산술 전에 FP32 확실화
            y1 = x1 + t2

            # y2 = exp(e(r(y1))) * x2 + y(y1)
            s1 = self.r(y1)              # r 내부 양자화
            t1 = self.y(y1)              # y 내부 양자화
            y2 = self.e(s1) * x2 + t1

        else:
            # y2 = (x2 - y(x1)) / exp(e(r(x1)))
            s1 = self.r(x1)
            t1 = self.y(x1)
            y2 = (x2 - t1) / self.e(s1)

            # y1 = x1 - f(y2)
            t2 = self.f(y2)
            t2 = self._to_fp32(t2)
            y1 = x1 - t2

        # cat은 항상 FP32에서만
        return self._cat_fp32([y1, y2], 1)