# invblock.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.ao.quantization as tq

from rrdb_denselayer import ResidualDenseBlock_out

__all__ = ["INV_block"]


class INV_block(nn.Module):
    """
    HiNet용 Invertible Block.
    - 내부 산술/concat은 FP32로 고정
    - 각 서브넷(f, r, y)은 Conv 경로만 양자화(QAT/INT8)되도록 설계
    - 입구에서 DeQuantStub를 항상 보유(스크립트/런타임 모두 안전)
    """

    def __init__(
        self,
        subnet_constructor=ResidualDenseBlock_out,
        clamp: float = 1.0,
        harr: bool = True,
        in_1: int = 3,
        in_2: int = 3,
    ):
        super().__init__()
        self.clamp = float(clamp)

        # Haar 사용 시 채널 4배(픽셀언셔플/파형분해 가정)
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        else:
            self.split_len1 = in_1
            self.split_len2 = in_2

        # 서브넷: 내부에서 Conv만 양자화되는 ResidualDenseBlock_out 사용
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        self.f = subnet_constructor(self.split_len2, self.split_len1)

        # 입구에서 항상 디양자화 (존재 보장: TorchScript에서 getattr/hasattr 회피)
        self.deq_in = tq.DeQuantStub()

        # 내부 산술 전에 쓸 ReLU (서브넷 안에도 있음)
        self.act = nn.ReLU(inplace=False)

    @torch.jit.export
    def e(self, s: torch.Tensor) -> torch.Tensor:
        # exp( clamp * 2 * (sigmoid(s) - 0.5) )
        return torch.exp(self.clamp * 2.0 * (torch.sigmoid(s) - 0.5))

    @staticmethod
    def _dq_if_q(x: torch.Tensor) -> torch.Tensor:
        # 양자화 텐서면 dequantize, 아니면 그대로
        if x.is_quantized:
            return x.dequantize()
        return x

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        # 입구에서 DeQuant → FP32 보장
        x = self.deq_in(x)
        x = self._dq_if_q(x)

        # 채널 분할 (총 채널 = split_len1 + split_len2 여야 함)
        # 예: harr=True, in_1=in_2=3이면 12 + 12 = 24 채널 입력 필요
        x1 = x.narrow(1, 0, self.split_len1)
        x2 = x.narrow(1, self.split_len1, self.split_len2)

        if not rev:
            # y1 = x1 + f(x2)
            t2 = self.f(x2)           # f 내부 Conv 경로는 QAT/INT8
            t2 = self._dq_if_q(t2)    # 산술 전에 FP32
            y1 = x1 + t2

            # y2 = exp(e(r(y1))) * x2 + y(y1)
            s1 = self.r(y1)           # r 내부 Conv 경로 QAT/INT8
            t1 = self.y(y1)           # y 내부 Conv 경로 QAT/INT8
            s1 = self._dq_if_q(s1)
            t1 = self._dq_if_q(t1)
            y2 = self.e(s1) * x2 + t1
        else:
            # 역변환
            # y2 = (x2 - y(x1)) / exp(e(r(x1)))
            s1 = self.r(x1)
            t1 = self.y(x1)
            s1 = self._dq_if_q(s1)
            t1 = self._dq_if_q(t1)
            y2 = (x2 - t1) / self.e(s1)

            # y1 = x1 - f(y2)
            t2 = self.f(y2)
            t2 = self._dq_if_q(t2)
            y1 = x1 - t2

        # 최종 concat (FP32)
        out = torch.cat((y1, y2), dim=1)
        return out