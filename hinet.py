import torch
import torch.nn as nn

import config as c
from invblock import INV_block


class Hinet(nn.Module):
    """
    HiNet core made of 16 invertible blocks.
    - 입력 채널: DWT(cover)+DWT(secret) = 8 * channels_in
    - 각 INV_block은 harr=True 기준으로 (4*channels_in, 4*channels_in) 분할
    """
    def __init__(self, channels_in: int = None, harr: bool = True):
        super().__init__()
        ch = int(c.channels_in if channels_in is None else channels_in)
        self.channels_in: int = ch
        self.harr: bool = harr

        # named modules (기존 state_dict 호환)
        self.inv1  = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv2  = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv3  = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv4  = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv5  = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv6  = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv7  = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv8  = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv9  = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv10 = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv11 = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv12 = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv13 = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv14 = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv15 = INV_block(harr=harr, in_1=ch, in_2=ch)
        self.inv16 = INV_block(harr=harr, in_1=ch, in_2=ch)

        # (선택) 순회용 모듈리스트 — state_dict 이름은 inv{n} 유지
        self._blocks = nn.ModuleList([
            self.inv1, self.inv2, self.inv3, self.inv4,
            self.inv5, self.inv6, self.inv7, self.inv8,
            self.inv9, self.inv10, self.inv11, self.inv12,
            self.inv13, self.inv14, self.inv15, self.inv16
        ])

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)
            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)
        else:
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)
            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)
        return out