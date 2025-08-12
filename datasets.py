# datasets.py
# -*- coding: utf-8 -*-

import os
import glob
from typing import Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from natsort import natsorted

import config as c


# ----------------------------
# Paired transforms
# ----------------------------
class PairedRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img_s: Image.Image, img_c: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if torch.rand(1).item() < self.p:
            img_s = F.hflip(img_s)
            img_c = F.hflip(img_c)
        return img_s, img_c


class PairedRandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img_s: Image.Image, img_c: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if torch.rand(1).item() < self.p:
            img_s = F.vflip(img_s)
            img_c = F.vflip(img_c)
        return img_s, img_c


class PairedRandomCrop:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img_s: Image.Image, img_c: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # secret 기준으로 파라미터 샘플, 같은 좌표/크기를 cover에도 적용
        i, j, h, w = T.RandomCrop.get_params(img_s, (self.size, self.size))
        # cover가 더 작을 가능성에 대비: 필요하면 먼저 center-crop/resize를 추가로 넣으세요.
        img_s = F.crop(img_s, i, j, h, w)
        img_c = F.crop(img_c, i, j, h, w)
        return img_s, img_c


class PairedCenterCrop:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img_s: Image.Image, img_c: Image.Image) -> Tuple[Image.Image, Image.Image]:
        img_s = F.center_crop(img_s, self.size)
        img_c = F.center_crop(img_c, self.size)
        return img_s, img_c


class EnsureEvenHW:
    """DWT(2x2) 위해 H, W를 짝수로 보정(필요 시 1픽셀 center crop)."""
    def __call__(self, img_s: Image.Image, img_c: Image.Image) -> Tuple[Image.Image, Image.Image]:
        def even_hw(im: Image.Image) -> Image.Image:
            w, h = im.size
            tw = w - (w % 2)
            th = h - (h % 2)
            if (tw, th) != (w, h):
                return F.center_crop(im, (th, tw))
            return im

        return even_hw(img_s), even_hw(img_c)


class ToTensorPair:
    def __call__(self, img_s: Image.Image, img_c: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.to_tensor(img_s), F.to_tensor(img_c)


class PairedCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_s, img_c):
        for t in self.transforms:
            img_s, img_c = t(img_s, img_c)
        return img_s, img_c


# ----------------------------
# Dataset
# ----------------------------
class HinetDataset(Dataset):
    """Return (secret, cover) image pair."""
    def __init__(self, secret_dir: str, cover_dir: str, transform=None, fmt: str = "png"):
        self.secret_files = natsorted(glob.glob(os.path.join(secret_dir, f"*.{fmt}")))
        self.cover_files  = natsorted(glob.glob(os.path.join(cover_dir,  f"*.{fmt}")))

        if not self.secret_files:
            raise FileNotFoundError(f"No secret images found in {secret_dir}")
        if not self.cover_files:
            raise FileNotFoundError(f"No cover images found in {cover_dir}")

        self.secret_files = self._filter_valid(self.secret_files)
        self.cover_files  = self._filter_valid(self.cover_files)
        if not self.secret_files or not self.cover_files:
            raise RuntimeError("No valid image files found in dataset")

        self.length = max(len(self.secret_files), len(self.cover_files))
        self.transform = transform

    @staticmethod
    def _filter_valid(files):
        valid = []
        for p in files:
            try:
                with Image.open(p) as img:
                    img.verify()
                valid.append(p)
            except Exception:
                continue
        return valid

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        secret_path = self.secret_files[idx % len(self.secret_files)]
        cover_path  = self.cover_files [idx % len(self.cover_files )]

        secret = Image.open(secret_path).convert("RGB")
        cover  = Image.open(cover_path ).convert("RGB")

        if self.transform:
            secret, cover = self.transform(secret, cover)

        return secret, cover


# ----------------------------
# Dataloaders builder
# ----------------------------
def build_dataloaders(seed: int = 1234):
    # Train: 무작위 동일 좌표 crop + 짝수 H/W 보정
    train_transform = PairedCompose([
        PairedRandomHorizontalFlip(),
        PairedRandomVerticalFlip(),
        PairedRandomCrop(c.cropsize),
        EnsureEvenHW(),
        ToTensorPair(),
    ])

    # Val/Calib: 중앙 동일 크기 crop으로 고정 크기 맞춤 + 짝수 보정
    val_transform = PairedCompose([
        PairedCenterCrop(c.cropsize_val),
        EnsureEvenHW(),
        ToTensorPair(),
    ])

    g = torch.Generator().manual_seed(seed)

    trainloader = DataLoader(
        HinetDataset(c.TRAIN_PATH, c.TRAIN_COVER_PATH, transform=train_transform, fmt=c.format_train),
        batch_size=c.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        generator=g,
    )

    testloader = DataLoader(
        HinetDataset(c.VAL_PATH, c.VAL_COVER_PATH, transform=val_transform, fmt=c.format_val),
        batch_size=c.batchsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
        generator=g,
    )
    return trainloader, testloader
