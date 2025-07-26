import glob
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted


class HinetDataset(Dataset):
    """Dataset returning pairs of secret and cover images."""

    def __init__(self, secret_dir: str, cover_dir: str, transform=None, fmt: str = "png"):
        self.secret_files = natsorted(glob.glob(os.path.join(secret_dir, f"*.{fmt}")))
        self.cover_files = natsorted(glob.glob(os.path.join(cover_dir, f"*.{fmt}")))
        if not self.secret_files:
            raise FileNotFoundError(f"No secret images found in {secret_dir}")
        if not self.cover_files:
            raise FileNotFoundError(f"No cover images found in {cover_dir}")
        self.secret_files = self._filter_valid(self.secret_files)
        self.cover_files = self._filter_valid(self.cover_files)
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
        cover_path = self.cover_files[idx % len(self.cover_files)]
        secret = Image.open(secret_path).convert("RGB")
        cover = Image.open(cover_path).convert("RGB")
        if self.transform:
            secret = self.transform(secret)
            cover = self.transform(cover)
        return secret, cover


transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor(),
])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])


trainloader = DataLoader(
    HinetDataset(c.TRAIN_PATH, c.TRAIN_COVER_PATH, transform, c.format_train),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True,
)

testloader = DataLoader(
    HinetDataset(c.VAL_PATH, c.VAL_COVER_PATH, transform_val, c.format_val),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True,
)
