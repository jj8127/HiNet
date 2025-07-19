import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == "train":
            # train
            self.files = natsorted(
                sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train))
            )
        else:
            # test
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))

        if not self.files:
            raise FileNotFoundError(
                f"No image files found for mode '{mode}' in "
                f"{'TRAIN_PATH' if mode == 'train' else 'VAL_PATH'}"
            )

    def __getitem__(self, index):
        """Return the transformed image at ``index``.

        The original implementation used recursion to skip files that could not
        be opened. When the end of ``self.files`` was reached, this caused an
        infinite recursion leading to ``RecursionError``. The new logic iterates
        forward until a valid image is found and raises ``IndexError`` if none is
        available.
        """

        while index < len(self.files):
            path = self.files[index]
            try:
                image = Image.open(path)
                image = to_rgb(image)
                return self.transform(image)
            except Exception:
                index += 1

        raise RuntimeError(
            f"No valid image found starting from index {index}. Check dataset "
            f"files for corruption."
        )

    def __len__(self):
        if self.mode == "shuffle":
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)


transform = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop(c.cropsize),
        T.ToTensor(),
    ]
)

transform_val = T.Compose(
    [
        T.CenterCrop(c.cropsize_val),
        T.ToTensor(),
    ]
)


# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True,
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True,
)
