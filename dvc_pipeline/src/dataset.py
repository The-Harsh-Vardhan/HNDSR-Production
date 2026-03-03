"""
SatelliteDataset — patch-based dataset for HR/LR satellite image pairs.

Extracted from MLFlow/HNDSR_MLflow.ipynb (Cell 6).
Supports training (random crop + horizontal flip) and evaluation (center crop).
Images are normalised to [-1, 1].
"""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SatelliteDataset(Dataset):
    """Dataset for satellite image super-resolution."""

    IMAGE_EXTENSIONS = [
        "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff",
        "*.PNG", "*.JPG", "*.JPEG", "*.TIF", "*.TIFF",
    ]

    def __init__(self, hr_dir: str, lr_dir: str, patch_size: int = 64, training: bool = True):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.patch_size = patch_size
        self.training = training

        # Collect images from both directories
        self.hr_images = []
        self.lr_images = []
        for ext in self.IMAGE_EXTENSIONS:
            self.hr_images.extend(list(self.hr_dir.glob(ext)))
            self.lr_images.extend(list(self.lr_dir.glob(ext)))

        self.hr_images = sorted(self.hr_images)
        self.lr_images = sorted(self.lr_images)

        if len(self.hr_images) == 0 or len(self.lr_images) == 0:
            raise ValueError(
                f"No images found!\n"
                f"HR Directory: {hr_dir} ({len(self.hr_images)} images)\n"
                f"LR Directory: {lr_dir} ({len(self.lr_images)} images)\n"
                f"Please check if the paths are correct."
            )

        # Match images by filename stem
        hr_names = {img.stem: img for img in self.hr_images}
        lr_names = {img.stem: img for img in self.lr_images}
        common_names = set(hr_names.keys()) & set(lr_names.keys())

        if len(common_names) == 0:
            # Fallback: positional pairing
            print("Warning: No matching filenames between HR and LR — using positional pairing.")
            min_len = min(len(self.hr_images), len(self.lr_images))
            self.hr_images = self.hr_images[:min_len]
            self.lr_images = self.lr_images[:min_len]
        else:
            self.hr_images = [hr_names[name] for name in sorted(common_names)]
            self.lr_images = [lr_names[name] for name in sorted(common_names)]

        print(f"Found {len(self.hr_images)} image pairs")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_images[idx]).convert("RGB")
        lr_img = Image.open(self.lr_images[idx]).convert("RGB")

        if self.training:
            hr_w, hr_h = hr_img.size
            lr_w, lr_h = lr_img.size
            scale = hr_w // lr_w

            lr_crop_size = self.patch_size // scale
            if lr_w > lr_crop_size and lr_h > lr_crop_size:
                x = random.randint(0, lr_w - lr_crop_size)
                y = random.randint(0, lr_h - lr_crop_size)

                lr_img = lr_img.crop((x, y, x + lr_crop_size, y + lr_crop_size))
                hr_img = hr_img.crop((
                    x * scale, y * scale,
                    (x + lr_crop_size) * scale,
                    (y + lr_crop_size) * scale,
                ))

            if random.random() > 0.5:
                lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            hr_img = transforms.CenterCrop(self.patch_size)(hr_img)
            lr_img = transforms.CenterCrop(self.patch_size // 4)(lr_img)

        lr_tensor = self.transform(lr_img)
        hr_tensor = self.transform(hr_img)

        return {"lr": lr_tensor, "hr": hr_tensor, "scale": 4}
