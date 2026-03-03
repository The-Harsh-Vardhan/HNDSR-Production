"""
Shared utilities for the HNDSR DVC training pipeline.

Functions:
    set_seed          – Deterministic seeding for Python, NumPy, PyTorch, CUDA.
    get_device        – Auto-detect CUDA / CPU.
    load_params       – Load params.yaml as a dict.
    denormalize       – Map tensors from [-1, 1] to [0, 1].
    calculate_psnr    – Batch PSNR via skimage.
    calculate_ssim    – Batch SSIM via skimage.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_gb:.2f} GB")
    return device


def load_params(path: str = "params.yaml") -> dict:
    """Load a YAML parameter file and return as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Map from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate mean PSNR over a batch of images (CHW tensors in [-1, 1])."""
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()

    psnr_vals = []
    for i in range(img1.shape[0]):
        im1 = (img1[i].transpose(1, 2, 0) + 1) / 2
        im2 = (img2[i].transpose(1, 2, 0) + 1) / 2
        psnr_vals.append(psnr(im1, im2, data_range=1.0))

    return float(np.mean(psnr_vals))


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate mean SSIM over a batch of images (CHW tensors in [-1, 1])."""
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()

    ssim_vals = []
    for i in range(img1.shape[0]):
        im1 = (img1[i].transpose(1, 2, 0) + 1) / 2
        im2 = (img2[i].transpose(1, 2, 0) + 1) / 2
        ssim_vals.append(ssim(im1, im2, data_range=1.0, channel_axis=2))

    return float(np.mean(ssim_vals))
