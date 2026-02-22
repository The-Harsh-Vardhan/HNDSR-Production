"""
backend/inference/tile_processor.py
=====================================
Handles large satellite images by splitting into overlapping tiles,
running inference per-tile, and stitching results with Hann-window blending.

Problem  : A 10,000x10,000 satellite image cannot fit in GPU VRAM as a single
           tensor (would require ~2.4 GB for FP32 RGB alone, before the model).
Risk     : Naive full-image inference causes CUDA OOM; tile boundaries cause
           visible seam artefacts in the output.
Solution : Overlap-tile-stitch with a Hann window blending mask eliminates
           seams; tiles are processed sequentially to bound peak VRAM usage.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hann window blending mask
# ─────────────────────────────────────────────────────────────────────────────

def _hann_window_2d(size: int, device: torch.device) -> torch.Tensor:
    """
    2-D Hann window of shape (1, 1, size, size).
    Weights are highest at the tile centre and taper to zero at edges,
    so overlapping tiles blend smoothly without hard seams.
    """
    w1d = torch.hann_window(size, periodic=False, device=device)
    w2d = w1d.unsqueeze(0) * w1d.unsqueeze(1)
    return w2d.unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Tile extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_tiles(
    image: torch.Tensor,
    tile_size: int,
    overlap: int,
    scale: float,
) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]], tuple[int, int]]:
    """
    Split a (C, H, W) image into overlapping LR tiles.

    Returns:
        tiles      : List of (C, tile_size, tile_size) tensors.
        coords     : List of (y0, x0, y1, x1) LR pixel coordinates per tile.
        hr_shape   : Expected (H_hr, W_hr) of the full super-resolved image.
    """
    _, H, W = image.shape
    stride = tile_size - overlap

    # Pad image so it's divisible by stride
    pad_h = math.ceil((H - overlap) / stride) * stride + overlap - H
    pad_w = math.ceil((W - overlap) / stride) * stride + overlap - W
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, H_pad, W_pad = image.shape

    tiles, coords = [], []
    for y in range(0, H_pad - overlap, stride):
        for x in range(0, W_pad - overlap, stride):
            y1 = min(y + tile_size, H_pad)
            x1 = min(x + tile_size, W_pad)
            tile = image[:, y:y1, x:x1]
            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                tile = F.pad(tile, (0, tile_size - tile.shape[2], 0, tile_size - tile.shape[1]))
            tiles.append(tile)
            coords.append((y, x, y1, x1))

    hr_h = int(H_pad * scale)
    hr_w = int(W_pad * scale)
    return tiles, coords, (hr_h, hr_w)


# ─────────────────────────────────────────────────────────────────────────────
# Tile stitching
# ─────────────────────────────────────────────────────────────────────────────

def stitch_tiles(
    hr_tiles: list[torch.Tensor],
    lr_coords: list[tuple[int, int, int, int]],
    hr_shape: tuple[int, int],
    scale: float,
    out_channels: int = 3,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Blend overlapping HR tiles back into a full image using Hann windowing."""
    H_hr, W_hr = hr_shape
    canvas = torch.zeros(out_channels, H_hr, W_hr, device=device)
    weight = torch.zeros(1, H_hr, W_hr, device=device)

    tile_h = hr_tiles[0].shape[1]
    tile_w = hr_tiles[0].shape[2]
    win = _hann_window_2d(tile_h, device).squeeze(0)

    for tile, (y0, x0, y1, x1) in zip(hr_tiles, lr_coords):
        hy0 = int(y0 * scale)
        hx0 = int(x0 * scale)
        hy1 = min(hy0 + tile_h, H_hr)
        hx1 = min(hx0 + tile_w, W_hr)

        th = hy1 - hy0
        tw = hx1 - hx0

        canvas[:, hy0:hy1, hx0:hx1] += tile[:, :th, :tw] * win[:, :th, :tw]
        weight[:, hy0:hy1, hx0:hx1] += win[:, :th, :tw]

    weight = weight.clamp(min=1e-8)
    return canvas / weight


# ─────────────────────────────────────────────────────────────────────────────
# High-level tile processor
# ─────────────────────────────────────────────────────────────────────────────

class SatelliteTileProcessor:
    """
    End-to-end processor for large satellite images.

    Usage:
        processor = SatelliteTileProcessor(engine, tile_size=256, overlap=32)
        hr_image = processor.process(lr_image_tensor, scale=4.0)
    """

    def __init__(
        self,
        inference_engine,
        tile_size: int = 256,
        overlap: int = 32,
        batch_size: int = 4,
        max_pixels: int = 16_000_000,
    ) -> None:
        self.engine = inference_engine
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.max_pixels = max_pixels

    def process(
        self,
        lr_image: torch.Tensor,
        scale: float = 4.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Super-resolve a full LR satellite image via tiling.

        Args:
            lr_image : (C, H, W) or (1, C, H, W) float tensor in [-1, 1].
            scale    : Upscaling factor.
            seed     : Optional seed for reproducibility.

        Returns:
            (C, H_out, W_out) HR image tensor in [-1, 1].
            Note: The output spatial size depends on the autoencoder's
            decode path. With the current architecture, the latent is
            the same spatial size as input, so output matches input.
        """
        if lr_image.dim() == 4:
            lr_image = lr_image.squeeze(0)

        C, H, W = lr_image.shape
        if H * W > self.max_pixels:
            raise ValueError(
                f"Image too large: {H}x{W} = {H*W:,} px exceeds limit "
                f"of {self.max_pixels:,} px."
            )

        logger.info("Processing %dx%d image at scale=%.1f", H, W, scale)

        tiles, coords, hr_shape = extract_tiles(
            lr_image, self.tile_size, self.overlap, scale
        )
        n_tiles = len(tiles)
        logger.info("Split into %d tiles (%dx%d, overlap=%d)",
                    n_tiles, self.tile_size, self.tile_size, self.overlap)

        hr_tiles: list[torch.Tensor] = []
        for i in range(0, n_tiles, self.batch_size):
            batch = torch.stack(tiles[i : i + self.batch_size])
            hr_batch = self.engine.infer_batch(batch, scale=scale, seed=seed)
            hr_tiles.extend(hr_batch.unbind(0))
            done = min(i + self.batch_size, n_tiles)
            logger.info("Tile %d/%d done", done, n_tiles)

        device = hr_tiles[0].device
        hr_image = stitch_tiles(hr_tiles, coords, hr_shape, scale,
                                out_channels=C, device=device)

        expected_h = int(H * scale)
        expected_w = int(W * scale)
        hr_image = hr_image[:, :expected_h, :expected_w]

        logger.info("Stitched HR image: %dx%d", expected_h, expected_w)
        return hr_image
