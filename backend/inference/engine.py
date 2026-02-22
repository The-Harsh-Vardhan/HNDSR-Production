"""
backend/inference/engine.py
=============================
Production inference engine for HNDSR.

Takes the composite HNDSR model and exposes:
  • infer_batch()  — synchronous batch inference (tiles)
  • infer_async()  — async wrapper for FastAPI
  • warmup()       — pre-run dummy passes

The engine delegates the full pipeline (FNO → ImplicitAmp → Diffusion → Decode)
to the HNDSR.super_resolve() method so inference matches training exactly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class HNDSRInferenceEngine:
    """
    Production inference engine wrapping the composite HNDSR model.

    Key design decisions:
      1. Calls model.super_resolve() so the inference pipeline matches
         the training notebook exactly (bicubic up → FNO → latent resize →
         ImplicitAmp → pool to 1-D context → DDIM → decode).
      2. DDIM steps configurable (default 10 for CPU, 50 for GPU).
      3. Concurrency semaphore prevents GPU OOM under load.
      4. Output spatial size = input_size × scale (true super-resolution).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        ddim_steps: int = 10,
        max_concurrent: int = 4,
    ) -> None:
        self.device = device
        self.ddim_steps = ddim_steps

        self.model = model.to(device).eval()

        # Concurrency guard
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            "HNDSRInferenceEngine ready | device=%s ddim_steps=%d",
            device, ddim_steps,
        )

    # ── Synchronous batch inference ────────────────────────────────────────

    @torch.no_grad()
    def infer_batch(
        self,
        lr_tiles: torch.Tensor,
        scale: float = 4.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run full HNDSR pipeline on a batch of LR tiles.

        Args:
            lr_tiles : (B, 3, H, W) float32 tensor, values in [-1, 1].
            scale    : Upscaling factor (2-8).
            seed     : Optional RNG seed for deterministic output.

        Returns:
            HR tiles (B, 3, H*scale, W*scale) in approx [-1, 1].
        """
        t0 = time.perf_counter()
        B = lr_tiles.shape[0]
        lr_tiles = lr_tiles.to(self.device)

        if seed is not None:
            torch.manual_seed(seed)

        hr_tiles = self.model.super_resolve(
            lr_tiles,
            scale_factor=scale,
            num_inference_steps=self.ddim_steps,
        )

        elapsed = time.perf_counter() - t0
        logger.debug(
            "infer_batch: B=%d scale=%.1f steps=%d elapsed=%.3fs",
            B, scale, self.ddim_steps, elapsed,
        )
        return hr_tiles

    # ── Async wrapper for FastAPI integration ──────────────────────────────

    async def infer_async(
        self,
        lr_tiles: torch.Tensor,
        scale: float = 4.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Async wrapper that acquires the concurrency semaphore before running
        inference in a thread pool executor (keeps the event loop unblocked).
        """
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.infer_batch(lr_tiles, scale, seed),
            )
        return result

    # ── Warmup ────────────────────────────────────────────────────────────

    def warmup(self, tile_size: int = 64, n_runs: int = 2) -> None:
        """
        Run dummy forward passes to warm up CUDA kernels / JIT caches.
        """
        logger.info("Warming up inference engine (%d runs, tile=%d)...",
                     n_runs, tile_size)
        dummy = torch.randn(1, 3, tile_size, tile_size)
        for _ in range(n_runs):
            self.infer_batch(dummy, scale=4.0, seed=0)
        logger.info("Warmup complete.")
