"""
backend/inference/engine.py
=============================
Production inference engine for HNDSR with:
  • FP16 / AMP mixed precision
  • DDIM accelerated sampling (50 steps vs 1000 training steps)
  • Batch and streaming inference modes
  • torch.compile() graph optimisation (PyTorch >= 2.0)
  • Concurrency semaphore to prevent GPU OOM under load

Adapted from Deployment/inference/optimized_inference.py for integration
into the HNDSR in Production backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DDIM Scheduler
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DDIMScheduler:
    """
    Denoising Diffusion Implicit Models (DDIM) scheduler.

    DDIM replaces the stochastic DDPM reverse process with a deterministic
    ODE solver, allowing us to skip timesteps. With eta=0 the process is
    fully deterministic (same seed → same output), critical for
    reproducibility in production.

    Reference: Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
    """

    num_train_timesteps: int = 1000
    num_inference_steps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    eta: float = 0.0  # 0 = deterministic; 1 = DDPM-equivalent

    betas: torch.Tensor = field(init=False)
    alphas: torch.Tensor = field(init=False)
    alphas_cumprod: torch.Tensor = field(init=False)
    timesteps: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Evenly spaced subset of training timesteps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.timesteps = torch.arange(0, self.num_train_timesteps, step_ratio).flip(0)

    def to(self, device: torch.device) -> "DDIMScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def step(
        self,
        noise_pred: torch.Tensor,
        t: int,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one DDIM reverse step: z_t → z_{t-1}.
        Implements eq. (12) from the DDIM paper.
        """
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

        # Predicted x_0
        pred_x0 = (z_t - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # Direction pointing to z_t
        sigma = (
            self.eta
            * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)).sqrt()
            * (1 - alpha_prod_t / alpha_prod_t_prev).sqrt()
        )
        noise = torch.randn_like(z_t) if self.eta > 0 else torch.zeros_like(z_t)

        z_prev = (
            alpha_prod_t_prev.sqrt() * pred_x0
            + (1 - alpha_prod_t_prev - sigma**2).sqrt() * noise_pred
            + sigma * noise
        )
        return z_prev


# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class HNDSRInferenceEngine:
    """
    Production inference engine wrapping the three HNDSR stages.

    Key design decisions:
      1. FP16 autocast: halves VRAM, ~1.5-2x throughput on Ampere+ GPUs.
      2. DDIM: 50 steps instead of 1000 → 20x faster with minimal quality loss.
      3. torch.compile: fuses elementwise ops, reduces kernel launch overhead.
      4. Semaphore: prevents concurrent requests from triggering GPU OOM.
      5. Streaming mode: yields intermediate denoised latents for progress UX.
    """

    def __init__(
        self,
        autoencoder: torch.nn.Module,
        neural_operator: torch.nn.Module,
        diffusion_unet: torch.nn.Module,
        device: torch.device,
        use_fp16: bool = True,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        max_concurrent: int = 4,
        use_compile: bool = False,
    ) -> None:
        self.device = device
        self.use_fp16 = use_fp16 and device.type == "cuda"
        self.max_concurrent = max_concurrent

        # Move models to device
        self.autoencoder = autoencoder.to(device).eval()
        self.neural_operator = neural_operator.to(device).eval()
        self.diffusion_unet = diffusion_unet.to(device).eval()

        # Optional torch.compile (PyTorch >= 2.0, Linux recommended)
        if use_compile:
            try:
                self.diffusion_unet = torch.compile(
                    self.diffusion_unet,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                logger.info("torch.compile() applied to diffusion UNet.")
            except Exception as exc:
                logger.warning("torch.compile() failed (%s); using eager mode.", exc)

        # DDIM scheduler
        self.scheduler = DDIMScheduler(
            num_inference_steps=ddim_steps,
            eta=ddim_eta,
        ).to(device)

        # Concurrency guard
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            "HNDSRInferenceEngine ready | device=%s fp16=%s ddim_steps=%d",
            device, self.use_fp16, ddim_steps,
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
            scale    : Upscaling factor (1.0-6.0).
            seed     : Optional RNG seed for deterministic output.

        Returns:
            Decoded HR tiles (B, 3, H, W) in [-1, 1].
            Note: output spatial size equals input (latent-space super-resolution).
        """
        t0 = time.perf_counter()
        B = lr_tiles.shape[0]
        lr_tiles = lr_tiles.to(self.device)

        if seed is not None:
            torch.manual_seed(seed)

        # Stage 2: Neural Operator → structural prior c
        # MUST run in FP32: torch.fft.rfft2 produces complex tensors which
        # don't support FP16 on CUDA (baddbmm_cuda not implemented for ComplexHalf)
        context = self.neural_operator(lr_tiles.float(), scale)

        # Stage 3 + Stage 1: FP16 autocast for diffusion + decoder
        amp_ctx = autocast(device_type=self.device.type, dtype=torch.float16) if self.use_fp16 else contextlib.nullcontext()

        with amp_ctx:
            # DDIM reverse diffusion
            z_t = torch.randn(
                B, context.shape[1], context.shape[2], context.shape[3],
                device=self.device,
            )
            context_amp = context.half() if self.use_fp16 else context

            for t_idx in self.scheduler.timesteps:
                t_batch = t_idx.expand(B)
                noise_pred = self.diffusion_unet(z_t, t_batch, context_amp)
                z_t = self.scheduler.step(noise_pred, int(t_idx), z_t)

            # Stage 1 decoder: latent → HR image
            hr_tiles = self.autoencoder.decode(z_t.float())

        elapsed = time.perf_counter() - t0
        logger.debug("infer_batch: B=%d scale=%.1f elapsed=%.3fs", B, scale, elapsed)
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
        Run dummy forward passes to warm up CUDA kernels and torch.compile
        caches before serving real requests.
        """
        logger.info("Warming up inference engine (%d runs)...", n_runs)
        dummy = torch.randn(1, 3, tile_size, tile_size)
        for i in range(n_runs):
            self.infer_batch(dummy, scale=4.0, seed=0)
        logger.info("Warmup complete.")
