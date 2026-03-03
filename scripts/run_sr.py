"""
scripts/run_sr.py
=================
Standalone script to run HNDSR super-resolution on a sample image
using the production checkpoints.

Model architecture matches dvc_pipeline/src/models.py (the training code
that produced these checkpoints). NOT model_stubs.py, which has different
key names and architecture.

Usage:
    python scripts/run_sr.py                              # defaults
    python scripts/run_sr.py --image path/to/img.png      # custom image
    python scripts/run_sr.py --scale 4 --steps 50         # custom params
    python scripts/run_sr.py --device cuda                # force GPU
    python scripts/run_sr.py --strength 0.2               # light SDEdit

Outputs are saved to ``outputs/`` by default:
    <name>_lr.png       — input LR image (resized)
    <name>_sr.png       — super-resolved image
    <name>_compare.png  — side-by-side: LR↑ (bicubic) | SR
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — Latent Autoencoder  (keys match checkpoint exactly)
# ═════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class LatentAutoencoder(nn.Module):
    """latent_dim=128, downsample_ratio=8 → 3 stride-2 convs + 4 ResBlocks."""

    def __init__(self, in_channels=3, latent_dim=128, num_res_blocks=4,
                 downsample_ratio=8):
        super().__init__()
        num_downs = int(math.log2(downsample_ratio))

        # Encoder
        enc = [nn.Conv2d(in_channels, latent_dim, 3, padding=1)]
        ch = latent_dim
        for _ in range(num_downs):
            out_ch = min(ch * 2, 128)
            enc += [nn.Conv2d(ch, out_ch, 4, stride=2, padding=1),
                    nn.ReLU(True)]
            ch = out_ch
        for _ in range(num_res_blocks):
            enc.append(ResidualBlock(ch))
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dec = [ResidualBlock(ch) for _ in range(num_res_blocks)]
        for _ in range(num_downs):
            out_ch = ch // 2
            dec += [nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                    nn.ReLU(True)]
            ch = out_ch
        dec += [nn.Conv2d(ch, in_channels, 3, padding=1), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — Fourier Neural Operator  (individual attrs: conv0-3, w0-3)
# ═════════════════════════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, 2))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, 2))

    def forward(self, x):
        x = x.float()
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x.size(-2),
            x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, x.size(-2))
        m2 = min(self.modes2, x.size(-1) // 2 + 1)
        if m1 > 0 and m2 > 0:
            w1 = torch.view_as_complex(self.weights1[:, :, :m1, :m2])
            w2 = torch.view_as_complex(self.weights2[:, :, :m1, :m2])
            out_ft[:, :, :m1, :m2] = torch.einsum(
                'bixy,ioxy->boxy', x_ft[:, :, :m1, :m2], w1)
            out_ft[:, :, -m1:, :m2] = torch.einsum(
                'bixy,ioxy->boxy', x_ft[:, :, -m1:, :m2], w2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class NeuralOperator(nn.Module):
    """FNO with conv0-3 / w0-3 individual attributes (matches checkpoint)."""

    def __init__(self, in_channels=3, out_channels=128, modes=8, width=32):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels + 1, width, 1)
        self.conv0 = SpectralConv2d(width, width, modes, modes)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.conv2 = SpectralConv2d(width, width, modes, modes)
        self.conv3 = SpectralConv2d(width, width, modes, modes)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Conv2d(width, 64, 1)
        self.fc2 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, scale_factor):
        b, c, h, w = x.shape
        scale_map = torch.ones(b, 1, h, w, device=x.device) * (scale_factor / 4.0)
        x = self.fc0(torch.cat([x, scale_map], 1))
        for conv, w_conv in zip(
            [self.conv0, self.conv1, self.conv2, self.conv3],
            [self.w0, self.w1, self.w2, self.w3],
        ):
            x = F.gelu(conv(x) + w_conv(x))
        return self.fc2(F.gelu(self.fc1(x)))


# ═════════════════════════════════════════════════════════════════════════════
# Implicit Amplification (no checkpoint — random init)
# ═════════════════════════════════════════════════════════════════════════════

class ImplicitAmplification(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim), nn.Sigmoid(),
        )

    def forward(self, latent, scale_factor):
        b, c, h, w = latent.shape
        if isinstance(scale_factor, (int, float)):
            s = torch.full((b, 1), float(scale_factor), device=latent.device)
        else:
            s = scale_factor.view(b, 1).float()
        gains = self.mlp(s).view(b, -1, 1, 1)
        return latent * (1 + gains)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — Diffusion UNet  (matches dvc_pipeline/src/models.py exactly)
#
# Key differences from the old buggy version:
#   - GroupNorm groups: min(8, ch) not min(32, ch)
#   - mid_attn is CrossAttentionBlock(128, context_dim=128), not SelfAttention
#   - forward(x, t, context)  — context is (B, 128) global vector
#   - DDPMScheduler.add_noise(x_start, noise, timesteps) — returns sample only
# ═════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep encoding (no learnable params)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlockWithTime(nn.Module):
    """Residual block with time embedding — matches training checkpoint keys.

    Checkpoint keys per block: norm1, conv1, time_emb.{0,1}, norm2, conv2,
    [shortcut].
    """

    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_emb(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class CrossAttentionBlock(nn.Module):
    """Cross-attention for FNO context conditioning.

    q comes from spatial features (via Conv2d), kv from 1-D context vector.
    Checkpoint keys: norm, q, kv, proj.
    """

    def __init__(self, channels, context_dim):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        """
        Args:
            x      : (B, C, H, W) spatial feature map
            context: (B, context_dim) global conditioning vector
        """
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)

        q = self.q(x).view(b, c, h * w).transpose(1, 2)   # (B, HW, C)

        kv = self.kv(context)                               # (B, 2C)
        k, v = kv.chunk(2, dim=1)                           # each (B, C)
        k = k.unsqueeze(1)                                  # (B, 1, C)
        v = v.unsqueeze(1)                                  # (B, 1, C)

        scale = c ** -0.5
        attn = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)  # (B, HW, 1)
        out = torch.bmm(attn, v)                              # (B, HW, C)

        out = out.transpose(1, 2).view(b, c, h, w)
        return self.proj(out) + residual


class DiffusionUNet(nn.Module):
    """
    UNet for latent-space denoising — matches training checkpoint.

    Input : (B, 128, H, W) noisy latent z_t
    Output: (B, 128, H, W) noise prediction

    Key: mid_attn is CrossAttentionBlock, conditioned on a (B, 128) context
    vector derived from the FNO prior via global average pooling.
    """

    def __init__(self, in_channels=128, model_channels=64,
                 out_channels=128, context_dim=128):
        super().__init__()
        t_dim = model_channels * 4  # 256

        # Time embedding: sin(64) → Linear(64, 256) → SiLU → Linear(256, 256)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(model_channels),         # idx 0
            nn.Linear(model_channels, t_dim),            # idx 1
            nn.SiLU(),                                   # idx 2
            nn.Linear(t_dim, t_dim),                     # idx 3
        )
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.down1 = ResidualBlockWithTime(
            model_channels, model_channels * 2, t_dim)
        self.down2 = nn.Conv2d(model_channels * 2, model_channels * 2, 3,
                               stride=2, padding=1)

        # Bottleneck — cross-attention, NOT self-attention
        self.mid1 = ResidualBlockWithTime(
            model_channels * 2, model_channels * 2, t_dim)
        self.mid_attn = CrossAttentionBlock(model_channels * 2, context_dim)
        self.mid2 = ResidualBlockWithTime(
            model_channels * 2, model_channels * 2, t_dim)

        # Decoder
        self.up1 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2,
                                      4, stride=2, padding=1)
        self.up2 = ResidualBlockWithTime(
            model_channels * 3, model_channels, t_dim)   # 192 → 64

        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, z_t, t, context):
        """
        Args:
            z_t    : (B, 128, H, W)  noisy latent
            t      : (B,)            integer timesteps
            context: (B, 128)        global FNO prior vector
        Returns:
            (B, 128, H, W) noise prediction
        """
        t_emb = self.time_embed(t)                     # (B, 256)
        h0 = self.input_proj(z_t)                      # (B, 64, H, W) — skip
        h = self.down1(h0, t_emb)                      # (B, 128, H, W)
        h = self.down2(h)                              # (B, 128, H/2, W/2)
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h, context)                  # cross-attention
        h = self.mid2(h, t_emb)
        h = self.up1(h)                                # (B, 128, H, W)
        h = torch.cat([h, h0], dim=1)                  # (B, 192, H, W)
        h = self.up2(h, t_emb)                         # (B, 64, H, W)
        return self.out(h)                             # (B, 128, H, W)


# ═════════════════════════════════════════════════════════════════════════════
# DDPM / DDIM Scheduler  (matches dvc_pipeline/src/models.py)
# ═════════════════════════════════════════════════════════════════════════════

class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod)

    def add_noise(self, x_start, noise, timesteps):
        """Forward diffusion: q(x_t | x_0).

        Args:
            x_start  : clean sample
            noise    : pre-sampled noise (same shape as x_start)
            timesteps: integer timestep tensor

        Returns:
            Noisy sample x_t  (single tensor, NOT a tuple).
        """
        t_cpu = timesteps.cpu()
        sqrt_a = self.sqrt_alphas_cumprod[t_cpu].to(x_start.device)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t_cpu].to(x_start.device)
        while sqrt_a.dim() < x_start.dim():
            sqrt_a = sqrt_a.unsqueeze(-1)
            sqrt_1ma = sqrt_1ma.unsqueeze(-1)
        return sqrt_a * x_start + sqrt_1ma * noise

    def ddim_sample(self, model_output, timestep, sample):
        """Deterministic DDIM reverse step."""
        t = timestep.item() if isinstance(timestep, torch.Tensor) and timestep.numel() == 1 else timestep
        a_t = self.alphas_cumprod[t].to(sample.device)
        a_prev = (self.alphas_cumprod_prev[t].to(sample.device)
                  if t > 0 else torch.tensor(1.0, device=sample.device))
        beta_prod_t = 1.0 - a_t
        x0_pred = (sample - torch.sqrt(beta_prod_t) * model_output
                   ) / torch.sqrt(a_t)
        pred_dir = torch.sqrt(1.0 - a_prev) * model_output
        return torch.sqrt(a_prev) * x0_pred + pred_dir


# ═════════════════════════════════════════════════════════════════════════════
# Complete HNDSR Pipeline
# ═════════════════════════════════════════════════════════════════════════════

class HNDSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = LatentAutoencoder(3, 128, 4, 8)
        self.neural_operator = NeuralOperator(3, 128, 8, 32)
        self.implicit_amp = ImplicitAmplification(128, 256)
        self.diffusion_unet = DiffusionUNet(
            in_channels=128, model_channels=64,
            out_channels=128, context_dim=128)
        self.scheduler = DDPMScheduler(1000)
        self.ae_downsample_ratio = 8

    def get_no_prior(self, lr, scale):
        up = F.interpolate(lr, scale_factor=scale, mode="bicubic",
                           align_corners=False)
        feat = self.neural_operator(up, scale)
        latent_size = up.shape[-1] // self.ae_downsample_ratio
        return F.interpolate(feat, size=(latent_size, latent_size),
                             mode="bilinear", align_corners=False)

    @torch.no_grad()
    def super_resolve(self, lr, scale_factor=4, num_inference_steps=50,
                      diffusion_strength=0.0):
        """
        Full HNDSR inference pipeline using SDEdit.

        1. Neural Operator → initial latent prior
        2. Implicit Amplification → channel-wise gain modulation
        3. If strength > 0: add noise + DDIM denoise with cross-attention
        4. Decode with autoencoder

        Args:
            lr                : (B, 3, H, W) LR input in [-1, 1]
            scale_factor      : upscaling factor
            num_inference_steps: DDIM steps at full strength
            diffusion_strength: 0.0 = skip diffusion (recommended default),
                                0.1–0.3 = light SDEdit,
                                1.0 = start from pure noise
        Returns:
            SR image (B, 3, H', W') in approx [-1, 1]
        """
        device = lr.device
        b = lr.shape[0]

        # 1 — FNO prior + implicit amplification
        no_prior = self.get_no_prior(lr, scale_factor)
        no_prior = self.implicit_amp(no_prior, scale_factor)

        if diffusion_strength <= 0.0:
            # No diffusion — direct autoencoder decode
            return self.autoencoder.decode(no_prior)

        # 2 — Compute context vector for cross-attention
        context = F.adaptive_avg_pool2d(no_prior, 1).view(b, -1)  # (B, 128)

        # 3 — SDEdit: add noise to the prior
        start_timestep = min(
            int(diffusion_strength * self.scheduler.num_timesteps),
            self.scheduler.num_timesteps - 1)
        actual_steps = max(int(num_inference_steps * diffusion_strength), 1)

        if diffusion_strength >= 1.0:
            z_t = torch.randn_like(no_prior)
            timesteps = torch.linspace(
                self.scheduler.num_timesteps - 1, 0,
                num_inference_steps, dtype=torch.long)
        else:
            noise = torch.randn_like(no_prior)
            t_tensor = torch.full((b,), start_timestep, dtype=torch.long)
            z_t = self.scheduler.add_noise(no_prior, noise, t_tensor)
            timesteps = torch.linspace(
                start_timestep, 0, actual_steps, dtype=torch.long)

        # 4 — DDIM denoise from t_start down to 0
        for t in timesteps:
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            eps = self.diffusion_unet(z_t, t_batch, context)
            z_t = self.scheduler.ddim_sample(eps, t, z_t)

        # 5 — Decode
        return self.autoencoder.decode(z_t)


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ═════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_dir: Path, device: torch.device) -> HNDSR:
    model = HNDSR()

    # Stage 1 — Autoencoder (raw OrderedDict)
    ae_path = checkpoint_dir / "autoencoder_best.pth"
    ae_raw = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae_state = (ae_raw.get("model_state_dict", ae_raw)
                if isinstance(ae_raw, dict) and "model_state_dict" in ae_raw
                else ae_raw)
    model.autoencoder.load_state_dict(ae_state, strict=True)
    print(f"  [OK] Autoencoder   <- {ae_path.name}")

    # Stage 2 — Neural Operator (raw OrderedDict, conv0/w0 naming)
    no_path = checkpoint_dir / "neural_operator_best.pth"
    no_raw = torch.load(no_path, map_location="cpu", weights_only=False)
    no_state = (no_raw.get("model_state_dict", no_raw)
                if isinstance(no_raw, dict) and "model_state_dict" in no_raw
                else no_raw)
    model.neural_operator.load_state_dict(no_state, strict=True)
    print(f"  [OK] NeuralOp      <- {no_path.name}")

    # Stage 3 — Diffusion UNet (dict with 'diffusion_unet' key)
    diff_path = checkpoint_dir / "diffusion_best.pth"
    diff_raw = torch.load(diff_path, map_location="cpu", weights_only=False)
    if isinstance(diff_raw, dict) and "ema_shadow" in diff_raw:
        diff_state = diff_raw["ema_shadow"]
        tag = "EMA shadow"
    elif isinstance(diff_raw, dict) and "diffusion_unet" in diff_raw:
        diff_state = diff_raw["diffusion_unet"]
        tag = "training"
    elif isinstance(diff_raw, dict) and "model_state_dict" in diff_raw:
        diff_state = diff_raw["model_state_dict"]
        tag = "state_dict"
    else:
        diff_state = diff_raw
        tag = "raw"
    model.diffusion_unet.load_state_dict(diff_state, strict=True)
    print(f"  [OK] DiffusionUNet <- {diff_path.name} ({tag} weights)")

    model.eval()
    model = model.to(device)
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [OK] Full HNDSR ready  {total_p:.1f}M params  device={device}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Image I/O helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_image(path: Path, max_size: int = 256) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load image, resize to ≤ max_size (divisible by 8), normalise to [-1,1]."""
    img = Image.open(path).convert("RGB")
    original_size = img.size  # (W, H)

    w, h = img.size
    ratio = min(max_size / w, max_size / h, 1.0)
    new_w = max((int(w * ratio) // 8) * 8, 8)
    new_h = max((int(h * ratio) // 8) * 8, 8)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
    return tensor.unsqueeze(0), original_size


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    img = tensor.squeeze(0).clamp(-1, 1)
    img = ((img + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))


def make_comparison(lr_tensor: torch.Tensor,
                    sr_tensor: torch.Tensor) -> Image.Image:
    """Side-by-side: LR (bicubic-upscaled) | SR."""
    sr_h, sr_w = sr_tensor.shape[2], sr_tensor.shape[3]
    lr_up = F.interpolate(lr_tensor, size=(sr_h, sr_w), mode="bicubic",
                          align_corners=False)
    lr_pil = tensor_to_pil(lr_up)
    sr_pil = tensor_to_pil(sr_tensor)
    gap = 4
    canvas = Image.new("RGB",
                        (lr_pil.width + sr_pil.width + gap, sr_pil.height),
                        (40, 40, 40))
    canvas.paste(lr_pil, (0, 0))
    canvas.paste(sr_pil, (lr_pil.width + gap, 0))
    return canvas


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run HNDSR super-resolution on a sample image.")
    parser.add_argument(
        "--image", type=str,
        default=str(PROJECT_ROOT / "tests" / "Sample Images"
                    / "HG_Satellite_LoRes_Pic1_TerraColor.avif"),
        help="Path to LR input image")
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50,
                        help="DDIM sampling steps")
    parser.add_argument("--max-size", type=int, default=128,
                        help="Max input side in px (for memory)")
    parser.add_argument("--strength", type=float, default=0.0,
                        help="SDEdit strength: 0=no diffusion (recommended), "
                             "0.1-0.3=light SDEdit, 1=pure noise")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable → falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  HNDSR Super-Resolution Inference")
    print(f"{'='*60}")
    print(f"  Image    : {Path(args.image).name}")
    print(f"  Scale    : {args.scale}x")
    print(f"  Steps    : {args.steps}")
    print(f"  Strength : {args.strength}")
    print(f"  Max size : {args.max_size}px")
    print(f"  Device   : {device}")
    print(f"  Seed     : {args.seed}")
    print(f"{'='*60}\n")

    # 1 — Load model
    print("[1/4] Loading HNDSR model...")
    model = load_model(PROJECT_ROOT / "checkpoints", device)

    # 2 — Load image
    print(f"\n[2/4] Loading image...")
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"  ERROR: not found: {image_path}")
        sys.exit(1)
    lr_tensor, orig_size = load_image(image_path, args.max_size)
    lr_tensor = lr_tensor.to(device)
    print(f"  Original : {orig_size[0]}x{orig_size[1]}")
    print(f"  Input    : {lr_tensor.shape[3]}x{lr_tensor.shape[2]}")

    # 3 — Super-resolve
    print(f"\n[3/4] Running SR ({args.steps} DDIM steps, "
          f"strength={args.strength})...")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    t0 = time.perf_counter()
    with torch.no_grad():
        sr_tensor = model.super_resolve(
            lr_tensor,
            scale_factor=args.scale,
            num_inference_steps=args.steps,
            diffusion_strength=args.strength,
        )
    elapsed = time.perf_counter() - t0
    print(f"  Output   : {sr_tensor.shape[3]}x{sr_tensor.shape[2]}")
    print(f"  Time     : {elapsed:.2f}s")

    # 4 — Save
    print(f"\n[4/4] Saving outputs...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    lr_pil = tensor_to_pil(lr_tensor)
    sr_pil = tensor_to_pil(sr_tensor)
    compare_pil = make_comparison(lr_tensor, sr_tensor)

    lr_path = out_dir / f"{stem}_lr.png"
    sr_path = out_dir / f"{stem}_sr.png"
    cmp_path = out_dir / f"{stem}_compare.png"

    lr_pil.save(lr_path)
    sr_pil.save(sr_path)
    compare_pil.save(cmp_path)

    print(f"  LR      -> {lr_path}")
    print(f"  SR      -> {sr_path}")
    print(f"  Compare -> {cmp_path}")

    print(f"\n{'='*60}")
    print(f"  Done! SR: {sr_pil.width}x{sr_pil.height}  ({elapsed:.1f}s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
