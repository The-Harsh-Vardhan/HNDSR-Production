"""
backend/model/model_stubs.py
=============================
HNDSR model architecture — reverse-engineered from the actual checkpoint
tensor shapes so that ``load_state_dict(strict=True)`` succeeds.

CRITICAL: Every nn.Module attribute name, channel count and kernel size
MUST match the saved ``state_dict`` keys.  A mismatch → silent weight-skip
(or RuntimeError with strict=True) → random noise output.

Checkpoint shapes (inspected with ``torch.load``):
  autoencoder_best.pth      — raw OrderedDict   (encoder.0 … decoder.10)
  neural_operator_best.pth  — raw OrderedDict   (fc0 … fc2)
  diffusion_best.pth        — {"model_state_dict": OrderedDict}
                               (t_embed.proj.0 … out)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Residual Block & Latent Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Two-conv residual block used inside the autoencoder."""
    def __init__(self, channels, use_bn=False):
        super().__init__()
        layers = [nn.Conv2d(channels, channels, 3, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(channels, channels, 3, padding=1)]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class LatentAutoencoder(nn.Module):
    """Encoder-Decoder that maps 3-ch images → compact latent z → back.

    Default config (matching checkpoint): latent_dim=128, downsample_ratio=8.
    Encoder: Conv(3→128) then 3× stride-2 Conv(128→128) + 4× ResBlock(128).
    Decoder: 4× ResBlock(128) then ConvT(128→64→32→16) → Conv(16→3) → Tanh.
    """
    def __init__(self, in_channels=3, latent_dim=64, num_res_blocks=4, downsample_ratio=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.downsample_ratio = downsample_ratio
        num_downs = int(math.log2(downsample_ratio))

        # Encoder
        enc = [nn.Conv2d(in_channels, latent_dim, 3, padding=1)]
        ch = latent_dim
        for _ in range(num_downs):
            out_ch = min(ch * 2, 128)
            enc += [nn.Conv2d(ch, out_ch, 4, stride=2, padding=1), nn.ReLU(True)]
            ch = out_ch
        for _ in range(num_res_blocks):
            enc.append(ResidualBlock(ch))
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dec = [ResidualBlock(ch) for _ in range(num_res_blocks)]
        for _ in range(num_downs):
            out_ch = ch // 2
            dec += [nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1), nn.ReLU(True)]
            ch = out_ch
        dec += [nn.Conv2d(ch, in_channels, 3, padding=1), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Fourier Neural Operator
# ─────────────────────────────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """2-D spectral convolution — real-valued 5-D weights (last dim=2)."""
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, 2))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(x.shape[0], self.out_channels, x.size(-2),
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

        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x_out.to(x_dtype) if x_dtype != torch.float32 else x_out


class NeuralOperator(nn.Module):
    """FNO with 4 spectral layers and scale-map conditioning."""
    def __init__(self, in_channels=3, out_channels=128, modes=8, width=32):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels + 1, width, 1)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(4)])
        self.ws = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Conv2d(width, 64, 1)
        self.fc2 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, scale_factor):
        b, c, h, w = x.shape
        scale_map = torch.ones(b, 1, h, w, device=x.device) * (scale_factor / 4.0)
        x = self.fc0(torch.cat([x, scale_map], 1))
        for conv, w_conv in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w_conv(x))
        return self.fc2(F.gelu(self.fc1(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Implicit Amplification MLP
# ─────────────────────────────────────────────────────────────────────────────

class ImplicitAmplification(nn.Module):
    """Scale-conditioned channel gain predictor."""
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim), nn.Sigmoid(),
        )

    def forward(self, latent, scale_factor):
        b = latent.shape[0]
        s = torch.full(
            (b, 1),
            float(scale_factor) if isinstance(scale_factor, (int, float))
            else scale_factor.item(),
            device=latent.device, dtype=torch.float32,
        )
        return latent * (1 + self.mlp(s).view(b, -1, 1, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Diffusion UNet (matching actual checkpoint shapes)
# ─────────────────────────────────────────────────────────────────────────────
#
# Checkpoint key naming:
#   t_embed.proj.{0,2}  — SinusoidalPositionEmbeddings with internal MLP
#   ctx_proj             — 1×1 Conv for spatial context
#   enc1 / enc2          — DiffResBlocks in encoder
#   down1 / down2        — strided Conv2d downsamples
#   mid                  — DiffResBlock at bottleneck
#   up2 / up1            — transposed Conv2d upsamples
#   dec2 / dec1          — DiffResBlocks in decoder (with skip connections)
#   out                  — final 3×3 Conv2d
#
# DiffResBlock attributes: norm1, conv1, t_proj, norm2, conv2, skip
#   (NOT time_emb / shortcut like the notebook version)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep encoding with an internal learned MLP projection.

    Checkpoint keys: ``t_embed.proj.0`` (Linear), ``t_embed.proj.2`` (Linear).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, time):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=time.device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.proj(emb)


class DiffResBlock(nn.Module):
    """Residual block conditioned on time embedding.

    Checkpoint keys per block: norm1, conv1, t_proj, norm2, conv2, [skip].
    """
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch
                     else nn.Identity())

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DiffusionUNet(nn.Module):
    """UNet for latent-space denoising — architecture matches checkpoint.

    Input is the *concatenation* of z_t (noise, ``out_channels`` ch) and
    projected context (``context_dim`` ch), giving ``in_channels`` ch total.
    Output predicts noise in ``out_channels`` ch.

    Default dims (from checkpoint):
        in_channels=128, model_channels=128, out_channels=64, context_dim=64
    """
    def __init__(self, in_channels=128, model_channels=128,
                 out_channels=64, context_dim=64):
        super().__init__()
        mc = model_channels
        t_dim = mc  # time-embedding dimensionality

        self.t_embed = SinusoidalPositionEmbeddings(mc)
        self.ctx_proj = nn.Conv2d(context_dim, context_dim, 1)

        # ── Encoder ──
        self.enc1 = DiffResBlock(in_channels, mc, t_dim)
        self.down1 = nn.Conv2d(mc, mc, 4, stride=2, padding=1)
        self.enc2 = DiffResBlock(mc, mc * 2, t_dim)
        self.down2 = nn.Conv2d(mc * 2, mc * 2, 4, stride=2, padding=1)

        # ── Bottleneck ──
        self.mid = DiffResBlock(mc * 2, mc * 2, t_dim)

        # ── Decoder (skip connections via concatenation) ──
        self.up2 = nn.ConvTranspose2d(mc * 2, mc * 2, 4, stride=2, padding=1)
        self.dec2 = DiffResBlock(mc * 4, mc, t_dim)       # cat(up2, enc2) = mc*2+mc*2
        self.up1 = nn.ConvTranspose2d(mc, mc, 4, stride=2, padding=1)
        self.dec1 = DiffResBlock(mc * 2, out_channels, t_dim)  # cat(up1, enc1) = mc+mc
        self.out = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, z_t, t, context):
        """
        Args:
            z_t:     (B, out_channels, H, W) — noisy latent
            t:       (B,) — integer timesteps
            context: (B, context_dim, H, W) — spatial context features
        Returns:
            noise prediction (B, out_channels, H, W)
        """
        t_emb = self.t_embed(t)
        ctx = self.ctx_proj(context)
        x = torch.cat([z_t, ctx], dim=1)          # (B, in_channels, H, W)

        h0 = self.enc1(x, t_emb)                  # skip-0
        h = self.down1(h0)
        h1 = self.enc2(h, t_emb)                  # skip-1
        h = self.down2(h1)

        h = self.mid(h, t_emb)

        h = self.up2(h)
        h = self.dec2(torch.cat([h, h1], 1), t_emb)
        h = self.up1(h)
        h = self.dec1(torch.cat([h, h0], 1), t_emb)
        return self.out(h)


# ─────────────────────────────────────────────────────────────────────────────
# DDPM Noise Scheduler (with DDIM sampling)
# ─────────────────────────────────────────────────────────────────────────────

class DDPMScheduler:
    """Linear-beta DDPM scheduler with DDIM sampling."""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)

    def ddim_sample(self, eps_pred, t, x_t):
        """Deterministic DDIM reverse step."""
        t_val = t.item() if isinstance(t, torch.Tensor) and t.numel() == 1 else t
        a_t = self.alphas_cumprod[t_val].to(x_t.device)
        a_prev = (self.alphas_cumprod_prev[t_val].to(x_t.device)
                  if t_val > 0 else torch.tensor(1.0, device=x_t.device))
        x0_pred = (x_t - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)
        return torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps_pred


# ─────────────────────────────────────────────────────────────────────────────
# Complete HNDSR Model
# ─────────────────────────────────────────────────────────────────────────────

class HNDSR(nn.Module):
    """Hybrid Neural Operator-Diffusion Super-Resolution.

    The diffusion UNet operates in a *reduced* 64-channel space while the
    autoencoder latent is 128 channels.  Two lightweight 1×1 convolutions
    (``no_proj`` and ``diff_to_latent``) bridge the gap.  These projections
    have **no saved checkpoint** — they're initialised to a sensible default
    (pair-wise channel averaging / duplication).
    """
    def __init__(self, ae_latent_dim=128, ae_downsample_ratio=8,
                 no_width=32, no_modes=8, diffusion_channels=64,
                 num_timesteps=1000):
        super().__init__()
        self.ae_downsample_ratio = ae_downsample_ratio
        self.ae_latent_dim = ae_latent_dim
        self.diff_channels = diffusion_channels

        self.autoencoder = LatentAutoencoder(
            3, ae_latent_dim, 4, ae_downsample_ratio)
        self.neural_operator = NeuralOperator(
            3, ae_latent_dim, no_modes, no_width)
        self.implicit_amp = ImplicitAmplification(ae_latent_dim, 256)
        self.diffusion_unet = DiffusionUNet(
            in_channels=ae_latent_dim,        # 128 (= diff_ch + context_ch)
            model_channels=ae_latent_dim,     # 128
            out_channels=diffusion_channels,  # 64  (noise prediction space)
            context_dim=diffusion_channels,   # 64
        )
        self.scheduler = DDPMScheduler(num_timesteps)

        # Projections between 128-ch latent and 64-ch diffusion space.
        # NOT saved in any checkpoint — initialised to channel-pair defaults.
        self.no_proj = nn.Conv2d(ae_latent_dim, diffusion_channels, 1, bias=False)
        self.diff_to_latent = nn.Conv2d(diffusion_channels, ae_latent_dim, 1, bias=False)
        self._init_projections()

    def _init_projections(self):
        """Channel-pair averaging (128→64) and duplication (64→128)."""
        dc, lc = self.diff_channels, self.ae_latent_dim
        with torch.no_grad():
            self.no_proj.weight.zero_()
            for i in range(dc):
                self.no_proj.weight[i, 2 * i, 0, 0] = 0.5
                self.no_proj.weight[i, 2 * i + 1, 0, 0] = 0.5
            self.diff_to_latent.weight.zero_()
            for i in range(dc):
                self.diff_to_latent.weight[2 * i, i, 0, 0] = 1.0
                self.diff_to_latent.weight[2 * i + 1, i, 0, 0] = 1.0

    def get_no_prior(self, lr, scale):
        up = F.interpolate(
            lr, scale_factor=scale, mode='bicubic', align_corners=False)
        feat = self.neural_operator(up, scale)
        s = up.shape[-1] // self.ae_downsample_ratio
        return F.interpolate(
            feat, size=(s, s), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def super_resolve(self, lr, scale_factor=4, num_inference_steps=50):
        """Full inference: LR → SR image via diffusion sampling."""
        b = lr.shape[0]

        # Neural-operator prior (128-ch spatial)
        no_prior = self.implicit_amp(
            self.get_no_prior(lr, scale_factor), scale_factor)

        # Project 128→64 for diffusion context
        context_64 = self.no_proj(no_prior)

        # Noise lives in 64-ch space matching UNet output
        z_t = torch.randn(
            b, self.diff_channels,
            context_64.shape[2], context_64.shape[3],
            device=lr.device)

        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0,
            num_inference_steps, dtype=torch.long)
        for t in timesteps:
            t_batch = torch.full(
                (b,), t, device=lr.device, dtype=torch.long)
            z_t = self.scheduler.ddim_sample(
                self.diffusion_unet(z_t, t_batch, context_64), t, z_t)

        # Expand 64→128-ch latent for autoencoder decoder
        z_128 = self.diff_to_latent(z_t)
        return self.autoencoder.decode(z_128)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat aliases
# ─────────────────────────────────────────────────────────────────────────────

HNDSRAutoencoder = LatentAutoencoder
HNDSRNeuralOperator = NeuralOperator
HNDSRDiffusionUNet = DiffusionUNet
HNDSRModel = HNDSR
