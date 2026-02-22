"""
backend/model/model_stubs.py
=============================
HNDSR model architecture — exact copy of training notebook definitions.

CRITICAL: These classes MUST match the training code identically, including
layer names, channel counts, and parameter shapes. Any mismatch causes
`load_state_dict` to silently skip weights (strict=False) → random noise.

Architecture (from HNDSR_Kaggle_Updated.ipynb):
  Stage 1 — LatentAutoencoder  (128-ch latent, 8× spatial reduction)
  Stage 2 — FourierNeuralOperator  (8 Fourier modes, width=32)
  Implicit Amplification  (scale-conditioned channel gain)
  Stage 3 — DiffusionUNet  (cross-attention, 64 model channels)
  DDPMScheduler  (1000 timesteps, DDIM sampling)
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
    """Encoder-Decoder that maps 3-ch images to a compact latent z and back.

    Default config: 128-ch latent, 3 downsamples (8x spatial reduction).
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
    """2-D spectral convolution — applies learnable weights in Fourier space.

    Uses real-valued 5D parameters (last dim=2 for real/imag) and
    view_as_complex at forward time, matching the training notebook exactly.
    Forces float32 FFT to avoid cuFFT half-precision limitations.
    """
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
# Stage 3: Diffusion UNet Components
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    """Maps integer timestep t -> sinusoidal embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class CrossAttentionBlock(nn.Module):
    """Cross-attention: UNet features attend to Neural-Operator context vector."""
    def __init__(self, channels, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        res = x
        x = self.norm(x)
        q = self.q(x).view(b, c, -1).transpose(1, 2)
        kv = self.kv(context)
        k, v = kv.chunk(2, 1)
        k, v = k.unsqueeze(1), v.unsqueeze(1)
        attn = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) * (c ** -0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(b, c, h, w)
        return self.proj(out) + res


class ResidualBlockWithTime(nn.Module):
    """ResBlock conditioned on a timestep embedding (added after first conv)."""
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch
                         else nn.Identity())

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class DiffusionUNet(nn.Module):
    """Simplified UNet for latent-space denoising.

    Architecture: input_proj -> Down -> Mid (cross-attn) -> Up -> output
    Skip connection from input_proj to Up via concatenation.
    """
    def __init__(self, in_channels=128, model_channels=64,
                 out_channels=128, context_dim=128):
        super().__init__()
        t_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down
        self.down1 = ResidualBlockWithTime(
            model_channels, model_channels * 2, t_dim)
        self.down2 = nn.Conv2d(
            model_channels * 2, model_channels * 2, 3, stride=2, padding=1)

        # Mid
        self.mid1 = ResidualBlockWithTime(
            model_channels * 2, model_channels * 2, t_dim)
        self.mid_attn = CrossAttentionBlock(model_channels * 2, context_dim)
        self.mid2 = ResidualBlockWithTime(
            model_channels * 2, model_channels * 2, t_dim)

        # Up
        self.up1 = nn.ConvTranspose2d(
            model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        self.up2 = ResidualBlockWithTime(
            model_channels * 3, model_channels, t_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels), nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, t, context):
        t_emb = self.time_embed(t)
        h = self.input_proj(x)    # skip source
        h0 = h
        h = self.down2(self.down1(h, t_emb))
        h = self.mid2(self.mid_attn(self.mid1(h, t_emb), context), t_emb)
        h = self.up2(torch.cat([self.up1(h), h0], dim=1), t_emb)
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

    This is the complete model matching the training notebook exactly.
    """
    def __init__(self, ae_latent_dim=128, ae_downsample_ratio=8,
                 no_width=32, no_modes=8, diffusion_channels=64,
                 num_timesteps=1000):
        super().__init__()
        self.ae_downsample_ratio = ae_downsample_ratio
        self.autoencoder = LatentAutoencoder(
            3, ae_latent_dim, 4, ae_downsample_ratio)
        self.neural_operator = NeuralOperator(
            3, ae_latent_dim, no_modes, no_width)
        self.implicit_amp = ImplicitAmplification(ae_latent_dim, 256)
        self.diffusion_unet = DiffusionUNet(
            ae_latent_dim, diffusion_channels,
            ae_latent_dim, ae_latent_dim)
        self.scheduler = DDPMScheduler(num_timesteps)

    def get_no_prior(self, lr, scale):
        up = F.interpolate(
            lr, scale_factor=scale, mode='bicubic', align_corners=False)
        feat = self.neural_operator(up, scale)
        s = up.shape[-1] // self.ae_downsample_ratio
        return F.interpolate(
            feat, size=(s, s), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def super_resolve(self, lr, scale_factor=4, num_inference_steps=50):
        """Full inference: LR -> SR image via diffusion sampling."""
        b = lr.shape[0]
        no_prior = self.implicit_amp(
            self.get_no_prior(lr, scale_factor), scale_factor)
        context = F.adaptive_avg_pool2d(no_prior, 1).view(b, -1)
        z_t = torch.randn(no_prior.shape, device=lr.device)

        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0,
            num_inference_steps, dtype=torch.long)
        for t in timesteps:
            t_batch = torch.full(
                (b,), t, device=lr.device, dtype=torch.long)
            z_t = self.scheduler.ddim_sample(
                self.diffusion_unet(z_t, t_batch, context), t, z_t)
        return self.autoencoder.decode(z_t)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat aliases used by generate_checkpoints.py
# ─────────────────────────────────────────────────────────────────────────────

HNDSRAutoencoder = LatentAutoencoder
HNDSRNeuralOperator = NeuralOperator
HNDSRDiffusionUNet = DiffusionUNet
HNDSRModel = HNDSR
