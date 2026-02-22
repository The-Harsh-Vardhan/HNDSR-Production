"""
backend/model/model_stubs.py
=============================
HNDSR model architecture — all three stages as PyTorch modules.

Architecture (from the HNDSR paper):
  Stage 1 – Convolutional Autoencoder  (Encoder E_θ + Decoder D_φ)
  Stage 2 – Fourier Neural Operator    (φ_NO)
  Stage 3 – Diffusion UNet             (ε_θ conditioned on c = φ_NO(I_LR))

These are production-ready model definitions imported from the original
Deployment/serialization/model_stubs.py. They are used by the inference
engine and checkpoint generation scripts.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Convolutional Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Basic residual block used in the autoencoder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class HNDSREncoder(nn.Module):
    """E_θ: maps HR image → latent z_HR."""

    def __init__(self, in_ch: int = 3, latent_ch: int = 64) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.down1 = nn.Sequential(ResBlock(64), nn.Conv2d(64, 128, 4, stride=2, padding=1))
        self.down2 = nn.Sequential(ResBlock(128), nn.Conv2d(128, latent_ch, 4, stride=2, padding=1))
        self.out_norm = nn.GroupNorm(8, latent_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        return self.out_norm(x)


class HNDSRDecoder(nn.Module):
    """D_φ: maps latent z → reconstructed image."""

    def __init__(self, latent_ch: int = 64, out_ch: int = 3) -> None:
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 128, 4, stride=2, padding=1),
            ResBlock(128),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            ResBlock(64),
        )
        self.head = nn.Conv2d(64, out_ch, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.up1(z)
        x = self.up2(x)
        return torch.tanh(self.head(x))


class HNDSRAutoencoder(nn.Module):
    """Full autoencoder: E_θ ∘ D_φ."""

    def __init__(self, in_ch: int = 3, latent_ch: int = 64) -> None:
        super().__init__()
        self.encoder = HNDSREncoder(in_ch, latent_ch)
        self.decoder = HNDSRDecoder(latent_ch, in_ch)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Fourier Neural Operator
# ─────────────────────────────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """
    Fourier layer: multiply in frequency domain with learnable weights W_k.
    Implements eq. (17)-(18) from the HNDSR paper.
    """

    def __init__(self, in_ch: int, out_ch: int, modes: int = 12) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.weights.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes, H), min(self.modes, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """Single FNO layer: spectral conv + pointwise residual (eq. 19)."""

    def __init__(self, channels: int, modes: int = 12) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))


class HNDSRNeuralOperator(nn.Module):
    """
    φ_NO: LR image + scale factor → latent z_NO  (eq. 3).
    Scale is embedded via a small MLP and added to the feature map.
    """

    def __init__(
        self,
        in_ch: int = 3,
        latent_ch: int = 64,
        fno_layers: int = 4,
        modes: int = 12,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_ch, latent_ch, 1)
        self.scale_embed = nn.Sequential(
            nn.Linear(1, latent_ch), nn.SiLU(), nn.Linear(latent_ch, latent_ch)
        )
        self.fno_blocks = nn.Sequential(*[FNOBlock(latent_ch, modes) for _ in range(fno_layers)])
        self.proj = nn.Conv2d(latent_ch, latent_ch, 1)

    def forward(self, x_lr: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
        B = x_lr.shape[0]
        h = self.lift(x_lr)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor([[scale]], dtype=x_lr.dtype, device=x_lr.device).expand(B, 1)
        s_emb = self.scale_embed(scale).view(B, -1, 1, 1)
        h = h + s_emb
        h = self.fno_blocks(h)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Diffusion UNet
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal positional embedding for diffusion timestep t."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class UNetBlock(nn.Module):
    """Residual block conditioned on time embedding and context."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class HNDSRDiffusionUNet(nn.Module):
    """
    ε_θ(z_t, t, c): predicts noise given noisy latent, timestep, and
    neural-operator context c.  Implements eq. (21) from the HNDSR paper.
    """

    def __init__(self, latent_ch: int = 64, t_dim: int = 128) -> None:
        super().__init__()
        self.t_embed = SinusoidalTimestepEmbedding(t_dim)
        self.ctx_proj = nn.Conv2d(latent_ch, latent_ch, 1)

        # Encoder path
        self.enc1 = UNetBlock(latent_ch * 2, 128, t_dim)
        self.down1 = nn.Conv2d(128, 128, 4, stride=2, padding=1)
        self.enc2 = UNetBlock(128, 256, t_dim)
        self.down2 = nn.Conv2d(256, 256, 4, stride=2, padding=1)

        # Bottleneck
        self.mid = UNetBlock(256, 256, t_dim)

        # Decoder path
        self.up2 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.dec2 = UNetBlock(512, 128, t_dim)
        self.up1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.dec1 = UNetBlock(256, latent_ch, t_dim)

        self.out = nn.Conv2d(latent_ch, latent_ch, 3, padding=1)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.t_embed(t)
        c = self.ctx_proj(context)
        x = torch.cat([z_t, c], dim=1)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        m = self.mid(self.down2(e2), t_emb)

        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Composite HNDSR model (all three stages)
# ─────────────────────────────────────────────────────────────────────────────

class HNDSRModel(nn.Module):
    """
    Full HNDSR pipeline for inference.
    Holds all three stage sub-models as named sub-modules so that
    state_dict keys are stable across serialisation formats.
    """

    def __init__(
        self,
        in_ch: int = 3,
        latent_ch: int = 64,
        fno_layers: int = 4,
        fno_modes: int = 12,
        t_dim: int = 128,
    ) -> None:
        super().__init__()
        self.autoencoder = HNDSRAutoencoder(in_ch, latent_ch)
        self.neural_operator = HNDSRNeuralOperator(in_ch, latent_ch, fno_layers, fno_modes)
        self.diffusion_unet = HNDSRDiffusionUNet(latent_ch, t_dim)

    def forward(
        self,
        x_lr: torch.Tensor,
        scale: float,
        t: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """Single forward pass (used for ONNX / TorchScript export)."""
        context = self.neural_operator(x_lr, scale)
        noise_pred = self.diffusion_unet(z_t, t, context)
        return noise_pred
