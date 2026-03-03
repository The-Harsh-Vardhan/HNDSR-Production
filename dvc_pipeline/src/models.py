"""
HNDSR model architecture — all components extracted from MLFlow/HNDSR_MLflow.ipynb.

Components:
    ResidualBlock          – Conv–ReLU–Conv residual block (Cell 7)
    LatentAutoencoder      – 8× downsample encoder + mirror decoder (Cell 7)
    SpectralConv2d         – Truncated spectral convolution via FFT (Cell 8)
    NeuralOperator         – 4-layer FNO with scale conditioning (Cell 8)
    ImplicitAmplification  – MLP channel-wise gain modulation (Cell 9)
    SinusoidalPositionEmbeddings – Timestep encoding (Cell 10)
    AttentionBlock         – Self-attention (Cell 10)
    CrossAttentionBlock    – Cross-attention for FNO context (Cell 10)
    ResidualBlockWithTime  – ResBlock conditioned on time embedding (Cell 10)
    DiffusionUNet          – Latent UNet with cross-attention (Cell 11)
    DDPMScheduler          – Linear beta schedule + DDIM sampling (Cell 12)
    HNDSR                  – Composite model with SDEdit inference (Cell 13)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Autoencoder components (Cell 7)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block for autoencoder."""

    def __init__(self, channels, use_bn=False):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class LatentAutoencoder(nn.Module):
    """Autoencoder for learning latent representation."""

    def __init__(self, in_channels=3, latent_dim=64, num_res_blocks=4, downsample_ratio=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.downsample_ratio = downsample_ratio

        num_downs = int(math.log2(downsample_ratio))

        # Encoder
        encoder_layers = [nn.Conv2d(in_channels, latent_dim, 3, padding=1)]
        channels = latent_dim
        for _ in range(num_downs):
            out_channels = min(channels * 2, 128)
            encoder_layers.extend([
                nn.Conv2d(channels, out_channels, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            channels = out_channels

        for _ in range(num_res_blocks):
            encoder_layers.append(ResidualBlock(channels))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for _ in range(num_res_blocks):
            decoder_layers.append(ResidualBlock(channels))

        for _ in range(num_downs):
            out_channels = channels // 2
            decoder_layers.extend([
                nn.ConvTranspose2d(channels, out_channels, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            channels = out_channels

        decoder_layers.append(nn.Conv2d(channels, in_channels, 3, padding=1))
        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ---------------------------------------------------------------------------
# Neural Operator components (Cell 8)
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """Spectral convolution for Fourier Neural Operator — fixed for mixed precision."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )

    def forward(self, x):
        batchsize = x.shape[0]

        # Force float32 for FFT (cuFFT limitation)
        x_dtype = x.dtype
        x = x.float()

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        modes1 = min(self.modes1, x.size(-2))
        modes2 = min(self.modes2, x.size(-1) // 2 + 1)

        if modes1 > 0 and modes2 > 0:
            out_ft[:, :, :modes1, :modes2] = self._compl_mul2d(
                x_ft[:, :, :modes1, :modes2],
                torch.view_as_complex(self.weights1[:, :, :modes1, :modes2]),
            )
            out_ft[:, :, -modes1:, :modes2] = self._compl_mul2d(
                x_ft[:, :, -modes1:, :modes2],
                torch.view_as_complex(self.weights2[:, :, :modes1, :modes2]),
            )

        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        if x_dtype != torch.float32:
            x_out = x_out.to(x_dtype)

        return x_out

    @staticmethod
    def _compl_mul2d(input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)


class NeuralOperator(nn.Module):
    """Neural Operator for structure-aware prior."""

    def __init__(self, in_channels=3, out_channels=128, modes=8, width=32):
        super().__init__()
        self.modes = modes
        self.width = width

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
        x = torch.cat([x, scale_map], dim=1)

        x = self.fc0(x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x


# ---------------------------------------------------------------------------
# Implicit Amplification (Cell 9)
# ---------------------------------------------------------------------------

class ImplicitAmplification(nn.Module):
    """MLP that predicts channel-wise gains for high-frequency enhancement."""

    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, latent, scale_factor):
        b, c, h, w = latent.shape

        if isinstance(scale_factor, (int, float)):
            scale_input = torch.full((b, 1), scale_factor, device=latent.device, dtype=torch.float32)
        else:
            scale_input = scale_factor.view(b, 1).float()

        gains = self.mlp(scale_input)
        gains = gains.view(b, c, 1, 1)

        return latent * (1 + gains)


# ---------------------------------------------------------------------------
# Diffusion UNet components (Cells 10–11)
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for diffusion."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """Self-attention block for UNet."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w).transpose(1, 2)
        v = v.view(b, c, h * w).transpose(1, 2)

        scale = c ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attn, v)

        out = out.transpose(1, 2).view(b, c, h, w)
        out = self.proj(out)

        return out + residual


class CrossAttentionBlock(nn.Module):
    """Cross-attention for conditioning."""

    def __init__(self, channels, context_dim):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)

        q = self.q(x).view(b, c, h * w).transpose(1, 2)

        kv = self.kv(context)
        k, v = kv.chunk(2, dim=1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        scale = c ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attn, v)

        out = out.transpose(1, 2).view(b, c, h, w)
        out = self.proj(out)

        return out + residual


class ResidualBlockWithTime(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.time_emb(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class DiffusionUNet(nn.Module):
    """Simplified UNet for latent diffusion — memory optimised."""

    def __init__(self, in_channels=128, model_channels=64, out_channels=128, context_dim=128):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Simplified architecture
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down
        self.down1 = ResidualBlockWithTime(model_channels, model_channels * 2, time_embed_dim)
        self.down2 = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)

        # Middle with cross-attention
        self.mid1 = ResidualBlockWithTime(model_channels * 2, model_channels * 2, time_embed_dim)
        self.mid_attn = CrossAttentionBlock(model_channels * 2, context_dim)
        self.mid2 = ResidualBlockWithTime(model_channels * 2, model_channels * 2, time_embed_dim)

        # Up — after concat: 128 + 64 = 192 channels
        self.up1 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        self.up2 = ResidualBlockWithTime(model_channels * 3, model_channels, time_embed_dim)  # 192 → 64

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, t, context):
        t_emb = self.time_embed(t)

        h = self.input_proj(x)
        h0 = h  # skip connection

        h = self.down1(h, t_emb)
        h = self.down2(h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid2(h, t_emb)

        h = self.up1(h)
        h = torch.cat([h, h0], dim=1)
        h = self.up2(h, t_emb)

        return self.out(h)


# ---------------------------------------------------------------------------
# DDPM Scheduler (Cell 12)
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """DDPM noise scheduler with DDIM sampling and proper device handling."""

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_start, noise, timesteps):
        timesteps_cpu = timesteps.cpu()

        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps_cpu].to(x_start.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps_cpu].to(x_start.device)

        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise

    def ddim_sample(self, model_output, timestep, sample):
        if isinstance(timestep, torch.Tensor):
            t = timestep.item() if timestep.numel() == 1 else timestep
        else:
            t = timestep

        alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
        alpha_prod_t_prev = (
            self.alphas_cumprod_prev[t].to(sample.device) if t > 0
            else torch.tensor(1.0).to(sample.device)
        )

        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (
            (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        )
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output
        pred_prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        return pred_prev_sample


# ---------------------------------------------------------------------------
# Composite HNDSR model (Cell 13)
# ---------------------------------------------------------------------------

class HNDSR(nn.Module):
    """Hybrid Neural Operator-Diffusion Super-Resolution."""

    def __init__(
        self,
        ae_latent_dim=128,
        ae_downsample_ratio=8,
        no_width=32,
        no_modes=8,
        diffusion_channels=64,
        num_timesteps=1000,
    ):
        super().__init__()

        self.autoencoder = LatentAutoencoder(
            in_channels=3,
            latent_dim=ae_latent_dim,
            num_res_blocks=4,
            downsample_ratio=ae_downsample_ratio,
        )

        self.neural_operator = NeuralOperator(
            in_channels=3,
            out_channels=ae_latent_dim,
            modes=no_modes,
            width=no_width,
        )

        self.implicit_amp = ImplicitAmplification(
            latent_dim=ae_latent_dim,
            hidden_dim=256,
        )

        self.diffusion_unet = DiffusionUNet(
            in_channels=ae_latent_dim,
            model_channels=diffusion_channels,
            out_channels=ae_latent_dim,
            context_dim=ae_latent_dim,
        )

        self.scheduler = DDPMScheduler(num_timesteps=num_timesteps)
        self.ae_downsample_ratio = ae_downsample_ratio

    def encode_hr(self, hr_img):
        _, z = self.autoencoder(hr_img)
        return z

    def decode_latent(self, z):
        return self.autoencoder.decode(z)

    def get_no_prior(self, lr_img, scale_factor):
        lr_upscaled = F.interpolate(lr_img, scale_factor=scale_factor, mode="bicubic", align_corners=False)
        no_features = self.neural_operator(lr_upscaled, scale_factor)

        latent_size = lr_upscaled.shape[-1] // self.ae_downsample_ratio
        no_prior = F.interpolate(no_features, size=(latent_size, latent_size), mode="bilinear", align_corners=False)

        return no_prior

    @torch.no_grad()
    def super_resolve(self, lr_img, scale_factor=4, num_inference_steps=50, diffusion_strength=0.0):
        """
        Full HNDSR inference with SDEdit-style diffusion.

        Args:
            lr_img: Low-resolution input (B, 3, H, W) in [-1, 1].
            scale_factor: Upscale factor (default 4).
            num_inference_steps: Total DDIM steps at full strength.
            diffusion_strength: 0.0 = skip diffusion (recommended),
                                0.1–0.3 = light SDEdit, 1.0 = pure noise start.
        """
        device = lr_img.device
        b = lr_img.shape[0]

        # Step 1: Compute FNO prior + implicit amplification
        no_prior = self.get_no_prior(lr_img, scale_factor)
        no_prior = self.implicit_amp(no_prior, scale_factor)

        # Step 2: Optionally apply diffusion refinement
        if diffusion_strength <= 0.0:
            hr_pred = self.decode_latent(no_prior)
            return hr_pred

        context = F.adaptive_avg_pool2d(no_prior, 1).view(b, -1)

        start_timestep = min(
            int(diffusion_strength * self.scheduler.num_timesteps),
            self.scheduler.num_timesteps - 1,
        )
        actual_steps = max(int(num_inference_steps * diffusion_strength), 1)

        if diffusion_strength >= 1.0:
            z_t = torch.randn_like(no_prior)
            timesteps = torch.linspace(
                self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long,
            )
        else:
            noise = torch.randn_like(no_prior)
            t_tensor = torch.full((b,), start_timestep, dtype=torch.long)
            z_t = self.scheduler.add_noise(no_prior, noise, t_tensor)
            timesteps = torch.linspace(start_timestep, 0, actual_steps, dtype=torch.long)

        # Step 3: DDIM denoising loop
        for t in tqdm(timesteps, desc="Diffusion sampling", leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            noise_pred = self.diffusion_unet(z_t, t_batch, context)
            z_t = self.scheduler.ddim_sample(noise_pred, t, z_t)

        # Step 4: Decode refined latent
        hr_pred = self.decode_latent(z_t)
        return hr_pred
