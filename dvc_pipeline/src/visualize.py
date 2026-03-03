"""
Visualize — generate before/after super-resolution comparison images.

Usage (invoked by DVC):
    python src/visualize.py
    python src/visualize.py --params params.yaml

Creates side-by-side comparison grids:
    LR (bicubic upscaled) | Bicubic baseline | SR (model) | HR (ground truth)

Outputs:
    visualize_results/comparison_grid.png          – full montage
    visualize_results/sample_NNN_comparison.png    – per-sample strips
    metrics/visualize_metrics.json                 – per-sample PSNR / SSIM
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure sibling modules are importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from dataset import SatelliteDataset
from models import HNDSR
from utils import set_seed, get_device, load_params, calculate_psnr, calculate_ssim


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same pattern as evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(params: dict) -> HNDSR:
    """Instantiate HNDSR from params."""
    ae = params["autoencoder"]
    no = params["neural_operator"]
    diff = params["diffusion"]
    return HNDSR(
        ae_latent_dim=ae["latent_dim"],
        ae_downsample_ratio=ae["downsample_ratio"],
        no_width=no["width"],
        no_modes=no["modes"],
        diffusion_channels=diff["model_channels"],
        num_timesteps=diff["num_timesteps"],
    )


def _load_checkpoints(model: HNDSR, ckpt_dir: str, device: torch.device) -> HNDSR:
    """Load all 3 stage checkpoints into the composite model."""
    ae_path = os.path.join(ckpt_dir, "autoencoder_best.pth")
    no_path = os.path.join(ckpt_dir, "neural_operator_best.pth")
    diff_path = os.path.join(ckpt_dir, "diffusion_best.pth")

    model.autoencoder.load_state_dict(
        torch.load(ae_path, map_location=device, weights_only=True)
    )
    no_ckpt = torch.load(no_path, map_location=device, weights_only=True)
    model.neural_operator.load_state_dict(no_ckpt["neural_operator"])
    model.implicit_amp.load_state_dict(no_ckpt["implicit_amp"])

    diff_ckpt = torch.load(diff_path, map_location=device, weights_only=True)
    model.diffusion_unet.load_state_dict(diff_ckpt["diffusion_unet"])

    print(f"Loaded checkpoints from {ckpt_dir}")
    return model


def _denorm(t: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] → [0, 1] and clamp."""
    return ((t + 1) / 2).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main visualize routine
# ─────────────────────────────────────────────────────────────────────────────

def visualize(params: dict, device: torch.device):
    """Generate comparison grids and per-sample metrics."""

    data = params["data"]
    vis_cfg = params["visualize"]
    ckpt_dir = params["checkpoints"]["dir"]

    num_samples = vis_cfg.get("num_samples", 8)
    output_dir = vis_cfg.get("output_dir", "visualize_results")
    diffusion_strength = vis_cfg.get("diffusion_strength", 0.0)
    save_individual = vis_cfg.get("save_individual", True)

    # Build & load model
    model = _build_model(params)
    model = _load_checkpoints(model, ckpt_dir, device)
    model.to(device)
    model.eval()

    # Val dataset (eval mode — center crop, no augmentation)
    val_dataset = SatelliteDataset(
        hr_dir=data["hr_dir"],
        lr_dir=data["lr_dir"],
        patch_size=data["patch_size"],
        training=False,
    )

    num_samples = min(num_samples, len(val_dataset))
    if len(val_dataset) > num_samples:
        indices = list(range(num_samples))
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)

    diff_cfg = params["diffusion"]
    num_inference_steps = diff_cfg.get("num_inference_steps", 50)

    # Collect strips and per-sample metrics
    strips = []
    sample_metrics = []

    print("\n" + "=" * 60)
    print("GENERATING VISUAL COMPARISONS")
    print(f"  Samples: {num_samples}")
    print(f"  Diffusion strength: {diffusion_strength}")
    print(f"  Output: {output_dir}/")
    print("=" * 60)

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)
            scale = batch["scale"][0].item()

            # Model SR
            sr_img = model.super_resolve(
                lr_img,
                scale_factor=scale,
                num_inference_steps=num_inference_steps,
                diffusion_strength=diffusion_strength,
            )

            # Bicubic baseline (naive upscale of LR)
            bicubic_img = F.interpolate(
                lr_img, scale_factor=scale, mode="bicubic", align_corners=False
            ).clamp(-1, 1)

            # Upscale LR for visual comparison (nearest so pixels stay blocky)
            lr_upscaled = F.interpolate(
                lr_img, scale_factor=scale, mode="nearest"
            )

            # Per-sample quality metrics
            psnr_sr = calculate_psnr(sr_img, hr_img)
            ssim_sr = calculate_ssim(sr_img, hr_img)
            psnr_bic = calculate_psnr(bicubic_img, hr_img)
            ssim_bic = calculate_ssim(bicubic_img, hr_img)

            sample_metrics.append({
                "sample": idx,
                "sr_psnr": round(psnr_sr, 4),
                "sr_ssim": round(ssim_sr, 4),
                "bicubic_psnr": round(psnr_bic, 4),
                "bicubic_ssim": round(ssim_bic, 4),
            })

            # Build horizontal strip: LR↑ | Bicubic | SR | HR  (all denormalized)
            strip = torch.cat([
                _denorm(lr_upscaled),
                _denorm(bicubic_img),
                _denorm(sr_img),
                _denorm(hr_img),
            ], dim=-1)  # concat along width

            strips.append(strip.squeeze(0))  # remove batch dim → (C, H, 4*W)

            # Save individual comparison strip
            if save_individual:
                save_image(
                    strip.squeeze(0),
                    os.path.join(output_dir, f"sample_{idx:03d}_comparison.png"),
                )

            print(f"  Sample {idx:03d}: SR PSNR={psnr_sr:.2f} dB, SSIM={ssim_sr:.4f}  |  "
                  f"Bicubic PSNR={psnr_bic:.2f} dB, SSIM={ssim_bic:.4f}")

            del lr_img, hr_img, sr_img, bicubic_img, lr_upscaled, strip
            torch.cuda.empty_cache()

    # Build full montage grid (stacked vertically)
    if strips:
        grid = make_grid(strips, nrow=1, padding=4, pad_value=1.0)
        save_image(grid, os.path.join(output_dir, "comparison_grid.png"))
        print(f"\nMontage saved → {output_dir}/comparison_grid.png")

    # Compute aggregated metrics
    sr_psnrs = [m["sr_psnr"] for m in sample_metrics]
    sr_ssims = [m["sr_ssim"] for m in sample_metrics]
    bic_psnrs = [m["bicubic_psnr"] for m in sample_metrics]
    bic_ssims = [m["bicubic_ssim"] for m in sample_metrics]

    metrics_summary = {
        "num_samples": len(sample_metrics),
        "diffusion_strength": diffusion_strength,
        "sr_psnr_mean": round(float(np.mean(sr_psnrs)), 4),
        "sr_psnr_std": round(float(np.std(sr_psnrs)), 4),
        "sr_ssim_mean": round(float(np.mean(sr_ssims)), 4),
        "sr_ssim_std": round(float(np.std(sr_ssims)), 4),
        "bicubic_psnr_mean": round(float(np.mean(bic_psnrs)), 4),
        "bicubic_ssim_mean": round(float(np.mean(bic_ssims)), 4),
        "psnr_improvement_over_bicubic": round(float(np.mean(sr_psnrs)) - float(np.mean(bic_psnrs)), 4),
        "per_sample": sample_metrics,
    }

    metrics_path = "metrics/visualize_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  SR  — PSNR: {metrics_summary['sr_psnr_mean']:.2f} ± {metrics_summary['sr_psnr_std']:.2f} dB, "
          f"SSIM: {metrics_summary['sr_ssim_mean']:.4f}")
    print(f"  Bic — PSNR: {metrics_summary['bicubic_psnr_mean']:.2f} dB, "
          f"SSIM: {metrics_summary['bicubic_ssim_mean']:.4f}")
    print(f"  PSNR improvement over bicubic: +{metrics_summary['psnr_improvement_over_bicubic']:.2f} dB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="HNDSR visual comparison generator")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    set_seed(params["seed"])
    device = get_device()

    visualize(params, device)


if __name__ == "__main__":
    main()
