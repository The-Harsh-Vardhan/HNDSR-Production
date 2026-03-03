"""
Evaluation script for the HNDSR DVC pipeline.

Usage (invoked by DVC):
    python src/evaluate.py
    python src/evaluate.py --params params.yaml

Loads all 3 checkpoints, runs inference on a test split,
computes PSNR / SSIM / LPIPS, and writes:
    metrics/eval_metrics.json   – DVC metric file
    evaluation_results/         – visual comparison PNGs
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
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import SatelliteDataset
from models import HNDSR
from utils import set_seed, get_device, load_params, calculate_psnr, calculate_ssim


def _build_model(params: dict) -> HNDSR:
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


def evaluate(params: dict, device: torch.device):
    """Run full evaluation and write metrics + visual samples."""
    data = params["data"]
    eval_cfg = params["evaluate"]
    ckpt_dir = params["checkpoints"]["dir"]

    # Build & load model
    model = _build_model(params)
    model = _load_checkpoints(model, ckpt_dir, device)
    model.to(device)
    model.eval()

    # Test dataset (eval mode — center crop, no augmentation)
    test_dataset = SatelliteDataset(
        hr_dir=data["hr_dir"],
        lr_dir=data["lr_dir"],
        patch_size=data["patch_size"],
        training=False,
    )
    num_samples = min(eval_cfg.get("num_test_samples", 50), len(test_dataset))
    if len(test_dataset) > num_samples:
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Optional LPIPS
    lpips_fn = None
    avg_lpips = 0.0
    std_lpips = 0.0
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        use_lpips = True
    except Exception:
        print("LPIPS not available — skipping perceptual metric.")
        use_lpips = False

    # Output dirs
    output_dir = "evaluation_results"
    for sub in ["lr", "sr", "hr"]:
        Path(f"{output_dir}/{sub}").mkdir(parents=True, exist_ok=True)

    psnr_values, ssim_values, lpips_values = [], [], []

    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 60)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)
            scale = batch["scale"][0].item()

            sr_img = model.super_resolve(lr_img, scale_factor=scale, num_inference_steps=50)

            psnr_val = calculate_psnr(sr_img, hr_img)
            ssim_val = calculate_ssim(sr_img, hr_img)
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)

            if use_lpips:
                lpips_val = lpips_fn(sr_img, hr_img).mean().item()
                lpips_values.append(lpips_val)

            # Save first 10 visual comparisons
            if idx < 10:
                save_image((lr_img + 1) / 2, f"{output_dir}/lr/sample_{idx:03d}.png")
                save_image((sr_img + 1) / 2, f"{output_dir}/sr/sample_{idx:03d}.png")
                save_image((hr_img + 1) / 2, f"{output_dir}/hr/sample_{idx:03d}.png")

            del lr_img, hr_img, sr_img
            torch.cuda.empty_cache()

    # Compute summary stats
    avg_psnr = float(np.mean(psnr_values))
    std_psnr = float(np.std(psnr_values))
    avg_ssim = float(np.mean(ssim_values))
    std_ssim = float(np.std(ssim_values))

    metrics = {
        "psnr_mean": round(avg_psnr, 4),
        "psnr_std": round(std_psnr, 4),
        "ssim_mean": round(avg_ssim, 4),
        "ssim_std": round(std_ssim, 4),
        "num_test_samples": len(psnr_values),
    }

    if use_lpips:
        avg_lpips = float(np.mean(lpips_values))
        std_lpips = float(np.std(lpips_values))
        metrics["lpips_mean"] = round(avg_lpips, 4)
        metrics["lpips_std"] = round(std_lpips, 4)

    # Check thresholds
    thresholds = eval_cfg.get("thresholds", {})
    metrics["pass_psnr"] = avg_psnr >= thresholds.get("min_psnr", 0)
    metrics["pass_ssim"] = avg_ssim >= thresholds.get("min_ssim", 0)
    if use_lpips:
        metrics["pass_lpips"] = avg_lpips <= thresholds.get("max_lpips", 1.0)

    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  PSNR:  {avg_psnr:.2f} +/- {std_psnr:.2f} dB")
    print(f"  SSIM:  {avg_ssim:.4f} +/- {std_ssim:.4f}")
    if use_lpips:
        print(f"  LPIPS: {avg_lpips:.4f} +/- {std_lpips:.4f}")
    print("=" * 60)

    # Write metrics JSON
    Path("metrics").mkdir(parents=True, exist_ok=True)
    metrics_path = "metrics/eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")
    print(f"Visual samples saved → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="HNDSR evaluation")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    set_seed(params["seed"])
    device = get_device()

    evaluate(params, device)


if __name__ == "__main__":
    main()
