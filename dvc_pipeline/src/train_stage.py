"""
CLI entry-point for per-stage HNDSR training.

Usage (invoked by DVC):
    python src/train_stage.py --stage autoencoder
    python src/train_stage.py --stage neural_operator
    python src/train_stage.py --stage diffusion

Each stage reads hyperparameters from params.yaml, trains the relevant
component(s), saves a best-validation checkpoint, and writes a JSON metrics
file consumed by ``dvc metrics``.

Optional: pass ``--mlflow`` to enable MLflow experiment logging.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Ensure sibling modules are importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import SatelliteDataset
from models import HNDSR
from utils import set_seed, get_device, load_params, calculate_psnr, calculate_ssim


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_loaders(params: dict):
    """Create train / val DataLoaders from params.yaml settings."""
    data = params["data"]
    dataset = SatelliteDataset(
        hr_dir=data["hr_dir"],
        lr_dir=data["lr_dir"],
        patch_size=data["patch_size"],
        training=True,
    )

    train_size = int((1 - data["val_split"]) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=data["batch_size"],
        shuffle=True,
        num_workers=data["num_workers"],
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data["batch_size"],
        shuffle=False,
        num_workers=data["num_workers"],
        pin_memory=False,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    return train_loader, val_loader


def _build_model(params: dict) -> HNDSR:
    """Instantiate HNDSR from params.yaml."""
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


def _ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save_metrics(metrics: dict, path: str):
    _ensure_dir(path)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {path}")


def _quick_check(model, val_loader, device, stage_name, num_samples=5):
    """Quick sanity PSNR / SSIM on a few val batches."""
    model.autoencoder.eval()
    model.neural_operator.eval()
    model.implicit_amp.eval()
    model.diffusion_unet.eval()

    model.to(device)
    psnr_vals, ssim_vals = [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            hr = batch["hr"].to(device)
            lr_img = batch["lr"].to(device)
            scale = batch["scale"][0].item()

            if stage_name == "autoencoder":
                recon, _ = model.autoencoder(hr)
                psnr_vals.append(calculate_psnr(recon, hr))
                ssim_vals.append(calculate_ssim(recon, hr))
            elif stage_name == "neural_operator":
                no_prior = model.get_no_prior(lr_img, scale)
                no_prior = model.implicit_amp(no_prior, scale)
                decoded = model.decode_latent(no_prior)
                psnr_vals.append(calculate_psnr(decoded, hr))
                ssim_vals.append(calculate_ssim(decoded, hr))
            elif stage_name == "diffusion":
                sr = model.super_resolve(lr_img, scale_factor=scale, num_inference_steps=20)
                psnr_vals.append(calculate_psnr(sr, hr))
                ssim_vals.append(calculate_ssim(sr, hr))

            torch.cuda.empty_cache()

    avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    avg_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    print(f"  Sanity check ({stage_name}): PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")
    return avg_psnr, avg_ssim


# ── Stage 1: Autoencoder ────────────────────────────────────────────────────

def train_autoencoder(model, train_loader, val_loader, params, device, ckpt_dir, use_mlflow):
    """Train the autoencoder with L1 loss (Stage 1)."""
    print("\n" + "=" * 50)
    print("STAGE 1: Training Autoencoder")
    print("=" * 50)

    cfg = params["autoencoder"]
    num_epochs = cfg["epochs"]
    lr = cfg["lr"]

    model.autoencoder.to(device)
    optimizer = torch.optim.AdamW(model.autoencoder.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    l1_loss_fn = nn.L1Loss()

    best_val_loss = float("inf")
    avg_train = 0.0
    avg_val = 0.0
    ckpt_path = os.path.join(ckpt_dir, "autoencoder_best.pth")

    for epoch in range(num_epochs):
        model.autoencoder.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            hr = batch["hr"].to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, z = model.autoencoder(hr)
            loss = l1_loss_fn(recon, hr)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            del hr, recon, z, loss

        # Validation
        model.autoencoder.eval()
        val_losses, val_psnr = [], []
        with torch.no_grad():
            for batch in val_loader:
                hr = batch["hr"].to(device)
                recon, z = model.autoencoder(hr)
                loss = l1_loss_fn(recon, hr)
                val_losses.append(loss.item())
                val_psnr.append(calculate_psnr(recon, hr))
                del hr, recon, z, loss
                torch.cuda.empty_cache()

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        avg_psnr = float(np.mean(val_psnr))

        if use_mlflow:
            import mlflow
            mlflow.log_metric("stage1_train_loss", avg_train, step=epoch)
            mlflow.log_metric("stage1_val_loss", avg_val, step=epoch)
            mlflow.log_metric("stage1_val_psnr", avg_psnr, step=epoch)
            mlflow.log_metric("stage1_lr", scheduler.get_last_lr()[0], step=epoch)

        print(f"Epoch {epoch + 1}: Train={avg_train:.4f}, Val={avg_val:.4f}, PSNR={avg_psnr:.2f} dB")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _ensure_dir(ckpt_path)
            torch.save(model.autoencoder.state_dict(), ckpt_path)
            print(f"  Saved best autoencoder → {ckpt_path}")

        scheduler.step()

    if use_mlflow:
        import mlflow
        mlflow.log_metric("stage1_best_val_loss", best_val_loss)
        mlflow.log_artifact(ckpt_path)

    # Sanity check
    check_psnr, check_ssim = _quick_check(model, val_loader, device, "autoencoder")

    _save_metrics({
        "best_val_loss": best_val_loss,
        "final_train_loss": avg_train,
        "final_val_loss": avg_val,
        "check_psnr": check_psnr,
        "check_ssim": check_ssim,
    }, "metrics/stage1_metrics.json")

    print("Stage 1 complete!\n")
    return model


# ── Stage 2: Neural Operator + Implicit Amplification ───────────────────────

def train_neural_operator(model, train_loader, val_loader, params, device, ckpt_dir, use_mlflow):
    """Train FNO + ImplicitAmp jointly; autoencoder is frozen (Stage 2)."""
    print("\n" + "=" * 50)
    print("STAGE 2: Training Neural Operator + Implicit Amplification")
    print("=" * 50)

    cfg_no = params["neural_operator"]
    cfg_ia = params.get("implicit_amp", {})
    num_epochs = cfg_no["epochs"]
    lr = cfg_no["lr"]

    # Load AE checkpoint
    ae_path = os.path.join(ckpt_dir, "autoencoder_best.pth")
    model.autoencoder.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
    print(f"  Loaded autoencoder from {ae_path}")

    # Freeze AE
    for p in model.autoencoder.parameters():
        p.requires_grad = False
    model.autoencoder.eval()

    model.neural_operator.to(device)
    model.implicit_amp.to(device)
    model.autoencoder.to(device)

    optimizer = torch.optim.AdamW(
        list(model.neural_operator.parameters()) + list(model.implicit_amp.parameters()),
        lr=lr,
        weight_decay=cfg_no["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    avg_train = 0.0
    avg_val = 0.0
    ckpt_path = os.path.join(ckpt_dir, "neural_operator_best.pth")

    for epoch in range(num_epochs):
        model.neural_operator.train()
        model.implicit_amp.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)
            scale = batch["scale"][0].item()

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                target_latent = model.encode_hr(hr_img)

            no_prior = model.get_no_prior(lr_img, scale)
            no_prior = model.implicit_amp(no_prior, scale)
            loss = mse_loss_fn(no_prior, target_latent)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            del lr_img, hr_img, target_latent, no_prior, loss

        # Validation
        model.neural_operator.eval()
        model.implicit_amp.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                lr_img = batch["lr"].to(device)
                hr_img = batch["hr"].to(device)
                scale = batch["scale"][0].item()

                target_latent = model.encode_hr(hr_img)
                no_prior = model.get_no_prior(lr_img, scale)
                no_prior = model.implicit_amp(no_prior, scale)
                loss = mse_loss_fn(no_prior, target_latent)
                val_losses.append(loss.item())

                del lr_img, hr_img, target_latent, no_prior, loss
                torch.cuda.empty_cache()

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))

        if use_mlflow:
            import mlflow
            mlflow.log_metric("stage2_train_loss", avg_train, step=epoch)
            mlflow.log_metric("stage2_val_loss", avg_val, step=epoch)
            mlflow.log_metric("stage2_lr", scheduler.get_last_lr()[0], step=epoch)

        print(f"Epoch {epoch + 1}: Train={avg_train:.4f}, Val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _ensure_dir(ckpt_path)
            torch.save({
                "neural_operator": model.neural_operator.state_dict(),
                "implicit_amp": model.implicit_amp.state_dict(),
            }, ckpt_path)
            print(f"  Saved best NO + ImplicitAmp → {ckpt_path}")

        scheduler.step()

    if use_mlflow:
        import mlflow
        mlflow.log_metric("stage2_best_val_loss", best_val_loss)
        mlflow.log_artifact(ckpt_path)

    check_psnr, check_ssim = _quick_check(model, val_loader, device, "neural_operator")

    _save_metrics({
        "best_val_loss": best_val_loss,
        "final_train_loss": avg_train,
        "final_val_loss": avg_val,
        "check_psnr": check_psnr,
        "check_ssim": check_ssim,
    }, "metrics/stage2_metrics.json")

    print("Stage 2 complete!\n")
    return model


# ── Stage 3: Diffusion UNet ─────────────────────────────────────────────────

def train_diffusion(model, train_loader, val_loader, params, device, ckpt_dir, use_mlflow):
    """Train the diffusion UNet; AE + FNO + ImplicitAmp are frozen (Stage 3)."""
    print("\n" + "=" * 50)
    print("STAGE 3: Training Diffusion Model")
    print("=" * 50)

    cfg = params["diffusion"]
    num_epochs = cfg["epochs"]
    lr = cfg["lr"]

    # Load prior-stage checkpoints
    ae_path = os.path.join(ckpt_dir, "autoencoder_best.pth")
    no_path = os.path.join(ckpt_dir, "neural_operator_best.pth")

    model.autoencoder.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
    no_ckpt = torch.load(no_path, map_location=device, weights_only=True)
    model.neural_operator.load_state_dict(no_ckpt["neural_operator"])
    model.implicit_amp.load_state_dict(no_ckpt["implicit_amp"])
    print(f"  Loaded AE from {ae_path}")
    print(f"  Loaded NO + ImplicitAmp from {no_path}")

    # Freeze everything except diffusion UNet
    for p in model.autoencoder.parameters():
        p.requires_grad = False
    for p in model.neural_operator.parameters():
        p.requires_grad = False
    for p in model.implicit_amp.parameters():
        p.requires_grad = False

    model.autoencoder.eval()
    model.neural_operator.eval()
    model.implicit_amp.eval()

    model.to(device)

    optimizer = torch.optim.AdamW(model.diffusion_unet.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    mse_loss_fn = nn.MSELoss()
    grad_clip = cfg.get("grad_clip", 1.0)

    best_val_loss = float("inf")
    avg_train = 0.0
    avg_val = 0.0
    ckpt_path = os.path.join(ckpt_dir, "diffusion_best.pth")

    for epoch in range(num_epochs):
        model.diffusion_unet.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)
            scale = batch["scale"][0].item()

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                target_latent = model.encode_hr(hr_img)
                no_prior = model.get_no_prior(lr_img, scale)
                no_prior = model.implicit_amp(no_prior, scale)
                b = lr_img.shape[0]
                context = F.adaptive_avg_pool2d(no_prior, 1).view(b, -1)

            timesteps = torch.randint(0, model.scheduler.num_timesteps, (b,), device=device).long()
            noise = torch.randn_like(target_latent)
            noisy_latent = model.scheduler.add_noise(target_latent, noise, timesteps)

            noise_pred = model.diffusion_unet(noisy_latent, timesteps, context)
            loss = mse_loss_fn(noise_pred, noise)

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.diffusion_unet.parameters(), grad_clip)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            del lr_img, hr_img, target_latent, no_prior, context, timesteps, noise, noisy_latent, noise_pred, loss

        # Validation
        model.diffusion_unet.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                lr_img = batch["lr"].to(device)
                hr_img = batch["hr"].to(device)
                scale = batch["scale"][0].item()

                target_latent = model.encode_hr(hr_img)
                no_prior = model.get_no_prior(lr_img, scale)
                no_prior = model.implicit_amp(no_prior, scale)
                b = lr_img.shape[0]
                context = F.adaptive_avg_pool2d(no_prior, 1).view(b, -1)

                timesteps = torch.randint(0, model.scheduler.num_timesteps, (b,), device=device).long()
                noise = torch.randn_like(target_latent)
                noisy_latent = model.scheduler.add_noise(target_latent, noise, timesteps)

                noise_pred = model.diffusion_unet(noisy_latent, timesteps, context)
                loss = mse_loss_fn(noise_pred, noise)
                val_losses.append(loss.item())

                del lr_img, hr_img, target_latent, no_prior, context, timesteps, noise, noisy_latent, noise_pred, loss
                torch.cuda.empty_cache()

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))

        if use_mlflow:
            import mlflow
            mlflow.log_metric("stage3_train_loss", avg_train, step=epoch)
            mlflow.log_metric("stage3_val_loss", avg_val, step=epoch)
            mlflow.log_metric("stage3_lr", scheduler.get_last_lr()[0], step=epoch)

        print(f"Epoch {epoch + 1}: Train={avg_train:.4f}, Val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _ensure_dir(ckpt_path)
            torch.save({"diffusion_unet": model.diffusion_unet.state_dict()}, ckpt_path)
            print(f"  Saved best diffusion → {ckpt_path}")

        scheduler.step()

    if use_mlflow:
        import mlflow
        mlflow.log_metric("stage3_best_val_loss", best_val_loss)
        mlflow.log_artifact(ckpt_path)

    check_psnr, check_ssim = _quick_check(model, val_loader, device, "diffusion")

    _save_metrics({
        "best_val_loss": best_val_loss,
        "final_train_loss": avg_train,
        "final_val_loss": avg_val,
        "check_psnr": check_psnr,
        "check_ssim": check_ssim,
    }, "metrics/stage3_metrics.json")

    print("Stage 3 complete!\n")
    return model


# ── CLI ──────────────────────────────────────────────────────────────────────

STAGE_FN = {
    "autoencoder": train_autoencoder,
    "neural_operator": train_neural_operator,
    "diffusion": train_diffusion,
}


def main():
    parser = argparse.ArgumentParser(description="HNDSR per-stage training")
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGE_FN.keys()),
        help="Which training stage to run.",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to params.yaml (default: params.yaml)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment logging.",
    )
    args = parser.parse_args()

    params = load_params(args.params)
    set_seed(params["seed"])
    device = get_device()

    # CUDA memory optimisation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    ckpt_dir = params["checkpoints"]["dir"]
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    model = _build_model(params)
    train_loader, val_loader = _build_loaders(params)

    use_mlflow = args.mlflow
    if use_mlflow:
        import mlflow
        mlflow.set_experiment("HNDSR_DVC_Training")

    stage_fn = STAGE_FN[args.stage]

    if use_mlflow:
        import mlflow
        with mlflow.start_run(run_name=f"dvc_{args.stage}"):
            mlflow.log_params({
                "stage": args.stage,
                "seed": params["seed"],
                "batch_size": params["data"]["batch_size"],
                "device": str(device),
            })
            stage_fn(model, train_loader, val_loader, params, device, ckpt_dir, use_mlflow)
    else:
        stage_fn(model, train_loader, val_loader, params, device, ckpt_dir, use_mlflow)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
