"""
training/train_pipeline.py
============================
Three-stage sequential training script for HNDSR.

What  : Trains the Autoencoder → Neural Operator → Diffusion UNet pipeline
        with MLflow logging, checkpoint saving, and early stopping.
Why   : The Jupyter notebook-based training is not reproducible or automatable.
        This script enables CLI-driven, CI-integrated training runs.
How   : Each stage freezes the previously trained components and only trains
        the current stage. MLflow logs everything for experiment comparison.

Stages:
  1. Autoencoder  — learns structural features via L1 reconstruction loss
  2. Neural Operator — learns continuous latent mappings via MSE loss
  3. Diffusion UNet — learns noise prediction for iterative refinement
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiment_tracking import (
    EarlyStopping,
    ExperimentConfig,
    HNDSRExperimentTracker,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """All hyperparameters for the training pipeline."""
    # Global
    seed: int = 42
    device: str = "auto"
    batch_size: int = 16
    num_workers: int = 4

    # Stage 1: Autoencoder
    ae_lr: float = 1e-4
    ae_epochs: int = 20
    ae_loss: str = "l1"  # l1 | l2 | perceptual
    ae_weight_decay: float = 0.0

    # Stage 2: Neural Operator
    no_lr: float = 1e-4
    no_epochs: int = 15
    no_loss: str = "mse"
    no_weight_decay: float = 1e-5

    # Stage 3: Diffusion
    diff_lr: float = 2e-4
    diff_epochs: int = 30
    diff_timesteps: int = 1000
    diff_loss: str = "mse"  # noise prediction MSE
    diff_weight_decay: float = 0.0

    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4

    # Paths
    data_dir: Path = Path("./data/processed")
    checkpoint_dir: Path = Path("./checkpoints")
    dataset_manifest: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: seed everything
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    """
    Set all random seeds for reproducibility.

    Why: PyTorch, NumPy, and Python have separate RNG states.
    Without seeding all three, results are non-deterministic.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Stage trainers
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    tracker: HNDSRExperimentTracker,
) -> nn.Module:
    """
    Stage 1: Train the autoencoder for structural feature learning.

    Loss: L1 reconstruction — chosen over L2 because L1 preserves sharper
    edges in satellite imagery. L2 tends to produce blurry reconstructions
    due to its implicit Gaussian assumption (mean-seeking behavior).
    """
    logger.info("═══ Stage 1: Autoencoder Training ═══")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.ae_lr, weight_decay=config.ae_weight_decay
    )
    criterion = nn.L1Loss() if config.ae_loss == "l1" else nn.MSELoss()
    early_stop = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ae_epochs)

    for epoch in range(config.ae_epochs):
        # ── Training ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            hr_images = batch["hr"].to(config.device)
            reconstructed = model(hr_images)
            loss = criterion(reconstructed, hr_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                hr_images = batch["hr"].to(config.device)
                reconstructed = model(hr_images)
                val_loss += criterion(reconstructed, hr_images).item()
        val_loss /= len(val_loader)

        # ── Logging ──────────────────────────────────────────────────
        tracker.log_metrics({
            "ae_train_loss": train_loss,
            "ae_val_loss": val_loss,
            "ae_lr": scheduler.get_last_lr()[0],
        }, step=epoch)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
            epoch + 1, config.ae_epochs, train_loss, val_loss,
        )

        scheduler.step()

        # ── Early stopping ───────────────────────────────────────────
        if early_stop.step(val_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Save checkpoint
    ckpt_path = config.checkpoint_dir / "autoencoder_best.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    tracker.log_model_checkpoint(str(ckpt_path), "autoencoder")

    return model


def train_neural_operator(
    model: nn.Module,
    autoencoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    tracker: HNDSRExperimentTracker,
) -> nn.Module:
    """
    Stage 2: Train the Neural Operator on frozen autoencoder latents.

    The autoencoder is frozen — only the Neural Operator parameters are updated.
    MSE loss in latent space forces the NO to learn continuous scale mappings.

    Why MSE over L1: In latent space (lower dimensional, smoother manifold),
    MSE provides more stable gradients. L1's gradient discontinuity at zero
    causes training oscillation in low-dimensional latent spaces.
    """
    logger.info("═══ Stage 2: Neural Operator Training ═══")

    # Freeze autoencoder
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.no_lr, weight_decay=config.no_weight_decay
    )
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.no_epochs)

    for epoch in range(config.no_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            lr_images = batch["lr"].to(config.device)
            hr_images = batch["hr"].to(config.device)

            # Get autoencoder latents (frozen)
            with torch.no_grad():
                z_hr = autoencoder.encode(hr_images)
                z_lr = autoencoder.encode(lr_images)

            # Neural operator maps z_lr → z_hr
            z_pred = model(z_lr)
            loss = criterion(z_pred, z_hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                lr_images = batch["lr"].to(config.device)
                hr_images = batch["hr"].to(config.device)
                z_hr = autoencoder.encode(hr_images)
                z_lr = autoencoder.encode(lr_images)
                z_pred = model(z_lr)
                val_loss += criterion(z_pred, z_hr).item()
        val_loss /= len(val_loader)

        tracker.log_metrics({
            "no_train_loss": train_loss,
            "no_val_loss": val_loss,
            "no_lr": scheduler.get_last_lr()[0],
        }, step=epoch)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
            epoch + 1, config.no_epochs, train_loss, val_loss,
        )

        scheduler.step()

        if early_stop.step(val_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    ckpt_path = config.checkpoint_dir / "neural_operator_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    tracker.log_model_checkpoint(str(ckpt_path), "neural_operator")

    return model


def train_diffusion(
    model: nn.Module,
    autoencoder: nn.Module,
    neural_operator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    tracker: HNDSRExperimentTracker,
) -> nn.Module:
    """
    Stage 3: Train the Diffusion UNet for iterative latent refinement.

    Both autoencoder and Neural Operator are frozen. The UNet learns to
    predict noise ε given a noisy latent z_t and timestep t.

    Loss: MSE between predicted noise ε_θ(z_t, t) and actual noise ε.
    This is the standard DDPM training objective.

    Why noise prediction over x0 prediction: Noise prediction provides
    more stable gradients across timesteps. x0 prediction concentrates
    gradient signal at low noise levels, causing unstable training.
    """
    logger.info("═══ Stage 3: Diffusion UNet Training ═══")

    # Freeze prior stages
    autoencoder.eval()
    neural_operator.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    for param in neural_operator.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.diff_lr, weight_decay=config.diff_weight_decay
    )
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.diff_epochs)

    # Precompute diffusion schedule
    betas = torch.linspace(1e-4, 0.02, config.diff_timesteps, device=config.device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    for epoch in range(config.diff_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            lr_images = batch["lr"].to(config.device)
            hr_images = batch["hr"].to(config.device)

            with torch.no_grad():
                z_hr = autoencoder.encode(hr_images)
                z_lr = autoencoder.encode(lr_images)
                z_no = neural_operator(z_lr)  # NO output as conditioning

            # Sample random timestep
            t = torch.randint(0, config.diff_timesteps, (z_hr.size(0),), device=config.device)

            # Add noise
            noise = torch.randn_like(z_hr)
            ab_t = alpha_bar[t].view(-1, 1, 1, 1)
            z_noisy = torch.sqrt(ab_t) * z_hr + torch.sqrt(1 - ab_t) * noise

            # Predict noise (conditioned on z_no)
            noise_pred = model(z_noisy, t, z_no)
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — prevents training collapse from rare
            # high-loss samples (e.g., images with extreme frequency content)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                lr_images = batch["lr"].to(config.device)
                hr_images = batch["hr"].to(config.device)
                z_hr = autoencoder.encode(hr_images)
                z_lr = autoencoder.encode(lr_images)
                z_no = neural_operator(z_lr)

                t = torch.randint(0, config.diff_timesteps, (z_hr.size(0),), device=config.device)
                noise = torch.randn_like(z_hr)
                ab_t = alpha_bar[t].view(-1, 1, 1, 1)
                z_noisy = torch.sqrt(ab_t) * z_hr + torch.sqrt(1 - ab_t) * noise
                noise_pred = model(z_noisy, t, z_no)
                val_loss += criterion(noise_pred, noise).item()

        val_loss /= len(val_loader)

        tracker.log_metrics({
            "diff_train_loss": train_loss,
            "diff_val_loss": val_loss,
            "diff_lr": scheduler.get_last_lr()[0],
        }, step=epoch)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
            epoch + 1, config.diff_epochs, train_loss, val_loss,
        )

        scheduler.step()

        if early_stop.step(val_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    ckpt_path = config.checkpoint_dir / "diffusion_unet_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    tracker.log_model_checkpoint(str(ckpt_path), "diffusion_unet")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(config: TrainingConfig):
    """
    Run the complete three-stage training pipeline.

    Stage execution order is critical:
      1. Autoencoder MUST be trained first (it defines the latent space)
      2. Neural Operator trains on frozen autoencoder latents
      3. Diffusion UNet trains on frozen NO + autoencoder
    """
    seed_everything(config.seed)

    # Resolve device
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", config.device)

    # Initialize tracker
    tracker_config = ExperimentConfig(
        experiment_name="hndsr-training",
        tags={"pipeline": "full", "seed": str(config.seed)},
    )
    tracker = HNDSRExperimentTracker(tracker_config)

    with tracker.start_run(run_name=f"pipeline_seed{config.seed}"):
        # Log all config
        tracker.log_params({
            "seed": config.seed, "batch_size": config.batch_size,
            "ae_lr": config.ae_lr, "ae_epochs": config.ae_epochs,
            "no_lr": config.no_lr, "no_epochs": config.no_epochs,
            "diff_lr": config.diff_lr, "diff_epochs": config.diff_epochs,
            "diff_timesteps": config.diff_timesteps,
        })

        if config.dataset_manifest:
            tracker.log_dataset_info(config.dataset_manifest)

        # TODO: Initialize actual model architectures and data loaders
        # autoencoder = HNDSRAutoencoder(...).to(config.device)
        # neural_operator = HNDSRNeuralOperator(...).to(config.device)
        # diffusion_unet = HNDSRDiffusionUNet(...).to(config.device)
        # train_loader = create_dataloader(config.data_dir / "train", ...)
        # val_loader = create_dataloader(config.data_dir / "val", ...)

        # Stage 1
        # autoencoder = train_autoencoder(autoencoder, train_loader, val_loader, config, tracker)

        # Stage 2
        # neural_operator = train_neural_operator(neural_operator, autoencoder, train_loader, val_loader, config, tracker)

        # Stage 3
        # diffusion_unet = train_diffusion(diffusion_unet, autoencoder, neural_operator, train_loader, val_loader, config, tracker)

        logger.info("Training pipeline complete!")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HNDSR Training Pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ae-lr", type=float, default=1e-4)
    parser.add_argument("--ae-epochs", type=int, default=20)
    parser.add_argument("--no-lr", type=float, default=1e-4)
    parser.add_argument("--no-epochs", type=int, default=15)
    parser.add_argument("--diff-lr", type=float, default=2e-4)
    parser.add_argument("--diff-epochs", type=int, default=30)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--dataset-manifest", type=str, default=None)

    args = parser.parse_args()

    config = TrainingConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        ae_lr=args.ae_lr,
        ae_epochs=args.ae_epochs,
        no_lr=args.no_lr,
        no_epochs=args.no_epochs,
        diff_lr=args.diff_lr,
        diff_epochs=args.diff_epochs,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        dataset_manifest=args.dataset_manifest,
    )

    run_training_pipeline(config)
