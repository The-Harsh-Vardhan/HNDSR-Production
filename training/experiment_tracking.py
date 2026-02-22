"""
training/experiment_tracking.py
=================================
MLflow experiment tracking wrapper for HNDSR training.

What  : Wraps MLflow to auto-log hyperparameters, metrics, and artifacts
        for every training run across all three HNDSR stages.
Why   : Manual experiment tracking fails at scale. Every run must be
        reproducible and comparable against prior experiments.
How   : Context manager that logs to MLflow on entry/exit, plus utility
        functions for metric recording and HPO integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    experiment_name: str = "hndsr-training"
    run_name: Optional[str] = None
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "./mlruns"
    tags: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


# ─────────────────────────────────────────────────────────────────────────────
# MLflow tracking wrapper
# ─────────────────────────────────────────────────────────────────────────────

class HNDSRExperimentTracker:
    """
    MLflow experiment tracker for HNDSR.

    Tracks:
      - Hyperparameters: LR, batch size, epochs, DDIM steps, latent dim
      - Architecture: model variant, layer counts, hidden dimensions
      - Metrics: loss, PSNR, SSIM, LPIPS (per epoch)
      - Artifacts: model checkpoints, training curves, config files
      - System: GPU type, VRAM, training time, dataset hash

    Usage:
        tracker = HNDSRExperimentTracker(ExperimentConfig())

        with tracker.start_run(run_name="autoencoder_v2") as run:
            tracker.log_params({
                "learning_rate": 1e-4,
                "batch_size": 16,
                "epochs": 20,
            })

            for epoch in range(20):
                loss = train_one_epoch(...)
                tracker.log_metrics({"train_loss": loss}, step=epoch)

            tracker.log_artifact("checkpoints/autoencoder_v2.pth")
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._active_run = None

        # Lazy import — MLflow is optional for development
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(config.tracking_uri)
            mlflow.set_experiment(config.experiment_name)
            logger.info("MLflow tracking URI: %s", config.tracking_uri)
        except ImportError:
            self._mlflow = None
            logger.warning("MLflow not installed. Experiment tracking disabled.")

    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Context manager for an MLflow run.

        Auto-logs system info (GPU, Python version, git hash) on entry.
        Auto-logs total training time on exit.
        """
        if self._mlflow is None:
            yield None
            return

        name = run_name or self.config.run_name or f"run_{int(time.time())}"
        start_time = time.time()

        with self._mlflow.start_run(run_name=name, nested=nested) as run:
            self._active_run = run

            # Auto-log system info
            self._log_system_info()

            # Set tags
            for key, value in self.config.tags.items():
                self._mlflow.set_tag(key, value)

            try:
                yield run
            finally:
                # Auto-log training duration
                duration_s = time.time() - start_time
                self._mlflow.log_metric("training_duration_seconds", duration_s)
                self._active_run = None

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self._mlflow and self._active_run:
            # MLflow has a 500-char limit per param value
            sanitized = {k: str(v)[:500] for k, v in params.items()}
            self._mlflow.log_params(sanitized)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for the current step."""
        if self._mlflow and self._active_run:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact."""
        if self._mlflow and self._active_run:
            self._mlflow.log_artifact(path, artifact_path)

    def log_model_checkpoint(self, checkpoint_path: str, stage: str):
        """Log a model checkpoint with metadata."""
        if self._mlflow and self._active_run:
            self._mlflow.log_artifact(checkpoint_path, f"checkpoints/{stage}")
            # Also log the checkpoint hash for integrity verification
            sha = self._compute_file_hash(checkpoint_path)
            self._mlflow.log_param(f"{stage}_checkpoint_sha256", sha[:32])

    def log_dataset_info(self, manifest_path: str):
        """Log dataset metadata for reproducibility."""
        if self._mlflow and self._active_run:
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                self._mlflow.log_param("dataset_version", manifest.get("version", "unknown"))
                self._mlflow.log_param("dataset_hash", manifest.get("dataset_hash", "unknown")[:32])
                self._mlflow.log_param("dataset_num_images", manifest.get("num_images", 0))
                self._mlflow.log_artifact(manifest_path, "dataset")
            except Exception as exc:
                logger.warning("Could not log dataset info: %s", exc)

    def _log_system_info(self):
        """Auto-log system information."""
        if self._mlflow is None:
            return

        import platform
        self._mlflow.set_tag("python_version", platform.python_version())
        self._mlflow.set_tag("os", platform.system())

        # GPU info
        try:
            import torch
            if torch.cuda.is_available():
                self._mlflow.set_tag("gpu_name", torch.cuda.get_device_name(0))
                free, total = torch.cuda.mem_get_info(0)
                self._mlflow.set_tag("gpu_vram_gb", f"{total / 1e9:.1f}")
                self._mlflow.log_param("torch_version", torch.__version__)
                self._mlflow.log_param("cuda_version", torch.version.cuda or "N/A")
        except Exception:
            pass

        # Git hash
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            self._mlflow.set_tag("git_hash", git_hash)
        except Exception:
            pass

    @staticmethod
    def _compute_file_hash(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Why patience=5: Too low (1-2) causes premature stopping on noisy loss
    landscapes. Too high (>10) wastes compute on clearly plateaued models.
    5 strikes a balance for HNDSR's smooth loss curves.

    Why monitor val_loss: Training loss always decreases; val_loss reveals
    overfitting. Using PSNR instead of loss is risky because PSNR can
    plateau while the model is still learning perceptual quality.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",  # "min" for loss, "max" for PSNR
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Update early stopping state.

        Returns True if training should stop.
        """
        improved = (
            (value < self.best_value - self.min_delta)
            if self.mode == "min"
            else (value > self.best_value + self.min_delta)
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered: no improvement for %d epochs "
                    "(best=%.6f, current=%.6f)",
                    self.patience, self.best_value, value,
                )

        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# HPO integration (Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def create_hpo_study(
    study_name: str = "hndsr-hpo",
    direction: str = "maximize",  # maximize PSNR
    n_trials: int = 20,
    timeout_hours: float = 24.0,
):
    """
    Create and run an Optuna hyperparameter optimization study.

    What  : Automated search over learning rate, batch size, DDIM steps,
            latent dimension, and architecture variants.
    Why   : Manual tuning is slow and biased toward recent experiments.
            Bayesian optimization (TPE sampler) finds better configs in
            fewer trials than grid/random search.
    How   : Optuna samples from the search space, trains a lightweight
            model (reduced epochs), evaluates PSNR on validation set.
    """
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        return None

    def objective(trial):
        # ── Define search space ──────────────────────────────────────
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
            "latent_dim": trial.suggest_categorical("latent_dim", [64, 128, 256]),
            "ddim_steps": trial.suggest_int("ddim_steps", 20, 100, step=10),
            "diffusion_timesteps": trial.suggest_categorical("diffusion_timesteps", [500, 1000, 2000]),
            "autoencoder_epochs": trial.suggest_int("autoencoder_epochs", 10, 30, step=5),
            "neural_operator_modes": trial.suggest_int("neural_operator_modes", 8, 32, step=4),
            "unet_channels": trial.suggest_categorical("unet_channels", [64, 128, 256]),
        }

        # Log to MLflow
        tracker = HNDSRExperimentTracker(ExperimentConfig())
        with tracker.start_run(run_name=f"hpo_trial_{trial.number}", nested=True):
            tracker.log_params(params)

            # TODO: Replace with actual training call
            # psnr = train_and_evaluate(params)
            psnr = 25.0 + trial.number * 0.1  # Placeholder

            tracker.log_metrics({"val_psnr": psnr})

        return psnr

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_hours * 3600,
        n_jobs=1,  # GPU training is single-job
    )

    logger.info("Best trial: %s", study.best_trial.params)
    return study
