"""
model_registry/registry_integration.py
=========================================
DVC artifacts → MLflow Model Registry connector.

What  : Bridges the gap between DVC-tracked training artifacts and MLflow's
        model registry for version management and deployment workflows.
Why   : DVC tracks reproducibility (data + code + params → artifacts).
        MLflow tracks deployment lifecycle (dev → staging → production).
        Both are needed; this module connects them.
How   : After DVC produces checkpoints, this module registers them in
        MLflow with metadata, enabling promotion through deployment stages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a registered model version."""
    name: str
    version: int
    stage: str       # dev | staging | canary | production | archived
    metrics: Dict[str, float]
    dataset_hash: str
    checkpoint_hash: str
    registered_at: str
    promoted_at: Optional[str] = None
    promoted_by: Optional[str] = None


class ModelRegistry:
    """
    HNDSR Model Registry — manages model lifecycle.

    Stages:
      1. dev       — any completed training run
      2. staging   — passes automated quality gates
      3. canary    — serving 10% of production traffic
      4. production — serving 100% of production traffic
      5. archived  — replaced by newer version

    Promotion rules:
      - dev → staging:     Automated (PSNR > threshold, shape tests pass)
      - staging → canary:  Manual trigger, auto-reverts if error rate spikes
      - canary → production: Manual approval after monitoring period
    """

    def __init__(
        self,
        model_name: str = "hndsr",
        tracking_uri: str = "sqlite:///mlflow.db",
    ):
        self.model_name = model_name
        self._versions: List[ModelVersion] = []

        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow registry connected: %s", tracking_uri)
        except ImportError:
            self._mlflow = None
            logger.warning("MLflow not installed. Registry operations will be logged only.")

    def register_model(
        self,
        checkpoint_dir: str,
        metrics: Dict[str, float],
        dataset_hash: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        """
        Register a new model version from DVC checkpoints.

        Registers all three HNDSR components as a single model version:
          - autoencoder_best.pth
          - neural_operator_best.pth
          - diffusion_unet_best.pth

        Args:
            checkpoint_dir: Directory containing the three checkpoint files.
            metrics: Evaluation metrics (psnr, ssim, lpips).
            dataset_hash: SHA-256 hash of the training dataset.
            run_id: MLflow run ID that produced these checkpoints.
            tags: Additional metadata tags.

        Returns:
            ModelVersion with the assigned version number.
        """
        ckpt_path = Path(checkpoint_dir)

        # Verify all three checkpoints exist
        required_files = [
            "autoencoder_best.pth",
            "neural_operator_best.pth",
            "diffusion_unet_best.pth",
        ]
        for fname in required_files:
            if not (ckpt_path / fname).exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path / fname}")

        # Compute combined checkpoint hash
        checkpoint_hash = self._compute_combined_hash(ckpt_path, required_files)

        # Check for duplicate registration
        for v in self._versions:
            if v.checkpoint_hash == checkpoint_hash:
                logger.warning("Model already registered as version %d", v.version)
                return v

        # Register in MLflow
        version_num = len(self._versions) + 1

        if self._mlflow:
            try:
                from mlflow.models import infer_signature

                # Log model artifacts
                with self._mlflow.start_run(run_id=run_id) as run:
                    for fname in required_files:
                        self._mlflow.log_artifact(
                            str(ckpt_path / fname),
                            artifact_path="model",
                        )
                    self._mlflow.log_metrics(metrics)

                    if tags:
                        for k, v in tags.items():
                            self._mlflow.set_tag(k, v)

                # Register model version
                model_uri = f"runs:/{run.info.run_id}/model"
                result = self._mlflow.register_model(model_uri, self.model_name)
                version_num = int(result.version)

            except Exception as exc:
                logger.error("MLflow registration failed: %s", exc)

        model_version = ModelVersion(
            name=self.model_name,
            version=version_num,
            stage="dev",
            metrics=metrics,
            dataset_hash=dataset_hash,
            checkpoint_hash=checkpoint_hash,
            registered_at=datetime.now(timezone.utc).isoformat(),
        )
        self._versions.append(model_version)

        logger.info(
            "Registered %s v%d | PSNR=%.2f | stage=dev",
            self.model_name, version_num, metrics.get("psnr", 0),
        )
        return model_version

    def promote(
        self,
        model_name: str,
        version: int,
        stage: str,
        promoted_by: str = "system",
    ) -> bool:
        """
        Promote a model version to a new deployment stage.

        Transition rules:
          - dev → staging:     Requires PSNR > acceptance threshold
          - staging → canary:  Requires manual approval
          - canary → production: Requires monitoring period passed
          - any → archived:    Always allowed

        Returns True if promotion succeeded.
        """
        valid_transitions = {
            "dev": ["staging", "archived"],
            "staging": ["canary", "archived"],
            "canary": ["production", "staging"],  # can roll back to staging
            "production": ["archived"],
            "archived": [],
        }

        # Find the version
        target = None
        for v in self._versions:
            if v.name == model_name and v.version == version:
                target = v
                break

        if target is None:
            logger.error("Version %d not found for %s", version, model_name)
            return False

        # Check valid transition
        if stage not in valid_transitions.get(target.stage, []):
            logger.error(
                "Invalid transition: %s → %s (allowed: %s)",
                target.stage, stage, valid_transitions.get(target.stage, []),
            )
            return False

        # Auto-gate: dev → staging requires minimum metrics
        if target.stage == "dev" and stage == "staging":
            if not self._check_staging_gates(target):
                return False

        old_stage = target.stage
        target.stage = stage
        target.promoted_at = datetime.now(timezone.utc).isoformat()
        target.promoted_by = promoted_by

        # Update MLflow
        if self._mlflow:
            try:
                client = self._mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=str(version),
                    stage=self._mlflow_stage(stage),
                )
            except Exception as exc:
                logger.error("MLflow stage transition failed: %s", exc)

        logger.info(
            "Promoted %s v%d: %s → %s (by %s)",
            model_name, version, old_stage, stage, promoted_by,
        )
        return True

    def rollback(self, model_name: str) -> Optional[ModelVersion]:
        """
        Roll back to the previous production version.

        Action: Archive current production, re-promote the most recent
        archived version that was previously in production.
        """
        # Find current production version
        current_prod = None
        for v in reversed(self._versions):
            if v.name == model_name and v.stage == "production":
                current_prod = v
                break

        if current_prod is None:
            logger.error("No production version found for %s", model_name)
            return None

        # Find previous production version (now archived)
        previous = None
        for v in reversed(self._versions):
            if (v.name == model_name
                and v.stage == "archived"
                and v.version < current_prod.version):
                previous = v
                break

        if previous is None:
            logger.error("No previous version available for rollback")
            return None

        # Archive current, re-promote previous
        self.promote(model_name, current_prod.version, "archived", promoted_by="rollback")
        previous.stage = "production"
        previous.promoted_at = datetime.now(timezone.utc).isoformat()
        previous.promoted_by = "rollback"

        logger.info(
            "Rolled back %s: v%d → v%d",
            model_name, current_prod.version, previous.version,
        )
        return previous

    def get_production_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the current production model version."""
        for v in reversed(self._versions):
            if v.name == model_name and v.stage == "production":
                return v
        return None

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        return [v for v in self._versions if v.name == model_name]

    # ── Private helpers ──────────────────────────────────────────────────

    def _check_staging_gates(self, version: ModelVersion) -> bool:
        """Check if a model meets staging promotion criteria."""
        gates = {
            "psnr": (">=", 26.0),
            "ssim": (">=", 0.75),
            "lpips": ("<=", 0.30),
        }

        for metric, (op, threshold) in gates.items():
            value = version.metrics.get(metric)
            if value is None:
                logger.error("Missing metric '%s' for staging gate", metric)
                return False

            passed = value >= threshold if op == ">=" else value <= threshold
            if not passed:
                logger.error(
                    "Staging gate failed: %s=%g %s %g",
                    metric, value, op, threshold,
                )
                return False

        logger.info("All staging gates passed")
        return True

    @staticmethod
    def _compute_combined_hash(ckpt_dir: Path, filenames: list) -> str:
        """Compute a combined hash of all checkpoint files."""
        hashes = []
        for fname in sorted(filenames):
            h = hashlib.sha256()
            with open(ckpt_dir / fname, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            hashes.append(h.hexdigest())
        combined = "".join(hashes)
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def _mlflow_stage(stage: str) -> str:
        """Map internal stage names to MLflow stage names."""
        mapping = {
            "dev": "None",
            "staging": "Staging",
            "canary": "Staging",      # MLflow doesn't have canary
            "production": "Production",
            "archived": "Archived",
        }
        return mapping.get(stage, "None")
