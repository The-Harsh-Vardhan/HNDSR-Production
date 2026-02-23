"""
backend/inference/model_loader.py
===================================
Thread-safe singleton model loader for the composite HNDSR model.

Builds one HNDSR() model and loads individual sub-model checkpoints
(autoencoder, neural operator, diffusion UNet) into it.

Checkpoint formats (from training notebook):
  autoencoder_best.pth   → raw state_dict (OrderedDict of tensors)
  neural_operator_best.pth → raw state_dict
  diffusion_best.pth     → {'diffusion_unet': state_dict, 'ema_shadow': state_dict}
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(preference: str = "auto") -> torch.device:
    """
    Resolve the compute device.
    "auto" → CUDA if available, else CPU.
    "cuda" → CUDA; raises RuntimeError if unavailable.
    "cpu"  → Always CPU.
    """
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Auto-selected device: %s", device)
        return device
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "Set DEVICE=cpu or ensure CUDA drivers are installed."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def log_memory_stats(tag: str = "") -> None:
    """Log current GPU and CPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info("[%s] GPU memory: allocated=%.2f GB reserved=%.2f GB",
                     tag, alloc, reserved)


def free_gpu_memory() -> None:
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.debug("GPU memory freed.")


# ─────────────────────────────────────────────────────────────────────────────
# Model loader singleton
# ─────────────────────────────────────────────────────────────────────────────

class HNDSRModelLoader:
    """
    Thread-safe singleton that builds one HNDSR model and loads
    all checkpoint weights into it.

    Usage::

        loader = get_model_loader()
        loader.initialize(model_dir, device=device)
        loader.warm_start()
        model = loader.model  # full HNDSR with all weights loaded
    """

    _instance: Optional["HNDSRModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "HNDSRModelLoader":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def initialize(
        self,
        model_dir: Path,
        autoencoder_ckpt: str = "autoencoder_best.pth",
        neural_operator_ckpt: str = "neural_operator_best.pth",
        diffusion_ckpt: str = "diffusion_best.pth",
        device: torch.device = torch.device("cpu"),
        use_fp16: bool = True,
        manifest_path: Optional[Path] = None,
    ) -> None:
        """Configure the loader. Call once at application startup."""
        if self._initialized:
            logger.warning("ModelLoader already initialized; ignoring re-init.")
            return

        self.model_dir = Path(model_dir)
        self.autoencoder_ckpt = autoencoder_ckpt
        self.neural_operator_ckpt = neural_operator_ckpt
        self.diffusion_ckpt = diffusion_ckpt
        self.device = device
        self.use_fp16 = use_fp16
        self.manifest_path = Path(manifest_path) if manifest_path else (self.model_dir / "manifest.json")

        self._model: Optional[torch.nn.Module] = None
        self._load_lock = threading.Lock()
        self.checkpoint_hashes: dict[str, str] = {}
        self.checkpoint_manifest_match: bool = False
        self.manifest_validation_error: Optional[str] = None

        self._initialized = True
        logger.info(
            "ModelLoader initialized. model_dir=%s device=%s manifest=%s",
            model_dir, device, self.manifest_path,
        )

    def _checkpoint_paths(self) -> dict[str, Path]:
        return {
            self.autoencoder_ckpt: self.model_dir / self.autoencoder_ckpt,
            self.neural_operator_ckpt: self.model_dir / self.neural_operator_ckpt,
            self.diffusion_ckpt: self.model_dir / self.diffusion_ckpt,
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def compute_checkpoint_hashes(self) -> dict[str, str]:
        """
        Compute SHA256 hashes for required checkpoint files.

        Raises:
            FileNotFoundError: If any checkpoint file is missing.
        """
        hashes: dict[str, str] = {}
        for name, path in self._checkpoint_paths().items():
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            hashes[name] = self._sha256_file(path)
        self.checkpoint_hashes = hashes
        return hashes

    def _read_manifest_hashes(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Checkpoint manifest not found: {self.manifest_path}")

        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Manifest must be a JSON object")

        if isinstance(data.get("files"), dict):
            raw_files = data["files"]
        elif isinstance(data.get("checkpoints"), dict):
            raw_files = data["checkpoints"]
        else:
            raw_files = {
                k: v for k, v in data.items()
                if k in {self.autoencoder_ckpt, self.neural_operator_ckpt, self.diffusion_ckpt}
            }

        expected: dict[str, str] = {}
        for name in self._checkpoint_paths():
            raw_hash = raw_files.get(name)
            if not isinstance(raw_hash, str) or not raw_hash.strip():
                raise ValueError(f"Manifest missing hash for {name}")
            expected[name] = raw_hash.strip().lower()

        return expected

    def validate_checkpoint_manifest(self) -> bool:
        """
        Validate local checkpoint files against the manifest SHA256 hashes.

        Returns:
            True if manifest exists and all hashes match; False otherwise.
        """
        try:
            expected = self._read_manifest_hashes()
            actual = self.compute_checkpoint_hashes()
        except Exception as exc:
            self.checkpoint_manifest_match = False
            self.manifest_validation_error = str(exc)
            logger.error("Checkpoint manifest validation failed: %s", exc)
            return False

        mismatches: list[str] = []
        for name, expected_hash in expected.items():
            actual_hash = actual.get(name, "")
            if actual_hash != expected_hash:
                mismatches.append(
                    f"{name}: expected={expected_hash[:12]}... actual={actual_hash[:12]}..."
                )

        if mismatches:
            self.checkpoint_manifest_match = False
            self.manifest_validation_error = "; ".join(mismatches)
            logger.error("Checkpoint manifest mismatch: %s", self.manifest_validation_error)
            return False

        self.checkpoint_manifest_match = True
        self.manifest_validation_error = None
        logger.info("Checkpoint manifest validated successfully: %s", self.manifest_path)
        return True

    # ── Internal loader ───────────────────────────────────────────────────

    def _load_full_model(self) -> torch.nn.Module:
        """Build the composite HNDSR model and load all checkpoint files."""
        from backend.model.model_stubs import HNDSR

        t0 = time.perf_counter()
        model = HNDSR()

        # ── Autoencoder ──
        ae_path = self.model_dir / self.autoencoder_ckpt
        if not ae_path.exists():
            raise FileNotFoundError(
                f"Autoencoder checkpoint not found: {ae_path}")
        ae_raw = torch.load(ae_path, map_location="cpu", weights_only=False)
        ae_state = (ae_raw.get("model_state_dict", ae_raw)
                    if isinstance(ae_raw, dict) and "model_state_dict" in ae_raw
                    else ae_raw)
        model.autoencoder.load_state_dict(ae_state, strict=True)
        logger.info("Loaded autoencoder from %s", ae_path.name)

        # ── Neural Operator ──
        no_path = self.model_dir / self.neural_operator_ckpt
        if not no_path.exists():
            raise FileNotFoundError(
                f"Neural operator checkpoint not found: {no_path}")
        no_raw = torch.load(no_path, map_location="cpu", weights_only=False)
        no_state = (no_raw.get("model_state_dict", no_raw)
                    if isinstance(no_raw, dict) and "model_state_dict" in no_raw
                    else no_raw)
        model.neural_operator.load_state_dict(no_state, strict=True)
        logger.info("Loaded neural operator from %s", no_path.name)

        # ── Diffusion UNet ──
        diff_path = self.model_dir / self.diffusion_ckpt
        if not diff_path.exists():
            raise FileNotFoundError(
                f"Diffusion checkpoint not found: {diff_path}")
        diff_raw = torch.load(diff_path, map_location="cpu", weights_only=False)

        # Prefer EMA shadow weights (moving average → better quality)
        if isinstance(diff_raw, dict) and "ema_shadow" in diff_raw:
            diff_state = diff_raw["ema_shadow"]
            logger.info("Using EMA shadow weights for diffusion UNet")
        elif isinstance(diff_raw, dict) and "diffusion_unet" in diff_raw:
            diff_state = diff_raw["diffusion_unet"]
            logger.info("Using training weights for diffusion UNet")
        elif isinstance(diff_raw, dict) and "model_state_dict" in diff_raw:
            diff_state = diff_raw["model_state_dict"]
        else:
            diff_state = diff_raw
        model.diffusion_unet.load_state_dict(diff_state, strict=True)
        logger.info("Loaded diffusion UNet from %s", diff_path.name)

        # Note: ImplicitAmplification has no saved checkpoint because it was
        # never optimised independently during training. It uses fresh random
        # init. The diffusion model's cross-attention is robust to this since
        # the context vector primarily captures global structure.
        logger.warning(
            "ImplicitAmplification uses random init (no checkpoint). "
            "Quality may differ slightly from the training session."
        )

        model.eval()
        model = model.to(self.device)

        elapsed = time.perf_counter() - t0
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(
            "Full HNDSR model loaded in %.2f s (%.1f M params) → %s",
            elapsed, total_params, self.device,
        )
        return model

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def model(self) -> torch.nn.Module:
        """Return the full HNDSR model (lazy-loaded on first access)."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._model = self._load_full_model()
        return self._model

    # Backward-compat sub-model accessors
    @property
    def autoencoder(self) -> torch.nn.Module:
        return self.model.autoencoder

    @property
    def neural_operator(self) -> torch.nn.Module:
        return self.model.neural_operator

    @property
    def diffusion_unet(self) -> torch.nn.Module:
        return self.model.diffusion_unet

    def warm_start(self) -> None:
        """Pre-load the full model. Call at server startup."""
        logger.info("Warm-starting model loader...")
        log_memory_stats("before_warm_start")
        _ = self.model
        log_memory_stats("after_warm_start")
        logger.info("All model stages loaded and ready.")

    def unload_all(self) -> None:
        """Unload all models from memory."""
        self._model = None
        free_gpu_memory()
        logger.info("All models unloaded.")


# Module-level singleton accessor
_loader = HNDSRModelLoader()


def get_model_loader() -> HNDSRModelLoader:
    """Return the global singleton model loader."""
    return _loader
