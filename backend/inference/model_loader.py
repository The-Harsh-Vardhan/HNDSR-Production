"""
backend/inference/model_loader.py
===================================
Thread-safe singleton model loader with lazy loading, warm-start caching,
and GPU/CPU memory pressure mitigation.

Adapted from Deployment/memory/model_loader.py.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


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
        logger.info("[%s] GPU memory: allocated=%.2f GB reserved=%.2f GB", tag, alloc, reserved)


def free_gpu_memory() -> None:
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.debug("GPU memory freed.")


class HNDSRModelLoader:
    """
    Thread-safe singleton model loader with lazy loading.

    Usage:
        loader = get_model_loader()
        loader.initialize(model_dir, ckpt_names, device)
        loader.warm_start()
        engine = HNDSRInferenceEngine(
            autoencoder=loader.autoencoder,
            neural_operator=loader.neural_operator,
            diffusion_unet=loader.diffusion_unet,
            device=device,
        )
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

        self._autoencoder: Optional[torch.nn.Module] = None
        self._neural_operator: Optional[torch.nn.Module] = None
        self._diffusion_unet: Optional[torch.nn.Module] = None
        self._load_lock = threading.Lock()

        self._initialized = True
        logger.info("ModelLoader initialized. model_dir=%s device=%s", model_dir, device)

    def _load_checkpoint(self, filename: str) -> torch.nn.Module:
        """Load a single model stage from its checkpoint file."""
        from backend.model.model_stubs import HNDSRAutoencoder, HNDSRNeuralOperator, HNDSRDiffusionUNet

        ckpt_path = self.model_dir / filename
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Ensure model files are present in the model directory. "
                "Run: python backend/inference/generate_checkpoints.py"
            )

        # Determine model class from filename convention
        if "autoencoder" in filename:
            model = HNDSRAutoencoder()
        elif "neural_operator" in filename:
            model = HNDSRNeuralOperator()
        elif "diffusion" in filename:
            model = HNDSRDiffusionUNet()
        else:
            raise ValueError(f"Cannot infer model class from filename: {filename}")

        t0 = time.perf_counter()
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = raw.get("model_state_dict", raw)
        model.load_state_dict(state, strict=False)
        model.eval()

        # Apply FP16 if requested
        if self.use_fp16 and self.device.type == "cuda":
            model = model.half()

        model = model.to(self.device)

        elapsed = time.perf_counter() - t0
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("Loaded %s in %.2f s → %s (%.1fM params)", filename, elapsed, self.device, param_count)
        return model

    @property
    def autoencoder(self) -> torch.nn.Module:
        if self._autoencoder is None:
            with self._load_lock:
                if self._autoencoder is None:
                    self._autoencoder = self._load_checkpoint(self.autoencoder_ckpt)
        return self._autoencoder

    @property
    def neural_operator(self) -> torch.nn.Module:
        if self._neural_operator is None:
            with self._load_lock:
                if self._neural_operator is None:
                    self._neural_operator = self._load_checkpoint(self.neural_operator_ckpt)
        return self._neural_operator

    @property
    def diffusion_unet(self) -> torch.nn.Module:
        if self._diffusion_unet is None:
            with self._load_lock:
                if self._diffusion_unet is None:
                    self._diffusion_unet = self._load_checkpoint(self.diffusion_ckpt)
        return self._diffusion_unet

    def warm_start(self) -> None:
        """Pre-load all model stages. Call at server startup."""
        logger.info("Warm-starting model loader...")
        log_memory_stats("before_warm_start")
        _ = self.autoencoder
        _ = self.neural_operator
        _ = self.diffusion_unet
        log_memory_stats("after_warm_start")
        logger.info("All model stages loaded and ready.")

    def unload_all(self) -> None:
        """Unload all models from memory."""
        self._autoencoder = None
        self._neural_operator = None
        self._diffusion_unet = None
        free_gpu_memory()
        logger.info("All models unloaded.")


# Module-level singleton accessor
_loader = HNDSRModelLoader()


def get_model_loader() -> HNDSRModelLoader:
    """Return the global singleton model loader."""
    return _loader
