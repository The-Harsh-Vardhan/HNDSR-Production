"""
data_pipeline/storage_config.py
=================================
Storage backend configuration for the HNDSR data pipeline.

What  : Defines S3/MinIO connection settings, Parquet schema, and versioning.
Why   : Centralised config prevents hard-coded paths scattered across scripts.
How   : Pydantic settings model reads from environment variables or .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class StorageSettings(BaseSettings):
    """
    Storage backend configuration.

    Supports:
      - AWS S3 (production)
      - MinIO (local development, S3-compatible)
      - Local filesystem (testing)
    """

    # ── Backend selection ─────────────────────────────────────────────────
    backend: Literal["s3", "minio", "local"] = Field(
        default="local",
        description="Storage backend type.",
    )

    # ── S3 / MinIO settings ───────────────────────────────────────────────
    s3_bucket: str = Field(
        default="hndsr-data",
        description="S3 bucket name for raw and processed data.",
    )
    s3_region: str = Field(default="us-east-1")
    s3_endpoint_url: Optional[str] = Field(
        default=None,
        description="Custom endpoint for MinIO. None = use AWS default.",
    )
    s3_access_key: Optional[str] = Field(default=None)
    s3_secret_key: Optional[str] = Field(default=None)

    # ── Local filesystem settings ─────────────────────────────────────────
    local_data_dir: Path = Field(
        default=Path("./data"),
        description="Local directory for data storage.",
    )

    # ── Paths within the bucket ───────────────────────────────────────────
    raw_prefix: str = Field(default="raw/", description="Prefix for raw HR images.")
    processed_prefix: str = Field(default="processed/", description="Prefix for processed train/val/test splits.")
    models_prefix: str = Field(default="models/", description="Prefix for model checkpoints.")
    metadata_prefix: str = Field(default="metadata/", description="Prefix for Parquet metadata files.")

    # ── DVC remote settings ───────────────────────────────────────────────
    dvc_remote_name: str = Field(default="s3-remote")
    dvc_cache_dir: Optional[str] = Field(
        default=None,
        description="Custom DVC cache directory. None = use default ~/.dvc/cache.",
    )

    # ── Versioning ────────────────────────────────────────────────────────
    enable_versioning: bool = Field(
        default=True,
        description="Enable S3 object versioning for rollback capability.",
    )
    retention_days: int = Field(
        default=90,
        description="Days to retain previous data versions.",
    )

    model_config = {
        "env_prefix": "HNDSR_STORAGE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parquet schema definitions
# ─────────────────────────────────────────────────────────────────────────────

METADATA_SCHEMA = {
    "filename": "string",
    "width": "int32",
    "height": "int32",
    "channels": "int8",
    "file_size_bytes": "int64",
    "sha256": "string",
    "scale_factor": "float32",
    "split": "string",       # train | val | test
    "source_file": "string",
    "sensor": "string",
    "capture_date": "string",  # ISO 8601
    "geographic_region": "string",
    "cloud_cover_pct": "float32",
}

CATALOG_SCHEMA = {
    "dataset_version": "string",
    "created_at": "string",
    "num_images": "int32",
    "total_size_bytes": "int64",
    "dataset_hash": "string",
    "scales": "list<float32>",
    "split_counts": "struct<train: int32, val: int32, test: int32>",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get S3 client
# ─────────────────────────────────────────────────────────────────────────────

def get_s3_client(settings: StorageSettings):
    """
    Create a boto3 S3 client from settings.

    Why a factory function: Avoids importing boto3 at module load time,
    which would break local-only development environments.
    """
    import boto3

    kwargs = {"region_name": settings.s3_region}

    if settings.s3_endpoint_url:
        kwargs["endpoint_url"] = settings.s3_endpoint_url

    if settings.s3_access_key and settings.s3_secret_key:
        kwargs["aws_access_key_id"] = settings.s3_access_key
        kwargs["aws_secret_access_key"] = settings.s3_secret_key

    return boto3.client("s3", **kwargs)


def get_storage_settings() -> StorageSettings:
    """Return a fresh StorageSettings instance."""
    return StorageSettings()
