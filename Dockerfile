# ═══════════════════════════════════════════════════════════════════════════
# HNDSR Hugging Face Spaces Dockerfile
# ═══════════════════════════════════════════════════════════════════════════

# Stage 1: Builder
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt


# Stage 2: Runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 libgl1-mesa-glx libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Create a non-root user for Hugging Face (UID 1000 is required)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy application code
COPY --chown=user backend/ ./backend/
COPY --chown=user checkpoints/ ./checkpoints/

# Set defaults for Hugging Face (CPU-only free tier)
ENV DEVICE=cpu
ENV MODEL_DIR=./checkpoints
ENV USE_FP16=false
ENV DDIM_STEPS=10
ENV TILE_SIZE=128
ENV TILE_OVERLAP=16
ENV REQUEST_TIMEOUT_S=300
ENV MAX_INPUT_DIM=512
ENV CHECKPOINT_MANIFEST_PATH=./checkpoints/manifest.json
ENV ENFORCE_CHECKPOINT_MANIFEST=true
ENV ALLOW_FALLBACK_ON_INVALID_CKPT=true
ENV ENABLE_QUALITY_PROBE=true
ENV QUALITY_PROBE_DDIM_STEPS=10
ENV QUALITY_PROBE_INPUT_SIZE=64
ENV QUALITY_PROBE_MIN_STD=0.05

# Hugging Face Spaces listen on port 7860
EXPOSE 7860

# Run with uvicorn on port 7860
CMD ["python3.11", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
