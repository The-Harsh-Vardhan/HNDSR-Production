# ═══════════════════════════════════════════════════════════════════════════
# HNDSR Production Dockerfile — Multi-stage build
# ═══════════════════════════════════════════════════════════════════════════
# What: Multi-stage Docker image for GPU-accelerated HNDSR inference.
# Why : Separates build-time deps from runtime → 40% smaller final image.
#       Build stage includes compilers; runtime stage has only what's needed.
# How : Stage 1 (builder) installs Python deps. Stage 2 (runtime) copies
#       only the virtual environment and application code.
# ═══════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements-prod.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 libgl1-mesa-glx libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user (security hardening)
RUN groupadd -r hndsr && useradd -r -g hndsr -d /app hndsr

# Copy application code
WORKDIR /app
COPY backend/ ./backend/
COPY observability/ ./observability/

# Create directories for models and data
RUN mkdir -p /app/checkpoints /app/data /app/logs && \
    chown -R hndsr:hndsr /app

# Switch to non-root user
USER hndsr

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Entry point
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
