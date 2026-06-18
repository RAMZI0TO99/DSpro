# ============================================================
# Base Image: Official PyTorch 2.5.1 + CUDA 12.1 image
# PyTorch is ALREADY INSTALLED here — no 780 MB wheel download!
# Python 3.11 is the default interpreter in this image.
# ============================================================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── 1. System dependencies ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Working directory ───────────────────────────────────
WORKDIR /app

# Pre-create runtime directories as fallback when running without Compose
RUN mkdir -p /app/media /app/qdrant_db

# ── 3. Install all Python dependencies ─────────────────────
# PyTorch/torchvision/torchaudio are already in the base image.
# pip will recognize them as already satisfied and skip re-downloading.
# --timeout 300 guards against slow connections on other packages.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# ── 4. Copy application source ─────────────────────────────
COPY . .

# ── 5. Expose FastAPI port ──────────────────────────────────
EXPOSE 8000

# ── 6. Launch Uvicorn ───────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]