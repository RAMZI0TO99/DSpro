# ============================================================
# Base Image: Official PyTorch 2.5.1 + CUDA 12.1 image
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

RUN mkdir -p /app/media /app/qdrant_db /app/graph_db

# ── 4. Install Python dependencies ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# ── 5. Install packages with --no-deps ──────────────────────
RUN pip install --no-deps \
    pyannote-audio==4.0.4 \
    git+https://github.com/m-bain/whisperX.git@5f2f9d4320dd93a7d12f5ba2495eef7e0a5af963

# ── 6. Download NLTK data ─────────────────────────────────
RUN python -m nltk.downloader -d /root/nltk_data punkt_tab

# ── 7. Set cache directories BEFORE downloading models ────
ENV HF_HOME=/app/models/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/models/huggingface/hub
ENV TORCH_HOME=/app/models/torch
ENV EASYOCR_HOME=/app/models/easyocr
ENV FASTEMBED_CACHE_PATH=/app/models/fastembed
ENV NLTK_DATA=/root/nltk_data
# NOTE: This is 0 during BUILD to allow pip/nltk to work normally.
# docker-compose.yml overrides this to 1 at RUNTIME to enforce strict offline mode.
ENV HF_HUB_LOCAL_FILES_ONLY=0

# ── 8. Copy application source and models ─────────────────────────────
# Because .dockerignore is perfectly configured, this safely copies the app code, 
# Frontend, and the entire 10GB models/ folder into the image in one step!
COPY . .

RUN echo "✓ Source code and all models successfully baked into the image from the host."

# ── 10. Expose FastAPI port ──────────────────────────────────
EXPOSE 8000

# ── 11. Launch Uvicorn ──────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
