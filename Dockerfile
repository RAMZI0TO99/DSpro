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

RUN mkdir -p /app/media /app/qdrant_db

# ── 3. Lock PyTorch versions ─────────────────────────────
RUN echo "torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\ntorchaudio==2.5.1+cu121" > /tmp/torch_constraints.txt

# ── 4. Install Python dependencies ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 \
    --constraint /tmp/torch_constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

# ── 5. Install packages with --no-deps ──────────────────────
RUN pip install --no-deps \
    pyannote-audio==4.0.4 \
    git+https://github.com/m-bain/whisperX.git@5f2f9d4320dd93a7d12f5ba2495eef7e0a5af963

# ── 6. Download NLTK data ─────────────────────────────────
RUN python -m nltk.downloader -d /root/nltk_data punkt_tab

# ── 7. Set cache directories BEFORE downloading models ────
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV EASYOCR_HOME=/root/.EasyOCR
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_LOCAL_FILES_ONLY=0

# ── 8. Pre-download ML models with cleanup on failure ──────
# Create a script to download models reliably with retries

RUN mkdir -p /root/.cache/huggingface /root/.cache/torch /root/.EasyOCR

# Download CLIP model with retry logic
RUN python -c "import os; os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface'; import open_clip; print('[MODEL 1/5] Downloading CLIP ViT-B-16-SigLIP...'); open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli'); print('✓ CLIP cached')"

# Download Whisper Base model
RUN python -c "import os; os.environ['TORCH_HOME'] = '/root/.cache/torch'; import whisperx; print('[MODEL 2/5] Downloading Whisper Base...'); whisperx.load_model('base', device='cpu', compute_type='float32'); print('✓ Whisper cached')"

# Download Alignment models (EN + AR)
RUN python -c "import os; os.environ['TORCH_HOME'] = '/root/.cache/torch'; import whisperx; print('[MODEL 3/5] Downloading alignment models...'); whisperx.load_align_model(language_code='en', device='cpu'); whisperx.load_align_model(language_code='ar', device='cpu'); print('✓ Alignment models cached')"

# Download EasyOCR models
RUN python -c "import os; os.environ['EASYOCR_HOME'] = '/root/.EasyOCR'; import easyocr; print('[MODEL 4/5] Downloading EasyOCR (EN + AR)...'); easyocr.Reader(['en', 'ar'], gpu=False, verbose=False); print('✓ EasyOCR cached')"

# Download BM25 and tokenizer
RUN python -c "import os; os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface'; from fastembed import SparseTextEmbedding; from open_clip.tokenizer import HFTokenizer; print('[MODEL 5/5] Downloading BM25 + Tokenizer...'); SparseTextEmbedding(model_name='Qdrant/bm25'); HFTokenizer('google/siglip-base-patch16-256', context_length=64); print('✓ BM25 + Tokenizer cached')"

RUN echo "✓ All models pre-cached successfully"

# ── 9. Copy application source ─────────────────────────────
COPY . .

# ── 10. Expose FastAPI port ──────────────────────────────────
EXPOSE 8000

# ── 11. Launch Uvicorn ──────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
