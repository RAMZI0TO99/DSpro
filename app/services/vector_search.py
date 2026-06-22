import os
import torch
import open_clip
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

# --- Qdrant Initialization (fast) ---
_qdrant_url = os.getenv("QDRANT_URL")
if _qdrant_url:
    print(f"Connecting to Qdrant service at {_qdrant_url}")
    qdrant_client = QdrantClient(url=_qdrant_url)
else:
    print("Falling back to local Qdrant file-based database at ./qdrant_db")
    qdrant_client = QdrantClient(path="./qdrant_db")

# --- Lazy-load ML models on first use ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Models will be loaded on {device} on first request...")

clip_model = None
clip_tokenizer = None
clip_transform = None
bm25_model = None
ingestion_engine = None

def _load_models():
    """Lazy-load ML models on first request."""
    global clip_model, clip_tokenizer, clip_transform, bm25_model, ingestion_engine
    
    if clip_model is not None:
        return  # Already loaded
    
    print(f"[STARTUP] Loading CLIP & BM25 models on {device}...")
    
    clip_model, _, clip_transform = open_clip.create_model_and_transforms(
        'ViT-B-16-SigLIP', pretrained='webli'
    )
    clip_model = clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP-256')
    
    print("[STARTUP] Loading BM25 Lexical Engine...")
    bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    from ingestion import LocalHotIngestionPipeline
    
    ingestion_engine = LocalHotIngestionPipeline(
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        clip_transform=clip_transform,
        db_client=qdrant_client
    )
    
    print("[STARTUP] All models loaded successfully.")
