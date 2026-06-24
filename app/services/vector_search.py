import os
import threading
import torch
import open_clip
from fastembed import SparseTextEmbedding, TextEmbedding
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
dense_text_model = None
ingestion_engine = None

_model_lock = threading.Lock()

def _load_models():
    """Lazy-load ML models on first request. Thread-safe."""
    global clip_model, clip_tokenizer, clip_transform, bm25_model, dense_text_model, ingestion_engine

    if clip_model is not None:
        return  # Already loaded (fast path, no lock needed)

    with _model_lock:
        # Double-checked locking: re-check after acquiring lock
        if clip_model is not None:
            return

        print(f"[STARTUP] Loading CLIP & BM25 models on {device}...")

        import os
        siglip_snapshots_dir = os.path.join(
            os.environ.get("HF_HOME", ""),
            "hub", "models--timm--ViT-B-16-SigLIP", "snapshots"
        )
        
        pretrained_arg = 'webli'
        if os.path.exists(siglip_snapshots_dir):
            snapshots = os.listdir(siglip_snapshots_dir)
            if snapshots:
                siglip_cache_path = os.path.join(siglip_snapshots_dir, snapshots[0], "open_clip_model.safetensors")
                if os.path.exists(siglip_cache_path):
                    pretrained_arg = siglip_cache_path

        clip_model, _, clip_transform = open_clip.create_model_and_transforms(
            'ViT-B-16-SigLIP', pretrained=pretrained_arg
        )
        clip_model = clip_model.to(device).eval()
        
        from open_clip.tokenizer import HFTokenizer
        tokenizer_snapshots_dir = os.path.join(
            os.environ.get("HF_HOME", ""),
            "hub", "models--google--siglip-base-patch16-256", "snapshots"
        )
        tokenizer_path = 'google/siglip-base-patch16-256'
        if os.path.exists(tokenizer_snapshots_dir):
            tokenizer_snapshots = os.listdir(tokenizer_snapshots_dir)
            if tokenizer_snapshots:
                tokenizer_path = os.path.join(tokenizer_snapshots_dir, tokenizer_snapshots[0])
                
        clip_tokenizer = HFTokenizer(tokenizer_path, context_length=64)

        print("[STARTUP] Loading BM25 Lexical Engine...")
        bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        print("[STARTUP] Loading Multilingual Dense Text Engine...")
        dense_text_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        from ingestion import LocalHotIngestionPipeline

        ingestion_engine = LocalHotIngestionPipeline(
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
            clip_transform=clip_transform,
            dense_text_model=dense_text_model,
            db_client=qdrant_client
        )

        print("[STARTUP] All models loaded successfully.")

