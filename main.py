import os

# --- FALLBACK FOR LOCAL OFFLINE MODE ---
# If these environment variables aren't set (e.g. running locally via `uvicorn main:app` instead of Docker),
# point them to the local `models/` directory to ensure 100% offline functionality.
local_models_dir = os.path.abspath("./models")
os.environ["HF_HOME"] = os.path.join(local_models_dir, "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(local_models_dir, "huggingface", "hub")
os.environ["TORCH_HOME"] = os.path.join(local_models_dir, "torch")
os.environ["EASYOCR_HOME"] = os.path.join(local_models_dir, "easyocr")
os.environ["FASTEMBED_CACHE_PATH"] = os.path.join(local_models_dir, "fastembed")

# If you want to force offline mode after downloading models, uncomment these:
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

from app.main import app

if __name__ == "__main__":
    import uvicorn
    import asyncio
    import sys

    # Silences the WinError 10054 connection drops on Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8000)