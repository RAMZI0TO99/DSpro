import os

# Only set local ./models paths when running locally (not inside Docker).
# In Docker, these env vars are already set correctly by the Dockerfile ENV instructions.
if not os.getenv("HF_HOME"):
    os.makedirs("./models", exist_ok=True)
    os.environ["HF_HOME"] = os.path.abspath("./models/huggingface")
    os.environ["TORCH_HOME"] = os.path.abspath("./models/torch")
    os.environ["FASTEMBED_CACHE_PATH"] = os.path.abspath("./models/fastembed")

# Ensure HF_HUB_OFFLINE is disabled locally so whisperx can download models
os.environ["HF_HUB_OFFLINE"] = "0"

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.llm_service import llm_service
from app.core.config import load_settings

app = FastAPI(title="Agentic Spatio-Temporal RAG API")

# Ensure static directories exist before mounting
os.makedirs("media", exist_ok=True)
os.makedirs("Frontend/css", exist_ok=True)
os.makedirs("Frontend/js", exist_ok=True)

# Mount static directories
app.mount("/media", StaticFiles(directory="media"), name="media")
app.mount("/css", StaticFiles(directory="Frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="Frontend/js"), name="js")

# Initialize global LLM settings on boot
llm_settings = load_settings()
llm_service.init_clients(llm_settings)

app.include_router(router)
