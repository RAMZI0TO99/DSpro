import os

os.makedirs("./models", exist_ok=True)

os.environ["HF_HOME"] = os.path.abspath("./models/huggingface")
os.environ["TORCH_HOME"] = os.path.abspath("./models/torch")
os.environ["FASTEMBED_CACHE_PATH"] = os.path.abspath("./models/fastembed")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.llm_service import llm_service
from app.core.config import load_settings

app = FastAPI(title="Agentic Spatio-Temporal RAG API")

# Mount static directories
app.mount("/media", StaticFiles(directory="media"), name="media")
app.mount("/css", StaticFiles(directory="Frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="Frontend/js"), name="js")

# Initialize global LLM settings on boot
llm_settings = load_settings()
llm_service.init_clients(llm_settings)

app.include_router(router)
