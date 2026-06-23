import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

SETTINGS_FILE = "settings.json"

class LLMSettings(BaseModel):
    provider: str = "local"
    # IMPORTANT: Use host.docker.internal, NOT localhost/127.0.0.1.
    # Inside Docker, localhost points to the container itself, not the host PC.
    base_url: str = "http://host.docker.internal:1234/v1"
    model: str = "Llama-3.2-3B-Instruct-Q4_K_M"
    api_key: str = ""

def load_settings() -> LLMSettings:
    # When running inside Docker, QDRANT_URL is always set.
    # In that case, skip settings.json entirely and use env vars so
    # docker-compose.yml is the single source of truth for configuration.
    is_docker = bool(os.getenv("QDRANT_URL"))

    if not is_docker and os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return LLMSettings(**data)
        except Exception as e:
            print(f"Error loading settings: {e}. Falling back to env vars.")

    # Build settings from environment variables (works both locally and in Docker)
    env_provider = os.getenv("LLM_PROVIDER", "local").lower()
    env_api_key = os.getenv(f"{env_provider.upper()}_API_KEY", "")
    env_base_url = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
    env_model = os.getenv("LLM_MODEL", "Llama-3.2-3B-Instruct-Q4_K_M")

    return LLMSettings(
        provider=env_provider,
        api_key=env_api_key,
        base_url=env_base_url,
        model=env_model,
    )

def save_settings(settings: LLMSettings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.write(settings.model_dump_json(indent=4))

llm_settings = load_settings()
