import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

SETTINGS_FILE = "settings.json"

class LLMSettings(BaseModel):
    provider: str = "local"
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "Llama-3.2-3B-Instruct-Q4_K_M"
    api_key: str = ""

def load_settings() -> LLMSettings:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return LLMSettings(**data)
        except Exception as e:
            print(f"Error loading settings: {e}. Falling back to defaults.")
    
    env_provider = os.getenv("LLM_PROVIDER", "local").lower()
    env_api_key = os.getenv(f"{env_provider.upper()}_API_KEY", "")
    return LLMSettings(
        provider=env_provider,
        api_key=env_api_key,
        base_url="http://host.docker.internal:1234/v1",
        model="Llama-3.2-3B-Instruct-Q4_K_M"
    )

def save_settings(settings: LLMSettings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.write(settings.model_dump_json(indent=4))

llm_settings = load_settings()
