import os
import json
import pytest
import httpx

@pytest.fixture(scope="session")
def dataset():
    """
    Safely opens and parses tests/eval_dataset.json
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(dir_path, "eval_dataset.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def backend_url():
    """
    Returns http://localhost:8000/query with environment variable fallback
    """
    return os.getenv("BACKEND_URL", "http://localhost:8000/query")

@pytest.fixture(scope="function")
def http_client():
    """
    Returns an httpx Client with a 15-second timeout
    """
    with httpx.Client(timeout=60.0) as client:
        yield client
