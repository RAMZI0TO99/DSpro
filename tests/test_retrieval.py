import os
import json
import pytest

# Load the dataset for parameterization at test collection time
dir_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(dir_path, "eval_dataset.json")

try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
except Exception as e:
    # Fail-safe empty cases if dataset is missing or corrupt during discovery
    eval_cases = []

@pytest.mark.parametrize(
    "dataset",
    eval_cases,
    ids=[case["query_id"] for case in eval_cases] if eval_cases else None
)
def test_context_recall(dataset, backend_url, http_client):
    """
    RAG Context Recall test suite:
    1. Sends the query to the configured RAG backend url (default: /query).
    2. Extracts chunk IDs from response (supporting both 'id' fields and search payload mappings).
    3. Computes the recall score against expectations.
    4. Asserts recall >= 0.80, listing missing IDs on failure.
    """
    query_id = dataset["query_id"]
    query_text = dataset["query"]
    expected_chunk_ids = dataset["expected_chunk_ids"]

    # 1. Send POST request to our FastAPI query endpoint
    payload = {
        "query": query_text,
        "top_k": 5
    }

    print(f"\n[*] Sending request for {query_id} to: {backend_url}")
    response = http_client.post(backend_url, json=payload)

    # Output details to stdout for debug assistance
    if response.status_code == 404:
        raise AssertionError(
            f"Endpoint 404 Not Found at: {backend_url}. "
            "Please ensure the backend is running and the BACKEND_URL environment variable "
            "is set to the correct endpoint (e.g. http://localhost:8000/search)."
        )

    response.raise_for_status()
    res_data = response.json()

    # 2. Extract returned ID fields from response chunks
    retrieved_chunk_ids = []
    chunks = []

    # Safe parsing based on standard variations of FastAPI responses
    if isinstance(res_data, list):
        chunks = res_data
    elif isinstance(res_data, dict):
        chunks = res_data.get("results") or res_data.get("chunks") or res_data.get("data")
        if chunks is None:
            # Fallback: scan for any list in the response dict keys
            for val in res_data.values():
                if isinstance(val, list):
                    chunks = val
                    break
            if chunks is None:
                chunks = [res_data]

    for chunk in chunks:
        if isinstance(chunk, dict):
            # A. Extract 'id' directly as primary field (matches /query spec)
            if "id" in chunk:
                retrieved_chunk_ids.append(str(chunk["id"]))
            # B. Fallback to video_id_start_timestamp if no 'id' but video_id/start_timestamp exist
            # (Matches current /search schema to allow seamless transition testing)
            elif "video_id" in chunk and "start_timestamp" in chunk:
                retrieved_chunk_ids.append(f"{chunk['video_id']}_{chunk['start_timestamp']}")
            elif "video_id" in chunk:
                retrieved_chunk_ids.append(str(chunk["video_id"]))
        elif isinstance(chunk, (str, int)):
            retrieved_chunk_ids.append(str(chunk))

    # 3. Calculate Context Recall Score
    retrieved_set = set(retrieved_chunk_ids)
    expected_set = set(expected_chunk_ids)

    if not expected_set:
        recall = 1.0
    else:
        recall = len(retrieved_set.intersection(expected_set)) / len(expected_set)

    missing_chunk_ids = expected_set - retrieved_set

    # 4. Assert recall satisfies 80% threshold
    assert recall >= 0.80, (
        f"Context Recall Assertion Failed for {query_id}!\n"
        f"Query: '{query_text}'\n"
        f"Expected Chunk IDs: {list(expected_set)}\n"
        f"Retrieved Chunk IDs: {list(retrieved_set)}\n"
        f"Recall Score: {recall:.2%} (Required: >= 80.00%)\n"
        f"Missing Chunk IDs: {list(missing_chunk_ids)}"
    )
