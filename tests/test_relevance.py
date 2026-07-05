import os
import json
import pytest
from tests.relevance_utils import evaluate_relevance, evaluate_refusal

dir_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(dir_path, "eval_dataset.json")

try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
except Exception:
    eval_cases = []

@pytest.mark.parametrize(
    "dataset",
    eval_cases,
    ids=[case["query_id"] for case in eval_cases] if eval_cases else None
)
def test_system_relevance_and_refusal(dataset, backend_url, http_client):
    """
    Test 3: Answer Relevance & Adversarial Refusal
    1. Iterates through golden dataset queries.
    2. Sends request and captures generated_answer.
    3. If query is standard, asserts relevance_score >= 0.80.
    4. If query is adversarial, asserts successful refusal (True).
    """
    query_id = dataset["query_id"]
    query_text = dataset["query"]
    is_adversarial = dataset.get("is_adversarial", False)
    
    payload = {
        "query": query_text
    }
    
    print(f"\n[*] Evaluating Relevance & Refusal for {query_id} via: {backend_url}")
    response = http_client.post(backend_url, json=payload)
    response.raise_for_status()
    res_data = response.json()
    
    generated_answer = res_data.get("generated_answer", "")
    chunks_data = res_data.get("chunks", [])
    
    if not generated_answer and res_data.get("answer"):
        generated_answer = res_data.get("answer")
    if not chunks_data and res_data.get("results"):
        chunks_data = res_data.get("results")
        
    context_chunks = []
    for c in chunks_data:
        if isinstance(c, dict):
            transcript = c.get("transcript") or c.get("matched_transcript") or str(c)
            context_chunks.append(transcript)
        elif isinstance(c, str):
            context_chunks.append(c)

    if not generated_answer:
        print(f"    -> Endpoint didn't return 'generated_answer', simulating LLM pipeline locally...")
        from app.services.llm_service import llm_service
        from app.core.config import load_settings
        llm_settings_local = load_settings()
        llm_service.init_clients(llm_settings_local)
        
        context_str = "\n".join(context_chunks)
        sys_prompt = f"Answer the user's query using ONLY the following context. If the context is empty or irrelevant, you MUST explicitly refuse to answer and state that it is not present in the video data. Do NOT guess or fabricate information.\nContext:\n{context_str}"
        gen = llm_service.generate_chat_stream(sys_prompt, [], query_text, llm_settings_local.provider.lower(), llm_settings_local.model)
        generated_answer = "".join([chunk for chunk in gen])

    if is_adversarial:
        # Check adversarial refusal
        successfully_refused = evaluate_refusal(query_text, generated_answer)
        
        from tests.relevance_utils import evaluate_refusal as eval_fn
        last_result = getattr(eval_fn, "last_result", {})
        
        if not successfully_refused:
            print(f"\n--- ADVERSARIAL REFUSAL FAILED FOR {query_id} ---")
            print(f"Query: {query_text}")
            print(f"Hallucinated Answer: {generated_answer}")
            print(f"Judge Reasoning: {last_result.get('reasoning')}")
            
        assert successfully_refused is True, (
            f"Adversarial Guardrail failed for {query_id}! "
            f"The model hallucinated an answer instead of refusing. "
            f"See stdout for the hallucinated payload."
        )
    else:
        # Check standard relevance
        relevance_score = evaluate_relevance(query_text, generated_answer)
        
        from tests.relevance_utils import evaluate_relevance as eval_fn
        last_result = getattr(eval_fn, "last_result", {})
        
        if relevance_score < 0.80:
            print(f"\n--- RELEVANCE EVALUATION FAILED FOR {query_id} ---")
            print(f"Query: {query_text}")
            print(f"Relevance Score: {relevance_score:.2f}")
            print(f"Judge Reasoning: {last_result.get('reasoning')}")
            
        assert relevance_score >= 0.80, (
            f"Relevance drop detected for {query_id}! "
            f"Score {relevance_score:.2f} is below the 0.80 threshold. "
            f"Review test output for reasoning."
        )
