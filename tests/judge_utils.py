import json
import re
from typing import List
from app.services.llm_service import llm_service
from app.core.config import load_settings

# Initialize the LLM config globally for the judge
llm_settings = load_settings()
llm_service.init_clients(llm_settings)

def get_llm_response(system_instruction: str, query: str) -> str:
    provider = llm_settings.provider.lower()
    model_name = llm_settings.model
    generator = llm_service.generate_chat_stream(system_instruction, [], query, provider, model_name)
    response_text = ""
    for chunk in generator:
        response_text += chunk
    return response_text

def evaluate_faithfulness(question: str, context_chunks: List[str], generated_answer: str) -> float:
    """
    Prompts the Judge LLM to act as an impartial evaluator.
    Extracts factual claims from `generated_answer` and verifies if each claim is strictly entailed by `context_chunks`.
    Returns a float score between 0.0 (complete hallucination) and 1.0 (100% strictly grounded).
    """
    
    system_instruction = """
You are an impartial Judge evaluating the Faithfulness (Groundedness) of an AI assistant's answer.
Your task is to determine if the generated answer contains any hallucinated claims that are NOT supported by the provided context.

Follow these steps carefully:
1. Extract all factual atomic claims made in the generated answer.
2. For each claim, check if it is strictly entailed (supported) by the provided context chunks.
3. If the context does not support the claim, mark it as a hallucination.

You MUST output your evaluation in the following strict JSON format without any extra text or markdown code blocks:

{
    "claims": [
        {
            "claim": "extracted claim text",
            "is_supported": true_or_false,
            "reason": "explanation of why it is supported or not based on context"
        }
    ],
    "faithfulness_score": 0.0_to_1.0_float
}

CRITICAL: DO NOT include trailing commas in your JSON. The last item in a list or object must not have a comma.

Note: The faithfulness_score is the ratio of supported claims to total claims. For example, if there are 4 claims and 3 are supported, the score is 0.75. If there are 0 claims, return 1.0.
"""

    context_str = "\n\n".join([f"--- Context Chunk {i+1} ---\n{c}" for i, c in enumerate(context_chunks)])
    
    query = f"""
Question asked: {question}

Context provided to the assistant:
{context_str}

Generated Answer:
{generated_answer}

Evaluate the faithfulness of the generated answer. Output JSON only.
"""

    response_text = get_llm_response(system_instruction, query)
    
    # Try parsing JSON safely
    try:
        # Strip markdown json blocks if any
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        # Clean trailing commas (often hallucinated by lite models)
        response_text = re.sub(r',\s*}', '}', response_text)
        response_text = re.sub(r',\s*]', ']', response_text)
            
        result = json.loads(response_text)
        
        # Save diagnostics for the test to read if it fails
        evaluate_faithfulness.last_result = result
        
        score = float(result.get("faithfulness_score", 0.0))
        return score
        
    except Exception as e:
        print(f"Failed to parse LLM JSON response: {e}")
        print(f"Raw Response: {response_text}")
        evaluate_faithfulness.last_result = None
        return 0.0
