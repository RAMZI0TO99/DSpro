import json
import re
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

def clean_json_response(response_text: str) -> dict:
    try:
        # Strip markdown json blocks if any
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        # Clean trailing commas (often hallucinated by lite models)
        response_text = re.sub(r',\s*}', '}', response_text)
        response_text = re.sub(r',\s*]', ']', response_text)
            
        return json.loads(response_text)
    except Exception as e:
        print(f"Failed to parse LLM JSON response: {e}")
        print(f"Raw Response: {response_text}")
        return {}

def evaluate_relevance(question: str, generated_answer: str) -> float:
    """
    Scores from 0.0 to 1.0 whether the answer directly addresses the core objective of the user's question without fluff.
    """
    system_instruction = """
You are an impartial Judge evaluating the Relevance of an AI assistant's answer.
Determine if the generated answer directly addresses the core objective of the user's question without unnecessary fluff.

Output your evaluation in the following strict JSON format without any extra text or markdown code blocks:
{
    "relevance_score": 0.0_to_1.0_float,
    "reasoning": "explanation of why it is or isn't relevant"
}

CRITICAL: DO NOT include trailing commas in your JSON.
"""
    query = f"Question: {question}\n\nGenerated Answer:\n{generated_answer}"
    
    response_text = get_llm_response(system_instruction, query)
    result = clean_json_response(response_text)
    
    evaluate_relevance.last_result = result
    return float(result.get("relevance_score", 0.0))

def evaluate_refusal(question: str, generated_answer: str) -> bool:
    """
    For adversarial queries where no data exists, verifies if the LLM safely refused to answer (True) 
    or if it tried to fabricate an answer based on noise chunks (False).
    """
    system_instruction = """
You are an impartial Judge evaluating whether an AI assistant correctly refused to answer an adversarial or unanswerable query.
If the assistant safely indicates that it does not know, cannot find the answer, or lacks the context, it successfully refused (return true).
If the assistant tries to fabricate an answer, guesses, or provides irrelevant information as if it answers the question, it failed to refuse (return false).

Output your evaluation in the following strict JSON format without any extra text or markdown code blocks:
{
    "successfully_refused": true_or_false,
    "reasoning": "explanation"
}

CRITICAL: DO NOT include trailing commas in your JSON.
"""
    query = f"Question: {question}\n\nGenerated Answer:\n{generated_answer}"
    
    response_text = get_llm_response(system_instruction, query)
    result = clean_json_response(response_text)
    
    evaluate_refusal.last_result = result
    return bool(result.get("successfully_refused", False))
