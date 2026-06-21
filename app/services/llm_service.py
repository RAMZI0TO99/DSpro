import os
from openai import OpenAI
import google.generativeai as genai
from app.core.config import LLMSettings

class LLMService:
    def __init__(self):
        self.local_client = None
        self.gemini_model = None
        self.openai_client = None
    
    def init_clients(self, llm_settings: LLMSettings):
        self.local_client = None
        self.gemini_model = None
        self.openai_client = None
        
        provider = llm_settings.provider.lower()
        
        if provider == "gemini":
            api_key = llm_settings.api_key.strip()
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(llm_settings.model)
            print(f"[BOOT] LLM Engine: Gemini Cloud ({llm_settings.model})")
        elif provider == "openai":
            api_key = llm_settings.api_key.strip()
            self.openai_client = OpenAI(api_key=api_key)
            print(f"[BOOT] LLM Engine: OpenAI Cloud ({llm_settings.model})")
        elif provider == "ollama":
            self.local_client = OpenAI(base_url=llm_settings.base_url.strip(), api_key="ollama")
            print(f"[BOOT] LLM Engine: Ollama Local ({llm_settings.model})")
        elif provider == "custom":
            api_key = llm_settings.api_key.strip()
            self.local_client = OpenAI(base_url=llm_settings.base_url.strip(), api_key=api_key)
            print(f"[BOOT] LLM Engine: Custom API ({llm_settings.model})")
        else:
            # Default to LM Studio (local)
            llm_settings.provider = "local"
            self.local_client = OpenAI(base_url=llm_settings.base_url, api_key="lm-studio")
            print(f"[BOOT] LLM Engine: LM Studio Local ({llm_settings.model})")

    def optimize_query(self, raw_query: str, provider: str, model_name: str) -> str:
        SYSTEM_PROMPT = """
        You are a highly precise Search Query Optimizer for a Tri-Modal Vector Database.
        Your ONLY job is to take the user's raw input and convert it into a perfect, English-only keyword search string.
        
        RULES:
        1. Extract the core semantic intent.
        2. Remove conversational filler (e.g., "find the part where", "show me").
        3. Convert questions into declarative statements of what is happening visually or being spoken.
        4. Output ONLY the raw optimized string. No quotes, no intro, no punctuation at the end.
        5. If the query is already a simple keyword search, just return it as is.
        """
        
        try:
            if provider == "gemini":
                try:
                    gemini_model_with_sys = genai.GenerativeModel(
                        model_name,
                        system_instruction=SYSTEM_PROMPT
                    )
                    res = gemini_model_with_sys.generate_content(
                        raw_query,
                        generation_config={"temperature": 0.1}
                    )
                except TypeError:
                    # Fallback for older SDK
                    gemini_model_with_sys = genai.GenerativeModel(model_name)
                    res = gemini_model_with_sys.generate_content(
                        f"{SYSTEM_PROMPT}\n\nUser Query:\n{raw_query}",
                        generation_config={"temperature": 0.1}
                    )
                return res.text.strip().replace('"', '')
                
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_query}
                ]
                client = self.openai_client if provider == "openai" else self.local_client
                res = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1
                )
                return res.choices[0].message.content.strip().replace('"', '')
                
        except Exception as e:
            print(f"Error during LLM query optimization: {e}")
            return raw_query

    def generate_chat_stream(self, system_instruction: str, chat_history: list, query: str, provider: str, model_name: str):
        try:
            if provider == "gemini":
                gemini_history = []
                for msg in chat_history:
                    role = "user" if msg["role"] == "user" else "model"
                    # Handle both dictionary and Pydantic model formats
                    content = msg["content"] if isinstance(msg, dict) else msg.content
                    gemini_history.append({"role": role, "parts": [content]})
                
                try:
                    gemini_model_with_sys = genai.GenerativeModel(
                        model_name, 
                        system_instruction=system_instruction
                    )
                    query_text = query
                except TypeError:
                    # Fallback for older SDKs
                    gemini_model_with_sys = genai.GenerativeModel(model_name)
                    query_text = f"{system_instruction}\n\nUser Question:\n{query}"
                
                chat = gemini_model_with_sys.start_chat(history=gemini_history)
                response_stream = chat.send_message(query_text, generation_config={"temperature": 0.7}, stream=True)
                
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                
            else:
                # OpenAI / LM Studio / Ollama Format
                messages = [{"role": "system", "content": system_instruction}]
                for msg in chat_history:
                    role_str = msg.role if hasattr(msg, "role") else msg.get("role", "user")
                    role = "user" if role_str == "user" else "assistant"
                    content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                    messages.append({"role": role, "content": content})
                messages.append({"role": "user", "content": query})

                llm_client = self.openai_client if provider == "openai" else self.local_client
                response_stream = llm_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                
                for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"\n\n[{provider.upper()} API Error] Details: {e}"

llm_service = LLMService()
