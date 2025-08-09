import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import re
import tiktoken
import sys
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.llm_utils import get_model_provider, is_reasoning_model, uses_max_completion_tokens, supports_temperature

# if "LANGCHAIN_API_KEY" not in os.environ:
#     print("Warning: LANGCHAIN_API_KEY not set. Set it to trace runs in LangSmith.")
# 
# langchain_tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").lower()
# if langchain_tracing not in ["true", "1"]:
#     os.environ["LANGCHAIN_TRACING_V2"] = "true"
#     print("Info: LANGCHAIN_TRACING_V2 was not set to 'true'. It has been set automatically.")
# 
# if "LANGCHAIN_PROJECT" not in os.environ:
#     os.environ["LANGCHAIN_PROJECT"] = "FCM-Extraction"
#     print("Info: LANGCHAIN_PROJECT not set. Defaulting to 'FCM-Extraction'.")


class UnifiedLLMClient:
    
    def __init__(self):
        self.openai_client = None
        self.google_client = None
        self._gpt5_warning_shown = False
        
    def _get_openai_client(self):
        if self.openai_client is None:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
            self.openai_client = openai.OpenAI(api_key=api_key)
        return self.openai_client
    
    def _get_google_client(self, model: str = "gemini-pro"):
        if self.google_client is None:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            self.google_client = genai
        return self.google_client
    
    def chat_completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 2000, **kwargs) -> Tuple[str, float]:
        
        provider = get_model_provider(model)
        
        if provider == 'openai':
            client = self._get_openai_client()
            
            # Handle temperature for different models
            actual_temperature = temperature
            include_temperature = supports_temperature(model)
            
            if not is_reasoning_model(model):
                if model.startswith('gpt-5') and temperature != 1.0:
                    if not self._gpt5_warning_shown:
                        print(f"Warning: {model} only supports temperature=1.0, adjusting from {temperature}")
                        self._gpt5_warning_shown = True
                    actual_temperature = 1.0
            
            # Build API call parameters
            api_params = {
                "model": model,
                "messages": messages
            }
            
            # Add temperature if supported
            if include_temperature:
                api_params["temperature"] = actual_temperature
            
            # Add max_tokens or max_completion_tokens based on model
            if uses_max_completion_tokens(model):
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens
            
            response = client.chat.completions.create(**api_params)
            
            content = response.choices[0].message.content
            confidence = 1.0
            
            return content, confidence
        
        elif provider == 'google':
            try:
                genai = self._get_google_client()
                
                # Convert messages to Google format
                prompt_parts = []
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "system":
                        prompt_parts.append(f"Instructions: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                
                prompt = "\n\n".join(prompt_parts)
                
                model_instance = genai.GenerativeModel(
                    model_name=model,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                response = model_instance.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                
                content = response.text
                
                confidence_match = re.search(r'confidence:\s*([0-1]\.\d+)', content)
                confidence = float(confidence_match.group(1)) if confidence_match else 1.0
                
                return content, confidence
            except Exception as e:
                print("--- ERROR DURING GEMINI API CALL ---")
                print(f"An exception of type {type(e).__name__} occurred: {e}")
                # Log the prompt that caused the error for debugging
                print("--- PROMPT THAT CAUSED ERROR ---")
                for msg in messages:
                    print(f"ROLE: {msg['role']}")
                    print(f"CONTENT:\n{msg['content'][:500]}{'...' if len(msg['content']) > 500 else ''}\n")
                print("---------------------------------")
                return "", 0.0
        
        else:
            raise ValueError(f"Unsupported model provider for {model}")
    
    def _count_tokens(self, content: any) -> int:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            if isinstance(content, str):
                return len(encoding.encode(content))
            elif isinstance(content, list):
                return sum(len(encoding.encode(msg["content"])) for msg in content)
            else:
                return 0
        except Exception:
            if isinstance(content, str):
                return len(content) // 4  # Rough estimate
            elif isinstance(content, list):
                return sum(len(msg["content"]) // 4 for msg in content)
            else:
                return 0
    
    def _openai_completion(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int]) -> str:
        content, _ = self.chat_completion(model, messages, temperature, max_tokens or 2000)
        return content
    
    def _google_completion(self, model: str, messages: List[Dict], temperature: float, max_tokens: Optional[int]) -> str:
        content, _ = self.chat_completion(model, messages, temperature, max_tokens or 2000)
        return content


llm_client = UnifiedLLMClient() 