import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import re
import tiktoken
import sys
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.llm_utils import get_model_provider, is_reasoning_model

if "LANGCHAIN_API_KEY" not in os.environ:
    print("Warning: LANGCHAIN_API_KEY not set. Set it to trace runs in LangSmith.")

langchain_tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").lower()
if langchain_tracing not in ["true", "1"]:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    print("Info: LANGCHAIN_TRACING_V2 was not set to 'true'. It has been set automatically.")

if "LANGCHAIN_PROJECT" not in os.environ:
    os.environ["LANGCHAIN_PROJECT"] = "FCM-Extraction"
    print("Info: LANGCHAIN_PROJECT not set. Defaulting to 'FCM-Extraction'.")


class UnifiedLLMClient:
    
    def __init__(self):
        self.openai_client = None
        self.google_client = None
        self._gpt5_warning_shown = False
        
    def _get_openai_client(self):
        if self.openai_client is None:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
            self.openai_client = ChatOpenAI(api_key=api_key)
        return self.openai_client
    
    def _get_google_client(self, model: str = "gemini-pro"):
        if self.google_client is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")
            self.google_client = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key
            )
        return self.google_client
    
    def chat_completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 2000, **kwargs) -> Tuple[str, float]:
        
        provider = get_model_provider(model)
        
        # Convert messages to LangChain format
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        if provider == 'openai':
            client = self._get_openai_client()
            client.model_name = model
            if not is_reasoning_model(model):
                if model.startswith('gpt-5') and temperature != 1.0:
                    if not self._gpt5_warning_shown:
                        print(f"Warning: {model} only supports temperature=1.0, adjusting from {temperature}")
                        self._gpt5_warning_shown = True
                    client.temperature = 1.0
                else:
                    client.temperature = temperature
            client.max_tokens = max_tokens
            
            response = client.invoke(langchain_messages)
            content = response.content
            
            confidence = 1.0
            
            input_tokens = self._count_tokens(messages)
            output_tokens = self._count_tokens(content)
            
            response.usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            
            return content, confidence
        
        elif provider == 'google':
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")
                
                client = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                response = client.invoke(langchain_messages)
                content = response.content
                
                confidence_match = re.search(r'confidence:\s*([0-1]\.\d+)', content)
                confidence = float(confidence_match.group(1)) if confidence_match else 1.0
                
                input_tokens = self._count_tokens(messages)
                output_tokens = self._count_tokens(content)
                
                response.usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                
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
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict]) -> str:

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
        
        return "\n\n".join(prompt_parts)


llm_client = UnifiedLLMClient() 