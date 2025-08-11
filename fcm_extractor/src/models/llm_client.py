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
        self.deepseek_client = None
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
    
    def _get_deepseek_client(self):
        if self.deepseek_client is None:
            import openai
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise EnvironmentError("DEEPSEEK_API_KEY environment variable not set.")
            self.deepseek_client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
        return self.deepseek_client
    
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
                
                # Handle Gemini response properly
                content = ""
                
                # Try to get text using the quick accessor first
                try:
                    if hasattr(response, 'text') and response.text:
                        content = response.text
                except ValueError as ve:
                    # This happens when response.text fails due to safety filtering
                    # Check the specific error and candidates
                    if "finish_reason" in str(ve) and hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = candidate.finish_reason
                            if finish_reason == 2:  # SAFETY
                                print(f"Warning: Content blocked by Gemini safety filters (finish_reason=2)")
                                print("This often happens with long or complex prompts.")
                                
                                # Suggest fallback options
                                print("Recommended solutions:")
                                print("  1. Use a different Gemini model (e.g., gemini-1.5-flash)")
                                print("  2. Switch to OpenAI model (e.g., gpt-4o, gpt-4o-mini)")
                                print("  3. Simplify/shorten the prompt")
                                print("  4. Remove potentially sensitive content")
                                
                                return "", 0.0
                            elif finish_reason == 3:  # RECITATION  
                                print(f"Warning: Content blocked for recitation (finish_reason=3)")
                                return "", 0.0
                            elif finish_reason == 4:  # OTHER
                                print(f"Warning: Content blocked for other reasons (finish_reason=4)")
                                return "", 0.0
                            else:
                                print(f"Warning: Unexpected finish_reason: {finish_reason}")
                        
                        # Try to extract any partial content if available
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        content += part.text
                    else:
                        # Re-raise the original error if we can't handle it
                        raise ve
                
                # Fallback: try to extract from candidates structure
                if not content and hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    content += part.text
                
                if not content:
                    print("Warning: Gemini returned empty response")
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            print(f"Finish reason: {candidate.finish_reason}")
                        if hasattr(candidate, 'safety_ratings'):
                            print(f"Safety ratings: {candidate.safety_ratings}")
                
                confidence_match = re.search(r'confidence:\s*([0-1]\.\d+)', content)
                confidence = float(confidence_match.group(1)) if confidence_match else 1.0
                
                return content, confidence
            except Exception as e:
                print("--- ERROR DURING GEMINI API CALL ---")
                print(f"Model: {model}")
                print(f"An exception of type {type(e).__name__} occurred: {e}")
                print(f"Error details: {str(e)}")
                
                # Check for specific Gemini error types
                error_str = str(e).lower()
                if "model not found" in error_str or "not found" in error_str:
                    print(f"ERROR: Invalid Gemini model name '{model}'")
                    print("Valid Gemini model formats include:")
                    print("  - gemini-1.0-pro, gemini-1.5-pro, gemini-1.5-flash") 
                    print("  - gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp")
                    print("  - Check Google AI Studio for latest available models")
                elif "safety" in error_str:
                    print("ERROR: Content was blocked by Gemini safety filters")
                elif "quota" in error_str or "rate limit" in error_str:
                    print("ERROR: API quota exceeded or rate limited")
                elif "api key" in error_str or "authentication" in error_str:
                    print("ERROR: Invalid or missing Google API key (GOOGLE_API_KEY environment variable)")
                
                # Log the prompt that caused the error for debugging
                print("--- PROMPT THAT CAUSED ERROR ---")
                for msg in messages:
                    print(f"ROLE: {msg['role']}")
                    print(f"CONTENT:\n{msg['content'][:500]}{'...' if len(msg['content']) > 500 else ''}\n")
                print("---------------------------------")
                return "", 0.0
        
        elif provider == 'deepseek':
            client = self._get_deepseek_client()
            
            # DeepSeek uses OpenAI-compatible API
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            try:
                response = client.chat.completions.create(**api_params)
                content = response.choices[0].message.content
                confidence = 1.0
                return content, confidence
                
            except Exception as e:
                print("--- ERROR DURING DEEPSEEK API CALL ---")
                print(f"Model: {model}")
                print(f"An exception of type {type(e).__name__} occurred: {e}")
                print(f"Error details: {str(e)}")
                
                # Check for specific DeepSeek error types
                error_str = str(e).lower()
                if "model not found" in error_str or "not found" in error_str:
                    print(f"ERROR: Invalid DeepSeek model name '{model}'")
                    print("Valid DeepSeek model formats include:")
                    print("  - deepseek-chat, deepseek-coder")
                    print("  - Check DeepSeek API documentation for latest available models")
                elif "quota" in error_str or "rate limit" in error_str:
                    print("ERROR: API quota exceeded or rate limited")
                elif "api key" in error_str or "authentication" in error_str:
                    print("ERROR: Invalid or missing DeepSeek API key (DEEPSEEK_API_KEY environment variable)")
                
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