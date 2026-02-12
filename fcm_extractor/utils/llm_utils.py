def get_model_provider(model_name: str) -> str:
    if model_name.startswith("gemini"):
        return "google"
    elif model_name.startswith(("gpt", "o1", "o3", "chatgpt")):
        return "openai"
    elif model_name.startswith("deepseek"):
        return "deepseek"
    elif model_name.startswith(("claude", "anthropic")):
        return "anthropic"
    else:
        return "openai"


def is_reasoning_model(model_name: str) -> bool:
    return model_name.startswith(("o1", "o3"))

def uses_max_completion_tokens(model_name: str) -> bool:
    """Check if model uses max_completion_tokens instead of max_tokens"""
    return model_name.startswith((
        "o3",           # OpenAI o3 reasoning family
        "o1",           # OpenAI o1 reasoning family
        "gpt-4.1",      # New GPT-4.1 family (uses max_completion_tokens)
        "gpt-5"         # GPT-5 family also expects max_completion_tokens
    ))

def supports_temperature(model_name: str) -> bool:
    """Check if model supports temperature parameter"""
    # o3 models don't support temperature parameter
    return not model_name.startswith(("o3",))

def get_model_capabilities(model_name: str) -> dict:
    """Get model capabilities for prompt and processing strategy selection."""
    
    # Normalize model name for consistent matching
    model_lower = model_name.lower()
    
    # Define model capabilities based on model families
    if model_lower.startswith("gemini"):
        # Extract version info if available (e.g., gemini-1.5, gemini-2.0)
        version_info = model_lower.replace("gemini-", "").replace("gemini", "")
        
        # Set default Gemini capabilities (works for all versions)
        capabilities = {
            "reasoning": True,  # All Gemini models have good reasoning capabilities
            "structured_output": True,
            "json_mode": True,
            "supports_system_messages": True,
            "requires_detailed_prompts": False,
            "provider": "google"
        }
        
        # Version-specific optimizations
        is_future_version = False
        if version_info and version_info[0].isdigit():
            try:
                major_version = int(version_info.split('.')[0])
                is_future_version = major_version >= 3
            except (ValueError, IndexError):
                is_future_version = any(x in version_info for x in ["3.", "3-", "4.", "4-"])
        
        if is_future_version:
            # Gemini 3.x+ series (future versions) - assume even better capabilities
            capabilities.update({
                "max_tokens": 65536,  # Future Gemini models may have higher limits
                "context_window": 4000000,  # Assume larger context windows
            })
        elif any(x in version_info for x in ["2.", "2-"]):
            # Gemini 2.x series (2.0, 2.5, etc.)
            if "2.5" in version_info:
                # Gemini 2.5 has enhanced capabilities
                capabilities.update({
                    "max_tokens": 65536,  # Gemini 2.5 supports higher output tokens
                    "context_window": 2000000,  # 2M context window
                })
            else:
                # Other Gemini 2.x models
                capabilities.update({
                    "max_tokens": 32768,  # Gemini 2.x supports up to 32K output tokens
                    "context_window": 2000000,  # Gemini 2.x has up to 2M context window
                })
        elif "1.5" in version_info:
            # Gemini 1.5 series
            capabilities.update({
                "max_tokens": 32768,
                "context_window": 1000000,  # 1M context window
            })
        elif any(x in version_info for x in ["1.", "1-"]):
            # Gemini 1.0/1.x series
            capabilities.update({
                "max_tokens": 8192,
                "context_window": 30720,
            })
        else:
            # Unknown Gemini version - use reasonable modern defaults
            # Assume it's at least as good as Gemini 1.5
            capabilities.update({
                "max_tokens": 32768,
                "context_window": 1000000,
            })
        
        return capabilities
        
    elif model_lower.startswith(("o3", "o1")):
        # OpenAI reasoning models
        return {
            "reasoning": True,  # Reasoning models
            "structured_output": True,
            "json_mode": False,  # o1/o3 don't have explicit JSON mode
            "max_tokens": 100000 if model_lower.startswith("o3") else 32768,
            "context_window": 200000,
            "supports_system_messages": False,  # o1/o3 models don't use system messages
            "requires_detailed_prompts": False,
            "provider": "openai"
        }
    elif model_lower.startswith(("gpt-4", "gpt-5")):
        # GPT-4 and GPT-5 series
        max_tokens = 16384
        context_window = 128000
        
        # Special cases for specific models
        if "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower:
            max_tokens = 16384
            context_window = 128000
        elif "gpt-4" in model_lower and "32k" in model_lower:
            context_window = 32768
        elif "gpt-5" in model_lower:
            # GPT-5 series - assume higher capabilities
            max_tokens = 32768
            context_window = 200000
            
        return {
            "reasoning": True,
            "structured_output": True,
            "json_mode": True,
            "max_tokens": max_tokens,
            "context_window": context_window,
            "supports_system_messages": True,
            "requires_detailed_prompts": False,
            "provider": "openai"
        }
    elif model_lower.startswith(("gpt-3.5", "gpt-3")):
        # GPT-3.5 and GPT-3 series
        return {
            "reasoning": False,  # Less capable reasoning
            "structured_output": False,
            "json_mode": True,
            "max_tokens": 4096,
            "context_window": 16385 if "gpt-3.5" in model_lower else 4096,
            "supports_system_messages": True,
            "requires_detailed_prompts": True,
            "provider": "openai"
        }
    elif model_lower.startswith(("claude", "anthropic")):
        # Anthropic Claude models
        return {
            "reasoning": True,
            "structured_output": True,
            "json_mode": True,
            "max_tokens": 8192,
            "context_window": 200000,
            "supports_system_messages": True,
            "requires_detailed_prompts": False,
            "provider": "anthropic"
        }
    elif model_lower.startswith("deepseek"):
        # DeepSeek models
        return {
            "reasoning": True,
            "structured_output": True,
            "json_mode": True,
            "max_tokens": 8192,
            "context_window": 128000,
            "supports_system_messages": True,
            "requires_detailed_prompts": False,
            "provider": "deepseek"
        }
    else:
        # Default capabilities for unknown models - assume mid-tier OpenAI-like
        return {
            "reasoning": True,
            "structured_output": True,
            "json_mode": True,
            "max_tokens": 4096,
            "context_window": 8192,
            "supports_system_messages": True,
            "requires_detailed_prompts": False,
            "provider": "openai"
        }

def requires_detailed_prompts(model_name: str) -> bool:
    """Check if model requires detailed prompts for good performance."""
    capabilities = get_model_capabilities(model_name)
    return capabilities.get("requires_detailed_prompts", False) 