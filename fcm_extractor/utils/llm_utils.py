"""
LLM utility functions for model provider detection and other LLM-related helpers.
"""

def get_model_provider(model_name: str) -> str:
    """
    Determine the provider based on model name.
    
    Args:
        model_name: The name of the model (e.g., "gpt-4", "gemini-2.0-flash-exp")
        
    Returns:
        Provider name ("openai" or "google")
    """
    if model_name.startswith("gemini"):
        return "google"
    elif model_name.startswith(("gpt", "o1", "o3", "chatgpt")):
        return "openai"
    else:
        return "openai"  # default fallback


def is_reasoning_model(model_name: str) -> bool:
    """
    Determine if the model is a reasoning model that doesn't support temperature parameter.
    
    Args:
        model_name: The name of the model (e.g., "o1-preview", "o3-mini")
        
    Returns:
        True if it's a reasoning model (o1/o3 series), False otherwise
    """
    return model_name.startswith(("o1", "o3")) 