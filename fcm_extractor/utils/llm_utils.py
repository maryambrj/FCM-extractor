def get_model_provider(model_name: str) -> str:
    if model_name.startswith("gemini"):
        return "google"
    elif model_name.startswith(("gpt", "o1", "o3", "chatgpt")):
        return "openai"
    else:
        return "openai"


def is_reasoning_model(model_name: str) -> bool:
    return model_name.startswith(("o1", "o3"))

def uses_max_completion_tokens(model_name: str) -> bool:
    """Check if model uses max_completion_tokens instead of max_tokens"""
    return model_name.startswith(("o3",)) 