"""
Unified prompt inference system that automatically selects appropriate prompts
based on model capabilities without requiring manual specification.
"""

from typing import Dict, Any, Optional
from config.model_prompts import get_prompt_for_task, get_system_message_for_task, should_use_json_output
from utils.llm_utils import get_model_capabilities

class PromptInferenceError(Exception):
    """Raised when prompt inference fails."""
    pass

def infer_prompt_and_system_message(
    task: str, 
    model: str, 
    **format_kwargs
) -> Dict[str, Any]:
    """
    Automatically infer the best prompt and system message for a task and model.
    
    Args:
        task: The task to perform (e.g., "concept_extraction", "edge_inference_batch")
        model: The model name
        **format_kwargs: Arguments to format the prompt template
        
    Returns:
        Dict containing:
        - prompt: The formatted prompt string
        - system_message: The appropriate system message
        - use_json: Whether JSON output is expected
        - model_class: The determined model class
        
    Raises:
        PromptInferenceError: If prompt inference fails
    """
    try:
        # Get model capabilities
        capabilities = get_model_capabilities(model)
        
        # Determine if JSON should be used
        use_json = should_use_json_output(model, task)
        
        # Get appropriate prompt and system message
        prompt = get_prompt_for_task(task, model, use_json=use_json, **format_kwargs)
        system_message = get_system_message_for_task(task, model)
        
        # Determine model class for logging
        model_class = "high_reasoning" if capabilities.get("reasoning", True) else "low_reasoning"
        
        return {
            "prompt": prompt,
            "system_message": system_message,
            "use_json": use_json,
            "model_class": model_class,
            "capabilities": capabilities
        }
        
    except Exception as e:
        raise PromptInferenceError(f"Failed to infer prompt for task '{task}' with model '{model}': {e}")

def infer_messages_for_task(
    task: str,
    model: str,
    **format_kwargs
) -> list:
    """
    Automatically create complete message list for a task and model.
    
    Returns:
        List of message dicts ready for LLM API call
    """
    inference_result = infer_prompt_and_system_message(task, model, **format_kwargs)
    
    return [
        {"role": "system", "content": inference_result["system_message"]},
        {"role": "user", "content": inference_result["prompt"]}
    ]

def get_task_configuration(task: str, model: str) -> Dict[str, Any]:
    """
    Get complete configuration for a task-model combination.
    
    Returns:
        Dict with model capabilities, expected output format, etc.
    """
    capabilities = get_model_capabilities(model)
    use_json = should_use_json_output(model, task)
    model_class = "high_reasoning" if capabilities.get("reasoning", True) else "low_reasoning"
    
    return {
        "model": model,
        "model_class": model_class,
        "capabilities": capabilities,
        "use_json": use_json,
        "batch_size": capabilities.get("batch_size", 5),
        "context_length": capabilities.get("context_length", 32000),
        "reasoning": capabilities.get("reasoning", True),
        "provider": capabilities.get("provider", "unknown")
    }

# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON TASKS
# =============================================================================

def get_concept_extraction_setup(model: str, text: str) -> Dict[str, Any]:
    """Get complete setup for concept extraction."""
    return infer_prompt_and_system_message("concept_extraction", model, text=text)

def get_edge_inference_setup(model: str, text: str, pairs: str) -> Dict[str, Any]:
    """Get complete setup for edge inference."""
    return infer_prompt_and_system_message("edge_inference_batch", model, text=text, pairs=pairs)

def get_single_edge_setup(model: str, text: str, concept1: str, concept2: str) -> Dict[str, Any]:
    """Get complete setup for single edge query."""
    return infer_prompt_and_system_message("edge_inference_single", model, 
                                          text=text, concept1=concept1, concept2=concept2)

def get_cluster_edge_setup(model: str, text: str, pairs: str, cluster_type: str = "inter") -> Dict[str, Any]:
    """Get complete setup for cluster edge inference."""
    task = f"{cluster_type}_cluster_edges"
    return infer_prompt_and_system_message(task, model, text=text, pairs=pairs)

# =============================================================================
# PROMPT DEBUGGING AND INSPECTION
# =============================================================================

def debug_prompt_selection(task: str, model: str, **format_kwargs) -> Dict[str, Any]:
    """
    Debug prompt selection process for a given task and model.
    
    Returns detailed information about the selection process.
    """
    try:
        config = get_task_configuration(task, model)
        inference_result = infer_prompt_and_system_message(task, model, **format_kwargs)
        
        return {
            "task": task,
            "model": model,
            "model_class": config["model_class"],
            "use_json": config["use_json"],
            "capabilities": config["capabilities"],
            "prompt_length": len(inference_result["prompt"]),
            "system_message_length": len(inference_result["system_message"]),
            "prompt_preview": inference_result["prompt"][:200] + "..." if len(inference_result["prompt"]) > 200 else inference_result["prompt"],
            "format_kwargs": format_kwargs,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "task": task,
            "model": model,
            "success": False,
            "error": str(e),
            "format_kwargs": format_kwargs
        }

def compare_prompts_across_models(task: str, models: list, **format_kwargs) -> Dict[str, Dict[str, Any]]:
    """Compare how prompts differ across multiple models for the same task."""
    results = {}
    
    for model in models:
        results[model] = debug_prompt_selection(task, model, **format_kwargs)
    
    return results

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_prompt_inference():
    """Validate that prompt inference works for all common task-model combinations."""
    test_cases = [
        ("concept_extraction", "gpt-4o", {"text": "Test text"}),
        ("concept_extraction", "o3-mini", {"text": "Test text"}),
        ("edge_inference_batch", "gpt-4o", {"text": "Test text", "pairs": "Test pairs"}),
        ("edge_inference_batch", "o3-mini", {"text": "Test text", "pairs": "Test pairs"}),
        ("edge_inference_single", "gpt-4o", {"text": "Test text", "concept1": "A", "concept2": "B"}),
        ("inter_cluster_edges", "o3-mini", {"text": "Test text", "pairs": "Test pairs"}),
        ("intra_cluster_edges", "o3-mini", {"text": "Test text", "pairs": "Test pairs", "cluster_name": "test"})
    ]
    
    errors = []
    successes = 0
    
    for task, model, kwargs in test_cases:
        try:
            result = infer_prompt_and_system_message(task, model, **kwargs)
            if not result["prompt"] or not result["system_message"]:
                errors.append(f"{task}/{model}: Empty prompt or system message")
            else:
                successes += 1
        except Exception as e:
            errors.append(f"{task}/{model}: {e}")
    
    if errors:
        raise ValueError(f"Prompt inference validation failed ({successes}/{len(test_cases)} passed):\\n" + "\\n".join(errors))
    
    return True

# Example usage
if __name__ == "__main__":
    print("=== PROMPT INFERENCE SYSTEM TEST ===\\n")
    
    # Test basic inference
    test_models = ["gpt-4o", "o3-mini"]
    
    for model in test_models:
        print(f"Testing model: {model}")
        
        # Test concept extraction
        setup = get_concept_extraction_setup(model, "Social isolation causes depression.")
        print(f"  Concept extraction: {setup['model_class']}, JSON: {setup['use_json']}")
        
        # Test edge inference
        setup = get_edge_inference_setup(model, "Test text", "Pair 1: A and B")
        print(f"  Edge inference: {setup['model_class']}, JSON: {setup['use_json']}")
        print()
    
    # Test prompt comparison
    print("=== PROMPT COMPARISON ===")
    comparison = compare_prompts_across_models(
        "edge_inference_batch", 
        ["gpt-4o", "o3-mini"],
        text="Stress causes anxiety",
        pairs="Pair 1: stress and anxiety"
    )
    
    for model, info in comparison.items():
        print(f"{model}: {info['model_class']}, {info['prompt_length']} chars, JSON: {info.get('use_json', False)}")
    
    # Validate system
    print("\\n=== VALIDATION ===")
    try:
        validate_prompt_inference()
        print("✅ All prompt inference tests passed!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")