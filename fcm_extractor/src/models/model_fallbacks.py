"""
Model fallback strategies for different tasks when primary models fail.
"""
import os
from typing import List, Dict
from config.constants import CONCEPT_EXTRACTION_MODEL


class ModelFallbackStrategy:
    """Provides fallback model sequences for different tasks."""
    
    def __init__(self, primary_model: str):
        self.primary_model = primary_model
    
    def get_fallback_models_for_task(self, task: str) -> List[str]:
        """Get ordered list of fallback models for a specific task."""
        
        if task == "concept_extraction":
            return self._get_concept_extraction_fallbacks()
        elif task == "edge_inference":
            return self._get_edge_inference_fallbacks()
        else:
            return self._get_general_fallbacks()
    
    def _get_concept_extraction_fallbacks(self) -> List[str]:
        """Fallback models optimized for concept extraction tasks."""
        
        # Start with environment-specified fallback
        env_fallback = os.getenv("FCM_ALT_MODEL")
        fallbacks = []
        
        # Add environment fallback if different from primary
        if env_fallback and env_fallback != self.primary_model:
            fallbacks.append(env_fallback)
        
        # Gemini fallback sequence
        if self.primary_model.startswith("gemini-2.5"):
            candidates = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        elif self.primary_model.startswith("gemini-2.0"):
            candidates = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-flash"]
        elif self.primary_model.startswith("gemini-1.5"):
            candidates = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        # OpenAI fallback sequence
        elif self.primary_model.startswith("gpt-4o"):
            candidates = ["gpt-4o-mini", "gpt-4-turbo", "gemini-2.0-flash"]
        elif self.primary_model.startswith("gpt-4"):
            candidates = ["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash"]
        elif self.primary_model.startswith("gpt-3.5"):
            candidates = ["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash"]
        else:
            # Default fallback for unknown models
            candidates = ["gpt-4o-mini", "gemini-2.0-flash", "gemini-1.5-flash"]
        
        # Add candidates that aren't already in fallbacks and aren't the primary
        for candidate in candidates:
            if candidate != self.primary_model and candidate not in fallbacks:
                fallbacks.append(candidate)
        
        return fallbacks[:3]  # Limit to 3 fallbacks to prevent excessive retries
    
    def _get_edge_inference_fallbacks(self) -> List[str]:
        """Fallback models optimized for edge inference tasks."""
        # Similar logic but optimized for reasoning tasks
        return self._get_concept_extraction_fallbacks()  # Same strategy for now
    
    def _get_general_fallbacks(self) -> List[str]:
        """General fallback sequence for unknown tasks."""
        return self._get_concept_extraction_fallbacks()


def apply_concept_extraction_fallback(model: str, messages: List[Dict], temperature: float, max_tokens: int = 2000) -> tuple[str, float]:
    """
    Apply fallback strategy specifically for concept extraction.
    Returns (text, confidence) or raises RuntimeError if all fallbacks fail.
    """
    from src.models.llm_client import llm_client
    
    # Try primary model
    text, conf = llm_client.chat_completion(model, messages, temperature, max_tokens)
    
    # Fail fast on empty extraction to prevent silent pipeline "success"
    if not text.strip():
        print(f"    ✗ {model} returned empty extraction text")
    else:
        return text, conf
    
    print(f"    Warning: {model} returned empty response, trying fallbacks...")
    
    # Get fallback models
    fallback_strategy = ModelFallbackStrategy(model)
    fallback_models = fallback_strategy.get_fallback_models_for_task("concept_extraction")
    
    # Try each fallback
    for fallback_model in fallback_models:
        print(f"    [fallback] Retrying concept extraction with {fallback_model}...")
        try:
            text, conf = llm_client.chat_completion(fallback_model, messages, temperature, max_tokens)
            if text.strip():
                print(f"    ✓ Fallback {fallback_model} succeeded")
                return text, conf
            else:
                print(f"    ✗ Fallback {fallback_model} also returned empty")
        except Exception as e:
            print(f"    ✗ Fallback {fallback_model} failed with error: {e}")
    
    # If we get here, all models failed
    raise RuntimeError(f"Concept extraction returned empty output across all fallbacks. Primary: {model}, Fallbacks: {fallback_models}")