"""
Capability-aware fallback strategies for weak models.
Provides progressive fallback mechanisms based on model capabilities.
"""

from typing import List, Tuple, Dict, Optional
from utils.llm_utils import get_model_capabilities, requires_detailed_prompts

class ModelFallbackStrategy:
    """Defines fallback strategies based on model capabilities."""
    
    def __init__(self, model: str):
        self.model = model
        self.capabilities = get_model_capabilities(model)
        self.is_weak_model = not self.capabilities.get("reasoning", True)
        
    def should_use_progressive_fallback(self) -> bool:
        """Determine if this model needs progressive fallback handling."""
        return (
            self.is_weak_model or 
            self.capabilities.get("context_length", 32000) < 64000 or
            self.capabilities.get("batch_size", 5) < 3
        )
    
    def get_max_context_per_call(self) -> int:
        """Get maximum context length for a single call."""
        base_context = self.capabilities.get("context_length", 32000)
        
        if self.is_weak_model:
            # Weak models get much smaller context to avoid confusion
            return min(4000, base_context // 4)
        else:
            # Strong models can handle more context
            return min(16000, base_context // 2)
    
    def get_max_pairs_per_batch(self) -> int:
        """Get maximum concept pairs per batch call."""
        if self.is_weak_model:
            return 1  # Force pair-by-pair for weak models
        else:
            return self.capabilities.get("batch_size", 5)
    
    def should_simplify_text(self) -> bool:
        """Whether to simplify/truncate text for this model."""
        return self.is_weak_model or self.capabilities.get("context_length", 32000) < 32000
    
    def get_text_simplification_strategy(self) -> str:
        """How to simplify text for weak models."""
        if self.is_weak_model:
            return "aggressive"  # Heavy simplification
        elif self.capabilities.get("context_length", 32000) < 64000:
            return "moderate"   # Some simplification
        else:
            return "minimal"    # Light simplification

def simplify_text_for_weak_models(text: str, strategy: str, target_concepts: List[str] = None) -> str:
    """Simplify text based on the specified strategy."""
    
    if strategy == "minimal":
        return text[:8000] if len(text) > 8000 else text
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if strategy == "aggressive":
        # Keep only sentences that mention target concepts
        if target_concepts:
            concept_words = set()
            for concept in target_concepts:
                concept_words.update(concept.lower().split())
            
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in concept_words):
                    relevant_sentences.append(sentence)
            
            # Limit to most relevant sentences
            simplified = '. '.join(relevant_sentences[:5]) + '.'
            return simplified if simplified.strip() else text[:1000]
        else:
            # Just take first few sentences
            return '. '.join(sentences[:3]) + '.'
    
    elif strategy == "moderate":
        # Keep sentences that are most informative
        important_keywords = ['causes', 'leads to', 'results in', 'affects', 'influences', 'because']
        scored_sentences = []
        
        for sentence in sentences:
            score = sum(1 for keyword in important_keywords if keyword in sentence.lower())
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        selected = [s[1] for s in scored_sentences[:8]]
        
        simplified = '. '.join(selected) + '.'
        return simplified if len(simplified) < 4000 else simplified[:4000]
    
    return text

def create_simplified_prompt_for_weak_model(
    concept1: str, 
    concept2: str, 
    text: str,
    use_structured_output: bool = False
) -> str:
    """Create a very simple, explicit prompt for weak models."""
    
    base_prompt = f"""You must analyze this text and determine if there is a causal relationship between two specific concepts.

TEXT TO ANALYZE:
"{text}"

CONCEPTS TO ANALYZE:
Concept A: {concept1}
Concept B: {concept2}

TASK:
Look for evidence that one concept directly causes or influences the other concept.

STEPS:
1. Find mentions of "{concept1}" in the text
2. Find mentions of "{concept2}" in the text  
3. Look for words like: causes, leads to, results in, affects, influences, because
4. Determine if Concept A causes Concept B, or Concept B causes Concept A, or neither

"""

    if use_structured_output:
        base_prompt += """REQUIRED JSON RESPONSE FORMAT:
{
  "edges": [
    {
      "pair_number": 1,
      "source": "concept that causes",
      "target": "concept that is affected", 
      "relationship": "positive",
      "confidence": 0.8,
      "has_relationship": true
    }
  ],
  "no_relationships": [
    {
      "pair_number": 1,
      "has_relationship": false,
      "reason": "no clear causal connection found"
    }
  ]
}

IMPORTANT: 
- Use "positive" if the source concept increases the target concept
- Use "negative" if the source concept decreases the target concept
- Put confidence between 0.0 and 1.0 (0.0 = no confidence, 1.0 = completely certain)
- If no relationship exists, use the "no_relationships" array instead

RESPOND WITH ONLY THE JSON:"""
    else:
        base_prompt += """RESPONSE FORMAT:
If you find a relationship, respond exactly like this:
"Pair 1: 'source concept' -> 'target concept' (positive, confidence: 0.8)"

If you find NO relationship, respond exactly like this:
"Pair 1: no relationship"

RULES:
- Use "positive" if first concept increases second concept
- Use "negative" if first concept decreases second concept  
- Put confidence between 0.0 and 1.0
- Be conservative - only report clear relationships

YOUR RESPONSE:"""

    return base_prompt.strip()

def get_fallback_chain_for_model(model: str) -> List[Dict]:
    """Get a progressive chain of fallback strategies for a model."""
    strategy = ModelFallbackStrategy(model)
    
    fallbacks = []
    
    if strategy.should_use_progressive_fallback():
        # Level 1: Try with reduced batch size
        fallbacks.append({
            "level": 1,
            "description": "Reduced batch size",
            "max_pairs": min(3, strategy.get_max_pairs_per_batch()),
            "simplify_text": False,
            "use_simple_prompts": False
        })
        
        # Level 2: Simplify text but keep batching
        fallbacks.append({
            "level": 2, 
            "description": "Simplified text",
            "max_pairs": min(2, strategy.get_max_pairs_per_batch()),
            "simplify_text": True,
            "use_simple_prompts": False
        })
        
        # Level 3: Pair-by-pair with simple prompts
        fallbacks.append({
            "level": 3,
            "description": "Pair-by-pair with simple prompts", 
            "max_pairs": 1,
            "simplify_text": True,
            "use_simple_prompts": True
        })
    else:
        # Strong models just need basic fallback
        fallbacks.append({
            "level": 1,
            "description": "Individual pairs",
            "max_pairs": 1,
            "simplify_text": False,
            "use_simple_prompts": False
        })
    
    return fallbacks

def should_trigger_fallback(
    model: str,
    response: str, 
    num_pairs: int,
    is_already_fallback: bool = False
) -> bool:
    """Determine if we should trigger fallback processing."""
    
    if is_already_fallback:
        return False  # Don't recurse indefinitely
    
    strategy = ModelFallbackStrategy(model)
    
    # Trigger fallback for weak models more aggressively
    if strategy.is_weak_model:
        return (
            not response.strip() or  # Empty response
            len(response.strip()) < 20 or  # Very short response
            num_pairs > 1  # Multiple pairs for weak model
        )
    else:
        # Standard fallback conditions for strong models
        return not response.strip() and num_pairs > 1

# Example usage and testing
if __name__ == "__main__":
    # Test fallback strategies for different models
    models = ["gpt-4o", "o3-mini", "gemini-pro"]
    
    for model in models:
        print(f"=== {model.upper()} FALLBACK STRATEGY ===")
        strategy = ModelFallbackStrategy(model)
        
        print(f"Is weak model: {strategy.is_weak_model}")
        print(f"Max context per call: {strategy.get_max_context_per_call()}")
        print(f"Max pairs per batch: {strategy.get_max_pairs_per_batch()}")
        print(f"Text simplification: {strategy.get_text_simplification_strategy()}")
        
        fallbacks = get_fallback_chain_for_model(model)
        print(f"Fallback levels: {len(fallbacks)}")
        for fallback in fallbacks:
            print(f"  Level {fallback['level']}: {fallback['description']}")
        print()