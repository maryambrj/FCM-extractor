"""
Model class-based prompt templates for FCM extraction.
Provides different prompt styles based on model reasoning capabilities.
"""

from typing import Dict, Any
from utils.llm_utils import get_model_capabilities

# =============================================================================
# UNIFIED PROMPT VARIANTS SYSTEM
# =============================================================================

# Task-based prompts with automatic model class selection
PROMPT_VARIANTS = {
    "concept_extraction": {
        "high_reasoning": """Extract the most important concepts from this text. Focus on key entities, processes, and factors that could be part of a causal relationship network. Be selective and avoid minor details.

Return only a comma-separated list.""",
        
        "low_reasoning": """TASK: Extract important concepts from the text below.

INSTRUCTIONS:
1. Read the text carefully
2. Identify key concepts, entities, processes, and factors
3. Focus on concepts that could be causes or effects
4. Avoid minor details or very specific items
5. List each concept on a new line with a comma

IMPORTANT: Return ONLY a comma-separated list with no explanations.

TEXT:
{text}

CONCEPTS:"""
    },
    
    "edge_inference_batch": {
        "high_reasoning": """Analyze causal relationships between concept pairs using structured format.

Text: {text}

Pairs: {pairs}

For each pair, determine:
- i: index (0-based, matching input order)
- dir: "A->B" (A causes B), "B->A" (B causes A), or "none" (no relationship)
- sign: "positive" (increases/improves) or "negative" (decreases/worsens) - only if dir ≠ "none"
- conf: confidence 0.0-1.0 based on text evidence

IMPORTANT: Return exactly {pair_count} items in the same order as input pairs.

Respond with JSON only:
{{
  "edges": [
    {{"i": 0, "dir": "A->B", "sign": "positive", "conf": 0.8}},
    {{"i": 1, "dir": "none", "sign": null, "conf": 0.2}}
  ]
}}""",
        
        "low_reasoning": """TASK: Find causal relationships between concept pairs.

STEPS:
1. Read text carefully
2. For each pair, look for causal words: causes, leads to, results in, affects, influences
3. Choose direction and sign
4. Rate confidence 0.0-1.0

TEXT:
{text}

PAIRS:
{pairs}

FOR EACH PAIR, CHOOSE:
- i: index (0-based, matching input order)
- dir: "A->B" (A causes B), "B->A" (B causes A), or "none" (no relationship)
- sign: "positive" (A increases B) or "negative" (A decreases B) - ONLY if dir ≠ "none"
- conf: confidence 0.0-1.0

IMPORTANT: 
- Use EXACT concept names as given. Do NOT rewrite them.
- Return exactly {pair_count} items in the same order as input pairs.
- Index 'i' must start at 0 and be sequential.

RESPOND WITH JSON ONLY:
{{
  "edges": [
    {{"i": 0, "dir": "A->B", "sign": "positive", "conf": 0.8}},
    {{"i": 1, "dir": "none", "sign": null, "conf": 0.2}}
  ]
}}""",
        
        "low_reasoning_json": """TASK: Find causal relationships. Return JSON only.

TEXT: {text}
PAIRS: {pairs}

FOR EACH PAIR:
- i: index (0-based, matching input order)
- dir: "A->B", "B->A", or "none"
- sign: "positive" or "negative" (ONLY if dir ≠ "none")
- conf: confidence 0.0-1.0

IMPORTANT: Return exactly {pair_count} items in the same order as input pairs.

JSON RESPONSE:
{{
  "edges": [
    {{"i": 0, "dir": "A->B", "sign": "positive", "conf": 0.8}},
    {{"i": 1, "dir": "none", "sign": null, "conf": 0.2}}
  ]
}}""",
        
        "basic": """Analyze causal relationships between concept pairs.

TEXT: {text}

PAIRS: {pairs}

For each pair, determine:
- i: index (0-based, matching input order)
- dir: "A->B" (A causes B), "B->A" (B causes A), or "none" (no relationship)
- sign: "positive" (increases/improves) or "negative" (decreases/worsens) - only if dir ≠ "none"
- conf: confidence 0.0-1.0

IMPORTANT: Return exactly {pair_count} items in the same order as input pairs.

RESPOND WITH JSON ONLY:
{{
  "edges": [
    {{"i": 0, "dir": "A->B", "sign": "positive", "conf": 0.8}},
    {{"i": 1, "dir": "none", "sign": null, "conf": 0.2}}
  ]
}}"""
    },
    
    "edge_inference_single": {
        "high_reasoning": """Does '{concept1}' causally influence '{concept2}' in this text?

Text: {text}

Respond: "concept1 -> concept2 (positive/negative, confidence: 0.XX)" or "no relationship""",
        
        "low_reasoning": """TASK: Determine if one concept causes another.

CONCEPT A: {concept1}
CONCEPT B: {concept2}

TEXT TO ANALYZE:
{text}

STEP-BY-STEP ANALYSIS:
1. Find mentions of "{concept1}" in the text
2. Find mentions of "{concept2}" in the text
3. Look for causal words between them: causes, leads to, results in, affects, influences, because, due to
4. Determine direction: Does A cause B, or does B cause A, or neither?
5. If causal relationship exists, is it positive (increases) or negative (decreases)?
6. Rate confidence 0.0-1.0 based on how clear the evidence is

OUTPUT FORMAT:
If relationship found: "{concept1} -> {concept2} (positive, confidence: 0.8)"
If no relationship: "no relationship"

YOUR ANSWER:"""
    },
    
    "inter_cluster_edges": {
        "high_reasoning": """Analyze causal relationships between concept clusters:

Text: {text}

{pairs}

For each pair, identify if one cluster causally influences the other. Focus on relationships explicitly mentioned in the text.""",
        
        "low_reasoning": """TASK: Find causal relationships between concept groups.

TEXT: {text}

GROUPS: {pairs}

STEPS:
1. Read text carefully
2. Look for causal connections between concepts from different groups
3. Use format: "Pair X: group1 -> group2 (positive/negative, confidence: 0.X)"
4. If no relationship: "Pair X: no relationship"

CONCEPT GROUPS:
{pairs}

YOUR ANALYSIS:""",
        
        "low_reasoning_json": """TASK: Find causal relationships between concept groups. Return JSON only.

TEXT: {text}
GROUPS: {pairs}

STEPS:
1. Read text carefully
2. Look for causal connections between different groups
3. Rate confidence 0.0-1.0

JSON FORMAT:
{{
  "edges": [
    {{"pair_number": 1, "source": "group_name", "target": "group_name", "relationship": "positive", "confidence": 0.8, "has_relationship": true}}
  ],
  "no_relationships": [
    {{"pair_number": 2, "has_relationship": false, "reason": "no causal connection"}}
  ]
}}

JSON RESPONSE:"""
    },
    
    "intra_cluster_edges": {
        "high_reasoning": """Identify causal relationships within this concept cluster:

Text: {text}

{pairs}

Look for direct causal connections between concepts in the same semantic group.""",
        
        "low_reasoning": """TASK: Find causal relationships within concept groups.

TEXT: {text}
CLUSTER: {cluster_name}
PAIRS: {pairs}

STEPS:
1. Read text carefully
2. Look for causal connections between concepts in same group
3. Use format: "Pair X: concept1 -> concept2 (positive/negative, confidence: 0.X)"
4. If no relationship: "Pair X: no relationship"

YOUR ANALYSIS:""",
        
        "low_reasoning_json": """TASK: Find causal relationships within concept groups. Return JSON only.

TEXT: {text}
CLUSTER: {cluster_name}
PAIRS: {pairs}

STEPS:
1. Read text carefully
2. Look for causal connections within same group
3. Rate confidence 0.0-1.0

JSON FORMAT:
{{
  "edges": [
    {{"pair_number": 1, "source": "concept1", "target": "concept2", "relationship": "positive", "confidence": 0.8, "has_relationship": true}}
  ],
  "no_relationships": [
    {{"pair_number": 2, "has_relationship": false, "reason": "no causal connection"}}
  ]
}}

JSON RESPONSE:"""
    }
}

# =============================================================================
# PROMPT SELECTION UTILITIES
# =============================================================================

def get_model_class(model: str) -> str:
    """Determine model class based on reasoning capability."""
    capabilities = get_model_capabilities(model)
    
    if capabilities.get("reasoning", True):
        return "high_reasoning"
    else:
        return "low_reasoning"

def get_prompt_for_task(task: str, model: str, use_json: bool = None, **kwargs) -> str:
    """Get the appropriate prompt template for a task and model with automatic selection."""
    model_class = get_model_class(model)
    
    if task not in PROMPT_VARIANTS:
        # Try to find similar task or fallback
        task_mappings = {
            "edge_inference": "edge_inference_batch",
            "inter_cluster": "inter_cluster_edges", 
            "intra_cluster": "intra_cluster_edges"
        }
        task = task_mappings.get(task, task)
    
    if task not in PROMPT_VARIANTS:
        raise ValueError(f"Task '{task}' not found in PROMPT_VARIANTS")
    
    task_variants = PROMPT_VARIANTS[task]
    
    # Auto-determine if JSON should be used for weak models
    if use_json is None:
        use_json = should_use_json_output(model, task)
    
    # Select appropriate variant
    if use_json and model_class == "low_reasoning" and f"{model_class}_json" in task_variants:
        variant_key = f"{model_class}_json"
    elif model_class in task_variants:
        variant_key = model_class
    else:
        # Fallback to high_reasoning
        variant_key = "high_reasoning"
    
    if variant_key not in task_variants:
        raise ValueError(f"Variant '{variant_key}' not found for task '{task}'")
    
    prompt_template = task_variants[variant_key]
    
    # Format the template with provided kwargs
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        # Return template as-is if formatting fails
        print(f"Warning: Could not format prompt template for {task}/{variant_key}, missing key: {e}")
        return prompt_template

def should_use_json_output(model: str, task: str) -> bool:
    """Determine if a model/task combination should use JSON output."""
    model_class = get_model_class(model)
    
    # Low-reasoning models benefit from structured JSON for edge inference
    if model_class == "low_reasoning":
        return task in ["edge_inference_batch", "inter_cluster_edges", "intra_cluster_edges"]
    
    return False

def get_system_message_for_task(task: str, model: str) -> str:
    """Get appropriate system message based on model class and task with automatic selection."""
    model_class = get_model_class(model)
    
    if model_class == "high_reasoning":
        return "You are an expert at analyzing causal relationships in text."
    else:
        # Low-reasoning models benefit from very simple, direct system messages
        return "You analyze text to find cause-and-effect relationships between concepts. Follow the instructions exactly."

def get_all_prompts_for_model(model: str) -> Dict[str, str]:
    """Get all available prompts for a specific model."""
    model_class = get_model_class(model)
    result = {}
    
    for task, variants in PROMPT_VARIANTS.items():
        if model_class in variants:
            result[task] = variants[model_class]
        elif "high_reasoning" in variants:
            result[task] = variants["high_reasoning"]
    
    return result

# =============================================================================
# PROMPT ENHANCEMENT UTILITIES
# =============================================================================

def enhance_prompt_for_weak_model(base_prompt: str, task_context: str = None) -> str:
    """Add additional guidance for weak models."""
    enhancements = [
        "\nIMPORTANT REMINDERS:",
        "- Read carefully and follow instructions exactly",
        "- Look for explicit causal language in the text",
        "- Only report relationships clearly supported by evidence",
        "- Use the exact format specified"
    ]
    
    if task_context == "confidence":
        enhancements.extend([
            "- Confidence 0.9-1.0: Very clear evidence",
            "- Confidence 0.7-0.8: Good evidence", 
            "- Confidence 0.5-0.6: Some evidence",
            "- Confidence 0.0-0.4: Weak or unclear evidence"
        ])
    
    return base_prompt + "\n" + "\n".join(enhancements)

def create_few_shot_examples(task: str) -> str:
    """Create few-shot examples for specific tasks."""
    examples = {
        "edge_inference_batch": '''
EXAMPLE:
Text: "Stress at work causes sleep problems, which leads to fatigue the next day."
Pairs: Pair 1: 'stress' and 'sleep problems', Pair 2: 'sleep problems' and 'fatigue'
Response:
Pair 1: stress -> sleep problems (positive, confidence: 0.9)
Pair 2: sleep problems -> fatigue (positive, confidence: 0.8)''',

        "edge_inference_single": '''
EXAMPLE:
Text: "Regular exercise reduces anxiety and improves mood."
Concept A: exercise, Concept B: anxiety
Response: exercise -> anxiety (negative, confidence: 0.8)'''
    }
    
    return examples.get(task, "")

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def validate_prompts():
    """Validate that all prompt templates are properly formatted."""
    errors = []
    
    for task, variants in PROMPT_VARIANTS.items():
        for variant_key, template in variants.items():
            # Check for common formatting issues
            if not template.strip():
                errors.append(f"{task}.{variant_key}: Empty template")
            
            # Check for unclosed braces  
            open_braces = template.count('{')
            close_braces = template.count('}')
            if open_braces != close_braces:
                errors.append(f"{task}.{variant_key}: Unmatched braces")
    
    if errors:
        raise ValueError("Prompt validation failed:\n" + "\n".join(errors))
    
    return True

# Example usage and testing
if __name__ == "__main__":
    # Test prompt selection
    models = ["gpt-4o", "o3-mini", "gemini-pro"]
    
    print("=== MODEL CLASS ASSIGNMENT ===")
    for model in models:
        model_class = get_model_class(model) 
        print(f"{model}: {model_class}")
    print()
    
    print("=== SAMPLE PROMPTS ===")
    for model in ["gpt-4o", "o3-mini"]:
        print(f"Model: {model} ({get_model_class(model)})")
        print("-" * 40)
        
        # Show edge inference prompt
        prompt = get_prompt_for_task(
            model, 
            "edge_inference_batch",
            text="Social isolation causes depression.",
            pairs="Pair 1: 'isolation' and 'depression'"
        )
        print("Edge inference prompt:")
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
        print()
    
    # Validate all prompts
    validate_prompts()
    print("✅ All prompts validated successfully")