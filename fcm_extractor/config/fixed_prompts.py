"""
Fixed-shape prompts that require exactly N items in input order.
Eliminates all parsing ambiguity through strict index-based alignment.
"""

from typing import List, Tuple
from config.model_prompts import get_model_class

def generate_fixed_edge_prompt(
    model: str,
    pairs: List[Tuple[str, str]], 
    text: str,
    task_type: str = "edge_inference_batch"
) -> str:
    """
    Generate fixed-shape prompt requiring exactly len(pairs) items in order.
    
    Args:
        model: Model name for capability-aware prompting
        pairs: List of concept pairs in order
        text: Text to analyze
        task_type: Type of edge inference task
        
    Returns:
        Formatted prompt requiring exact N responses
    """
    n_pairs = len(pairs)
    model_class = get_model_class(model)
    
    # Generate indexed pairs list
    pairs_list = []
    for i, (concept_a, concept_b) in enumerate(pairs, 1):
        pairs_list.append(f"{i}. A='{concept_a}' B='{concept_b}'")
    
    pairs_text = "\n".join(pairs_list)
    
    # Create example response template
    example_edges = []
    for i in range(1, min(3, n_pairs + 1)):  # Show 1-2 examples
        if i == 1:
            example_edges.append(f'  {{"i": {i}, "dir": "A->B", "sign": "positive", "conf": 0.8}}')
        else:
            example_edges.append(f'  {{"i": {i}, "dir": "none", "sign": null, "conf": 0.2}}')
    
    if n_pairs > 2:
        example_edges.append(f'  {{"i": {n_pairs}, "dir": "B->A", "sign": "negative", "conf": 0.6}}')
    
    example_json = "{\n  \"edges\": [\n" + ",\n".join(example_edges) + "\n  ]\n}"
    
    # Generate model-appropriate prompt
    if model_class == "low_reasoning":
        return _generate_low_reasoning_fixed_prompt(text, pairs_text, n_pairs, example_json)
    else:
        return _generate_high_reasoning_fixed_prompt(text, pairs_text, n_pairs, example_json)

def _generate_high_reasoning_fixed_prompt(
    text: str, 
    pairs_text: str, 
    n_pairs: int, 
    example_json: str
) -> str:
    """High-reasoning model prompt for fixed output."""
    return f"""Analyze causal relationships between concept pairs in this text.

TEXT:
{text}

PAIRS:
{pairs_text}

For each pair, determine:
- dir: "A->B" (A causes B), "B->A" (B causes A), or "none"
- sign: "positive" (increases) or "negative" (decreases) - null if dir="none"
- conf: confidence 0.0-1.0

CRITICAL: Return exactly {n_pairs} edges with indices 1-{n_pairs} in the same order as the pairs above.

JSON FORMAT:
{example_json}

Return only valid JSON."""

def _generate_low_reasoning_fixed_prompt(
    text: str, 
    pairs_text: str, 
    n_pairs: int, 
    example_json: str
) -> str:
    """Low-reasoning model prompt with explicit instructions."""
    return f"""TASK: Analyze {n_pairs} concept pairs for causal relationships.

TEXT TO ANALYZE:
{text}

CONCEPT PAIRS (ANSWER IN THIS EXACT ORDER):
{pairs_text}

INSTRUCTIONS:
1. For each pair, look for causal words: causes, leads to, results in, affects, influences
2. Choose exactly one option for each pair:
   - dir: "A->B" if A causes B
   - dir: "B->A" if B causes A  
   - dir: "none" if no relationship
3. If dir is NOT "none", choose sign:
   - "positive" if the cause increases the effect
   - "negative" if the cause decreases the effect
4. If dir is "none", set sign to null
5. Rate confidence 0.0 (no confidence) to 1.0 (completely sure)

CRITICAL RULES:
- Return EXACTLY {n_pairs} edges
- Use indices 1, 2, 3, ..., {n_pairs} in order
- Each edge must have: "i", "dir", "sign", "conf"
- Return ONLY valid JSON, no explanations

REQUIRED FORMAT:
{example_json}

JSON RESPONSE:"""

def generate_fixed_single_edge_prompt(
    model: str,
    concept_a: str, 
    concept_b: str,
    text: str
) -> str:
    """Generate fixed-shape prompt for single edge query."""
    model_class = get_model_class(model)
    
    example_json = '{"edges": [{"i": 1, "dir": "A->B", "sign": "positive", "conf": 0.8}]}'
    
    if model_class == "low_reasoning":
        return f"""TASK: Analyze relationship between two concepts.

TEXT TO ANALYZE:
{text}

CONCEPT PAIR:
1. A='{concept_a}' B='{concept_b}'

INSTRUCTIONS:
1. Look for causal words: causes, leads to, results in, affects, influences
2. Choose direction:
   - "A->B" if {concept_a} causes {concept_b}
   - "B->A" if {concept_b} causes {concept_a}
   - "none" if no relationship
3. If direction is NOT "none", choose sign:
   - "positive" if cause increases effect
   - "negative" if cause decreases effect
4. If direction is "none", set sign to null
5. Rate confidence 0.0-1.0

CRITICAL: Return exactly 1 edge with index 1.

REQUIRED FORMAT:
{example_json}

JSON RESPONSE:"""
    else:
        return f"""Analyze the causal relationship between '{concept_a}' and '{concept_b}'.

TEXT:
{text}

Determine:
- dir: "A->B", "B->A", or "none"
- sign: "positive" or "negative" (null if dir="none")
- conf: confidence 0.0-1.0

Return exactly 1 edge with index 1:
{example_json}"""

def generate_reask_prompt(original_response: str, schema_description: str) -> str:
    """Generate terse reask prompt for JSON validation failures."""
    return f"""You must return valid JSON matching this schema. No prose.

REQUIRED SCHEMA:
{schema_description}

YOUR PREVIOUS RESPONSE (INVALID):
{original_response[:500]}{'...' if len(original_response) > 500 else ''}

RETURN ONLY VALID JSON:"""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_json_schema_description(n_pairs: int) -> str:
    """Create human-readable schema description for reask prompts."""
    return f"""{{
  "edges": [
    {{"i": 1, "dir": "A->B|B->A|none", "sign": "positive|negative|null", "conf": 0.0-1.0}},
    {{"i": 2, "dir": "A->B|B->A|none", "sign": "positive|negative|null", "conf": 0.0-1.0}},
    ...
    {{"i": {n_pairs}, "dir": "A->B|B->A|none", "sign": "positive|negative|null", "conf": 0.0-1.0}}
  ]
}}

RULES:
- Exactly {n_pairs} edges with indices 1-{n_pairs}
- sign must be null when dir="none"
- sign required when dir="A->B" or dir="B->A\""""

def validate_fixed_format_requirements(pairs: List[Tuple[str, str]]) -> str:
    """Generate validation requirements text for fixed format."""
    n_pairs = len(pairs)
    return f"""VALIDATION REQUIREMENTS:
- Must return exactly {n_pairs} edges
- Indices must be 1, 2, 3, ..., {n_pairs} (in order)
- Each edge must have "i", "dir", "sign", "conf"
- dir must be "A->B", "B->A", or "none"
- sign must be "positive", "negative", or null
- sign must be null when dir="none"
- conf must be number between 0.0 and 1.0
- Must be valid JSON format"""

# =============================================================================
# TESTING
# =============================================================================

def test_fixed_prompts():
    """Test fixed prompt generation."""
    print("=== FIXED PROMPTS TEST ===")
    
    # Test data
    pairs = [
        ("social isolation", "depression"),
        ("exercise", "anxiety"),
        ("sleep quality", "stress")
    ]
    
    text = "Social isolation often leads to depression. Regular exercise can reduce anxiety."
    
    print(f"Testing with {len(pairs)} pairs")
    
    # Test high-reasoning model prompt
    print("\n--- High-reasoning model prompt ---")
    high_prompt = generate_fixed_edge_prompt("gpt-4o", pairs, text)
    print(high_prompt)
    
    # Test low-reasoning model prompt  
    print("\n--- Low-reasoning model prompt ---")
    low_prompt = generate_fixed_edge_prompt("o3-mini", pairs, text)
    print(low_prompt[:600] + "..." if len(low_prompt) > 600 else low_prompt)
    
    # Test single edge prompt
    print("\n--- Single edge prompt ---")
    single_prompt = generate_fixed_single_edge_prompt("o3-mini", "exercise", "depression", text)
    print(single_prompt[:400] + "..." if len(single_prompt) > 400 else single_prompt)
    
    # Test reask prompt
    print("\n--- Reask prompt ---")
    bad_response = "I think there might be relationships but I'm not sure..."
    schema_desc = create_json_schema_description(len(pairs))
    reask = generate_reask_prompt(bad_response, schema_desc)
    print(reask)
    
    print("\n✅ Fixed prompts test completed")
    return True

if __name__ == "__main__":
    test_fixed_prompts()