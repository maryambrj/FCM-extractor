"""
Utilities for structured output parsing with JSON schema enforcement.
Provides helpers for generating schemas and handling structured LLM responses.
"""

import json
from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel
from utils.llm_utils import get_model_capabilities, requires_detailed_prompts

def should_use_structured_output(model: str) -> bool:
    """Determine if a model should use structured output based on its capabilities."""
    capabilities = get_model_capabilities(model)
    
    # Models that benefit most from structured output:
    # 1. Less capable models (like o3-mini) that need explicit schemas
    # 2. Models that don't support dynamic prompting well
    return (
        requires_detailed_prompts(model) or 
        not capabilities.get("dynamic_prompting", True) or
        not capabilities.get("reasoning", True)
    )

def get_structured_output_instructions(model: str, schema: Dict[str, Any]) -> str:
    """Generate model-specific instructions for structured output."""
    base_instructions = f"""
IMPORTANT: Your response must be valid JSON that exactly matches this schema:

```json
{json.dumps(schema, indent=2)}
```

Rules:
1. Respond with ONLY valid JSON, no additional text
2. All required fields must be present
3. Follow the exact field names and types specified
4. Confidence values must be between 0.0 and 1.0
5. Return exactly N items in the same order as the input pairs
6. Index 'i' must start at 0 and be sequential
"""

    if requires_detailed_prompts(model):
        # Add extra guidance for models that need it
        base_instructions += """
7. For each pair, carefully analyze if a causal relationship exists
8. Only report relationships that are explicitly supported by the text
9. Use "A->B" for relationships where concept A influences concept B
10. Use "B->A" for relationships where concept B influences concept A
11. Use "none" for pairs with no causal relationship
12. Use "positive" for relationships where source increases target
13. Use "negative" for relationships where source decreases target
14. Be conservative with confidence scores - use lower values if uncertain
15. Ensure each edge has the correct index matching the input pair order
"""

    return base_instructions.strip()

def create_json_prompt(
    base_prompt: str, 
    model: str, 
    schema_model: Type[BaseModel],
    additional_context: Optional[str] = None
) -> str:
    """Create a prompt that requests structured JSON output."""
    
    schema = schema_model.model_json_schema()
    structured_instructions = get_structured_output_instructions(model, schema)
    
    # Build the complete prompt
    parts = [base_prompt.strip()]
    
    if additional_context:
        parts.append(additional_context.strip())
    
    parts.append(structured_instructions)
    
    if requires_detailed_prompts(model):
        parts.append("\nExample JSON structure:")
        # Add a minimal example
        example = create_example_json(schema_model)
        if example:
            parts.append(f"```json\n{example}\n```")
    
    parts.append("\nYour JSON response:")
    
    return "\n\n".join(parts)

def create_example_json(schema_model: Type[BaseModel]) -> Optional[str]:
    """Create a minimal example JSON for the schema."""
    try:
        # Try to create a minimal valid example
        if hasattr(schema_model, '__name__'):
            if 'EdgeInference' in schema_model.__name__:
                example = {
                    "edges": [
                        {
                            "i": 0,
                            "dir": "A->B",
                            "sign": "positive",
                            "conf": 0.8
                        },
                        {
                            "i": 1,
                            "dir": "none",
                            "sign": None,
                            "conf": 0.9
                        }
                    ]
                }
                return json.dumps(example, indent=2)
    except Exception:
        pass
    
    return None

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from a response that may contain additional text."""
    import re
    
    # First, try to parse the entire response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in code blocks
    code_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'`(\{.*?\})`'
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find any JSON-like structure
    json_patterns = [
        r'\{[^{}]*"edges"[^{}]*\}',
        r'\{.*?"edges".*?\}',
        r'\{.*?\}',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    return None

def parse_with_fallback(
    response: str, 
    schema_model: Type[BaseModel],
    fallback_parser: Optional[callable] = None
) -> Any:
    """Parse structured response with fallback to legacy parsing."""
    
    # Try structured parsing first
    json_data = extract_json_from_response(response)
    if json_data:
        try:
            return schema_model.model_validate(json_data)
        except Exception as e:
            print(f"Structured parsing failed: {e}")
    
    # Fall back to legacy parser if provided
    if fallback_parser:
        print("Falling back to legacy parsing")
        return fallback_parser(response)
    
    # Return empty instance if all else fails
    try:
        return schema_model()
    except Exception:
        return None

def validate_structured_response(response: str, expected_schema: Dict[str, Any]) -> bool:
    """Validate that a response matches the expected schema structure."""
    json_data = extract_json_from_response(response)
    if not json_data:
        return False
    
    # Basic structure validation
    required_keys = expected_schema.get('required', [])
    return all(key in json_data for key in required_keys)

# Schema templates for common use cases
EDGE_INFERENCE_TEMPLATE = """
For each concept pair, analyze the text and determine if there's a causal relationship.
Respond with a JSON object containing:
- "edges": array of relationships found
- "no_relationships": array of pairs with no causal connection

Each edge should specify:
- pair_number (1-indexed)
- source and target concept names
- relationship type ("positive" or "negative") 
- confidence score (0.0 to 1.0)
"""

CLUSTER_EDGE_TEMPLATE = """
Analyze relationships between concept clusters.
Respond with JSON containing cluster-level relationships.
"""