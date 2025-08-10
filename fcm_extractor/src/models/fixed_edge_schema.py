"""
Fixed-shape, index-based edge inference schema.
Returns exactly N items in the same order as input pairs for perfect alignment.
"""

from typing import List, Dict, Tuple, Literal, Optional
from pydantic import BaseModel, Field, validator
import json

# Literal types for forced selection
DirectionType = Literal["A->B", "B->A", "none"]
SignType = Literal["positive", "negative"]

class FixedEdgeResult(BaseModel):
    """Single edge result with fixed index-based format."""
    i: int = Field(..., ge=1, description="Index matching input pair order (1-based)")
    dir: DirectionType = Field(..., description="Direction: A->B, B->A, or none")
    sign: Optional[SignType] = Field(None, description="positive or negative (null if dir=none)")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    
    @validator('sign')
    def sign_validation(cls, v, values):
        """Validate sign based on direction."""
        dir_val = values.get('dir')
        if dir_val == 'none':
            if v is not None:
                raise ValueError("sign must be null when dir is 'none'")
        else:
            if v is None:
                raise ValueError("sign is required when dir is not 'none'")
        return v

class FixedEdgeResponse(BaseModel):
    """Fixed-shape response with exactly N edges in input order."""
    edges: List[FixedEdgeResult] = Field(..., description="Exactly N edges in input pair order")
    
    @validator('edges')
    def validate_edges_count_and_order(cls, v, values):
        """Validate that edges are properly indexed and ordered."""
        if not v:
            raise ValueError("edges list cannot be empty")
        
        # Check indices are sequential starting from 1
        expected_indices = list(range(1, len(v) + 1))
        actual_indices = [edge.i for edge in v]
        
        if actual_indices != expected_indices:
            raise ValueError(f"Edge indices must be sequential 1-{len(v)}, got {actual_indices}")
        
        return v
    
    def to_legacy_format(self, expected_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Convert to legacy edge format using index-based alignment."""
        if len(self.edges) != len(expected_pairs):
            raise ValueError(f"Edge count {len(self.edges)} doesn't match expected pairs {len(expected_pairs)}")
        
        legacy_edges = []
        
        for edge in self.edges:
            # Get the corresponding pair using index (1-based to 0-based)
            pair_idx = edge.i - 1
            if pair_idx >= len(expected_pairs):
                raise ValueError(f"Edge index {edge.i} exceeds pair count {len(expected_pairs)}")
            
            concept_a, concept_b = expected_pairs[pair_idx]
            
            # Skip edges with no relationship
            if edge.dir == "none":
                continue
            
            # Determine source and target based on direction
            if edge.dir == "A->B":
                source, target = concept_a, concept_b
            else:  # B->A
                source, target = concept_b, concept_a
            
            # Create legacy edge
            legacy_edge = {
                "source": source,
                "target": target,
                "weight": 1.0 if edge.sign == "positive" else -1.0,
                "relationship": edge.sign,
                "confidence": edge.conf,
                "pair_index": edge.i,
                "original_pair": (concept_a, concept_b),
                "original_direction": edge.dir,
                "validation_method": "fixed_index"
            }
            
            legacy_edges.append(legacy_edge)
        
        return legacy_edges
    
    @classmethod
    def get_json_schema(cls, n_pairs: int) -> Dict:
        """Get JSON schema for exactly N pairs."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "minItems": n_pairs,
                    "maxItems": n_pairs,
                    "items": {
                        "type": "object",
                        "properties": {
                            "i": {
                                "type": "integer", 
                                "minimum": 1, 
                                "maximum": n_pairs,
                                "description": f"Index 1-{n_pairs} matching input pair order"
                            },
                            "dir": {
                                "type": "string", 
                                "enum": ["A->B", "B->A", "none"],
                                "description": "Causal direction or none"
                            },
                            "sign": {
                                "type": ["string", "null"], 
                                "enum": ["positive", "negative", None],
                                "description": "Effect sign (null if dir=none)"
                            },
                            "conf": {
                                "type": "number", 
                                "minimum": 0.0, 
                                "maximum": 1.0,
                                "description": "Confidence score"
                            }
                        },
                        "required": ["i", "dir", "conf"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["edges"],
            "additionalProperties": False
        }

def parse_fixed_edge_response(
    response_text: str, 
    expected_pairs: List[Tuple[str, str]]
) -> Tuple[bool, List[Dict], List[str]]:
    """
    Parse fixed-shape edge response with strict index validation.
    
    Returns:
        Tuple of (success, legacy_edges, errors)
    """
    errors = []
    n_pairs = len(expected_pairs)
    
    try:
        # Extract JSON from response
        json_text = _extract_json_from_response(response_text)
        if not json_text:
            errors.append("No JSON found in response")
            return False, [], errors
        
        # Parse and validate with Pydantic
        edge_response = FixedEdgeResponse.model_validate_json(json_text)
        
        # Validate count matches expected pairs
        if len(edge_response.edges) != n_pairs:
            errors.append(f"Expected exactly {n_pairs} edges, got {len(edge_response.edges)}")
            return False, [], errors
        
        # Convert to legacy format using index alignment
        legacy_edges = edge_response.to_legacy_format(expected_pairs)
        
        return True, legacy_edges, errors
        
    except json.JSONDecodeError as e:
        errors.append(f"JSON parsing failed: {e}")
        return False, [], errors
    except Exception as e:
        errors.append(f"Validation failed: {e}")
        return False, [], errors

def _extract_json_from_response(response: str) -> Optional[str]:
    """Extract JSON from mixed text response."""
    import re
    
    # Try to find JSON in code blocks first
    json_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_block_patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                json.loads(match)  # Validate JSON
                return match
            except json.JSONDecodeError:
                continue
    
    # Try to find standalone JSON
    json_patterns = [
        r'\{[^{}]*"edges"[^{}]*\}',
        r'\{.*?"edges".*?\}',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)  # Validate JSON
                return match
            except json.JSONDecodeError:
                continue
    
    return None

def create_fixed_format_example(expected_pairs: List[Tuple[str, str]]) -> str:
    """Create example fixed format response for given pairs."""
    edges = []
    
    for i, (concept_a, concept_b) in enumerate(expected_pairs, 1):
        # Create sample edge (alternate patterns for demonstration)
        if i % 3 == 1:
            edge = {"i": i, "dir": "A->B", "sign": "positive", "conf": 0.8}
        elif i % 3 == 2:
            edge = {"i": i, "dir": "B->A", "sign": "negative", "conf": 0.6}
        else:
            edge = {"i": i, "dir": "none", "sign": None, "conf": 0.2}
        
        edges.append(edge)
    
    return json.dumps({"edges": edges}, indent=2)

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def test_fixed_edge_schema():
    """Test the fixed edge schema system."""
    print("=== FIXED EDGE SCHEMA TEST ===")
    
    # Test data
    expected_pairs = [
        ("social isolation", "depression"),
        ("exercise", "anxiety"),
        ("sleep quality", "stress")
    ]
    
    print(f"Testing with {len(expected_pairs)} pairs:")
    for i, (a, b) in enumerate(expected_pairs, 1):
        print(f"  {i}: {a} <-> {b}")
    
    # Create example response
    example = create_fixed_format_example(expected_pairs)
    print(f"\nExample fixed format:\n{example}")
    
    # Test parsing
    response_text = f"Here's the analysis:\n\n```json\n{example}\n```\n\nCompleted."
    
    success, edges, errors = parse_fixed_edge_response(response_text, expected_pairs)
    
    print(f"\nParsing results:")
    print(f"  Success: {success}")
    print(f"  Errors: {errors}")
    print(f"  Edges found: {len(edges)}")
    
    # Show edge details
    for edge in edges:
        pair_idx = edge["pair_index"]
        original_pair = edge["original_pair"]
        print(f"    {pair_idx}: {edge['source']} -> {edge['target']} ({edge['relationship']}, {edge['confidence']:.2f})")
        print(f"         Original: {original_pair[0]} <-> {original_pair[1]} ({edge['original_direction']})")
    
    # Test schema generation
    schema = FixedEdgeResponse.get_json_schema(len(expected_pairs))
    print(f"\nJSON Schema generated for {len(expected_pairs)} pairs")
    print(f"  Required edges: {schema['properties']['edges']['minItems']}-{schema['properties']['edges']['maxItems']}")
    
    return success

if __name__ == "__main__":
    test_fixed_edge_schema()