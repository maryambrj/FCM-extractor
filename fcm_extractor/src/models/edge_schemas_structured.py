"""
Structured edge inference schemas with forced selection format.
Eliminates fuzzy matching by requiring models to choose from fixed options.
"""

from typing import List, Dict, Tuple, Literal, Optional
from pydantic import BaseModel, Field, validator
import json

# Literal types for forced selection
DirectionType = Literal["A->B", "B->A", "none"]
SignType = Literal["positive", "negative"]

class StructuredEdgeResult(BaseModel):
    """Single edge result with forced selection format."""
    pair_number: int = Field(..., ge=1, description="Pair number from the input")
    concept_A: str = Field(..., description="First concept (exactly as given)")
    concept_B: str = Field(..., description="Second concept (exactly as given)")
    dir: DirectionType = Field(..., description="Direction: A->B, B->A, or none")
    sign: Optional[SignType] = Field(None, description="positive or negative (ignored if dir=none)")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    
    @validator('sign')
    def sign_required_if_direction(cls, v, values):
        """Sign is required when direction is not 'none'."""
        dir_val = values.get('dir')
        if dir_val and dir_val != 'none' and v is None:
            raise ValueError("sign is required when dir is not 'none'")
        if dir_val == 'none' and v is not None:
            raise ValueError("sign should be null when dir is 'none'")
        return v
    
    def to_legacy_format(self) -> Dict:
        """Convert to legacy edge format for backward compatibility."""
        if self.dir == "none":
            return {
                "source": self.concept_A,
                "target": self.concept_B,
                "weight": 0,
                "relationship": "none",
                "confidence": self.conf,
                "pair_number": self.pair_number,
                "validation_method": "structured_selection"
            }
        
        # Determine source and target based on direction
        if self.dir == "A->B":
            source, target = self.concept_A, self.concept_B
        else:  # B->A
            source, target = self.concept_B, self.concept_A
        
        # Determine weight based on sign
        weight = 1.0 if self.sign == "positive" else -1.0
        
        return {
            "source": source,
            "target": target,
            "weight": weight,
            "relationship": self.sign,
            "confidence": self.conf,
            "pair_number": self.pair_number,
            "original_direction": self.dir,
            "validation_method": "structured_selection"
        }

class StructuredEdgeResponse(BaseModel):
    """Complete response with structured edge results."""
    edges: List[StructuredEdgeResult] = Field(default_factory=list, description="List of edge results")
    
    def to_legacy_format(self, expected_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Convert all edges to legacy format, including all expected pairs."""
        legacy_edges = []
        
        # Create a mapping of expected pairs to their results
        pair_results = {}
        for edge in self.edges:
            pair_key = (edge.concept_A, edge.concept_B)
            pair_results[pair_key] = edge
            # Also add reverse pair for bidirectional matching
            pair_results[(edge.concept_B, edge.concept_A)] = edge
        
        # Process all expected pairs
        for i, (concept_A, concept_B) in enumerate(expected_pairs):
            pair_key = (concept_A, concept_B)
            reverse_key = (concept_B, concept_A)
            
            if pair_key in pair_results or reverse_key in pair_results:
                # Use the found edge result
                edge = pair_results.get(pair_key) or pair_results.get(reverse_key)
                legacy_edge = edge.to_legacy_format()
                legacy_edges.append(legacy_edge)
            else:
                # Create a "no relationship" entry for missing pairs
                legacy_edges.append({
                    "source": concept_A,
                    "target": concept_B,
                    "weight": 0,
                    "relationship": "no relationship",
                    "confidence": 0.0,  # Default confidence for no relationship
                    "pair_number": i + 1,
                    "validation_method": "structured_selection_missing"
                })
        
        return legacy_edges
    
    @classmethod
    def get_json_schema(cls) -> Dict:
        """Get JSON schema for the response format."""
        return {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair_number": {"type": "integer", "minimum": 1},
                            "concept_A": {"type": "string"},
                            "concept_B": {"type": "string"},
                            "dir": {"type": "string", "enum": ["A->B", "B->A", "none"]},
                            "sign": {"type": "string", "enum": ["positive", "negative"], "nullable": True},
                            "conf": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["pair_number", "concept_A", "concept_B", "dir", "conf"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["edges"],
            "additionalProperties": False
        }

def parse_structured_edge_response(response_text: str, expected_pairs: List[Tuple[str, str]]) -> Tuple[bool, List[Dict], List[str]]:
    """
    Parse structured edge response with validation.
    
    Returns:
        Tuple of (success, edges, errors)
    """
    errors = []
    
    try:
        # Extract JSON from response
        json_text = _extract_json_from_response(response_text)
        if not json_text:
            errors.append("No JSON found in response")
            return False, [], errors
        
        # Parse and validate with Pydantic
        edge_response = StructuredEdgeResponse.model_validate_json(json_text)
        
        # Convert to legacy format
        legacy_edges = edge_response.to_legacy_format(expected_pairs)
        
        # Validate that all pairs are covered
        expected_count = len(expected_pairs)
        found_count = len(edge_response.edges)
        
        if found_count != expected_count:
            errors.append(f"Expected {expected_count} pairs, found {found_count}")
        
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
    
    # Try to find JSON without code blocks
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

# =============================================================================
# EXAMPLE AND TESTING
# =============================================================================

def create_example_structured_response() -> str:
    """Create example structured response for testing."""
    example = {
        "edges": [
            {
                "pair_number": 1,
                "concept_A": "social isolation",
                "concept_B": "depression",
                "dir": "A->B",
                "sign": "positive",
                "conf": 0.85
            },
            {
                "pair_number": 2,
                "concept_A": "exercise",
                "concept_B": "anxiety", 
                "dir": "B->A",
                "sign": "negative",
                "conf": 0.72
            },
            {
                "pair_number": 3,
                "concept_A": "sleep quality",
                "concept_B": "stress levels",
                "dir": "none",
                "sign": None,
                "conf": 0.15
            }
        ]
    }
    
    return json.dumps(example, indent=2)

def test_structured_parsing():
    """Test the structured parsing system."""
    print("=== STRUCTURED EDGE PARSING TEST ===")
    
    # Test data
    expected_pairs = [
        ("social isolation", "depression"),
        ("exercise", "anxiety"),
        ("sleep quality", "stress levels")
    ]
    
    # Create example response
    example_response = f"Here's the analysis:\n\n```json\n{create_example_structured_response()}\n```\n\nThis completes the analysis."
    
    print("Example response:")
    print(example_response)
    print("\n" + "-" * 50)
    
    # Parse the response
    success, edges, errors = parse_structured_edge_response(example_response, expected_pairs)
    
    print(f"Parsing success: {success}")
    print(f"Errors: {errors}")
    print(f"Extracted edges: {len(edges)}")
    
    for i, edge in enumerate(edges):
        print(f"  {i+1}: {edge['source']} -> {edge['target']} ({edge['relationship']}, conf: {edge['confidence']})")
    
    # Test schema generation
    schema = StructuredEdgeResponse.get_json_schema()
    print(f"\nJSON Schema properties: {list(schema['properties'].keys())}")
    
    return success

if __name__ == "__main__":
    test_structured_parsing()