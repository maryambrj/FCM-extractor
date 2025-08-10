"""
Pydantic models for structured edge inference output parsing.
These ensure consistent, validated output from LLM responses.
"""

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, validator
import json

class EdgeRelationship(BaseModel):
    """A single edge relationship between two concepts with index-based identification."""
    
    i: int = Field(..., description="Index of the pair (0-indexed, matching input order)")
    dir: Literal["A->B", "B->A"] = Field(..., description="Direction of causal relationship")
    sign: Literal["positive", "negative"] = Field(..., description="Type of causal relationship")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    
    @validator('i')
    def validate_index(cls, v):
        """Ensure index is non-negative."""
        if v < 0:
            raise ValueError("Index must be non-negative")
        return v
    
    @validator('conf')
    def validate_confidence(cls, v):
        """Ensure confidence is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

class NoRelationship(BaseModel):
    """Represents a pair with no causal relationship."""
    
    i: int = Field(..., description="Index of the pair (0-indexed, matching input order)")
    dir: Literal["none"] = Field("none", description="No relationship exists")
    sign: None = Field(None, description="No sign for non-existent relationship")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence that no relationship exists")
    
    @validator('i')
    def validate_index(cls, v):
        """Ensure index is non-negative."""
        if v < 0:
            raise ValueError("Index must be non-negative")
        return v

class EdgeInferenceResponse(BaseModel):
    """Complete response for edge inference containing exactly N items in input order."""
    
    edges: List[Union[EdgeRelationship, NoRelationship]] = Field(..., description="List of relationships, one per input pair in order")
    
    @validator('edges')
    def validate_edge_count(cls, v, values, **kwargs):
        """Ensure we have exactly the expected number of edges."""
        # This will be validated when we know the expected count
        return v
    
    def to_legacy_format(self, expected_pairs: List[tuple]) -> List[dict]:
        """Convert to the legacy dict format expected by existing code."""
        results = []
        
        for edge in self.edges:
            if edge.i < len(expected_pairs):
                expected_c1, expected_c2 = expected_pairs[edge.i]
                
                if isinstance(edge, EdgeRelationship):
                    # Has a relationship
                    result = {
                        "source": expected_c1 if edge.dir == "A->B" else expected_c2,
                        "target": expected_c2 if edge.dir == "A->B" else expected_c1,
                        "weight": 1.0 if edge.sign == "positive" else -1.0,
                        "relationship": edge.sign,
                        "confidence": edge.conf,
                        "expected_pair": f"{expected_c1} -> {expected_c2}",
                        "parsed_pair": f"{expected_c1 if edge.dir == 'A->B' else expected_c2} -> {expected_c2 if edge.dir == 'A->B' else expected_c1}",
                        "structured_parsing": True,
                        "direction": edge.dir
                    }
                else:
                    # No relationship
                    result = {
                        "source": expected_c1,
                        "target": expected_c2,
                        "weight": 0.0,
                        "relationship": "none",
                        "confidence": edge.conf,
                        "expected_pair": f"{expected_c1} -> {expected_c2}",
                        "parsed_pair": f"{expected_c1} -> {expected_c2}",
                        "structured_parsing": True,
                        "direction": "none"
                    }
                
                results.append(result)
        
        return results

class ClusterEdgeResponse(BaseModel):
    """Response for inter-cluster edge inference."""
    
    cluster_relationships: List[dict] = Field(default_factory=list, description="Relationships between clusters")
    
    class ClusterRelationship(BaseModel):
        pair_number: int
        cluster_a: str
        cluster_b: str
        direction: Literal["A_to_B", "B_to_A", "bidirectional", "none"]
        relationship_type: Literal["positive", "negative"]
        confidence: float = Field(..., ge=0.0, le=1.0)
        representative_concepts: dict = Field(..., description="Example concepts from each cluster")

def get_edge_inference_schema() -> dict:
    """Get JSON schema for edge inference structured output."""
    return EdgeInferenceResponse.model_json_schema()

def get_cluster_edge_schema() -> dict:
    """Get JSON schema for cluster edge inference structured output.""" 
    return ClusterEdgeResponse.model_json_schema()

def parse_structured_edge_response(json_str: str, expected_pairs: List[tuple]) -> List[dict]:
    """Parse structured JSON response and convert to legacy format."""
    try:
        # Try to parse as structured response
        response = EdgeInferenceResponse.model_validate_json(json_str)
        
        # Validate that we have exactly the expected number of edges
        if len(response.edges) != len(expected_pairs):
            raise ValueError(f"Expected {len(expected_pairs)} edges, got {len(response.edges)}")
        
        # Validate that indices are sequential and start from 0
        indices = [edge.i for edge in response.edges]
        if indices != list(range(len(expected_pairs))):
            raise ValueError(f"Indices must be sequential starting from 0, got {indices}")
        
        return response.to_legacy_format(expected_pairs)
    
    except Exception as e:
        print(f"Failed to parse structured response: {e}")
        # Fall back to extracting JSON from mixed content
        return parse_json_from_mixed_content(json_str, expected_pairs)

def parse_json_from_mixed_content(content: str, expected_pairs: List[tuple]) -> List[dict]:
    """Extract JSON from content that may contain additional text."""
    import re
    
    # Try to find JSON object in the content
    json_patterns = [
        r'\{[^{}]*"edges"[^{}]*\}',  # Simple JSON object
        r'\{.*?"edges".*?\}',        # More permissive
        r'```json\s*(\{.*?\})\s*```', # JSON in code blocks
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                response = EdgeInferenceResponse.model_validate_json(match)
                
                # Validate edge count and indices
                if len(response.edges) != len(expected_pairs):
                    continue
                
                indices = [edge.i for edge in response.edges]
                if indices != list(range(len(expected_pairs))):
                    continue
                
                return response.to_legacy_format(expected_pairs)
            except:
                continue
    
    # If no structured JSON found, return empty list
    print("Could not extract valid structured JSON from response")
    return []

# Example usage and schema generation
if __name__ == "__main__":
    # Print the schema for debugging
    schema = get_edge_inference_schema()
    print("Edge Inference JSON Schema:")
    print(json.dumps(schema, indent=2))
    
    # Example structured response
    example_response = EdgeInferenceResponse(
        edges=[
            EdgeRelationship(
                i=0,
                dir="A->B",
                sign="positive",
                conf=0.85
            ),
            NoRelationship(
                i=1,
                conf=0.9
            )
        ]
    )
    
    print("\nExample Response JSON:")
    print(example_response.model_dump_json(indent=2))