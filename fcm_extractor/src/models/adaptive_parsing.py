"""
Adaptive parsing and validation system for different model capabilities.
Provides stricter validation for weak models and hallucination detection.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import re
import json
from pydantic import BaseModel, Field, ValidationError, validator
from utils.llm_utils import get_model_capabilities
from src.models.edge_schemas import EdgeInferenceResponse, EdgeRelationship, NoRelationship
from src.models.edge_schemas_structured import StructuredEdgeResponse, parse_structured_edge_response

class ValidationResult(BaseModel):
    """Result of parsing and validation."""
    success: bool
    edges: List[Dict] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    parsing_method: str = "unknown"
    hallucination_detected: bool = False

class AdaptiveParser:
    """Parser that adapts validation strictness based on model capabilities."""
    
    def __init__(self, model: str):
        self.model = model
        self.capabilities = get_model_capabilities(model)
        self.is_weak_model = not self.capabilities.get("reasoning", True)
        
    def parse_edge_response(
        self, 
        response: str, 
        expected_pairs: List[Tuple[str, str]], 
        use_confidence: bool = True
    ) -> ValidationResult:
        """
        Parse edge inference response with model-appropriate validation.
        """
        result = ValidationResult(success=False, parsing_method="adaptive")
        
        if not response.strip():
            result.errors.append("Empty response")
            return result
        
        # Try parsing methods in order of reliability
        parsing_methods = self._get_parsing_methods()
        
        for method_name, method_func in parsing_methods:
            try:
                method_result = method_func(response, expected_pairs, use_confidence)
                if method_result.success and method_result.edges:
                    result = method_result
                    result.parsing_method = method_name
                    break
                elif method_result.errors:
                    result.warnings.extend([f"{method_name}: {e}" for e in method_result.errors])
            except Exception as e:
                result.warnings.append(f"{method_name} failed: {str(e)}")
        
        # Post-process validation
        if result.success:
            result = self._post_validate(result, expected_pairs)
        
        return result
    
    def _get_parsing_methods(self) -> List[Tuple[str, callable]]:
        """Get parsing methods in order of preference for this model."""
        methods = []
        
        if self.is_weak_model:
            # Weak models: prefer structured JSON parsing first
            methods.extend([
                ("structured_json", self._parse_structured_json),
                ("strict_regex", self._parse_strict_regex),
                ("lenient_regex", self._parse_lenient_regex),
                ("keyword_extraction", self._parse_keyword_extraction)
            ])
        else:
            # Strong models: can handle more flexible parsing
            methods.extend([
                ("structured_json", self._parse_structured_json),
                ("lenient_regex", self._parse_lenient_regex),
                ("strict_regex", self._parse_strict_regex),
                ("natural_language", self._parse_natural_language)
            ])
        
        return methods
    
    def _parse_structured_json(
        self, 
        response: str, 
        expected_pairs: List[Tuple[str, str]], 
        use_confidence: bool
    ) -> ValidationResult:
        """Parse structured JSON response with strict validation using new dir/sign/conf format."""
        result = ValidationResult(success=False, parsing_method="structured_json")
        
        try:
            # Try the new structured format first
            success, edges, errors = parse_structured_edge_response(response, expected_pairs)
            
            if success:
                result.edges = edges
                result.success = True
                if errors:
                    result.warnings.extend(errors)
                return result
            else:
                # Try legacy format as fallback
                json_match = self._extract_json_from_text(response)
                if not json_match:
                    result.errors.extend(errors or ["No valid JSON found"])
                    return result
                
                # Validate with legacy Pydantic model
                edge_response = EdgeInferenceResponse.model_validate_json(json_match)
                
                # Convert to legacy format with validation
                result.edges = self._convert_to_legacy_format(edge_response, expected_pairs, use_confidence)
                result.success = len(result.edges) > 0 or len(edge_response.no_relationships) > 0
                
                # Check for hallucinations in structured output
                result.hallucination_detected = self._detect_hallucinations_structured(edge_response, expected_pairs)
            
        except ValidationError as e:
            result.errors.append(f"JSON validation failed: {e}")
        except Exception as e:
            result.errors.append(f"JSON parsing failed: {e}")
        
        return result
    
    def _parse_strict_regex(
        self, 
        response: str, 
        expected_pairs: List[Tuple[str, str]], 
        use_confidence: bool
    ) -> ValidationResult:
        """Parse with strict regex patterns - good for weak models."""
        result = ValidationResult(success=False, parsing_method="strict_regex")
        
        # Very strict patterns
        relationship_pattern = re.compile(
            r"pair\s+(\d+):\s*['\"]?([^'\"]+?)['\"]?\s*->\s*['\"]?([^'\"]+?)['\"]?\s*\(\s*(positive|negative)\s*,\s*confidence:\s*([0-1]\.?\d*)\s*\)",
            re.IGNORECASE
        )
        
        no_relationship_pattern = re.compile(
            r"pair\s+(\d+):\s*no\s+relationship",
            re.IGNORECASE
        )
        
        lines = response.strip().split('\n')
        found_pairs = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try relationship pattern
            match = relationship_pattern.search(line)
            if match:
                pair_num = int(match.group(1)) - 1
                concept1 = match.group(2).strip().strip("'\"")
                concept2 = match.group(3).strip().strip("'\"")
                relationship = match.group(4).lower()
                
                try:
                    confidence = float(match.group(5))
                except ValueError:
                    result.warnings.append(f"Invalid confidence in line: {line}")
                    continue
                
                if 0 <= pair_num < len(expected_pairs):
                    found_pairs.add(pair_num)
                    
                    # Validate concept names against expected pairs
                    expected_c1, expected_c2 = expected_pairs[pair_num]
                    if not self._concepts_match(concept1, expected_c1, concept2, expected_c2):
                        if self.is_weak_model:
                            result.warnings.append(f"Concept mismatch in pair {pair_num + 1}: got '{concept1}' -> '{concept2}', expected '{expected_c1}' -> '{expected_c2}'")
                    
                    edge = {
                        "source": concept1,
                        "target": concept2,
                        "weight": 1.0 if relationship == "positive" else -1.0,
                        "relationship": relationship,
                        "confidence": confidence if use_confidence else 1.0,
                        "expected_pair": f"{expected_c1} -> {expected_c2}",
                        "parsed_pair": f"{concept1} -> {concept2}",
                        "validation_method": "strict_regex"
                    }
                    result.edges.append(edge)
                else:
                    result.warnings.append(f"Invalid pair number {pair_num + 1}")
            
            # Try no relationship pattern
            elif no_relationship_pattern.search(line):
                match = no_relationship_pattern.search(line)
                pair_num = int(match.group(1)) - 1
                if 0 <= pair_num < len(expected_pairs):
                    found_pairs.add(pair_num)
                    # Note: we don't add anything to results for "no relationship"
        
        # For weak models, warn about missing pairs
        if self.is_weak_model:
            missing_pairs = set(range(len(expected_pairs))) - found_pairs
            if missing_pairs:
                result.warnings.append(f"Missing responses for pairs: {sorted([p + 1 for p in missing_pairs])}")
        
        result.success = len(result.edges) > 0 or len(found_pairs) > 0
        return result
    
    def _parse_lenient_regex(
        self, 
        response: str, 
        expected_pairs: List[Tuple[str, str]], 
        use_confidence: bool
    ) -> ValidationResult:
        """Parse with more flexible regex - for stronger models."""
        result = ValidationResult(success=False, parsing_method="lenient_regex")
        
        # More flexible patterns
        patterns = [
            # Standard format
            r"pair\s*(\d+)[:\.]?\s*([^->\n]+?)\s*->\s*([^(\n]+?)\s*\(\s*(positive|negative)\s*,?\s*conf(?:idence)?[:\s]*([0-9.]+)",
            # Relaxed format
            r"(\d+)[:\.]?\s*([^->\n]+?)\s*(?:->|causes?|leads?\s+to|affects?)\s*([^(\n]+?)\s*\(\s*(positive|negative)",
            # Very relaxed
            r"([^->\n]+?)\s*(?:->|causes?|leads?\s+to)\s*([^(\n]+?)\s*\(\s*(positive|negative)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                for match in matches:
                    try:
                        if len(match) >= 4:
                            if pattern == patterns[0]:  # Has pair number
                                pair_num, concept1, concept2, relationship = match[:4]
                                confidence = float(match[4]) if len(match) > 4 else 0.7
                            else:
                                concept1, concept2, relationship = match[:3]
                                confidence = 0.7
                                pair_num = "1"
                            
                            edge = {
                                "source": concept1.strip().strip("'\""),
                                "target": concept2.strip().strip("'\""),
                                "weight": 1.0 if relationship.lower() == "positive" else -1.0,
                                "relationship": relationship.lower(),
                                "confidence": confidence if use_confidence else 1.0,
                                "validation_method": "lenient_regex"
                            }
                            result.edges.append(edge)
                    except Exception as e:
                        result.warnings.append(f"Failed to parse match {match}: {e}")
                
                if result.edges:
                    result.success = True
                    break
        
        return result
    
    def _parse_keyword_extraction(
        self, 
        response: str, 
        expected_pairs: List[Tuple[str, str]], 
        use_confidence: bool
    ) -> ValidationResult:
        """Fallback: extract relationships using keyword analysis."""
        result = ValidationResult(success=False, parsing_method="keyword_extraction")
        
        # Look for causal keywords near concept pairs
        causal_positive = ["causes", "leads to", "results in", "increases", "improves", "enhances"]
        causal_negative = ["reduces", "decreases", "hinders", "prevents", "worsens"]
        
        response_lower = response.lower()
        
        for i, (concept1, concept2) in enumerate(expected_pairs):
            c1_lower = concept1.lower()
            c2_lower = concept2.lower()
            
            # Check if both concepts appear in text
            if c1_lower in response_lower and c2_lower in response_lower:
                # Look for causal language
                for pos_word in causal_positive:
                    if pos_word in response_lower:
                        # Simple heuristic: if causal word appears between concepts
                        c1_pos = response_lower.find(c1_lower)
                        c2_pos = response_lower.find(c2_lower)
                        word_pos = response_lower.find(pos_word)
                        
                        if c1_pos < word_pos < c2_pos or c2_pos < word_pos < c1_pos:
                            direction = "positive"
                            source, target = (concept1, concept2) if c1_pos < c2_pos else (concept2, concept1)
                            
                            edge = {
                                "source": source,
                                "target": target,
                                "weight": 1.0,
                                "relationship": direction,
                                "confidence": 0.5 if use_confidence else 1.0,
                                "validation_method": "keyword_extraction"
                            }
                            result.edges.append(edge)
                            break
                
                # Check negative causation
                if not result.edges:  # Only if no positive found
                    for neg_word in causal_negative:
                        if neg_word in response_lower:
                            c1_pos = response_lower.find(c1_lower)
                            c2_pos = response_lower.find(c2_lower)
                            word_pos = response_lower.find(neg_word)
                            
                            if c1_pos < word_pos < c2_pos or c2_pos < word_pos < c1_pos:
                                direction = "negative"
                                source, target = (concept1, concept2) if c1_pos < c2_pos else (concept2, concept1)
                                
                                edge = {
                                    "source": source,
                                    "target": target,
                                    "weight": -1.0,
                                    "relationship": direction,
                                    "confidence": 0.5 if use_confidence else 1.0,
                                    "validation_method": "keyword_extraction"
                                }
                                result.edges.append(edge)
                                break
        
        result.success = len(result.edges) > 0
        if result.success:
            result.warnings.append("Used fallback keyword extraction - lower reliability")
        
        return result
    
    def _parse_natural_language(
        self, 
        response: str, 
        expected_pairs: List[Tuple[str, str]], 
        use_confidence: bool
    ) -> ValidationResult:
        """Parse natural language responses from strong models."""
        result = ValidationResult(success=False, parsing_method="natural_language")
        
        # This would use more sophisticated NLP parsing
        # For now, just a placeholder that tries to extract meaning
        
        sentences = response.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["causes", "leads", "affects", "influences"]):
                # Try to extract relationships using more flexible matching
                # This is a simplified implementation
                for concept1, concept2 in expected_pairs:
                    if concept1.lower() in sentence.lower() and concept2.lower() in sentence.lower():
                        # Determine polarity based on keywords
                        if any(word in sentence.lower() for word in ["positive", "increase", "improve", "enhance"]):
                            relationship = "positive"
                            weight = 1.0
                        elif any(word in sentence.lower() for word in ["negative", "decrease", "reduce", "worsen"]):
                            relationship = "negative"
                            weight = -1.0
                        else:
                            relationship = "positive"  # Default assumption
                            weight = 1.0
                        
                        edge = {
                            "source": concept1,
                            "target": concept2,
                            "weight": weight,
                            "relationship": relationship,
                            "confidence": 0.6 if use_confidence else 1.0,
                            "validation_method": "natural_language"
                        }
                        result.edges.append(edge)
        
        result.success = len(result.edges) > 0
        return result
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from mixed text content."""
        # Try to find JSON in code blocks first
        json_block_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'`(\{.*?\})`'
        ]
        
        for pattern in json_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
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
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)  # Validate JSON
                    return match
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _convert_to_legacy_format(
        self, 
        edge_response: EdgeInferenceResponse, 
        expected_pairs: List[Tuple[str, str]], 
        use_confidence: bool
    ) -> List[Dict]:
        """Convert structured response to legacy edge format."""
        return edge_response.to_legacy_format(expected_pairs)
    
    def _concepts_match(self, c1: str, expected_c1: str, c2: str, expected_c2: str) -> bool:
        """Check if parsed concepts match expected concepts (fuzzy matching)."""
        def normalize(concept: str) -> str:
            return concept.lower().strip().replace("_", " ").replace("-", " ")
        
        c1_norm, c2_norm = normalize(c1), normalize(c2)
        exp_c1_norm, exp_c2_norm = normalize(expected_c1), normalize(expected_c2)
        
        # Direct match
        if (c1_norm == exp_c1_norm and c2_norm == exp_c2_norm) or \
           (c1_norm == exp_c2_norm and c2_norm == exp_c1_norm):
            return True
        
        # Fuzzy match (contains)
        if (exp_c1_norm in c1_norm or c1_norm in exp_c1_norm) and \
           (exp_c2_norm in c2_norm or c2_norm in exp_c2_norm):
            return True
        
        return False
    
    def _detect_hallucinations_structured(
        self, 
        edge_response: EdgeInferenceResponse, 
        expected_pairs: List[Tuple[str, str]]
    ) -> bool:
        """Detect potential hallucinations in structured output."""
        hallucination_indicators = []
        
        # Check for unexpected concepts
        expected_concepts = set()
        for c1, c2 in expected_pairs:
            expected_concepts.add(c1.lower())
            expected_concepts.add(c2.lower())
        
        for edge in edge_response.edges:
            if edge.source.lower() not in expected_concepts:
                hallucination_indicators.append(f"Unexpected source concept: {edge.source}")
            if edge.target.lower() not in expected_concepts:
                hallucination_indicators.append(f"Unexpected target concept: {edge.target}")
        
        # Check for excessive confidence
        high_confidence_edges = [e for e in edge_response.edges if e.confidence > 0.95]
        if len(high_confidence_edges) > len(expected_pairs) * 0.8:
            hallucination_indicators.append("Suspiciously high confidence scores")
        
        return len(hallucination_indicators) > 0
    
    def _post_validate(self, result: ValidationResult, expected_pairs: List[Tuple[str, str]]) -> ValidationResult:
        """Post-process validation and quality checks."""
        
        # Check for duplicate edges
        seen_edges = set()
        unique_edges = []
        for edge in result.edges:
            edge_key = (edge["source"], edge["target"])
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(edge)
            else:
                result.warnings.append(f"Duplicate edge removed: {edge['source']} -> {edge['target']}")
        
        result.edges = unique_edges
        
        # Validate confidence scores for weak models
        if self.is_weak_model:
            for edge in result.edges:
                if edge["confidence"] > 0.9:
                    result.warnings.append(f"Suspiciously high confidence {edge['confidence']} for weak model")
                    edge["confidence"] = min(0.8, edge["confidence"])  # Cap confidence for weak models
        
        # Check response completeness
        if len(result.edges) == 0 and len(expected_pairs) > 0:
            result.warnings.append(f"No edges found for {len(expected_pairs)} pairs")
        
        return result

# Factory function for easy use
def create_adaptive_parser(model: str) -> AdaptiveParser:
    """Create an adaptive parser for a specific model."""
    return AdaptiveParser(model)

# Convenience function for backward compatibility
def parse_edge_response_adaptive(
    response: str, 
    expected_pairs: List[Tuple[str, str]], 
    model: str,
    use_confidence: bool = True
) -> List[Dict]:
    """Parse edge response with model-adaptive validation."""
    parser = create_adaptive_parser(model)
    result = parser.parse_edge_response(response, expected_pairs, use_confidence)
    
    if not result.success:
        print(f"Adaptive parsing failed for model {model}")
        for error in result.errors:
            print(f"  Error: {error}")
    
    for warning in result.warnings:
        print(f"  Warning: {warning}")
    
    if result.hallucination_detected:
        print(f"  ⚠️ Potential hallucination detected in {model} response")
    
    return result.edges

# Example usage
if __name__ == "__main__":
    # Test with different models
    models = ["gpt-4o", "o3-mini"]
    expected_pairs = [("stress", "anxiety"), ("exercise", "mood")]
    
    # Mock responses
    responses = {
        "json": '{"edges": [{"pair_number": 1, "source": "stress", "target": "anxiety", "relationship": "positive", "confidence": 0.8}]}',
        "text": "Pair 1: stress -> anxiety (positive, confidence: 0.8)\nPair 2: no relationship"
    }
    
    for model in models:
        print(f"\n=== Testing {model} ===")
        parser = create_adaptive_parser(model)
        
        for response_type, response in responses.items():
            print(f"\nResponse type: {response_type}")
            result = parser.parse_edge_response(response, expected_pairs)
            print(f"Success: {result.success}")
            print(f"Method: {result.parsing_method}")
            print(f"Edges: {len(result.edges)}")
            print(f"Warnings: {len(result.warnings)}")
            if result.hallucination_detected:
                print("⚠️ Hallucination detected")