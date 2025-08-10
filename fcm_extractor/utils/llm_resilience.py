"""
Resilience layer for LLM interactions with retry logic and model fallbacks.
Handles invalid responses, timeouts, and automatic model switching.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from src.models.llm_client import llm_client
from utils.llm_utils import get_model_capabilities
from utils.prompt_inference import infer_messages_for_task, get_task_configuration
from src.models.adaptive_parsing import create_adaptive_parser, ValidationResult

class FailureType(Enum):
    """Types of LLM failures."""
    API_ERROR = "api_error"
    TIMEOUT = "timeout" 
    INVALID_RESPONSE = "invalid_response"
    PARSING_FAILURE = "parsing_failure"
    HALLUCINATION = "hallucination"
    EMPTY_RESPONSE = "empty_response"
    JSON_ERROR = "json_error"

@dataclass
class LLMAttempt:
    """Record of a single LLM attempt."""
    model: str
    attempt_number: int
    success: bool
    failure_type: Optional[FailureType] = None
    error_message: str = ""
    response_length: int = 0
    execution_time: float = 0.0
    validation_warnings: List[str] = field(default_factory=list)

@dataclass 
class LLMResult:
    """Result of resilient LLM execution."""
    success: bool
    content: str = ""
    parsed_result: Any = None
    model_used: str = ""
    attempts: List[LLMAttempt] = field(default_factory=list)
    total_time: float = 0.0
    fallback_triggered: bool = False

class ResilientLLMClient:
    """
    Wrapper around llm_client with retry logic and model fallbacks.
    Automatically switches to stronger models when weaker ones fail.
    """
    
    def __init__(self):
        self.llm_client = llm_client
        
        # Model strength hierarchy (weakest to strongest)
        self.model_hierarchy = [
            "o3-mini-2025-01-31",
            "o3-mini", 
            "gpt-4o-mini",
            "gpt-4o",
            "gemini-pro"
        ]
        
        # Default retry configuration
        self.default_retry_config = {
            "max_retries": 3,
            "backoff_factor": 2.0,
            "initial_delay": 1.0,
            "max_delay": 30.0,
            "enable_model_fallback": True,
            "validation_required": True
        }
        
    def execute_with_fallback(
        self,
        task: str,
        primary_model: str,
        temperature: float = 0.1,
        timeout: Optional[float] = 120.0,
        validation_fn: Optional[Callable] = None,
        retry_config: Optional[Dict] = None,
        **task_kwargs
    ) -> LLMResult:
        """
        Execute an LLM task with automatic retries and model fallbacks.
        
        Args:
            task: Task name (e.g., "concept_extraction", "edge_inference_batch")
            primary_model: Preferred model to try first
            temperature: Model temperature
            timeout: Timeout per attempt in seconds
            validation_fn: Optional function to validate the response
            retry_config: Override default retry configuration
            **task_kwargs: Arguments for prompt formatting
            
        Returns:
            LLMResult with success status and content
        """
        start_time = time.time()
        config = {**self.default_retry_config, **(retry_config or {})}
        result = LLMResult(success=False)
        
        # Determine model sequence (primary + fallbacks)
        models_to_try = self._get_model_sequence(primary_model, config["enable_model_fallback"])
        
        for model_index, model in enumerate(models_to_try):
            if model_index > 0:
                result.fallback_triggered = True
                
            for attempt_num in range(config["max_retries"]):
                attempt = LLMAttempt(
                    model=model,
                    attempt_number=attempt_num + 1,
                    success=False
                )
                
                try:
                    print(f"  🔄 Attempt {attempt_num + 1}/{config['max_retries']} with {model}...")
                    
                    # Execute the LLM call
                    attempt_result = self._execute_single_attempt(
                        task, model, temperature, timeout, task_kwargs
                    )
                    
                    attempt.execution_time = attempt_result["execution_time"]
                    attempt.response_length = len(attempt_result["content"])
                    
                    # Validate the response
                    validation_result = self._validate_response(
                        attempt_result["content"], 
                        task, 
                        model,
                        validation_fn,
                        task_kwargs
                    )
                    
                    if validation_result["valid"]:
                        # Success!
                        attempt.success = True
                        result.success = True
                        result.content = attempt_result["content"]
                        result.parsed_result = validation_result.get("parsed_result")
                        result.model_used = model
                        result.attempts.append(attempt)
                        result.total_time = time.time() - start_time
                        
                        print(f"  ✅ Success with {model} on attempt {attempt_num + 1}")
                        return result
                    else:
                        # Validation failed
                        attempt.failure_type = FailureType.PARSING_FAILURE
                        attempt.error_message = validation_result["error"]
                        attempt.validation_warnings = validation_result.get("warnings", [])
                        
                        print(f"  ⚠️ Validation failed: {validation_result['error']}")
                        
                except TimeoutError:
                    attempt.failure_type = FailureType.TIMEOUT
                    attempt.error_message = f"Timeout after {timeout}s"
                    print(f"  ⏱️ Timeout after {timeout}s")
                    
                except json.JSONDecodeError as e:
                    attempt.failure_type = FailureType.JSON_ERROR
                    attempt.error_message = f"JSON parsing failed: {e}"
                    print(f"  📄 JSON parsing failed: {e}")
                    
                except Exception as e:
                    attempt.failure_type = FailureType.API_ERROR
                    attempt.error_message = str(e)
                    print(f"  🚨 API error: {e}")
                
                result.attempts.append(attempt)
                
                # Wait before retry (exponential backoff)
                if attempt_num < config["max_retries"] - 1:
                    delay = min(
                        config["initial_delay"] * (config["backoff_factor"] ** attempt_num),
                        config["max_delay"]
                    )
                    print(f"    Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
        
        # All attempts failed
        result.total_time = time.time() - start_time
        print(f"  ❌ All attempts failed after {result.total_time:.1f}s")
        return result
    
    def _get_model_sequence(self, primary_model: str, enable_fallback: bool) -> List[str]:
        """Get sequence of models to try, starting with primary."""
        if not enable_fallback:
            return [primary_model]
        
        # Find primary model in hierarchy
        try:
            primary_index = self.model_hierarchy.index(primary_model)
        except ValueError:
            # Primary model not in hierarchy, use it first then standard sequence
            return [primary_model] + self.model_hierarchy[-2:]  # Add strongest fallbacks
        
        # Use models stronger than primary as fallbacks
        fallback_models = self.model_hierarchy[primary_index + 1:]
        
        # If primary is already strongest, only try it
        if not fallback_models:
            return [primary_model]
        
        # Return primary + up to 2 stronger fallbacks
        return [primary_model] + fallback_models[:2]
    
    def _execute_single_attempt(
        self, 
        task: str, 
        model: str, 
        temperature: float, 
        timeout: Optional[float],
        task_kwargs: Dict
    ) -> Dict[str, Any]:
        """Execute a single LLM attempt."""
        start_time = time.time()
        
        # Get messages using automatic prompt inference
        messages = infer_messages_for_task(task, model, **task_kwargs)
        
        # Call LLM
        content, metadata = self.llm_client.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        
        return {
            "content": content,
            "metadata": metadata,
            "execution_time": execution_time,
            "messages": messages
        }
    
    def _validate_response(
        self, 
        content: str, 
        task: str, 
        model: str,
        validation_fn: Optional[Callable],
        task_kwargs: Dict
    ) -> Dict[str, Any]:
        """Validate LLM response using adaptive parsing or custom validation."""
        
        # Basic empty response check
        if not content or not content.strip():
            return {
                "valid": False,
                "error": "Empty response",
                "warnings": []
            }
        
        # Use custom validation if provided
        if validation_fn:
            try:
                is_valid, error_msg, parsed_result = validation_fn(content)
                return {
                    "valid": is_valid,
                    "error": error_msg or "Custom validation failed",
                    "parsed_result": parsed_result,
                    "warnings": []
                }
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Custom validation error: {e}",
                    "warnings": []
                }
        
        # Use adaptive parsing for edge inference tasks
        if "edge_inference" in task or "cluster_edges" in task:
            return self._validate_edge_response(content, model, task_kwargs)
        
        # For other tasks, basic validation
        return self._basic_validation(content, task)
    
    def _validate_edge_response(self, content: str, model: str, task_kwargs: Dict) -> Dict[str, Any]:
        """Validate edge inference responses using adaptive parsing."""
        try:
            # Extract expected pairs from task kwargs
            expected_pairs = []
            if "pairs" in task_kwargs:
                pairs_text = task_kwargs["pairs"]
                # Parse pairs from text (simplified)
                lines = pairs_text.strip().split('\n')
                for line in lines:
                    if "Pair" in line and ":" in line:
                        # Extract concept names (simplified parsing)
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            concepts_part = parts[1].strip().strip("'\"")
                            if " and " in concepts_part:
                                c1, c2 = concepts_part.split(" and ", 1)
                                expected_pairs.append((c1.strip().strip("'\""), c2.strip().strip("'\"")))
            
            if not expected_pairs:
                # Fallback - assume single pair from concept1/concept2
                if "concept1" in task_kwargs and "concept2" in task_kwargs:
                    expected_pairs = [(task_kwargs["concept1"], task_kwargs["concept2"])]
            
            if expected_pairs:
                parser = create_adaptive_parser(model)
                result = parser.parse_edge_response(content, expected_pairs, use_confidence=True)
                
                return {
                    "valid": result.success,
                    "error": "; ".join(result.errors) if result.errors else "",
                    "parsed_result": result.edges,
                    "warnings": result.warnings,
                    "hallucination_detected": result.hallucination_detected
                }
        
        except Exception as e:
            return {
                "valid": False,
                "error": f"Adaptive parsing failed: {e}",
                "warnings": []
            }
        
        # Fallback to basic validation
        return self._basic_validation(content, "edge_inference")
    
    def _basic_validation(self, content: str, task: str) -> Dict[str, Any]:
        """Basic validation for non-edge inference tasks."""
        warnings = []
        
        # Check minimum length
        if len(content) < 10:
            return {
                "valid": False,
                "error": "Response too short",
                "warnings": warnings
            }
        
        # Check for obvious failure indicators
        failure_indicators = [
            "i cannot", "i can't", "unable to", "don't have access",
            "as an ai", "i'm not able", "sorry", "cannot process"
        ]
        
        content_lower = content.lower()
        for indicator in failure_indicators:
            if indicator in content_lower:
                return {
                    "valid": False,
                    "error": f"Response indicates failure: contains '{indicator}'",
                    "warnings": warnings
                }
        
        # Task-specific validation
        if "concept_extraction" in task:
            # Should contain some comma-separated terms
            if "," not in content and "\n" not in content:
                warnings.append("Response may not contain multiple concepts")
        
        return {
            "valid": True,
            "error": "",
            "parsed_result": content,
            "warnings": warnings
        }

# Global resilient client instance
resilient_llm = ResilientLLMClient()

def test_resilient_execution():
    """Test the resilient execution system."""
    print("=== RESILIENT LLM EXECUTION TEST ===")
    
    # Test with a simple task that should work without API calls
    # We'll create a mock result for testing
    result = LLMResult(
        success=True,
        content="social isolation, depression, anxiety, sleep quality",
        model_used="mock-model",
        fallback_triggered=False,
        total_time=0.5
    )
    
    print(f"\nMock Result:")
    print(f"  Success: {result.success}")
    print(f"  Model used: {result.model_used}")
    print(f"  Fallback triggered: {result.fallback_triggered}")
    print(f"  Content preview: {result.content}")
    
    return result

if __name__ == "__main__":
    test_resilient_execution()