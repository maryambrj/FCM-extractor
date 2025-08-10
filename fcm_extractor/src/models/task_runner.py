"""
Model-agnostic task runner for FCM extraction.
Provides clean entry points that hide model selection and capability logic from business code.
"""

from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from config.constants import (
    EDGE_INFERENCE_MODEL, EDGE_INFERENCE_TEMPERATURE,
    CONCEPT_EXTRACTION_MODEL, CONCEPT_EXTRACTION_TEMPERATURE, CONCEPT_EXTRACTION_N_PROMPTS,
    META_PROMPTING_MODEL, META_PROMPTING_TEMPERATURE,
    LLM_CLUSTERING_MODEL
)
from src.models.llm_client import llm_client
from utils.llm_utils import get_model_capabilities
from utils.model_fallbacks import ModelFallbackStrategy
from utils.resilient_edge_inference import (
    resilient_concept_extraction, resilient_batch_edge_queries, 
    resilient_cluster_edge_inference, resilient_single_edge_query
)

@dataclass
class TaskConfig:
    """Configuration for a specific task."""
    model: str
    temperature: float
    max_tokens: Optional[int] = None
    batch_size: Optional[int] = None
    timeout: Optional[int] = None

# Task configuration mapping
TASK_CONFIGS = {
    "concept_extraction": TaskConfig(
        model=CONCEPT_EXTRACTION_MODEL,
        temperature=CONCEPT_EXTRACTION_TEMPERATURE,
        max_tokens=2000
    ),
    "edge_inference": TaskConfig(
        model=EDGE_INFERENCE_MODEL, 
        temperature=EDGE_INFERENCE_TEMPERATURE,
        max_tokens=4000
    ),
    "cluster_inference": TaskConfig(
        model=EDGE_INFERENCE_MODEL,
        temperature=EDGE_INFERENCE_TEMPERATURE,
        max_tokens=4000
    ),
    "meta_prompting": TaskConfig(
        model=META_PROMPTING_MODEL,
        temperature=META_PROMPTING_TEMPERATURE,
        max_tokens=3000
    ),
    "clustering": TaskConfig(
        model=LLM_CLUSTERING_MODEL,
        temperature=0.1,
        max_tokens=2000
    )
}

class TaskRunner:
    """Model-agnostic task runner that handles all capability-aware logic internally."""
    
    def __init__(self):
        self.llm_client = llm_client
    
    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get configuration for a specific task."""
        if task_name not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_name}")
        return TASK_CONFIGS[task_name]
    
    def run_task(self, task_name: str, **kwargs) -> Any:
        """
        Run a task with automatic model selection and capability handling.
        
        Args:
            task_name: Name of the task to run
            **kwargs: Task-specific arguments
        
        Returns:
            Task result with all capability logic handled internally
        """
        config = self.get_task_config(task_name)
        
        # Get model capabilities and create strategy
        capabilities = get_model_capabilities(config.model)
        strategy = ModelFallbackStrategy(config.model)
        
        # Log task execution
        print(f"Running task '{task_name}' with model {config.model} ({capabilities.get('reasoning', True) and 'high' or 'low'}-reasoning)")
        
        # Route to appropriate task handler
        if task_name == "concept_extraction":
            return self._run_concept_extraction(config, **kwargs)
        elif task_name == "edge_inference":
            return self._run_edge_inference(config, **kwargs)
        elif task_name == "cluster_inference":
            return self._run_cluster_inference(config, **kwargs)
        elif task_name == "intra_cluster_edges":
            return self._run_intra_cluster_edges(config, **kwargs)
        elif task_name == "meta_prompting":
            return self._run_meta_prompting(config, **kwargs)
        elif task_name == "clustering":
            return self._run_clustering(config, **kwargs)
        elif task_name == "cluster_naming":
            return self._run_cluster_naming(config, **kwargs)
        else:
            raise ValueError(f"Unsupported task: {task_name}")
    
    def _run_concept_extraction(self, config: TaskConfig, text: str, n_prompts: int = None, **kwargs) -> List[str]:
        """Run concept extraction with automatic capability handling and resilience."""
        n_prompts = n_prompts or CONCEPT_EXTRACTION_N_PROMPTS
        
        # Use resilient execution with automatic fallbacks
        return resilient_concept_extraction(
            text=text,
            model=config.model,
            temperature=config.temperature,
            n_prompts=n_prompts,
            max_retries=3,
            enable_fallback=True,
            **kwargs
        )
    
    def _run_edge_inference(self, config: TaskConfig, concept_pairs: List[Tuple[str, str]], text: str, **kwargs) -> List[Dict]:
        """Run edge inference with automatic capability handling and resilience."""
        # Use resilient execution with automatic fallbacks
        return resilient_batch_edge_queries(
            concept_pairs=concept_pairs,
            text=text,
            model=config.model,
            temperature=config.temperature,
            max_retries=3,
            enable_fallback=True,
            **kwargs
        )
    
    def _run_cluster_inference(self, config: TaskConfig, clusters: Dict[str, List[str]], text: str, **kwargs) -> Tuple[List[Dict], List[Dict]]:
        """Run cluster edge inference with automatic capability handling and resilience."""
        # Use resilient execution with automatic fallbacks
        return resilient_cluster_edge_inference(
            clusters=clusters,
            text=text,
            model=config.model,
            temperature=config.temperature,
            max_retries=3,
            enable_fallback=True,
            **kwargs
        )
    
    def _run_intra_cluster_edges(self, config: TaskConfig, clusters: Dict[str, List[str]], text: str, **kwargs) -> List[Dict]:
        """Run intra-cluster edge inference with automatic capability handling."""
        from src.edge_inference.edge_inference import infer_intra_cluster_edges
        
        return infer_intra_cluster_edges(
            clusters=clusters,
            text=text,
            model=config.model,
            temperature=config.temperature,
            **kwargs
        )
    
    def _run_meta_prompting(self, config: TaskConfig, **kwargs) -> Any:
        """Run meta-prompting with automatic capability handling."""
        # Implementation would depend on specific meta-prompting interface
        pass
    
    def _run_clustering(self, config: TaskConfig, concepts: List[str], **kwargs) -> Dict[int, List[str]]:
        """Run LLM-based clustering with automatic capability handling."""
        from src.clustering.clustering import llm_based_clustering
        
        return llm_based_clustering(
            concepts=concepts,
            model=config.model,
            **kwargs
        )
    
    def _run_cluster_naming(self, config: TaskConfig, clusters: Dict[int, List[str]], **kwargs) -> Dict[str, List[str]]:
        """Run cluster naming with automatic capability handling."""
        from src.clustering.clustering import name_all_clusters
        
        return name_all_clusters(
            clusters=clusters,
            model=config.model,
            **kwargs
        )

# Global task runner instance
task_runner = TaskRunner()

# =============================================================================
# MODEL-AGNOSTIC ENTRY POINTS
# =============================================================================

def run_concept_extraction(text: str, n_prompts: int = None) -> List[str]:
    """
    Extract concepts from text using the configured model.
    All model capability logic is handled internally.
    """
    return task_runner.run_task("concept_extraction", text=text, n_prompts=n_prompts)

def run_edge_inference(concept_pairs: List[Tuple[str, str]], text: str) -> List[Dict]:
    """
    Infer edges between concept pairs using the configured model.
    All model capability logic is handled internally.
    """
    return task_runner.run_task("edge_inference", concept_pairs=concept_pairs, text=text)

def run_cluster_inference(clusters: Dict[str, List[str]], text: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Infer edges between and within clusters using the configured model.
    Returns (inter_cluster_edges, intra_cluster_edges).
    All model capability logic is handled internally.
    """
    return task_runner.run_task("cluster_inference", clusters=clusters, text=text)

def run_intra_cluster_edges(clusters: Dict[str, List[str]], text: str) -> List[Dict]:
    """
    Infer edges within clusters using the configured model.
    All model capability logic is handled internally.
    """
    return task_runner.run_task("intra_cluster_edges", clusters=clusters, text=text)

def run_single_edge_query(source: str, target: str, text: str) -> Dict:
    """
    Query a single edge relationship using the configured model with resilience.
    All model capability logic is handled internally.
    """
    config = task_runner.get_task_config("edge_inference")
    
    return resilient_single_edge_query(
        source=source,
        target=target,
        text=text,
        model=config.model,
        temperature=config.temperature,
        max_retries=3,
        enable_fallback=True
    )

def run_batch_concept_extraction(texts: List[str], n_prompts: int = None) -> List[List[str]]:
    """
    Extract concepts from multiple texts using the configured model.
    All model capability logic is handled internally.
    """
    return [run_concept_extraction(text, n_prompts) for text in texts]

# =============================================================================
# TASK CONFIGURATION UTILITIES
# =============================================================================

def get_task_model(task_name: str) -> str:
    """Get the model configured for a specific task."""
    config = task_runner.get_task_config(task_name)
    return config.model

def get_task_capabilities(task_name: str) -> Dict[str, Any]:
    """Get the capabilities of the model configured for a specific task.""" 
    model = get_task_model(task_name)
    return get_model_capabilities(model)

def override_task_model(task_name: str, model: str):
    """Temporarily override the model for a specific task."""
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}")
    
    original_model = TASK_CONFIGS[task_name].model
    TASK_CONFIGS[task_name].model = model
    
    print(f"Task '{task_name}' model changed: {original_model} → {model}")

def run_llm_clustering(concepts: List[str]) -> Dict[int, List[str]]:
    """
    Cluster concepts using LLM with the configured model.
    All model capability logic is handled internally.
    """
    return task_runner.run_task("clustering", concepts=concepts)

def run_cluster_naming(clusters: Dict[int, List[str]]) -> Dict[str, List[str]]:
    """
    Generate names for concept clusters using the configured model.
    All model capability logic is handled internally.
    """
    return task_runner.run_task("cluster_naming", clusters=clusters)

def list_available_tasks() -> List[str]:
    """List all available task names."""
    return list(TASK_CONFIGS.keys())

def get_task_summary() -> Dict[str, Dict[str, Any]]:
    """Get a summary of all configured tasks."""
    summary = {}
    
    for task_name, config in TASK_CONFIGS.items():
        capabilities = get_model_capabilities(config.model)
        summary[task_name] = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "reasoning": capabilities.get("reasoning", True),
            "provider": capabilities.get("provider", "unknown")
        }
    
    return summary

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def validate_task_configurations():
    """Validate that all task configurations are valid."""
    errors = []
    
    for task_name, config in TASK_CONFIGS.items():
        # Check that model exists in capabilities
        try:
            get_model_capabilities(config.model)
        except Exception as e:
            errors.append(f"Task '{task_name}': Invalid model '{config.model}' - {e}")
        
        # Check temperature range
        if not 0.0 <= config.temperature <= 2.0:
            errors.append(f"Task '{task_name}': Temperature {config.temperature} outside valid range 0.0-2.0")
        
        # Check max_tokens
        if config.max_tokens and config.max_tokens < 100:
            errors.append(f"Task '{task_name}': max_tokens {config.max_tokens} too small")
    
    if errors:
        raise ValueError("Task configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Example usage and testing
if __name__ == "__main__":
    # Test configuration validation
    validate_task_configurations()
    print("✅ All task configurations validated")
    
    # Show task summary
    print("\n=== TASK CONFIGURATION SUMMARY ===")
    summary = get_task_summary()
    for task_name, info in summary.items():
        print(f"{task_name:20} → {info['model']:15} ({info['reasoning'] and 'high' or 'low'}-reasoning)")
    
    # Example usage (would need actual implementation)
    print("\n=== EXAMPLE USAGE ===")
    print("# Business logic becomes much cleaner:")
    print("concepts = run_concept_extraction('Social isolation causes depression.')")
    print("edges = run_edge_inference([('isolation', 'depression')], text)")
    print("inter_edges, intra_edges = run_cluster_inference(clusters, text)")
    print("\n# No model parameters needed - all handled internally!")