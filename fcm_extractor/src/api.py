"""
Model-agnostic API for FCM extraction tasks.
This module provides clean entry points that hide all model selection and capability logic.

Usage:
    from src.api import run_concept_extraction, run_edge_inference, run_cluster_inference
    
    # Business logic becomes simple:
    concepts = run_concept_extraction(text)
    edges = run_edge_inference(concept_pairs, text)
    inter_edges, intra_edges = run_cluster_inference(clusters, text)
"""

from typing import List, Dict, Tuple, Optional, Any

# Import model-agnostic functions from task runner
from src.models.task_runner import (
    run_concept_extraction as _run_concept_extraction,
    run_edge_inference as _run_edge_inference, 
    run_cluster_inference as _run_cluster_inference,
    run_intra_cluster_edges as _run_intra_cluster_edges,
    run_single_edge_query as _run_single_edge_query,
    run_batch_concept_extraction as _run_batch_concept_extraction,
    run_llm_clustering,
    run_cluster_naming,
    get_task_model,
    get_task_capabilities,
    override_task_model,
    get_task_summary
)

# =============================================================================
# CONCEPT EXTRACTION API
# =============================================================================

def run_concept_extraction(text: str, n_prompts: Optional[int] = None) -> List[str]:
    """
    Extract concepts from text using the best configured model.
    
    Args:
        text: Text to extract concepts from
        n_prompts: Number of extraction prompts (uses config default if None)
    
    Returns:
        List of extracted concept strings
        
    Example:
        concepts = run_concept_extraction("Social isolation causes depression and anxiety")
        # Returns: ["social isolation", "depression", "anxiety"]
    """
    return _run_concept_extraction(text, n_prompts)

def run_batch_concept_extraction(texts: List[str], n_prompts: Optional[int] = None) -> List[List[str]]:
    """
    Extract concepts from multiple texts using the best configured model.
    
    Args:
        texts: List of texts to process
        n_prompts: Number of extraction prompts per text
    
    Returns:
        List of concept lists, one per input text
    """
    return _run_batch_concept_extraction(texts, n_prompts)

# =============================================================================
# EDGE INFERENCE API  
# =============================================================================

def run_edge_inference(concept_pairs: List[Tuple[str, str]], text: str) -> List[Dict]:
    """
    Infer causal edges between concept pairs using the best configured model.
    
    Args:
        concept_pairs: List of (source, target) concept pairs
        text: Text to analyze for relationships
    
    Returns:
        List of edge dictionaries with 'source', 'target', 'weight', 'confidence'
        
    Example:
        pairs = [("stress", "anxiety"), ("exercise", "mood")]
        edges = run_edge_inference(pairs, "Stress increases anxiety, while exercise improves mood")
        # Returns: [{"source": "stress", "target": "anxiety", "weight": 1.0, ...}, ...]
    """
    return _run_edge_inference(concept_pairs, text)

def run_single_edge_query(source: str, target: str, text: str) -> Dict:
    """
    Query a single edge relationship using the best configured model.
    
    Args:
        source: Source concept name
        target: Target concept name  
        text: Text to analyze
    
    Returns:
        Edge dictionary or empty edge if no relationship found
    """
    return _run_single_edge_query(source, target, text)

# =============================================================================
# CLUSTER INFERENCE API
# =============================================================================

def run_cluster_inference(clusters: Dict[str, List[str]], text: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Infer edges between and within clusters using the best configured model.
    
    Args:
        clusters: Dictionary mapping cluster names to concept lists
        text: Text to analyze for relationships
    
    Returns:
        Tuple of (inter_cluster_edges, intra_cluster_edges)
        
    Example:
        clusters = {"mental_health": ["anxiety", "depression"], "lifestyle": ["exercise", "diet"]}
        inter_edges, intra_edges = run_cluster_inference(clusters, text)
    """
    return _run_cluster_inference(clusters, text)

def run_intra_cluster_edges(clusters: Dict[str, List[str]], text: str) -> List[Dict]:
    """
    Infer edges within clusters only using the best configured model.
    
    Args:
        clusters: Dictionary mapping cluster names to concept lists
        text: Text to analyze
    
    Returns:
        List of intra-cluster edge dictionaries
    """
    return _run_intra_cluster_edges(clusters, text)

# =============================================================================
# CLUSTERING API
# =============================================================================

def run_clustering(concepts: List[str]) -> Dict[int, List[str]]:
    """
    Cluster concepts using LLM with the best configured model.
    
    Args:
        concepts: List of concept strings to cluster
    
    Returns:
        Dictionary mapping cluster IDs to concept lists
        
    Example:
        concepts = ["anxiety", "depression", "exercise", "diet"]
        clusters = run_clustering(concepts)
        # Returns: {0: ["anxiety", "depression"], 1: ["exercise", "diet"]}
    """
    return run_llm_clustering(concepts)

def run_cluster_naming(clusters: Dict[int, List[str]]) -> Dict[str, List[str]]:
    """
    Generate semantic names for concept clusters using the best configured model.
    
    Args:
        clusters: Dictionary mapping cluster IDs to concept lists
    
    Returns:
        Dictionary mapping cluster names to concept lists
        
    Example:
        clusters = {0: ["anxiety", "depression"], 1: ["exercise", "diet"]}
        named = run_cluster_naming(clusters)
        # Returns: {"Mental Health": ["anxiety", "depression"], "Lifestyle": ["exercise", "diet"]}
    """
    return run_cluster_naming(clusters)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_info(task: str) -> Dict[str, Any]:
    """
    Get information about the model configured for a task.
    
    Args:
        task: Task name (e.g., "concept_extraction", "edge_inference")
    
    Returns:
        Dictionary with model name, capabilities, etc.
    """
    model_name = get_task_model(task)
    capabilities = get_task_capabilities(task)
    
    return {
        "model": model_name,
        "capabilities": capabilities,
        "reasoning_level": "high" if capabilities.get("reasoning", True) else "low",
        "provider": capabilities.get("provider", "unknown"),
        "batch_size": capabilities.get("batch_size", 1),
        "context_length": capabilities.get("context_length", 32000)
    }

def get_all_task_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all configured tasks."""
    return get_task_summary()

def temporarily_use_model(task: str, model: str):
    """
    Temporarily override the model for a task.
    
    Args:
        task: Task name to override
        model: New model to use
        
    Example:
        temporarily_use_model("edge_inference", "o3-mini")
        # Now all edge inference will use o3-mini until program restart
    """
    override_task_model(task, model)

# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON PATTERNS
# =============================================================================

def extract_and_infer_edges(text: str, n_prompts: Optional[int] = None) -> Tuple[List[str], List[Dict]]:
    """
    Extract concepts and infer all possible edges in one call.
    
    Args:
        text: Text to analyze
        n_prompts: Number of concept extraction prompts
    
    Returns:
        Tuple of (concepts, edges)
    """
    concepts = run_concept_extraction(text, n_prompts)
    
    # Generate all possible pairs
    from itertools import combinations
    concept_pairs = list(combinations(concepts, 2))
    
    edges = run_edge_inference(concept_pairs, text)
    
    return concepts, edges

def analyze_single_document(text: str, enable_clustering: bool = True) -> Dict[str, Any]:
    """
    Complete analysis of a single document.
    
    Args:
        text: Document text to analyze
        enable_clustering: Whether to perform clustering analysis
    
    Returns:
        Complete analysis results
    """
    # Extract concepts
    concepts = run_concept_extraction(text)
    
    result = {
        "concepts": concepts,
        "concept_count": len(concepts)
    }
    
    if enable_clustering:
        # For clustering, we'd need to import and use clustering functions
        # This is a placeholder for the complete workflow
        result["clustering_enabled"] = True
    else:
        # Simple pairwise analysis
        from itertools import combinations
        concept_pairs = list(combinations(concepts, 2))
        edges = run_edge_inference(concept_pairs, text)
        
        result.update({
            "edges": edges,
            "edge_count": len(edges),
            "pairs_analyzed": len(concept_pairs)
        })
    
    return result

# =============================================================================
# DEBUG AND INTROSPECTION
# =============================================================================

def debug_task_execution(task: str, **kwargs) -> Dict[str, Any]:
    """
    Run a task with detailed debug information.
    
    Returns both the task result and execution metadata.
    """
    import time
    
    model_info = get_model_info(task)
    start_time = time.time()
    
    try:
        if task == "concept_extraction":
            result = run_concept_extraction(**kwargs)
        elif task == "edge_inference":
            result = run_edge_inference(**kwargs)
        elif task == "cluster_inference":
            result = run_cluster_inference(**kwargs)
        else:
            raise ValueError(f"Debug not supported for task: {task}")
        
        execution_time = time.time() - start_time
        success = True
        error = None
        
    except Exception as e:
        execution_time = time.time() - start_time
        result = None
        success = False
        error = str(e)
    
    return {
        "task": task,
        "model_info": model_info,
        "execution_time": execution_time,
        "success": success,
        "error": error,
        "result": result,
        "input_args": kwargs
    }

# Example usage
if __name__ == "__main__":
    print("=== FCM EXTRACTION API DEMO ===")
    
    sample_text = "Social isolation causes poor sleep quality, which leads to depression and anxiety. Regular exercise can reduce both stress and depression."
    
    print(f"\nAnalyzing text: {sample_text[:50]}...")
    
    # Extract concepts
    print("\n1. Extracting concepts...")
    concepts = run_concept_extraction(sample_text)
    print(f"Found {len(concepts)} concepts: {concepts}")
    
    # Analyze single relationship
    print("\n2. Analyzing single relationship...")
    edge = run_single_edge_query("social isolation", "depression", sample_text)
    print(f"Edge: {edge}")
    
    # Get model information
    print("\n3. Model information...")
    for task in ["concept_extraction", "edge_inference"]:
        info = get_model_info(task)
        print(f"{task}: {info['model']} ({info['reasoning_level']}-reasoning)")
    
    print("\n✅ API demonstration complete!")
    print("\nBusiness logic can now use simple calls:")
    print("  concepts = run_concept_extraction(text)")
    print("  edges = run_edge_inference(pairs, text)")
    print("  # No model parameters needed!")