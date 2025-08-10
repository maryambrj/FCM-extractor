"""
Resilient edge inference functions that use the retry/fallback system.
These replace the existing edge inference functions with resilient versions.
"""

from typing import List, Dict, Tuple, Optional
from utils.llm_resilience import resilient_llm, LLMResult
from config.constants import EDGE_INFERENCE_MODEL, EDGE_INFERENCE_TEMPERATURE

def resilient_batch_edge_queries(
    concept_pairs: List[Tuple[str, str]], 
    text: str,
    model: str = EDGE_INFERENCE_MODEL,
    temperature: float = EDGE_INFERENCE_TEMPERATURE,
    max_retries: int = 3,
    enable_fallback: bool = True,
    **kwargs
) -> List[Dict]:
    """
    Execute batch edge queries with resilience and automatic fallbacks.
    
    Args:
        concept_pairs: List of (source, target) concept pairs
        text: Text to analyze
        model: Primary model to use
        temperature: Model temperature
        max_retries: Max retries per model
        enable_fallback: Enable model fallbacks on failure
        
    Returns:
        List of edge dictionaries with resilience metadata
    """
    
    # Format pairs for prompt
    pairs_text = ""
    for i, (c1, c2) in enumerate(concept_pairs):
        pairs_text += f"Pair {i+1}: '{c1}' and '{c2}'\\n"
    
    print(f"🔄 Resilient batch edge inference: {len(concept_pairs)} pairs with {model}")
    
    # Execute with resilience
    result = resilient_llm.execute_with_fallback(
        task="edge_inference_batch",
        primary_model=model,
        temperature=temperature,
        text=text,
        pairs=pairs_text.strip(),
        retry_config={
            "max_retries": max_retries,
            "enable_model_fallback": enable_fallback,
            "validation_required": True
        },
        **kwargs
    )
    
    # Process results
    edges = []
    if result.success and result.parsed_result:
        edges = result.parsed_result
        # Add resilience metadata
        for edge in edges:
            edge["resilience_metadata"] = {
                "model_used": result.model_used,
                "fallback_triggered": result.fallback_triggered,
                "attempts": len(result.attempts),
                "execution_time": result.total_time
            }
    else:
        print(f"❌ Resilient batch edge inference failed:")
        for attempt in result.attempts:
            print(f"  {attempt.model}: {attempt.failure_type} - {attempt.error_message}")
        
        # Return empty results with error metadata
        for i, (c1, c2) in enumerate(concept_pairs):
            edges.append({
                "source": c1,
                "target": c2,
                "weight": 0,
                "confidence": 0.0,
                "relationship": "unknown",
                "resilience_metadata": {
                    "failed": True,
                    "attempts": len(result.attempts),
                    "error": "All resilience attempts failed"
                }
            })
    
    return edges

def resilient_single_edge_query(
    source: str,
    target: str,
    text: str,
    model: str = EDGE_INFERENCE_MODEL,
    temperature: float = EDGE_INFERENCE_TEMPERATURE,
    max_retries: int = 3,
    enable_fallback: bool = True,
    **kwargs
) -> Dict:
    """
    Execute single edge query with resilience.
    """
    print(f"🔄 Resilient single edge query: {source} -> {target} with {model}")
    
    # Execute with resilience
    result = resilient_llm.execute_with_fallback(
        task="edge_inference_single",
        primary_model=model,
        temperature=temperature,
        text=text,
        concept1=source,
        concept2=target,
        retry_config={
            "max_retries": max_retries,
            "enable_model_fallback": enable_fallback,
            "validation_required": True
        },
        **kwargs
    )
    
    # Process result
    if result.success and result.parsed_result:
        edges = result.parsed_result
        if edges:
            edge = edges[0]
            edge["resilience_metadata"] = {
                "model_used": result.model_used,
                "fallback_triggered": result.fallback_triggered,
                "attempts": len(result.attempts),
                "execution_time": result.total_time
            }
            return edge
    
    # Return empty edge on failure
    return {
        "source": source,
        "target": target,
        "weight": 0,
        "confidence": 0.0,
        "relationship": "unknown",
        "resilience_metadata": {
            "failed": True,
            "attempts": len(result.attempts),
            "error": "All resilience attempts failed"
        }
    }

def resilient_cluster_edge_inference(
    clusters: Dict[str, List[str]],
    text: str,
    model: str = EDGE_INFERENCE_MODEL,
    temperature: float = EDGE_INFERENCE_TEMPERATURE,
    max_retries: int = 3,
    enable_fallback: bool = True,
    **kwargs
) -> Tuple[List[Dict], List[Dict]]:
    """
    Execute cluster edge inference (inter and intra) with resilience.
    
    Returns:
        Tuple of (inter_cluster_edges, intra_cluster_edges)
    """
    print(f"🔄 Resilient cluster edge inference: {len(clusters)} clusters with {model}")
    
    inter_edges = []
    intra_edges = []
    
    # Generate inter-cluster pairs
    cluster_names = list(clusters.keys())
    inter_pairs = []
    for i, cluster_a in enumerate(cluster_names):
        for j, cluster_b in enumerate(cluster_names):
            if i != j:
                inter_pairs.append((cluster_a, cluster_b))
    
    if inter_pairs:
        # Format inter-cluster pairs
        pairs_text = ""
        for i, (cluster_a, cluster_b) in enumerate(inter_pairs):
            concepts_a = ", ".join(clusters[cluster_a])
            concepts_b = ", ".join(clusters[cluster_b])
            pairs_text += f"Pair {i+1}: '{concepts_a}' and '{concepts_b}'\\n"
        
        # Execute inter-cluster inference
        result = resilient_llm.execute_with_fallback(
            task="inter_cluster_edges",
            primary_model=model,
            temperature=temperature,
            text=text,
            pairs=pairs_text.strip(),
            retry_config={
                "max_retries": max_retries,
                "enable_model_fallback": enable_fallback
            },
            **kwargs
        )
        
        if result.success and result.parsed_result:
            inter_edges = result.parsed_result
            for edge in inter_edges:
                edge["resilience_metadata"] = {
                    "model_used": result.model_used,
                    "fallback_triggered": result.fallback_triggered,
                    "attempts": len(result.attempts),
                    "execution_time": result.total_time,
                    "edge_type": "inter_cluster"
                }
    
    # Generate intra-cluster edges for each cluster
    for cluster_name, concepts in clusters.items():
        if len(concepts) < 2:
            continue
            
        # Generate concept pairs within cluster
        concept_pairs = []
        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts[i+1:], i+1):
                concept_pairs.append((concept_a, concept_b))
        
        if concept_pairs:
            # Format intra-cluster pairs
            pairs_text = ""
            for i, (concept_a, concept_b) in enumerate(concept_pairs):
                pairs_text += f"Pair {i+1}: '{concept_a}' and '{concept_b}'\\n"
            
            # Execute intra-cluster inference
            result = resilient_llm.execute_with_fallback(
                task="intra_cluster_edges",
                primary_model=model,
                temperature=temperature,
                text=text,
                pairs=pairs_text.strip(),
                cluster_name=cluster_name,
                retry_config={
                    "max_retries": max_retries,
                    "enable_model_fallback": enable_fallback
                },
                **kwargs
            )
            
            if result.success and result.parsed_result:
                cluster_edges = result.parsed_result
                for edge in cluster_edges:
                    edge["cluster"] = cluster_name
                    edge["resilience_metadata"] = {
                        "model_used": result.model_used,
                        "fallback_triggered": result.fallback_triggered,
                        "attempts": len(result.attempts),
                        "execution_time": result.total_time,
                        "edge_type": "intra_cluster"
                    }
                intra_edges.extend(cluster_edges)
    
    print(f"✅ Resilient cluster inference completed: {len(inter_edges)} inter, {len(intra_edges)} intra")
    return inter_edges, intra_edges

def resilient_concept_extraction(
    text: str,
    model: str = EDGE_INFERENCE_MODEL,
    temperature: float = EDGE_INFERENCE_TEMPERATURE,
    n_prompts: int = 1,
    max_retries: int = 3,
    enable_fallback: bool = True,
    **kwargs
) -> List[str]:
    """
    Execute concept extraction with resilience.
    """
    print(f"🔄 Resilient concept extraction with {model}")
    
    # Execute with resilience
    result = resilient_llm.execute_with_fallback(
        task="concept_extraction",
        primary_model=model,
        temperature=temperature,
        text=text,
        retry_config={
            "max_retries": max_retries,
            "enable_model_fallback": enable_fallback,
            "validation_required": False  # Concept extraction has simpler validation
        },
        **kwargs
    )
    
    concepts = []
    if result.success and result.content:
        # Parse concepts from response
        content = result.content.strip()
        
        # Try comma-separated format first
        if "," in content:
            concepts = [c.strip().strip('"').strip("'") for c in content.split(",")]
        # Try line-separated format
        elif "\\n" in content:
            concepts = [c.strip().strip('"').strip("'") for c in content.split("\\n") if c.strip()]
        else:
            # Single concept or space-separated
            concepts = [c.strip().strip('"').strip("'") for c in content.split() if len(c.strip()) > 2]
        
        # Filter out empty or too short concepts
        concepts = [c for c in concepts if c and len(c) > 1]
        
        print(f"✅ Extracted {len(concepts)} concepts with {result.model_used}")
        if result.fallback_triggered:
            print(f"  (Used fallback from original model)")
    else:
        print(f"❌ Concept extraction failed with all models")
        for attempt in result.attempts:
            print(f"  {attempt.model}: {attempt.failure_type} - {attempt.error_message}")
    
    return concepts

# =============================================================================\n# REPLACEMENT FUNCTIONS FOR EXISTING CODE\n# =============================================================================\n\ndef batch_llm_edge_queries_resilient(*args, **kwargs):\n    \"\"\"Drop-in replacement for batch_llm_edge_queries with resilience.\"\"\"\n    return resilient_batch_edge_queries(*args, **kwargs)\n\ndef single_llm_edge_query_resilient(*args, **kwargs):\n    \"\"\"Drop-in replacement for single LLM edge query with resilience.\"\"\"\n    return resilient_single_edge_query(*args, **kwargs)\n\ndef infer_edges_resilient(*args, **kwargs):\n    \"\"\"Drop-in replacement for infer_edges with resilience.\"\"\"\n    return resilient_cluster_edge_inference(*args, **kwargs)\n\ndef extract_concepts_resilient(*args, **kwargs):\n    \"\"\"Drop-in replacement for extract_concepts with resilience.\"\"\"\n    return resilient_concept_extraction(*args, **kwargs)\n\n# =============================================================================\n# TESTING\n# =============================================================================\n\ndef test_resilient_edge_inference():\n    \"\"\"Test resilient edge inference functions.\"\"\"\n    print(\"=== RESILIENT EDGE INFERENCE TEST ===\")\n    \n    text = \"Social isolation causes poor sleep quality, which leads to depression and anxiety. Regular exercise can reduce both stress and depression.\"\n    \n    # Test concept extraction\n    print(\"\\n1. Testing concept extraction...\")\n    concepts = resilient_concept_extraction(text, model=\"o3-mini\")\n    print(f\"   Concepts: {concepts}\")\n    \n    # Test single edge query\n    print(\"\\n2. Testing single edge query...\")\n    edge = resilient_single_edge_query(\"isolation\", \"depression\", text, model=\"o3-mini\")\n    print(f\"   Edge: {edge['relationship']} (confidence: {edge['confidence']})\")\n    \n    # Test batch edge queries\n    print(\"\\n3. Testing batch edge queries...\")\n    pairs = [(\"isolation\", \"depression\"), (\"exercise\", \"stress\")]\n    edges = resilient_batch_edge_queries(pairs, text, model=\"o3-mini\")\n    print(f\"   Edges: {len(edges)} found\")\n    \n    for edge in edges:\n        print(f\"     {edge['source']} -> {edge['target']}: {edge['relationship']} ({edge['confidence']})\")\n        if \"resilience_metadata\" in edge:\n            meta = edge[\"resilience_metadata\"]\n            print(f\"       Model: {meta.get('model_used', 'unknown')}, Fallback: {meta.get('fallback_triggered', False)}\")\n    \n    print(\"\\n✅ Resilient edge inference test completed\")\n\nif __name__ == \"__main__\":\n    test_resilient_edge_inference()"