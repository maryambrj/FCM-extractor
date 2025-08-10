"""
Efficient edge inference system with context reduction and structured output.
Combines pair-focused text extraction with structured response parsing.
"""

from typing import List, Dict, Tuple, Optional
from utils.text_extraction import PairFocusedExtractor, extract_focused_context
from utils.resilient_edge_inference import resilient_batch_edge_queries, resilient_single_edge_query
from src.models.edge_schemas_structured import parse_structured_edge_response
from config.constants import MAX_EDGE_INFERENCE_TEXT_LENGTH, EDGE_INFERENCE_MODEL, EDGE_INFERENCE_TEMPERATURE

class EfficientEdgeInference:
    """
    Efficient edge inference with automatic context reduction and structured parsing.
    """
    
    def __init__(
        self, 
        model: str = EDGE_INFERENCE_MODEL,
        temperature: float = EDGE_INFERENCE_TEMPERATURE,
        max_text_length: int = MAX_EDGE_INFERENCE_TEXT_LENGTH,
        context_sentences: int = 2
    ):
        self.model = model
        self.temperature = temperature
        self.text_extractor = PairFocusedExtractor(max_text_length)
        self.context_sentences = context_sentences
    
    def infer_edges_efficient(
        self, 
        concept_pairs: List[Tuple[str, str]], 
        text: str,
        **kwargs
    ) -> List[Dict]:
        """
        Efficient edge inference with automatic context reduction.
        
        Args:
            concept_pairs: List of (concept1, concept2) pairs
            text: Full text to analyze
            **kwargs: Additional arguments for resilient execution
            
        Returns:
            List of edge dictionaries with structured validation
        """
        print(f"🚀 Efficient edge inference: {len(concept_pairs)} pairs")
        print(f"   Original text: {len(text)} chars")
        
        # Step 1: Extract focused context
        focused_text = self.text_extractor.extract_pair_focused_text(
            text, concept_pairs, self.context_sentences
        )
        
        print(f"   Focused text: {len(focused_text)} chars ({100 * len(focused_text) / len(text):.1f}% of original)")
        
        # Step 2: Run resilient inference with focused text
        edges = resilient_batch_edge_queries(
            concept_pairs=concept_pairs,
            text=focused_text,
            model=self.model,
            temperature=self.temperature,
            validation_fn=self._structured_validation,
            **kwargs
        )
        
        # Step 3: Add efficiency metadata
        for edge in edges:
            edge["efficiency_metadata"] = {
                "original_text_length": len(text),
                "focused_text_length": len(focused_text),
                "context_reduction": 1 - (len(focused_text) / len(text)),
                "context_sentences": self.context_sentences
            }
        
        print(f"✅ Efficient inference completed: {len(edges)} edges found")
        return edges
    
    def infer_single_edge_efficient(
        self, 
        concept1: str, 
        concept2: str, 
        text: str,
        **kwargs
    ) -> Dict:
        """
        Efficient single edge inference with focused context extraction.
        """
        print(f"🔍 Efficient single edge: {concept1} -> {concept2}")
        print(f"   Original text: {len(text)} chars")
        
        # Extract focused context for single pair
        focused_text = self.text_extractor.extract_single_pair_text(
            text, concept1, concept2, context_sentences=3  # More generous for single pair
        )
        
        print(f"   Focused text: {len(focused_text)} chars")
        
        # Run resilient single query
        edge = resilient_single_edge_query(
            source=concept1,
            target=concept2,
            text=focused_text,
            model=self.model,
            temperature=self.temperature,
            validation_fn=self._structured_validation_single,
            **kwargs
        )
        
        # Add efficiency metadata
        edge["efficiency_metadata"] = {
            "original_text_length": len(text),
            "focused_text_length": len(focused_text),
            "context_reduction": 1 - (len(focused_text) / len(text))
        }
        
        return edge
    
    def _structured_validation(self, response: str) -> Tuple[bool, str, List[Dict]]:
        """
        Validation function for structured edge responses.
        
        Returns:
            Tuple of (is_valid, error_message, parsed_edges)
        """
        try:
            # This is a placeholder - we need expected_pairs from context
            # In practice, this would be passed through the resilient system
            success, edges, errors = parse_structured_edge_response(response, [])
            
            if success:
                return True, "", edges
            else:
                return False, "; ".join(errors), []
                
        except Exception as e:
            return False, f"Structured validation failed: {e}", []
    
    def _structured_validation_single(self, response: str) -> Tuple[bool, str, List[Dict]]:
        """Validation function for single edge structured responses."""
        return self._structured_validation(response)

# =============================================================================
# HIGH-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def efficient_batch_edge_inference(
    concept_pairs: List[Tuple[str, str]], 
    text: str,
    model: str = EDGE_INFERENCE_MODEL,
    temperature: float = EDGE_INFERENCE_TEMPERATURE,
    max_retries: int = 3,
    enable_fallback: bool = True,
    **kwargs
) -> List[Dict]:
    """
    High-level function for efficient batch edge inference.
    
    Combines context reduction, structured parsing, and resilient execution.
    """
    inferencer = EfficientEdgeInference(model, temperature)
    return inferencer.infer_edges_efficient(
        concept_pairs, 
        text, 
        max_retries=max_retries, 
        enable_fallback=enable_fallback,
        **kwargs
    )

def efficient_single_edge_query(
    concept1: str,
    concept2: str,
    text: str,
    model: str = EDGE_INFERENCE_MODEL,
    temperature: float = EDGE_INFERENCE_TEMPERATURE,
    max_retries: int = 3,
    enable_fallback: bool = True,
    **kwargs
) -> Dict:
    """
    High-level function for efficient single edge inference.
    
    Combines context reduction, structured parsing, and resilient execution.
    """
    inferencer = EfficientEdgeInference(model, temperature)
    return inferencer.infer_single_edge_efficient(
        concept1, 
        concept2, 
        text, 
        max_retries=max_retries, 
        enable_fallback=enable_fallback,
        **kwargs
    )

def efficient_cluster_edge_inference(
    clusters: Dict[str, List[str]],
    text: str,
    model: str = EDGE_INFERENCE_MODEL,
    temperature: float = EDGE_INFERENCE_TEMPERATURE,
    **kwargs
) -> Tuple[List[Dict], List[Dict]]:
    """
    Efficient cluster edge inference with context reduction.
    
    Returns:
        Tuple of (inter_cluster_edges, intra_cluster_edges)
    """
    inferencer = EfficientEdgeInference(model, temperature)
    
    print(f"🔄 Efficient cluster inference: {len(clusters)} clusters")
    
    # Generate inter-cluster pairs
    cluster_names = list(clusters.keys())
    inter_pairs = []
    for i, cluster_a in enumerate(cluster_names):
        for j, cluster_b in enumerate(cluster_names):
            if i != j:
                inter_pairs.append((cluster_a, cluster_b))
    
    # Process inter-cluster edges
    inter_edges = []
    if inter_pairs:
        # Create cluster-focused text extraction
        cluster_focused_text = extract_focused_context(text, inter_pairs)
        
        # Format pairs for display
        pairs_text = ""
        for i, (cluster_a, cluster_b) in enumerate(inter_pairs):
            concepts_a = ", ".join(clusters[cluster_a])
            concepts_b = ", ".join(clusters[cluster_b])
            pairs_text += f"Pair {i+1}: '{concepts_a}' and '{concepts_b}'\\n"
        
        print(f"   Inter-cluster: {len(inter_pairs)} pairs, {len(cluster_focused_text)} chars")
        
        # Use resilient inference with structured format
        inter_edges = resilient_batch_edge_queries(
            concept_pairs=inter_pairs,
            text=cluster_focused_text,
            model=model,
            temperature=temperature,
            **kwargs
        )
        
        # Add cluster metadata
        for edge in inter_edges:
            edge["edge_type"] = "inter_cluster"
    
    # Process intra-cluster edges
    intra_edges = []
    for cluster_name, concepts in clusters.items():
        if len(concepts) < 2:
            continue
            
        # Generate concept pairs within cluster
        concept_pairs = []
        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts[i+1:], i+1):
                concept_pairs.append((concept_a, concept_b))
        
        if concept_pairs:
            print(f"   Intra-cluster '{cluster_name}': {len(concept_pairs)} pairs")
            
            cluster_edges = inferencer.infer_edges_efficient(concept_pairs, text, **kwargs)
            
            # Add cluster metadata
            for edge in cluster_edges:
                edge["cluster"] = cluster_name
                edge["edge_type"] = "intra_cluster"
            
            intra_edges.extend(cluster_edges)
    
    print(f"✅ Efficient cluster inference: {len(inter_edges)} inter, {len(intra_edges)} intra")
    return inter_edges, intra_edges

# =============================================================================
# TESTING
# =============================================================================

def test_efficient_edge_inference():
    """Test the efficient edge inference system."""
    print("=== EFFICIENT EDGE INFERENCE TEST ===")
    
    # Sample data
    text = """
    Social isolation is a significant risk factor for depression and anxiety disorders. 
    Research has consistently shown that individuals who experience prolonged social isolation 
    are more likely to develop symptoms of depression. The lack of social connection can 
    create a cycle where depression further isolates individuals from their support networks.
    
    Regular physical exercise has been proven to be an effective intervention for managing 
    depression and anxiety. Exercise releases endorphins which improve mood and reduce stress 
    levels. Many mental health professionals now include exercise recommendations as part of 
    comprehensive treatment plans for patients dealing with anxiety and depression.
    
    Sleep quality plays a crucial role in mental health maintenance. Poor sleep patterns 
    can exacerbate symptoms of depression and make it more difficult for individuals to cope 
    with stress. Establishing healthy sleep habits is often recommended as a foundational 
    element in treating mood disorders.
    """
    
    concept_pairs = [
        ("social isolation", "depression"),
        ("exercise", "anxiety"),
        ("sleep quality", "stress"),
        ("depression", "social isolation")  # Test bidirectional
    ]
    
    print(f"Test data: {len(text)} chars, {len(concept_pairs)} pairs")
    
    try:
        # Test batch inference
        edges = efficient_batch_edge_inference(
            concept_pairs, 
            text,
            model="mock-model",  # Use mock for testing
            max_retries=1,
            enable_fallback=False
        )
        
        print(f"\\nResults: {len(edges)} edges")
        for edge in edges:
            print(f"  {edge['source']} -> {edge['target']}: {edge.get('relationship', 'unknown')}")
            if 'efficiency_metadata' in edge:
                meta = edge['efficiency_metadata']
                print(f"    Context reduction: {meta['context_reduction']:.1%}")
        
        # Test single edge
        single_edge = efficient_single_edge_query(
            "exercise", "depression", text,
            model="mock-model",
            max_retries=1,
            enable_fallback=False
        )
        
        print(f"\\nSingle edge test: {single_edge['source']} -> {single_edge['target']}")
        
        print("\\n✅ Efficient edge inference test completed")
        return True
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_efficient_edge_inference()