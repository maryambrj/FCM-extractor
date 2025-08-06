import json
import networkx as nx
from typing import Dict, List
import numpy as np

def build_fcm_graph(clusters: Dict[str, List[str]], inter_cluster_edges: List[Dict], intra_cluster_edges: List[Dict] = None) -> nx.DiGraph:
    """Build a NetworkX graph from clusters and edges with proper node labeling."""
    G = nx.DiGraph()
    
    # Add nodes for each cluster with rich metadata
    for cluster_name, concepts in clusters.items():
        # Create a proper concepts display
        concepts_str = ', '.join(concepts) if isinstance(concepts, list) else str(concepts)
        
        G.add_node(cluster_name, 
                  concepts=concepts_str,  # Store as string for JSON serialization
                  concept_list=concepts,  # Store as list for processing
                  type='cluster',
                  node_size=15 + len(concepts) * 3,  # Size based on number of concepts
                  label=cluster_name)
    
    # Add all concept nodes with cluster associations
    for cluster_name, concepts in clusters.items():
        for concept in concepts:
            if concept not in G:  # Only add if not already exists
                G.add_node(concept, 
                          concepts=concept,  # Concept is itself
                          type='concept',
                          cluster=cluster_name,  # Associate with cluster
                          node_size=10,
                          label=concept)
    
    # Add inter-cluster edges (between cluster nodes)
    for edge in inter_cluster_edges:
        source = edge.get('source')
        target = edge.get('target')
        weight = edge.get('weight', 0)
        confidence = edge.get('confidence', 0.8)
        edge_type = edge.get('type', 'inter_cluster')
        
        if source in G and target in G:
            G.add_edge(source, target, 
                      weight=weight, 
                      confidence=confidence,
                      type=edge_type,
                      style='solid')

    # Add intra-cluster edges (between concepts within clusters)
    if intra_cluster_edges:
        for edge in intra_cluster_edges:
            source = edge.get('source')
            target = edge.get('target')
            weight = edge.get('weight', 0)
            confidence = edge.get('confidence', 0.8)
            
            # Find which cluster these concepts belong to
            source_cluster = None
            target_cluster = None
            
            for cluster_name, concepts in clusters.items():
                if source in concepts:
                    source_cluster = cluster_name
                if target in concepts:
                    target_cluster = cluster_name
            
            # Ensure concept nodes exist with proper cluster association
            if source not in G:
                G.add_node(source, 
                          concepts=source,
                          type='concept',
                          cluster=source_cluster or 'unknown',
                          node_size=10,
                          label=source)
            
            if target not in G:
                G.add_node(target, 
                          concepts=target,
                          type='concept', 
                          cluster=target_cluster or 'unknown',
                          node_size=10,
                          label=target)
            
            # Add the intra-cluster edge
            G.add_edge(source, target, 
                      weight=weight, 
                      confidence=confidence,
                      type='intra_cluster',
                      cluster=source_cluster,
                      style='dashed')
            
    return G

def export_graph_to_json(G: nx.DiGraph, out_path: str):
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    data = {
        "nodes": [
            {"id": convert_numpy_types(n), "concepts": G.nodes[n]["concepts"]}
            for n in G.nodes if G.nodes[n].get("type") == "cluster"
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "weight": convert_numpy_types(d["weight"]),
                "confidence": convert_numpy_types(d.get("confidence", 1.0)),
                "type": d.get("type", "inter_cluster")  # Preserve edge type
            }
            for u, v, d in G.edges(data=True) 
            if d.get("type") in ["inter_cluster", "intra_cluster"] and "weight" in d
        ]
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    # Mock pipeline
    clusters = {0: ["social isolation"], 1: ["depression"]}
    edges = [
        {"source": "social isolation", "target": "depression", "weight": 1, "confidence": 0.92}
    ]
    G = build_fcm_graph(clusters, edges)
    export_graph_to_json(G, "fcm_output.json")
    print("Graph exported to fcm_output.json")

if __name__ == "__main__":
    main() 