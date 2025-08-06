"""
Post-clustering module for FCM extraction.

This module provides functionality to merge unconnected nodes (concepts or clusters 
without edges) with connected clusters based on semantic similarity. This helps 
ensure that synonym concepts are properly grouped together even if they were initially 
separated during the clustering phase or edge inference phase.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import logging

# Add parent directories to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import (
    POST_CLUSTERING_SIMILARITY_THRESHOLD,
    POST_CLUSTERING_EMBEDDING_MODEL,
    ENABLE_POST_CLUSTERING,
    POST_CLUSTERING_MAX_MERGES_PER_CLUSTER,
    POST_CLUSTERING_REQUIRE_MINIMUM_SIMILARITY
)
from src.models import get_global_prompt_agent, TaskType


class PostClusteringProcessor:
    """
    Handles post-clustering operations to merge unconnected nodes with connected clusters
    based on semantic similarity.
    """
    
    def __init__(self, 
                 similarity_threshold: float = POST_CLUSTERING_SIMILARITY_THRESHOLD,
                 embedding_model: str = POST_CLUSTERING_EMBEDDING_MODEL,
                 max_merges_per_cluster: int = POST_CLUSTERING_MAX_MERGES_PER_CLUSTER,
                 require_minimum_similarity: bool = POST_CLUSTERING_REQUIRE_MINIMUM_SIMILARITY):
        """
        Initialize the post-clustering processor.
        
        Args:
            similarity_threshold: Minimum cosine similarity to merge nodes (0.0-1.0)
            embedding_model: SentenceTransformer model name for computing embeddings
            max_merges_per_cluster: Maximum number of unconnected clusters to merge into each connected cluster (None = unlimited)
            require_minimum_similarity: Whether to require minimum similarity threshold for any merge
        """
        # Validate parameters
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}")
        
        if max_merges_per_cluster is not None and max_merges_per_cluster < 0:
            raise ValueError(f"max_merges_per_cluster must be non-negative or None, got {max_merges_per_cluster}")
        
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self.max_merges_per_cluster = max_merges_per_cluster
        self.require_minimum_similarity = require_minimum_similarity
        self.embedding_model = None
        self.logger = logging.getLogger(__name__)
        
    def _load_embedding_model(self):
        """Lazily load the embedding model when needed."""
        if self.embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
    def identify_unconnected_nodes(self, graph: nx.DiGraph, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Identify clusters that have no edges (neither incoming nor outgoing).
        
        Args:
            graph: NetworkX directed graph after edge inference
            clusters: Dictionary mapping cluster names to concept lists
            
        Returns:
            Dictionary of unconnected cluster names to their concepts
        """
        unconnected_clusters = {}
        
        for cluster_name, concepts in clusters.items():
            # Check if this cluster has any edges
            has_edges = (
                graph.has_node(cluster_name) and 
                (graph.in_degree(cluster_name) > 0 or graph.out_degree(cluster_name) > 0)
            )
            
            if not has_edges:
                unconnected_clusters[cluster_name] = concepts
                self.logger.debug(f"Found unconnected cluster: {cluster_name} with concepts: {concepts}")
        
        self.logger.info(f"Identified {len(unconnected_clusters)} unconnected clusters out of {len(clusters)} total clusters")
        return unconnected_clusters
    
    def identify_connected_clusters(self, graph: nx.DiGraph, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Identify clusters that have at least one edge (incoming or outgoing).
        
        Args:
            graph: NetworkX directed graph after edge inference
            clusters: Dictionary mapping cluster names to concept lists
            
        Returns:
            Dictionary of connected cluster names to their concepts
        """
        connected_clusters = {}
        
        for cluster_name, concepts in clusters.items():
            # Check if this cluster has any edges
            has_edges = (
                graph.has_node(cluster_name) and 
                (graph.in_degree(cluster_name) > 0 or graph.out_degree(cluster_name) > 0)
            )
            
            if has_edges:
                connected_clusters[cluster_name] = concepts
        
        self.logger.info(f"Identified {len(connected_clusters)} connected clusters")
        return connected_clusters
    
    def compute_cluster_similarities(self, 
                                   unconnected_clusters: Dict[str, List[str]],
                                   connected_clusters: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute semantic similarities between unconnected and connected clusters.
        
        Args:
            unconnected_clusters: Dictionary of unconnected cluster names to concepts
            connected_clusters: Dictionary of connected cluster names to concepts
            
        Returns:
            Dictionary mapping unconnected cluster names to list of (connected_cluster_name, similarity_score) tuples,
            sorted by similarity score in descending order
        """
        if not unconnected_clusters or not connected_clusters:
            return {}
        
        self._load_embedding_model()
        
        # Prepare text representations for embedding
        unconnected_texts = {}
        connected_texts = {}
        
        # Create text representations by joining concepts
        for cluster_name, concepts in unconnected_clusters.items():
            unconnected_texts[cluster_name] = " ".join(concepts)
        
        for cluster_name, concepts in connected_clusters.items():
            connected_texts[cluster_name] = " ".join(concepts)
        
        # Compute embeddings
        unconnected_embeddings = {}
        for cluster_name, text in unconnected_texts.items():
            embedding = self.embedding_model.encode([text])[0]
            unconnected_embeddings[cluster_name] = embedding
        
        connected_embeddings = {}
        for cluster_name, text in connected_texts.items():
            embedding = self.embedding_model.encode([text])[0]
            connected_embeddings[cluster_name] = embedding
        
        # Compute pairwise similarities
        similarities = {}
        
        for unconnected_name, unconnected_emb in unconnected_embeddings.items():
            cluster_similarities = []
            
            for connected_name, connected_emb in connected_embeddings.items():
                # Compute cosine similarity
                similarity = cosine_similarity(
                    unconnected_emb.reshape(1, -1), 
                    connected_emb.reshape(1, -1)
                )[0, 0]
                
                cluster_similarities.append((connected_name, float(similarity)))
            
            # Sort by similarity score (descending)
            cluster_similarities.sort(key=lambda x: x[1], reverse=True)
            similarities[unconnected_name] = cluster_similarities
            
            self.logger.debug(f"Similarities for {unconnected_name}: {cluster_similarities[:3]}")  # Log top 3
        
        return similarities
    
    def merge_clusters(self, 
                      unconnected_clusters: Dict[str, List[str]],
                      connected_clusters: Dict[str, List[str]],
                      similarities: Dict[str, List[Tuple[str, float]]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Merge unconnected clusters with their most similar connected clusters if similarity exceeds threshold.
        
        Args:
            unconnected_clusters: Dictionary of unconnected cluster names to concepts
            connected_clusters: Dictionary of connected cluster names to concepts
            similarities: Dictionary of similarity scores from compute_cluster_similarities
            
        Returns:
            Tuple of:
            - Updated clusters dictionary (merged clusters)
            - Mapping of old unconnected cluster names to their new merged cluster names
        """
        merged_clusters = connected_clusters.copy()
        merge_mapping = {}
        merge_count = 0
        
        # Track merges per connected cluster to enforce limits
        merges_per_cluster = {cluster_name: 0 for cluster_name in connected_clusters.keys()}
        
        # Sort unconnected clusters by their best similarity score (highest first) for fairer allocation
        sorted_unconnected = []
        for unconnected_name, cluster_similarities in similarities.items():
            if cluster_similarities:
                best_similarity = cluster_similarities[0][1]
                sorted_unconnected.append((unconnected_name, best_similarity, cluster_similarities))
        
        # Sort by best similarity descending
        sorted_unconnected.sort(key=lambda x: x[1], reverse=True)
        
        for unconnected_name, best_similarity, cluster_similarities in sorted_unconnected:
            merged = False
            
            # Try to find the best available connected cluster considering merge limits
            for target_cluster_name, similarity in cluster_similarities:
                # Check if this target cluster has reached its merge limit
                if (self.max_merges_per_cluster is not None and 
                    merges_per_cluster[target_cluster_name] >= self.max_merges_per_cluster):
                    continue  # Try next best option
                
                # Check similarity threshold
                threshold_met = similarity >= self.similarity_threshold
                if self.require_minimum_similarity and not threshold_met:
                    continue  # Skip if threshold not met and we require it
                
                # Perform the merge
                merged_clusters[target_cluster_name].extend(unconnected_clusters[unconnected_name])
                merge_mapping[unconnected_name] = target_cluster_name
                merges_per_cluster[target_cluster_name] += 1
                merge_count += 1
                merged = True
                
                self.logger.info(f"Merged '{unconnected_name}' into '{target_cluster_name}' (similarity: {similarity:.3f})")
                self.logger.debug(f"  Added concepts: {unconnected_clusters[unconnected_name]}")
                
                # Log merge limits if applicable
                if self.max_merges_per_cluster is not None:
                    remaining = self.max_merges_per_cluster - merges_per_cluster[target_cluster_name]
                    self.logger.debug(f"  '{target_cluster_name}' can accept {remaining} more merges")
                
                break  # Successfully merged, move to next unconnected cluster
            
            if not merged:
                # Keep the unconnected cluster as is
                merged_clusters[unconnected_name] = unconnected_clusters[unconnected_name]
                
                # Log why it wasn't merged
                if not cluster_similarities:
                    reason = "no similarity data"
                elif self.max_merges_per_cluster is not None:
                    all_targets_full = all(merges_per_cluster[target] >= self.max_merges_per_cluster 
                                         for target, _ in cluster_similarities)
                    if all_targets_full:
                        reason = f"all targets reached merge limit ({self.max_merges_per_cluster})"
                    else:
                        reason = f"best similarity {best_similarity:.3f} < threshold {self.similarity_threshold}"
                else:
                    reason = f"best similarity {best_similarity:.3f} < threshold {self.similarity_threshold}"
                
                self.logger.info(f"Kept '{unconnected_name}' separate ({reason})")
        
        # Log final merge statistics
        if self.max_merges_per_cluster is not None:
            self.logger.info(f"Merge distribution: {dict(merges_per_cluster)}")
        
        self.logger.info(f"Post-clustering complete: Merged {merge_count} unconnected clusters")
        return merged_clusters, merge_mapping
    
    def update_graph_after_merge(self, 
                                graph: nx.DiGraph, 
                                merge_mapping: Dict[str, str]) -> nx.DiGraph:
        """
        Update the graph structure after merging clusters by removing merged cluster nodes.
        
        Args:
            graph: Original NetworkX directed graph
            merge_mapping: Dictionary mapping old unconnected cluster names to new merged cluster names
            
        Returns:
            Updated NetworkX directed graph
        """
        updated_graph = graph.copy()
        
        # Remove nodes for merged clusters (they had no edges anyway)
        for old_cluster_name in merge_mapping.keys():
            if updated_graph.has_node(old_cluster_name):
                updated_graph.remove_node(old_cluster_name)
                self.logger.debug(f"Removed node '{old_cluster_name}' from graph")
        
        return updated_graph
    
    def process_post_clustering(self, 
                               graph: nx.DiGraph, 
                               clusters: Dict[str, List[str]]) -> Tuple[nx.DiGraph, Dict[str, List[str]], Dict[str, str]]:
        """
        Main post-clustering processing function.
        
        Args:
            graph: NetworkX directed graph after edge inference
            clusters: Dictionary mapping cluster names to concept lists
            
        Returns:
            Tuple of:
            - Updated graph after post-clustering
            - Updated clusters dictionary after merging
            - Mapping of old cluster names to new cluster names (for tracking changes)
        """
        if not ENABLE_POST_CLUSTERING:
            self.logger.info("Post-clustering is disabled in configuration")
            return graph, clusters, {}
        
        self.logger.info("Starting post-clustering process...")
        
        # Step 1: Identify unconnected and connected clusters
        unconnected_clusters = self.identify_unconnected_nodes(graph, clusters)
        connected_clusters = self.identify_connected_clusters(graph, clusters)
        
        if not unconnected_clusters:
            self.logger.info("No unconnected clusters found - post-clustering not needed")
            return graph, clusters, {}
        
        if not connected_clusters:
            self.logger.info("No connected clusters found - cannot perform post-clustering")
            return graph, clusters, {}
        
        # Step 2: Compute similarities
        similarities = self.compute_cluster_similarities(unconnected_clusters, connected_clusters)
        
        # Step 3: Merge clusters based on similarity threshold
        merged_clusters, merge_mapping = self.merge_clusters(
            unconnected_clusters, connected_clusters, similarities
        )
        
        # Step 4: Update graph structure
        updated_graph = self.update_graph_after_merge(graph, merge_mapping)
        
        self.logger.info(f"Post-clustering completed: {len(clusters)} → {len(merged_clusters)} clusters")
        
        return updated_graph, merged_clusters, merge_mapping


def apply_post_clustering(graph: nx.DiGraph, 
                         clusters: Dict[str, List[str]]) -> Tuple[nx.DiGraph, Dict[str, List[str]], Dict[str, str]]:
    """
    Convenience function to apply post-clustering with default settings.
    
    Args:
        graph: NetworkX directed graph after edge inference
        clusters: Dictionary mapping cluster names to concept lists
        
    Returns:
        Tuple of (updated_graph, updated_clusters, merge_mapping)
    """
    processor = PostClusteringProcessor()
    return processor.process_post_clustering(graph, clusters)