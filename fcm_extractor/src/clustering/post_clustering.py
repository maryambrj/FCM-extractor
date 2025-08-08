import numpy as np
from typing import Dict, List, Tuple, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import logging

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
    
    def __init__(self, 
                 similarity_threshold: float = POST_CLUSTERING_SIMILARITY_THRESHOLD,
                 embedding_model: str = POST_CLUSTERING_EMBEDDING_MODEL,
                 max_merges_per_cluster: int = POST_CLUSTERING_MAX_MERGES_PER_CLUSTER,
                 require_minimum_similarity: bool = POST_CLUSTERING_REQUIRE_MINIMUM_SIMILARITY):
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
        if self.embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
    def identify_unconnected_nodes(self, graph: nx.DiGraph, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        unconnected_clusters = {}
        
        for cluster_name, concepts in clusters.items():
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
        connected_clusters = {}
        
        for cluster_name, concepts in clusters.items():
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
        if not unconnected_clusters or not connected_clusters:
            return {}
        
        self._load_embedding_model()
        
        unconnected_texts = {}
        connected_texts = {}
        
        for cluster_name, concepts in unconnected_clusters.items():
            unconnected_texts[cluster_name] = " ".join(concepts)
        
        for cluster_name, concepts in connected_clusters.items():
            connected_texts[cluster_name] = " ".join(concepts)
        
        unconnected_embeddings = {}
        for cluster_name, text in unconnected_texts.items():
            embedding = self.embedding_model.encode([text])[0]
            unconnected_embeddings[cluster_name] = embedding
        
        connected_embeddings = {}
        for cluster_name, text in connected_texts.items():
            embedding = self.embedding_model.encode([text])[0]
            connected_embeddings[cluster_name] = embedding
        
        similarities = {}
        
        for unconnected_name, unconnected_emb in unconnected_embeddings.items():
            cluster_similarities = []
            
            for connected_name, connected_emb in connected_embeddings.items():
                similarity = cosine_similarity(
                    unconnected_emb.reshape(1, -1), 
                    connected_emb.reshape(1, -1)
                )[0, 0]
                
                cluster_similarities.append((connected_name, float(similarity)))
            
            cluster_similarities.sort(key=lambda x: x[1], reverse=True)
            similarities[unconnected_name] = cluster_similarities
            
            self.logger.debug(f"Similarities for {unconnected_name}: {cluster_similarities[:3]}")
        
        return similarities
    
    def merge_clusters(self, 
                      unconnected_clusters: Dict[str, List[str]],
                      connected_clusters: Dict[str, List[str]],
                      similarities: Dict[str, List[Tuple[str, float]]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        merged_clusters = connected_clusters.copy()
        merge_mapping = {}
        merge_count = 0
        
        merges_per_cluster = {cluster_name: 0 for cluster_name in connected_clusters.keys()}
        
        sorted_unconnected = []
        for unconnected_name, cluster_similarities in similarities.items():
            if cluster_similarities:
                best_similarity = cluster_similarities[0][1]
                sorted_unconnected.append((unconnected_name, best_similarity, cluster_similarities))
        
        sorted_unconnected.sort(key=lambda x: x[1], reverse=True)
        
        for unconnected_name, best_similarity, cluster_similarities in sorted_unconnected:
            merged = False
            
            for target_cluster_name, similarity in cluster_similarities:
                if (self.max_merges_per_cluster is not None and 
                    merges_per_cluster[target_cluster_name] >= self.max_merges_per_cluster):
                    continue
                
                threshold_met = similarity >= self.similarity_threshold
                if self.require_minimum_similarity and not threshold_met:
                    continue
                
                merged_clusters[target_cluster_name].extend(unconnected_clusters[unconnected_name])
                merge_mapping[unconnected_name] = target_cluster_name
                merges_per_cluster[target_cluster_name] += 1
                merge_count += 1
                merged = True
                
                self.logger.info(f"Merged '{unconnected_name}' into '{target_cluster_name}' (similarity: {similarity:.3f})")
                self.logger.debug(f"  Added concepts: {unconnected_clusters[unconnected_name]}")
                
                if self.max_merges_per_cluster is not None:
                    remaining = self.max_merges_per_cluster - merges_per_cluster[target_cluster_name]
                    self.logger.debug(f"  '{target_cluster_name}' can accept {remaining} more merges")
                
                break
            
            if not merged:
                merged_clusters[unconnected_name] = unconnected_clusters[unconnected_name]
                
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
        
        if self.max_merges_per_cluster is not None:
            self.logger.info(f"Merge distribution: {dict(merges_per_cluster)}")
        
        self.logger.info(f"Post-clustering complete: Merged {merge_count} unconnected clusters")
        return merged_clusters, merge_mapping
    
    def update_graph_after_merge(self, 
                                graph: nx.DiGraph, 
                                merge_mapping: Dict[str, str]) -> nx.DiGraph:
        updated_graph = graph.copy()
        
        for old_cluster_name in merge_mapping.keys():
            if updated_graph.has_node(old_cluster_name):
                updated_graph.remove_node(old_cluster_name)
                self.logger.debug(f"Removed node '{old_cluster_name}' from graph")
        
        return updated_graph
    
    def process_post_clustering(self, 
                               graph: nx.DiGraph, 
                               clusters: Dict[str, List[str]]) -> Tuple[nx.DiGraph, Dict[str, List[str]], Dict[str, str]]:
        if not ENABLE_POST_CLUSTERING:
            self.logger.info("Post-clustering is disabled in configuration")
            return graph, clusters, {}
        
        self.logger.info("Starting post-clustering process...")
        
        unconnected_clusters = self.identify_unconnected_nodes(graph, clusters)
        connected_clusters = self.identify_connected_clusters(graph, clusters)
        
        if not unconnected_clusters:
            self.logger.info("No unconnected clusters found - post-clustering not needed")
            return graph, clusters, {}
        
        if not connected_clusters:
            self.logger.info("No connected clusters found - cannot perform post-clustering")
            return graph, clusters, {}
        
        similarities = self.compute_cluster_similarities(unconnected_clusters, connected_clusters)
        
        merged_clusters, merge_mapping = self.merge_clusters(
            unconnected_clusters, connected_clusters, similarities
        )
        
        updated_graph = self.update_graph_after_merge(graph, merge_mapping)
        
        self.logger.info(f"Post-clustering completed: {len(clusters)} → {len(merged_clusters)} clusters")
        
        return updated_graph, merged_clusters, merge_mapping


def apply_post_clustering(graph: nx.DiGraph, 
                         clusters: Dict[str, List[str]]) -> Tuple[nx.DiGraph, Dict[str, List[str]], Dict[str, str]]:
    processor = PostClusteringProcessor()
    return processor.process_post_clustering(graph, clusters)