#!/usr/bin/env python3
"""
Cluster metadata management for FCM extraction.

This module defines data structures and utilities for storing rich metadata
about concept clusters, including source contexts, embeddings, and summaries.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import numpy as np

@dataclass
class ConceptMetadata:
    """Metadata for a single concept."""
    concept: str
    source_contexts: List[str]  # Text snippets where this concept appeared
    chunk_indices: List[int]    # Which text chunks contained this concept
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        result = asdict(self)
        # Convert numpy array to list for JSON serialization
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        return result

@dataclass
class ClusterMetadata:
    """Rich metadata for a concept cluster."""
    id: str                                    # Unique cluster identifier
    name: str                                  # Human-readable name
    concepts: List[str]                        # Member concepts
    concept_metadata: Dict[str, ConceptMetadata] # Detailed concept info
    summary: Optional[str] = None              # LLM-generated description
    embedding_centroid: Optional[np.ndarray] = None # Mean embedding
    confidence: float = 0.0                    # Naming/clustering confidence
    size: int = 0                             # Number of concepts
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.size = len(self.concepts)
        
        # Calculate embedding centroid if concept embeddings available
        if self.concept_metadata:
            embeddings = [meta.embedding for meta in self.concept_metadata.values() 
                         if meta.embedding is not None]
            if embeddings:
                self.embedding_centroid = np.mean(embeddings, axis=0)
    
    def get_all_contexts(self) -> List[str]:
        """Get all source contexts for concepts in this cluster."""
        contexts = []
        for concept_meta in self.concept_metadata.values():
            contexts.extend(concept_meta.source_contexts)
        return contexts
    
    def get_concept_contexts(self, concept: str) -> List[str]:
        """Get source contexts for a specific concept."""
        if concept in self.concept_metadata:
            return self.concept_metadata[concept].source_contexts
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        result = asdict(self)
        
        # Convert numpy arrays to lists
        if self.embedding_centroid is not None:
            result['embedding_centroid'] = self.embedding_centroid.tolist()
        
        # Convert concept metadata
        result['concept_metadata'] = {
            concept: meta.to_dict() 
            for concept, meta in self.concept_metadata.items()
        }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterMetadata':
        """Create from dictionary (for JSON deserialization)."""
        # Convert embedding back to numpy array
        if data.get('embedding_centroid'):
            data['embedding_centroid'] = np.array(data['embedding_centroid'])
        
        # Convert concept metadata
        concept_metadata = {}
        for concept, meta_data in data.get('concept_metadata', {}).items():
            if meta_data.get('embedding'):
                meta_data['embedding'] = np.array(meta_data['embedding'])
            concept_metadata[concept] = ConceptMetadata(**meta_data)
        
        data['concept_metadata'] = concept_metadata
        return cls(**data)

class ClusterMetadataManager:
    """Manages cluster metadata throughout the FCM pipeline."""
    
    def __init__(self):
        self.clusters: Dict[str, ClusterMetadata] = {}
    
    def add_cluster(self, cluster_id: str, concepts: List[str], 
                   concept_metadata: Dict[str, ConceptMetadata],
                   name: str = None) -> ClusterMetadata:
        """Add a new cluster with metadata."""
        if name is None:
            name = f"Cluster {cluster_id}"
        
        cluster = ClusterMetadata(
            id=cluster_id,
            name=name,
            concepts=concepts,
            concept_metadata=concept_metadata
        )
        
        self.clusters[cluster_id] = cluster
        return cluster
    
    def get_cluster(self, cluster_id: str) -> Optional[ClusterMetadata]:
        """Get cluster metadata by ID."""
        return self.clusters.get(cluster_id)
    
    def get_all_clusters(self) -> Dict[str, ClusterMetadata]:
        """Get all cluster metadata."""
        return self.clusters
    
    def update_cluster_name(self, cluster_id: str, name: str, 
                           summary: str = None, confidence: float = 0.0):
        """Update cluster name and summary."""
        if cluster_id in self.clusters:
            self.clusters[cluster_id].name = name
            if summary:
                self.clusters[cluster_id].summary = summary
            self.clusters[cluster_id].confidence = confidence
    
    def get_cluster_contexts_for_edge_inference(self, cluster_a_id: str, 
                                               cluster_b_id: str) -> List[str]:
        """Get relevant contexts for edge inference between two clusters."""
        contexts = []
        
        if cluster_a_id in self.clusters:
            contexts.extend(self.clusters[cluster_a_id].get_all_contexts())
        
        if cluster_b_id in self.clusters:
            contexts.extend(self.clusters[cluster_b_id].get_all_contexts())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_contexts = []
        for context in contexts:
            if context not in seen:
                seen.add(context)
                unique_contexts.append(context)
        
        return unique_contexts
    
    def save_to_file(self, filepath: str):
        """Save cluster metadata to JSON file."""
        data = {
            cluster_id: cluster.to_dict() 
            for cluster_id, cluster in self.clusters.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str):
        """Load cluster metadata from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.clusters = {
            cluster_id: ClusterMetadata.from_dict(cluster_data)
            for cluster_id, cluster_data in data.items()
        }
    
    def to_simple_clusters(self) -> Dict[str, List[str]]:
        """Convert to simple cluster format for backward compatibility."""
        return {
            cluster.name: cluster.concepts 
            for cluster in self.clusters.values()
        } 