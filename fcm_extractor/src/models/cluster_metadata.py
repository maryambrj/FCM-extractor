from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import numpy as np

@dataclass
class ConceptMetadata:
    concept: str
    source_contexts: List[str]  
    chunk_indices: List[int]   
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        result = asdict(self)
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
        self.size = len(self.concepts)
        
        if self.concept_metadata:
            embeddings = [meta.embedding for meta in self.concept_metadata.values() 
                         if meta.embedding is not None]
            if embeddings:
                self.embedding_centroid = np.mean(embeddings, axis=0)
    
    def get_all_contexts(self) -> List[str]:
        contexts = []
        for concept_meta in self.concept_metadata.values():
            contexts.extend(concept_meta.source_contexts)
        return contexts
    
    def get_concept_contexts(self, concept: str) -> List[str]:
        if concept in self.concept_metadata:
            return self.concept_metadata[concept].source_contexts
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        
        if self.embedding_centroid is not None:
            result['embedding_centroid'] = self.embedding_centroid.tolist()
        
        result['concept_metadata'] = {
            concept: meta.to_dict() 
            for concept, meta in self.concept_metadata.items()
        }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterMetadata':
        if data.get('embedding_centroid'):
            data['embedding_centroid'] = np.array(data['embedding_centroid'])
        
        concept_metadata = {}
        for concept, meta_data in data.get('concept_metadata', {}).items():
            if meta_data.get('embedding'):
                meta_data['embedding'] = np.array(meta_data['embedding'])
            concept_metadata[concept] = ConceptMetadata(**meta_data)
        
        data['concept_metadata'] = concept_metadata
        return cls(**data)

class ClusterMetadataManager:
    
    def __init__(self):
        self.clusters: Dict[str, ClusterMetadata] = {}
    
    def add_cluster(self, cluster_id: str, concepts: List[str], 
                   concept_metadata: Dict[str, ConceptMetadata],
                   name: str = None) -> ClusterMetadata:
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
        return self.clusters.get(cluster_id)
    
    def get_all_clusters(self) -> Dict[str, ClusterMetadata]:
        return self.clusters
    
    def update_cluster_name(self, cluster_id: str, name: str, 
                           summary: str = None, confidence: float = 0.0):
        if cluster_id in self.clusters:
            self.clusters[cluster_id].name = name
            if summary:
                self.clusters[cluster_id].summary = summary
            self.clusters[cluster_id].confidence = confidence
    
    def get_cluster_contexts_for_edge_inference(self, cluster_a_id: str, 
                                               cluster_b_id: str) -> List[str]:
        contexts = []
        
        if cluster_a_id in self.clusters:
            contexts.extend(self.clusters[cluster_a_id].get_all_contexts())
        
        if cluster_b_id in self.clusters:
            contexts.extend(self.clusters[cluster_b_id].get_all_contexts())
        
        seen = set()
        unique_contexts = []
        for context in contexts:
            if context not in seen:
                seen.add(context)
                unique_contexts.append(context)
        
        return unique_contexts
    
    def save_to_file(self, filepath: str):
        data = {
            cluster_id: cluster.to_dict() 
            for cluster_id, cluster in self.clusters.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.clusters = {
            cluster_id: ClusterMetadata.from_dict(cluster_data)
            for cluster_id, cluster_data in data.items()
        }
    
    def to_simple_clusters(self) -> Dict[str, List[str]]:
        return {
            cluster.name: cluster.concepts 
            for cluster in self.clusters.values()
        } 