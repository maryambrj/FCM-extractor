import os
import sys
import warnings

# Suppress numerical computation logs first
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.suppress_numba_logs import setup_clean_logging
setup_clean_logging()

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import hdbscan


from config.constants import CLUSTERING_EMBEDDING_MODEL, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES


def embed_concepts(concepts: List[str]) -> np.ndarray:
    """Embed concepts using SentenceTransformer."""
    model = SentenceTransformer(CLUSTERING_EMBEDDING_MODEL)
    return model.encode(concepts)

def tsne_reduce(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce embedding dimensionality using t-SNE."""
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
    return tsne.fit_transform(embeddings)

def hdbscan_cluster(reduced: np.ndarray) -> List[int]:
    """Cluster reduced embeddings using HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES if HDBSCAN_MIN_SAMPLES is not None else HDBSCAN_MIN_CLUSTER_SIZE
    )
    return clusterer.fit_predict(reduced)

def cluster_concepts(concepts: List[str]) -> Dict[int, List[str]]:
    """Cluster concepts into groups using embedding, t-SNE, and HDBSCAN.
    Cluster IDs are arbitrary integers assigned by HDBSCAN. -1 means 'noise' (unclustered concepts).
    """
    if len(concepts) < 4:
        # If we have very few concepts, just return them as separate clusters
        return {i: [concept] for i, concept in enumerate(concepts)}
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            embeddings = embed_concepts(concepts)
            reduced = tsne_reduce(embeddings)
            cluster_labels = hdbscan_cluster(reduced)
        clusters = {}
        for label, concept in zip(cluster_labels, concepts):
            clusters.setdefault(label, []).append(concept)
        
        # Handle noise points (-1 cluster)
        if -1 in clusters:
            noise_concepts = clusters.pop(-1)
            # Find the next available cluster ID
            max_cluster_id = max(clusters.keys()) if clusters else -1
            # Assign each noise concept to its own new cluster
            for i, concept in enumerate(noise_concepts):
                clusters[max_cluster_id + 1 + i] = [concept]
                
        # If all concepts are in cluster -1 (noise), treat each as its own cluster
        if not clusters:
            return {i: [concept] for i, concept in enumerate(concepts)}
        return clusters
    except Exception as e:
        print(f"Warning: Clustering failed with error: {e}")
        print("Falling back to treating each concept as its own cluster")
        return {i: [concept] for i, concept in enumerate(concepts)}

if __name__ == "__main__":
    test_concepts = ["social isolation", "poor sleep", "depression", "sleep problems"]
    clusters = cluster_concepts(test_concepts)
    print("Clusters:", clusters) 