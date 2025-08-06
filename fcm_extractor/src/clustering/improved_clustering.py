"""
Improved clustering methods for concept grouping.
Includes multiple approaches: better embeddings, LLM-based clustering, and hybrid methods.
"""

import os
import sys
import warnings

# Suppress numerical computation logs first
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.suppress_numba_logs import setup_clean_logging
setup_clean_logging()

import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
import warnings
import re
import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import (
    CLUSTERING_EMBEDDING_MODEL, CLUSTERING_METHOD, DIMENSIONALITY_REDUCTION, 
    CLUSTERING_ALGORITHM, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_COMPONENTS, 
    USE_LLM_CLUSTERING, LLM_CLUSTERING_MODEL, CLUSTER_NAMING_BATCH_SIZE,
    AGGLOMERATIVE_MAX_CLUSTERS, AGGLOMERATIVE_USE_ELBOW_METHOD,
    AGGLOMERATIVE_USE_DISTANCE_THRESHOLD, AGGLOMERATIVE_DISTANCE_THRESHOLD,
    TSNE_PERPLEXITY, TSNE_EARLY_EXAGGERATION, TSNE_LEARNING_RATE, TSNE_N_ITER
)
from src.models.llm_client import llm_client
from src.models.cluster_metadata import ClusterMetadata, ConceptMetadata, ClusterMetadataManager


def name_cluster(concepts: List[str], model: str) -> str:
    if not concepts:
        return "Unnamed Cluster"
    if len(concepts) == 1:
        return concepts[0].lstrip('*i.v. ').rstrip(':**').strip()
    concepts_str = ", ".join([c.lstrip('*i.v. ').rstrip(':**').strip() for c in concepts])
    prompt = f"""Generate a short, unique, and descriptive name (max 4 words) for this cluster of concepts. Do not include numbers, impacts (e.g., -2), markdown, or any extra text. Output ONLY the name.\nExamples:\n- Concepts: shrimp fishery, pleasure boating → Shrimp Fishery Impacts\n- Concepts: loneliness, social isolation → Social Isolation\n\nConcepts: {concepts_str}\nName:"""
    messages = [
        {"role": "system", "content": "You are an expert in creating concise, unique cluster names."},
        {"role": "user", "content": prompt}
    ]
    name, _ = llm_client.chat_completion(model, messages, temperature=0.2)
    name = name.strip().replace('"', '').lstrip('*i.v. ').rstrip(':**').strip()
    name = re.sub(r'\(\d+\)|-?\d+|\*\*', '', name).strip()
    if len(name.split()) > 4 or len(name) > 30 or name.lower().startswith('here are'):
        return "Unnamed Cluster" # Return a default name if it's invalid
    return name

def name_clusters_batch(clusters_list: List[Tuple[int, List[str]]], model: str) -> List[str]:
    """Name multiple clusters in a single API call for efficiency."""
    if not clusters_list:
        return []
    
    # If only one cluster, use the original function
    if len(clusters_list) == 1:
        return [name_cluster(clusters_list[0][1], model)]
    
    # Build batch prompt
    cluster_descriptions = []
    for i, (cluster_id, concepts) in enumerate(clusters_list):
        if not concepts:
            cluster_descriptions.append(f"Cluster {i+1}: (empty)")
            continue
        concepts_str = ", ".join([c.lstrip('*i.v. ').rstrip(':**').strip() for c in concepts])
        cluster_descriptions.append(f"Cluster {i+1}: {concepts_str}")
    
    batch_prompt = f"""Generate short, unique, and descriptive names (max 4 words each) for these clusters of concepts. 
Do not include numbers, impacts (e.g., -2), markdown, or any extra text. 
Return ONLY the names, one per line, in the same order as the clusters.

Examples:
- Concepts: shrimp fishery, pleasure boating → Shrimp Fishery Impacts
- Concepts: loneliness, social isolation → Social Isolation

{chr(10).join(cluster_descriptions)}

Names:"""

    messages = [
        {"role": "system", "content": "You are an expert in creating concise, unique cluster names."},
        {"role": "user", "content": batch_prompt}
    ]
    
    try:
        response, _ = llm_client.chat_completion(model, messages, temperature=0.2)
        names = response.strip().split('\n')
        
        # Clean up names
        cleaned_names = []
        for name in names:
            name = name.strip().replace('"', '').lstrip('*i.v. ').rstrip(':**').strip()
            name = re.sub(r'\(\d+\)|-?\d+|\*\*|^\d+\.\s*', '', name).strip()
            if len(name.split()) > 4 or len(name) > 30 or name.lower().startswith('here are') or not name:
                name = "Unnamed Cluster"
            cleaned_names.append(name)
        
        # Ensure we have the right number of names
        while len(cleaned_names) < len(clusters_list):
            cleaned_names.append("Unnamed Cluster")
        
        return cleaned_names[:len(clusters_list)]
        
    except Exception as e:
        print(f"Warning: Batch cluster naming failed: {e}")
        # Fallback to individual naming
        return [name_cluster(concepts, model) for _, concepts in clusters_list]

def name_all_clusters(clusters: Dict[int, List[str]], model: str = LLM_CLUSTERING_MODEL) -> Dict[str, List[str]]:
    """Name all clusters efficiently using batch processing."""
    if not clusters:
        return {}
    
    # Convert to list for batch processing
    clusters_list = [(cluster_id, concepts) for cluster_id, concepts in clusters.items() if concepts]
    
    if not clusters_list:
        return {}
    
    # Process in batches of 10 clusters per API call
    batch_size = 10
    all_names = []
    
    print(f"Naming {len(clusters_list)} clusters in batches of {batch_size}...")
    
    for i in range(0, len(clusters_list), batch_size):
        batch = clusters_list[i:i + batch_size]
        batch_names = name_clusters_batch(batch, model)
        all_names.extend(batch_names)
        print(f"  Named batch {i//batch_size + 1}/{(len(clusters_list) - 1)//batch_size + 1}")
    
    # Build final result with duplicate handling
    named_clusters = {}
    for (cluster_id, concepts), cluster_name in zip(clusters_list, all_names):
        # Handle duplicate names
        original_name = cluster_name
        counter = 2
        while cluster_name in named_clusters:
            cluster_name = f"{original_name} ({counter})"
            counter += 1
        named_clusters[cluster_name] = concepts
    
    return named_clusters


def embed_concepts_improved(concepts: List[str], model_name: str = CLUSTERING_EMBEDDING_MODEL) -> np.ndarray:
    """
    Embed concepts using improved embedding models.
    
    Args:
        concepts: List of concept strings
        model_name: Name of the embedding model to use
        
    Returns:
        Numpy array of embeddings
    """
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(concepts, show_progress_bar=False)
        return embeddings
    except Exception as e:
        print(f"Warning: Failed to load model {model_name}, falling back to all-mpnet-base-v2")
        model = SentenceTransformer("all-mpnet-base-v2")
        return model.encode(concepts, show_progress_bar=False)


def reduce_dimensions(embeddings: np.ndarray, method: str = DIMENSIONALITY_REDUCTION, 
                     n_components: int = UMAP_N_COMPONENTS) -> np.ndarray:
    """
    Reduce embedding dimensionality using various methods.
    
    Args:
        embeddings: Input embeddings
        method: Reduction method ("umap", "tsne", "pca", "none")
        n_components: Number of dimensions to reduce to
        
    Returns:
        Reduced embeddings
    """
    if method == "none" or len(embeddings) < 4:
        return embeddings
        
    try:
        if method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(UMAP_N_NEIGHBORS, len(embeddings) - 1),
                min_dist=UMAP_MIN_DIST,
                random_state=42
            )
            return reducer.fit_transform(embeddings)
            
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(TSNE_PERPLEXITY, len(embeddings) - 1),
                early_exaggeration=TSNE_EARLY_EXAGGERATION,
                learning_rate=TSNE_LEARNING_RATE,
                n_iter=TSNE_N_ITER
            )
            return reducer.fit_transform(embeddings)
            
        elif method == "pca":
            reducer = PCA(n_components=min(n_components, len(embeddings) - 1))
            return reducer.fit_transform(embeddings)
            
        else:
            print(f"Unknown reduction method: {method}, using UMAP")
            return reduce_dimensions(embeddings, "umap", n_components)
            
    except Exception as e:
        print(f"Warning: Dimensionality reduction failed: {e}")
        return embeddings


def cluster_embeddings(embeddings: np.ndarray, algorithm: str = CLUSTERING_ALGORITHM) -> List[int]:
    """
    Cluster embeddings using various algorithms.
    
    Args:
        embeddings: Input embeddings
        algorithm: Clustering algorithm ("hdbscan", "kmeans", "agglomerative", "spectral")
        
    Returns:
        Cluster labels
    """
    try:
        if algorithm == "hdbscan":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES if HDBSCAN_MIN_SAMPLES is not None else HDBSCAN_MIN_CLUSTER_SIZE
            )
            return clusterer.fit_predict(embeddings)
            
        elif algorithm == "kmeans":
            # Estimate number of clusters using elbow method
            n_clusters = estimate_optimal_clusters(embeddings)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            return clusterer.fit_predict(embeddings)
            
        elif algorithm == "agglomerative":
            if AGGLOMERATIVE_USE_DISTANCE_THRESHOLD:
                # Use distance threshold instead of fixed number of clusters
                clusterer = AgglomerativeClustering(
                    distance_threshold=AGGLOMERATIVE_DISTANCE_THRESHOLD,
                    n_clusters=None  # Must be None when using distance_threshold
                )
                print(f"  Using distance threshold: {AGGLOMERATIVE_DISTANCE_THRESHOLD}")
            elif AGGLOMERATIVE_USE_ELBOW_METHOD:
                n_clusters = estimate_optimal_clusters(embeddings, max_clusters=AGGLOMERATIVE_MAX_CLUSTERS)
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                print(f"  Using elbow method: {n_clusters} clusters")
            else:
                n_clusters = AGGLOMERATIVE_MAX_CLUSTERS
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                print(f"  Using fixed number: {n_clusters} clusters")
            return clusterer.fit_predict(embeddings)
            
        elif algorithm == "spectral":
            n_clusters = estimate_optimal_clusters(embeddings)
            clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)
            return clusterer.fit_predict(embeddings)
            
        else:
            print(f"Unknown clustering algorithm: {algorithm}, using HDBSCAN")
            return cluster_embeddings(embeddings, "hdbscan")
            
    except Exception as e:
        print(f"Warning: Clustering failed: {e}")
        # Fallback to simple clustering
        return list(range(len(embeddings)))


def estimate_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10) -> int:
    """
    Estimate optimal number of clusters using elbow method.
    
    Args:
        embeddings: Input embeddings
        max_clusters: Maximum number of clusters to consider
        
    Returns:
        Estimated optimal number of clusters
    """
    n_samples = len(embeddings)
    max_clusters = min(max_clusters, n_samples // 2, 8)  # Reasonable upper bound
    
    if max_clusters < 2:
        return max(2, n_samples // 3)
    
    inertias = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        except:
            break
    
    if len(inertias) < 2:
        return max(2, n_samples // 3)
    
    # Simple elbow detection
    diffs = np.diff(inertias)
    if len(diffs) > 1:
        second_diffs = np.diff(diffs)
        elbow_idx = np.argmax(second_diffs) + 2
        return k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[0]
    
    return k_range[0]


def llm_based_clustering(concepts: List[str], model: str = LLM_CLUSTERING_MODEL) -> Dict[int, List[str]]:
    """
    Use LLM to semantically cluster concepts.
    
    Args:
        concepts: List of concept strings
        model: LLM model to use for clustering
        
    Returns:
        Dictionary mapping cluster IDs to concept lists
    """
    if len(concepts) <= 3:
        return {i: [concept] for i, concept in enumerate(concepts)}
    
    # Create prompt for LLM clustering
    concepts_str = "\n".join([f"{i+1}. {concept}" for i, concept in enumerate(concepts)])
    
    prompt = f"""Group the following concepts into semantically related clusters. Concepts in the same cluster should be closely related in meaning or domain.

Concepts:
{concepts_str}

Instructions:
1. Create clusters where concepts are semantically similar
2. Each concept should belong to exactly one cluster
3. Aim for 2-{len(concepts)//2} clusters
4. Provide output as: Cluster X: concept_number, concept_number, ...

Example format:
Cluster 1: 1, 3, 5
Cluster 2: 2, 4
Cluster 3: 6, 7, 8

Your clustering:"""

    try:
        messages = [
            {"role": "system", "content": "You are an expert at semantic clustering. Group concepts by their meaning and relationships."},
            {"role": "user", "content": prompt}
        ]
        
        content, _ = llm_client.chat_completion(model, messages, temperature=0.1)
        
        # Parse the response
        clusters = {}
        cluster_id = 0
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('Cluster') and ':' in line:
                try:
                    # Extract concept numbers
                    numbers_part = line.split(':', 1)[1].strip()
                    concept_indices = [int(x.strip()) - 1 for x in numbers_part.split(',') if x.strip().isdigit()]
                    
                    if concept_indices:
                        clusters[cluster_id] = [concepts[i] for i in concept_indices if 0 <= i < len(concepts)]
                        cluster_id += 1
                except:
                    continue
        
        # Ensure all concepts are assigned
        assigned_concepts = set()
        for cluster_concepts in clusters.values():
            assigned_concepts.update(cluster_concepts)
        
        unassigned = [c for c in concepts if c not in assigned_concepts]
        for concept in unassigned:
            clusters[cluster_id] = [concept]
            cluster_id += 1
        
        return clusters
        
    except Exception as e:
        print(f"Warning: LLM clustering failed: {e}")
        return {i: [concept] for i, concept in enumerate(concepts)}

def batch_llm_cluster_refinement(clusters_to_refine: List[Tuple[int, List[str]]], model: str = LLM_CLUSTERING_MODEL) -> Dict[int, Dict[int, List[str]]]:
    """
    Refine multiple clusters in a single API call for efficiency.
    
    Args:
        clusters_to_refine: List of (original_cluster_id, concepts) tuples
        model: LLM model to use
        
    Returns:
        Dictionary mapping original cluster IDs to their refined sub-clusters
    """
    if not clusters_to_refine:
        return {}
    
    # If only one cluster, use the individual function
    if len(clusters_to_refine) == 1:
        original_id, concepts = clusters_to_refine[0]
        refined = llm_based_clustering(concepts, model)
        return {original_id: refined}
    
    # Build batch prompt for multiple clusters
    cluster_descriptions = []
    for i, (cluster_id, concepts) in enumerate(clusters_to_refine):
        concepts_str = "\n".join([f"  {j+1}. {concept}" for j, concept in enumerate(concepts)])
        cluster_descriptions.append(f"Group {i+1} ({len(concepts)} concepts):\n{concepts_str}")
    
    batch_prompt = f"""Refine the following concept groups by splitting them into more focused sub-clusters. Each group should be split into 2-4 semantically related sub-clusters.

{chr(10).join(cluster_descriptions)}

Instructions:
1. For each group, create 2-4 focused sub-clusters
2. Each concept should belong to exactly one sub-cluster within its group
3. Provide output as: Group X.Y: concept_number, concept_number, ...

Example format:
Group 1.1: 1, 3
Group 1.2: 2, 4, 5
Group 2.1: 1, 2
Group 2.2: 3

Your refined clustering:"""

    try:
        messages = [
            {"role": "system", "content": "You are an expert at semantic clustering. Refine concept groups into focused sub-clusters."},
            {"role": "user", "content": batch_prompt}
        ]
        
        content, _ = llm_client.chat_completion(model, messages, temperature=0.1)
        
        # Parse the response
        refined_results = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if 'Group' in line and ':' in line:
                try:
                    # Extract group info: "Group 1.2: 1, 3, 5"
                    parts = line.split(':', 1)
                    group_part = parts[0].strip()
                    numbers_part = parts[1].strip()
                    
                    # Parse group number: "Group 1.2" -> group_idx=0, subcluster_id=2
                    if '.' in group_part:
                        group_info = group_part.replace('Group ', '').strip()
                        group_idx = int(group_info.split('.')[0]) - 1
                        subcluster_id = int(group_info.split('.')[1]) - 1
                        
                        if 0 <= group_idx < len(clusters_to_refine):
                            original_cluster_id, original_concepts = clusters_to_refine[group_idx]
                            
                            # Parse concept numbers
                            concept_indices = [int(x.strip()) - 1 for x in numbers_part.split(',') if x.strip().isdigit()]
                            
                            if concept_indices:
                                refined_concepts = [original_concepts[i] for i in concept_indices if 0 <= i < len(original_concepts)]
                                
                                if original_cluster_id not in refined_results:
                                    refined_results[original_cluster_id] = {}
                                
                                refined_results[original_cluster_id][subcluster_id] = refined_concepts
                except:
                    continue
        
        # Ensure all concepts are assigned for each original cluster
        for original_cluster_id, original_concepts in clusters_to_refine:
            if original_cluster_id not in refined_results:
                refined_results[original_cluster_id] = {0: original_concepts}
                continue
                
            # Check for unassigned concepts
            assigned_concepts = set()
            for sub_concepts in refined_results[original_cluster_id].values():
                assigned_concepts.update(sub_concepts)
            
            unassigned = [c for c in original_concepts if c not in assigned_concepts]
            if unassigned:
                max_sub_id = max(refined_results[original_cluster_id].keys()) if refined_results[original_cluster_id] else -1
                refined_results[original_cluster_id][max_sub_id + 1] = unassigned
        
        return refined_results
        
    except Exception as e:
        print(f"Warning: Batch LLM refinement failed: {e}")
        # Fallback to individual processing
        fallback_results = {}
        for original_cluster_id, concepts in clusters_to_refine:
            try:
                refined = llm_based_clustering(concepts, model)
                fallback_results[original_cluster_id] = refined
            except:
                fallback_results[original_cluster_id] = {0: concepts}
        return fallback_results


def hybrid_clustering(concepts: List[str]) -> Dict[int, List[str]]:
    """
    Combine embedding-based and LLM-based clustering for best results.
    
    Args:
        concepts: List of concept strings
        
    Returns:
        Dictionary mapping cluster IDs to concept lists
    """
    if len(concepts) <= 3:
        return {i: [concept] for i, concept in enumerate(concepts)}
    
    # Step 1: Get initial clustering using embeddings
    embeddings = embed_concepts_improved(concepts)
    reduced_embeddings = reduce_dimensions(embeddings)
    initial_labels = cluster_embeddings(reduced_embeddings)
    
    # Convert to cluster dictionary
    initial_clusters = {}
    for label, concept in zip(initial_labels, concepts):
        if label not in initial_clusters:
            initial_clusters[label] = []
        initial_clusters[label].append(concept)
    
    # Step 2: Collect large clusters for batch refinement
    large_clusters = []
    final_clusters = {}
    cluster_id = 0
    
    # Separate large clusters that need refinement
    for initial_id, cluster_concepts in initial_clusters.items():
        if len(cluster_concepts) > 4:  # Collect large clusters for batch processing
            large_clusters.append((initial_id, cluster_concepts))
        else:
            final_clusters[cluster_id] = cluster_concepts
            cluster_id += 1
    
    # Batch refine large clusters to reduce API calls
    if large_clusters:
        print(f"  Refining {len(large_clusters)} large clusters in batch to reduce API calls...")
        try:
            # Process in batches of 5 clusters per API call
            batch_size = 5
            for batch_start in range(0, len(large_clusters), batch_size):
                batch = large_clusters[batch_start:batch_start + batch_size]
                refined_results = batch_llm_cluster_refinement(batch)
                
                # Add refined clusters to final result
                for original_id, sub_clusters in refined_results.items():
                    for sub_concepts in sub_clusters.values():
                        if sub_concepts:  # Only add non-empty clusters
                            final_clusters[cluster_id] = sub_concepts
                            cluster_id += 1
                            
            print(f"  Completed batch refinement of large clusters")
        except Exception as e:
            print(f"  Warning: Batch refinement failed: {e}")
            # Fallback: add large clusters as-is
            for _, cluster_concepts in large_clusters:
                final_clusters[cluster_id] = cluster_concepts
                cluster_id += 1
    
    return final_clusters


def generate_cluster_summary(concepts: List[str], concept_metadata: Dict[str, ConceptMetadata], 
                           cluster_name: str = None, model: str = LLM_CLUSTERING_MODEL) -> Tuple[str, float]:
    """
    Generate a short, descriptive name for a cluster using LLM.
    
    Returns:
        Tuple of (short_name, confidence)
    """
    
    # Collect sample contexts from the concepts (reduced for brevity)
    sample_contexts = []
    for concept in concepts[:3]:  # Limit to first 3 concepts
        if concept in concept_metadata:
            contexts = concept_metadata[concept].source_contexts
            if contexts:
                sample_contexts.extend(contexts[:1])  # Max 1 context per concept
    
    # Limit total context length
    context_text = ". ".join(sample_contexts[:3])
    if len(context_text) > 200:
        context_text = context_text[:200] + "..."
    
    concepts_text = ", ".join(concepts)
    
    prompt = f"""Given these related concepts from an interview: [{concepts_text}]

Sample contexts: "{context_text}"

Create a SHORT, concise name (2-4 words maximum) that captures the main theme.

Examples:
- Concepts: stress, anxiety, worry → "Stress Management"
- Concepts: meeting, discussion, decision → "Decision Making" 
- Concepts: water, pollution, quality → "Water Quality"

Concepts: [{concepts_text}]
Short name:"""

    messages = [
        {"role": "system", "content": "Create SHORT, concise cluster names (2-4 words maximum). Focus on brevity and clarity."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        content, _ = llm_client.chat_completion(model, messages, 0.3)
        summary = content.strip().strip('"').strip("'")
        
        # Remove common unwanted phrases and clean up
        summary = summary.replace("Short name:", "").strip()
        summary = summary.split('\n')[0].strip()  # Take only first line
        
        # Basic confidence scoring based on length and content
        word_count = len(summary.split())
        if word_count <= 4 and len(summary) <= 25 and not summary.lower().startswith('this cluster'):
            confidence = 0.9
        elif word_count <= 6 and len(summary) <= 40:
            confidence = 0.7
        else:
            confidence = 0.5
            
        return summary, confidence
        
    except Exception as e:
        print(f"    Warning: Could not generate summary for cluster: {e}")
        return f"Cluster containing {', '.join(concepts[:3])}{'...' if len(concepts) > 3 else ''}", 0.5

def batch_generate_cluster_summaries(clusters_data: List[Tuple[int, List[str], Dict[str, ConceptMetadata]]], 
                                    model: str = LLM_CLUSTERING_MODEL) -> Dict[int, Tuple[str, float]]:
    """
    Generate summaries for multiple clusters in a single API call for maximum efficiency.
    
    Args:
        clusters_data: List of (cluster_id, concepts, concept_metadata) tuples
        model: LLM model to use
        
    Returns:
        Dictionary mapping cluster_id to (summary, confidence) tuple
    """
    
    if not clusters_data:
        return {}
    
    # Build batch prompt for multiple clusters
    clusters_text = ""
    for i, (cluster_id, concepts, concept_metadata) in enumerate(clusters_data):
        concepts_text = ", ".join(concepts)
        
        # Collect sample contexts from the concepts (limit to avoid huge prompts)
        sample_contexts = []
        for concept in concepts[:3]:  # Limit to first 3 concepts
            if concept in concept_metadata:
                contexts = concept_metadata[concept].source_contexts
                if contexts:
                    sample_contexts.extend(contexts[:1])  # Max 1 context per concept
        
        # Limit total context length per cluster
        context_text = ". ".join(sample_contexts[:3])
        if len(context_text) > 200:
            context_text = context_text[:200] + "..."
        
        clusters_text += f"""
Cluster {i+1} (ID: {cluster_id}):
- Concepts: [{concepts_text}]
- Sample contexts: "{context_text}"
"""
    
    prompt = f"""Given these concept clusters from an interview, create SHORT, concise names for each cluster (2-4 words maximum).

{clusters_text}

For each cluster, create a brief NAME that captures the main theme. Use simple, clear terms.

Examples:
- "Stress Management"
- "Social Relationships" 
- "Work Performance"
- "Health Issues"
- "Learning Skills"

Respond with exactly one line per cluster:
"Cluster X: [short name]"

Response:"""

    messages = [
        {"role": "system", "content": "Create SHORT, concise cluster names (2-4 words maximum). Focus on brevity and clarity."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        content, _ = llm_client.chat_completion(model, messages, 0.3)
        
        if not content.strip():
            print(f"    Warning: Empty response for batch cluster naming")
            # Fallback to individual processing
            return _fallback_individual_naming(clusters_data, model)
        
        # Parse the batched response
        results = {}
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # Extract cluster number and description
            try:
                cluster_part, description = line.split(':', 1)
                cluster_num = int(cluster_part.strip().split()[-1]) - 1  # Convert to 0-based index
                
                if cluster_num >= len(clusters_data):
                    continue
                
                cluster_id, _, _ = clusters_data[cluster_num]
                description = description.strip()
                
                # Clean up the description
                if description.startswith('"') and description.endswith('"'):
                    description = description[1:-1]
                
                # Fixed confidence scoring - favor SHORT names
                confidence = 0.9
                word_count = len(description.split())
                
                if word_count <= 4 and len(description) <= 25:
                    confidence = 0.95  # High confidence for short, appropriate names
                elif word_count <= 6 and len(description) <= 40:
                    confidence = 0.8   # Medium confidence for moderately short names  
                elif "cluster" in description.lower() or "this cluster" in description.lower():
                    confidence = 0.3   # Low confidence for verbose descriptions
                else:
                    confidence = 0.6   # Default for other cases
                
                results[cluster_id] = (description, confidence)
                
            except (ValueError, IndexError) as e:
                print(f"    Warning: Could not parse line '{line}': {e}")
                continue
        
        print(f"    Batch processed {len(clusters_data)} clusters → {len(results)} descriptions in 1 API call")
        
        # Fill in any missing clusters with fallback
        missing_clusters = [(cid, concepts, meta) for cid, concepts, meta in clusters_data if cid not in results]
        if missing_clusters:
            print(f"    Processing {len(missing_clusters)} missing clusters individually...")
            fallback_results = _fallback_individual_naming(missing_clusters, model)
            results.update(fallback_results)
        
        return results
        
    except Exception as e:
        print(f"    Error in batch cluster naming: {e}")
        # Fallback to individual processing
        return _fallback_individual_naming(clusters_data, model)

def _fallback_individual_naming(clusters_data: List[Tuple[int, List[str], Dict[str, ConceptMetadata]]], 
                               model: str) -> Dict[int, Tuple[str, float]]:
    """Fallback to individual cluster naming if batch fails."""
    results = {}
    for cluster_id, concepts, concept_metadata in clusters_data:
        try:
            summary, confidence = generate_cluster_summary(concepts, concept_metadata, model=model)
            results[cluster_id] = (summary, confidence)
        except Exception as e:
            print(f"    Warning: Failed to name cluster {cluster_id}: {e}")
            results[cluster_id] = (f"Cluster containing {', '.join(concepts[:3])}{'...' if len(concepts) > 3 else ''}", 0.5)
    return results

def cluster_concepts_improved(concepts: List[str]) -> Dict[int, List[str]]:
    """
    Main improved clustering function (backward compatibility version).
    
    Args:
        concepts: List of concept strings
        
    Returns:
        Dictionary mapping cluster IDs to concept lists
    """
    if len(concepts) < 4:
        return {i: [concept] for i, concept in enumerate(concepts)}
    
    print(f"Clustering {len(concepts)} concepts using method: {CLUSTERING_METHOD}")
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            if CLUSTERING_METHOD == "llm_only" and USE_LLM_CLUSTERING:
                print("  Using LLM-only clustering (1 API call)")
                clusters = llm_based_clustering(concepts)
            elif CLUSTERING_METHOD == "hybrid":
                print("  Using hybrid clustering (embedding + LLM refinement)")
                clusters = hybrid_clustering(concepts)
            elif CLUSTERING_METHOD == "embedding_enhanced":
                print("  Using enhanced embedding-based clustering (0 API calls)")
                # Enhanced embedding-based pipeline
                embeddings = embed_concepts_improved(concepts)
                reduced_embeddings = reduce_dimensions(embeddings)
                cluster_labels = cluster_embeddings(reduced_embeddings)
                
                clusters = {}
                for label, concept in zip(cluster_labels, concepts):
                    clusters.setdefault(label, []).append(concept)
            elif CLUSTERING_METHOD == "no_clustering":
                print("  Skipping clustering - treating each concept as its own cluster")
                clusters = {i: [concept] for i, concept in enumerate(concepts)}
            else:
                print(f"  Unknown clustering method: {CLUSTERING_METHOD}")
                print("  Supported methods: 'llm_only', 'hybrid', 'embedding_enhanced', 'no_clustering'")
                print("  Using original clustering method as fallback")
                from embed_and_cluster import cluster_concepts
                return cluster_concepts(concepts)
        
        # Handle noise points (-1 cluster) if they exist
        if -1 in clusters:
            noise_concepts = clusters.pop(-1)
            max_cluster_id = max(clusters.keys()) if clusters else -1
            for i, concept in enumerate(noise_concepts):
                clusters[max_cluster_id + 1 + i] = [concept]
        
        # Ensure all concepts are assigned
        if not clusters:
            print(f"  No clusters created, falling back to individual concepts")
            return {i: [concept] for i, concept in enumerate(concepts)}
        
        print(f"  Created {len(clusters)} clusters")
        return clusters
        
    except Exception as e:
        print(f"Clustering failed with error: {e}")
        print("Falling back to treating each concept as its own cluster")
        return {i: [concept] for i, concept in enumerate(concepts)}

def cluster_concepts_with_metadata(concepts: List[str], concept_metadata: Dict[str, ConceptMetadata]) -> ClusterMetadataManager:
    """
    Cluster concepts and return rich metadata.
    
    Args:
        concepts: List of concept strings
        concept_metadata: Metadata for each concept
        
    Returns:
        ClusterMetadataManager with full cluster metadata
    """
    if len(concepts) < 4:
        # Handle small concept sets
        manager = ClusterMetadataManager()
        for i, concept in enumerate(concepts):
            cluster_id = f"cluster_{i}"
            concept_meta = {concept: concept_metadata.get(concept, ConceptMetadata(concept=concept, source_contexts=[], chunk_indices=[]))}
            manager.add_cluster(cluster_id, [concept], concept_meta, name=concept)
        return manager
    
    print(f"Clustering {len(concepts)} concepts with metadata using method: {CLUSTERING_METHOD}")
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Get basic clusters first (without metadata)
            if CLUSTERING_METHOD == "llm_only" and USE_LLM_CLUSTERING:
                print("  Using LLM-only clustering (1 API call)")
                basic_clusters = llm_based_clustering(concepts)
            elif CLUSTERING_METHOD == "hybrid":
                print("  Using hybrid clustering (embedding + LLM refinement)")
                basic_clusters = hybrid_clustering(concepts)
            elif CLUSTERING_METHOD == "embedding_enhanced":
                print("  Using enhanced embedding-based clustering (0 API calls)")
                # Enhanced embedding-based pipeline
                embeddings = embed_concepts_improved(concepts)
                reduced_embeddings = reduce_dimensions(embeddings)
                cluster_labels = cluster_embeddings(reduced_embeddings)
                
                basic_clusters = {}
                for label, concept in zip(cluster_labels, concepts):
                    basic_clusters.setdefault(label, []).append(concept)
            elif CLUSTERING_METHOD == "no_clustering":
                print("  Skipping clustering - treating each concept as its own cluster")
                basic_clusters = {i: [concept] for i, concept in enumerate(concepts)}
            else:
                print(f"  Unknown clustering method: {CLUSTERING_METHOD}")
                print("  Supported methods: 'llm_only', 'hybrid', 'embedding_enhanced', 'no_clustering'")
                print("  Using original clustering method as fallback")
                from embed_and_cluster import cluster_concepts
                basic_clusters = cluster_concepts(concepts)
        
        # Convert to metadata-rich clusters
        manager = ClusterMetadataManager()
        
        print(f"  Created {len(basic_clusters)} clusters, generating metadata...")
        
        # Prepare data for batch cluster naming
        clusters_for_naming = []
        single_concept_clusters = []
        
        for cluster_id, cluster_concepts in basic_clusters.items():
            cluster_concept_metadata = {}
            for concept in cluster_concepts:
                if concept in concept_metadata:
                    cluster_concept_metadata[concept] = concept_metadata[concept]
                else:
                    cluster_concept_metadata[concept] = ConceptMetadata(
                        concept=concept, source_contexts=[], chunk_indices=[]
                    )
            
            if len(cluster_concepts) == 1:
                # Handle single concept clusters separately
                single_concept_clusters.append((cluster_id, cluster_concepts, cluster_concept_metadata))
            else:
                # Add to batch naming list
                clusters_for_naming.append((cluster_id, cluster_concepts, cluster_concept_metadata))
        
        # Batch process multi-concept clusters in configurable batch sizes
        batch_summaries = {}
        total_api_calls = 0
        
        if clusters_for_naming:
            total_clusters = len(clusters_for_naming)
            total_batches = (total_clusters + CLUSTER_NAMING_BATCH_SIZE - 1) // CLUSTER_NAMING_BATCH_SIZE
            
            print(f"  Batch naming {total_clusters} multi-concept clusters in {total_batches} batches (size: {CLUSTER_NAMING_BATCH_SIZE})...")
            
            for batch_idx in range(0, total_clusters, CLUSTER_NAMING_BATCH_SIZE):
                batch_end = min(batch_idx + CLUSTER_NAMING_BATCH_SIZE, total_clusters)
                batch_data = clusters_for_naming[batch_idx:batch_end]
                
                print(f"    Batch {batch_idx//CLUSTER_NAMING_BATCH_SIZE + 1}: Processing {len(batch_data)} clusters...")
                batch_results = batch_generate_cluster_summaries(batch_data)
                batch_summaries.update(batch_results)
                total_api_calls += 1
        
        # Process all clusters (single and multi-concept)
        all_clusters_data = single_concept_clusters + clusters_for_naming
        
        for cluster_id, cluster_concepts, cluster_concept_metadata in all_clusters_data:
            cluster_id_str = f"cluster_{cluster_id}"
            
            # Generate name and summary
            if len(cluster_concepts) == 1:
                # Single concept clusters
                cluster_name = cluster_concepts[0]
                summary = f"Individual concept: {cluster_concepts[0]}"
                confidence = 1.0
            else:
                # Multi-concept clusters (from batch)
                if cluster_id in batch_summaries:
                    short_name, confidence = batch_summaries[cluster_id]
                    cluster_name = short_name  # Use the short name directly
                    summary = f"Cluster of concepts related to {short_name.lower()}"
                else:
                    # Fallback if batch failed
                    cluster_name = f"Cluster {cluster_id}"
                    summary = f"Cluster containing {', '.join(cluster_concepts[:3])}{'...' if len(cluster_concepts) > 3 else ''}"
                    confidence = 0.5
            
            # Add cluster to manager
            cluster = manager.add_cluster(cluster_id_str, cluster_concepts, cluster_concept_metadata, cluster_name)
            cluster.summary = summary
            cluster.confidence = confidence
            
            print(f"    Cluster {cluster_id}: {len(cluster_concepts)} concepts - {summary}")
        
        # Report efficiency gains
        original_calls = len([c for c in all_clusters_data if len(c[1]) > 1])
        if original_calls > 0:
            efficiency_gain = ((original_calls - total_api_calls) / original_calls) * 100
            print(f"  🚀 Cluster naming efficiency: {total_api_calls} API calls (vs {original_calls} individual calls)")
            print(f"  📊 Efficiency gain: {efficiency_gain:.1f}% fewer API calls")
        
        return manager
        
    except Exception as e:
        print(f"Warning: Clustering failed with error: {e}")
        print("Falling back to individual clusters")
        
        manager = ClusterMetadataManager()
        for i, concept in enumerate(concepts):
            cluster_id = f"cluster_{i}"
            concept_meta = {concept: concept_metadata.get(concept, ConceptMetadata(concept=concept, source_contexts=[], chunk_indices=[]))}
            manager.add_cluster(cluster_id, [concept], concept_meta, name=concept)
        
        return manager


if __name__ == "__main__":
    # Test the improved clustering
    test_concepts = [
        "social isolation", "loneliness", "depression", "anxiety", 
        "poor sleep", "insomnia", "fatigue", "stress", 
        "social support", "friendship", "family relationships"
    ]
    
    print("Testing improved clustering methods:")
    
    # Test different methods
    for method in ["improved_pipeline", "llm_based", "hybrid"]:
        print(f"\n--- {method.upper()} ---")
        # Temporarily change method
        original_method = CLUSTERING_METHOD
        import constants
        constants.CLUSTERING_METHOD = method
        
        try:
            clusters = cluster_concepts_with_metadata(test_concepts, {}) # Pass an empty dict for concept_metadata for now
            for cluster_id, cluster_metadata in clusters.clusters.items():
                print(f"Cluster {cluster_id}: {', '.join(cluster_metadata.concepts)}")
        except Exception as e:
            print(f"Error with {method}: {e}")
        
        # Restore original method
        constants.CLUSTERING_METHOD = original_method 