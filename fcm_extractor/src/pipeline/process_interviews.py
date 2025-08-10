import os
import glob
import json
from typing import List, Dict
from docx import Document
from pathlib import Path
import pandas as pd
import tempfile
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import (
    OUTPUT_DIRECTORY, INTERVIEWS_DIRECTORY, DEFAULT_INTERVIEW_FILE, PROCESS_ALL_FILES,
    CLUSTERING_METHOD, EDGE_INFERENCE_MODEL, EDGE_INFERENCE_TEMPERATURE,
    CLUSTER_EDGE_BATCH_SIZE, EDGE_INFERENCE_BATCH_SIZE, CLUSTER_NAMING_BATCH_SIZE,
    CONCEPT_EXTRACTION_N_PROMPTS, CONCEPT_EXTRACTION_MODEL, CONCEPT_EXTRACTION_TEMPERATURE,
    HDBSCAN_MIN_CLUSTER_SIZE, UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_COMPONENTS,
    DIMENSIONALITY_REDUCTION, CLUSTERING_ALGORITHM, ENABLE_INTRA_CLUSTER_EDGES,
    ENABLE_FILE_LOGGING, LOG_DIRECTORY, SEPARATE_LOG_PER_DOCUMENT, INCLUDE_TIMESTAMP_IN_LOGS, LOG_LEVEL,
    INTERVIEW_FILE_NAME,
    CLUSTERING_EMBEDDING_MODEL, HDBSCAN_MIN_SAMPLES, USE_LLM_CLUSTERING, LLM_CLUSTERING_MODEL,
    USE_CONFIDENCE_FILTERING, EDGE_CONFIDENCE_THRESHOLD, DEFAULT_CONCEPT_EXTRACTION_PROMPT,
    DEFAULT_EDGE_INFERENCE_PROMPT, DEFAULT_INTER_CLUSTER_EDGE_PROMPT, DEFAULT_INTRA_CLUSTER_EDGE_PROMPT,
    MAX_EDGE_INFERENCE_TEXT_LENGTH, AGGLOMERATIVE_MAX_CLUSTERS, AGGLOMERATIVE_USE_ELBOW_METHOD,
    AGGLOMERATIVE_USE_DISTANCE_THRESHOLD, AGGLOMERATIVE_DISTANCE_THRESHOLD,
    TSNE_PERPLEXITY, TSNE_EARLY_EXAGGERATION, TSNE_LEARNING_RATE, TSNE_N_ITER,
    EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES, EVALUATION_INCLUDE_INTRA_CLUSTER_NODES,
    LLM_CLUSTERING_PROMPT_TEMPLATE, LLM_CLUSTER_REFINEMENT_PROMPT_TEMPLATE,
    ENABLE_POST_CLUSTERING, POST_CLUSTERING_SIMILARITY_THRESHOLD, POST_CLUSTERING_EMBEDDING_MODEL,
    META_PROMPTING_MODEL, META_PROMPTING_TEMPERATURE, META_PROMPTING_ENABLED, META_PROMPTING_VERBOSE,
    DYNAMIC_PROMPTING_ENABLED, DYNAMIC_PROMPTING_USE_CACHE, DYNAMIC_PROMPTING_USE_REFLECTION,
    DYNAMIC_PROMPTING_TRACK_PERFORMANCE, POST_CLUSTERING_MAX_MERGES_PER_CLUSTER,
    POST_CLUSTERING_REQUIRE_MINIMUM_SIMILARITY, ACO_MAX_ITERATIONS, ACO_SAMPLES_PER_ITERATION,
    ACO_EVAPORATION_RATE, ACO_INITIAL_PHEROMONE, ACO_CONVERGENCE_THRESHOLD, ACO_GUARANTEE_COVERAGE
)
from src.core import extract_concepts, extract_concepts_with_metadata, build_fcm_graph, export_graph_to_json
from src.clustering import cluster_concepts_improved, name_all_clusters, cluster_concepts_with_metadata, apply_post_clustering
from src.models import ClusterMetadataManager, ClusterMetadata
from src.edge_inference import infer_edges, infer_cluster_edge_grounded, infer_edges_original
from src.edge_inference.aco_edge_inference import ACOEdgeInference
from utils.logging_utils import setup_logging, finalize_logging, get_log_file_path
from utils.visualize_fcm import create_interactive_visualization
import networkx as nx
from itertools import combinations

def read_document(file_path: str) -> str:
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.docx', '.doc']:
        return read_word_document(file_path)
    elif file_extension == '.txt':
        return read_text_file(file_path)
    else:
        print(f"Unsupported file format: {file_extension}")
        return None

def read_word_document(file_path: str) -> str:
    try:
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text.strip())
        return '\n\n'.join(text)
    except Exception as e:
        print(f"Error reading Word document {file_path}: {e}")
        return None

def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            return text
        except Exception as e:
            print(f"Error reading text file {file_path} with latin-1 encoding: {e}")
            return None
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return None

def get_constants_config():
    """Extract constants configuration for inclusion in output files."""
    return {
        "CONCEPT_EXTRACTION_MODEL": CONCEPT_EXTRACTION_MODEL,
        "CONCEPT_EXTRACTION_TEMPERATURE": CONCEPT_EXTRACTION_TEMPERATURE,
        "CONCEPT_EXTRACTION_N_PROMPTS": CONCEPT_EXTRACTION_N_PROMPTS,
        "DEFAULT_CONCEPT_EXTRACTION_PROMPT": DEFAULT_CONCEPT_EXTRACTION_PROMPT,
        "META_PROMPTING_MODEL": META_PROMPTING_MODEL,
        "META_PROMPTING_TEMPERATURE": META_PROMPTING_TEMPERATURE,
        "META_PROMPTING_ENABLED": META_PROMPTING_ENABLED,
        "META_PROMPTING_VERBOSE": META_PROMPTING_VERBOSE,
        "DYNAMIC_PROMPTING_ENABLED": DYNAMIC_PROMPTING_ENABLED,
        "DYNAMIC_PROMPTING_USE_CACHE": DYNAMIC_PROMPTING_USE_CACHE,
        "DYNAMIC_PROMPTING_USE_REFLECTION": DYNAMIC_PROMPTING_USE_REFLECTION,
        "DYNAMIC_PROMPTING_TRACK_PERFORMANCE": DYNAMIC_PROMPTING_TRACK_PERFORMANCE,
        "CLUSTERING_METHOD": CLUSTERING_METHOD,
        "CLUSTERING_ALGORITHM": CLUSTERING_ALGORITHM,
        "CLUSTERING_EMBEDDING_MODEL": CLUSTERING_EMBEDDING_MODEL,
        "DIMENSIONALITY_REDUCTION": DIMENSIONALITY_REDUCTION,
        "USE_LLM_CLUSTERING": USE_LLM_CLUSTERING,
        "LLM_CLUSTERING_MODEL": LLM_CLUSTERING_MODEL,
        "CLUSTER_NAMING_BATCH_SIZE": CLUSTER_NAMING_BATCH_SIZE,
        "HDBSCAN_MIN_CLUSTER_SIZE": HDBSCAN_MIN_CLUSTER_SIZE,
        "HDBSCAN_MIN_SAMPLES": HDBSCAN_MIN_SAMPLES,
        "UMAP_N_NEIGHBORS": UMAP_N_NEIGHBORS,
        "UMAP_MIN_DIST": UMAP_MIN_DIST,
        "UMAP_N_COMPONENTS": UMAP_N_COMPONENTS,
        "TSNE_PERPLEXITY": TSNE_PERPLEXITY,
        "TSNE_EARLY_EXAGGERATION": TSNE_EARLY_EXAGGERATION,
        "TSNE_LEARNING_RATE": TSNE_LEARNING_RATE,
        "TSNE_N_ITER": TSNE_N_ITER,
        "AGGLOMERATIVE_MAX_CLUSTERS": AGGLOMERATIVE_MAX_CLUSTERS,
        "AGGLOMERATIVE_USE_ELBOW_METHOD": AGGLOMERATIVE_USE_ELBOW_METHOD,
        "AGGLOMERATIVE_USE_DISTANCE_THRESHOLD": AGGLOMERATIVE_USE_DISTANCE_THRESHOLD,
        "AGGLOMERATIVE_DISTANCE_THRESHOLD": AGGLOMERATIVE_DISTANCE_THRESHOLD,
        "EDGE_INFERENCE_MODEL": EDGE_INFERENCE_MODEL,
        "EDGE_INFERENCE_TEMPERATURE": EDGE_INFERENCE_TEMPERATURE,
        "EDGE_INFERENCE_BATCH_SIZE": EDGE_INFERENCE_BATCH_SIZE,
        "CLUSTER_EDGE_BATCH_SIZE": CLUSTER_EDGE_BATCH_SIZE,
        "ENABLE_INTRA_CLUSTER_EDGES": ENABLE_INTRA_CLUSTER_EDGES,
        "USE_CONFIDENCE_FILTERING": USE_CONFIDENCE_FILTERING,
        "EDGE_CONFIDENCE_THRESHOLD": EDGE_CONFIDENCE_THRESHOLD,
        "MAX_EDGE_INFERENCE_TEXT_LENGTH": MAX_EDGE_INFERENCE_TEXT_LENGTH,
        "DEFAULT_EDGE_INFERENCE_PROMPT": DEFAULT_EDGE_INFERENCE_PROMPT,
        "DEFAULT_INTER_CLUSTER_EDGE_PROMPT": DEFAULT_INTER_CLUSTER_EDGE_PROMPT,
        "DEFAULT_INTRA_CLUSTER_EDGE_PROMPT": DEFAULT_INTRA_CLUSTER_EDGE_PROMPT,
        "EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES": EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES,
        "EVALUATION_INCLUDE_INTRA_CLUSTER_NODES": EVALUATION_INCLUDE_INTRA_CLUSTER_NODES,
        "ENABLE_POST_CLUSTERING": ENABLE_POST_CLUSTERING,
        "POST_CLUSTERING_SIMILARITY_THRESHOLD": POST_CLUSTERING_SIMILARITY_THRESHOLD,
        "POST_CLUSTERING_EMBEDDING_MODEL": POST_CLUSTERING_EMBEDDING_MODEL,
        "POST_CLUSTERING_MAX_MERGES_PER_CLUSTER": POST_CLUSTERING_MAX_MERGES_PER_CLUSTER,
        "POST_CLUSTERING_REQUIRE_MINIMUM_SIMILARITY": POST_CLUSTERING_REQUIRE_MINIMUM_SIMILARITY,
        "ACO_MAX_ITERATIONS": ACO_MAX_ITERATIONS,
        "ACO_SAMPLES_PER_ITERATION": ACO_SAMPLES_PER_ITERATION,
        "ACO_EVAPORATION_RATE": ACO_EVAPORATION_RATE,
        "ACO_INITIAL_PHEROMONE": ACO_INITIAL_PHEROMONE,
        "ACO_CONVERGENCE_THRESHOLD": ACO_CONVERGENCE_THRESHOLD,
        "ACO_GUARANTEE_COVERAGE": ACO_GUARANTEE_COVERAGE,
        "LLM_CLUSTERING_PROMPT_TEMPLATE": LLM_CLUSTERING_PROMPT_TEMPLATE,
        "LLM_CLUSTER_REFINEMENT_PROMPT_TEMPLATE": LLM_CLUSTER_REFINEMENT_PROMPT_TEMPLATE,
        "DEFAULT_INTERVIEW_FILE": DEFAULT_INTERVIEW_FILE,
        "PROCESS_ALL_FILES": PROCESS_ALL_FILES,
        "INTERVIEWS_DIRECTORY": INTERVIEWS_DIRECTORY,
        "OUTPUT_DIRECTORY": OUTPUT_DIRECTORY,
        "ENABLE_FILE_LOGGING": ENABLE_FILE_LOGGING,
        "LOG_DIRECTORY": LOG_DIRECTORY,
        "SEPARATE_LOG_PER_DOCUMENT": SEPARATE_LOG_PER_DOCUMENT,
        "INCLUDE_TIMESTAMP_IN_LOGS": INCLUDE_TIMESTAMP_IN_LOGS,
        "LOG_LEVEL": LOG_LEVEL
    }

def save_run_parameters(output_dir, base_name):
    print(f"Saving run parameters to {output_dir} for {base_name}")
    try:
        params = {
            "CONCEPT_EXTRACTION_MODEL": CONCEPT_EXTRACTION_MODEL,
            "CONCEPT_EXTRACTION_TEMPERATURE": CONCEPT_EXTRACTION_TEMPERATURE,
            "CONCEPT_EXTRACTION_N_PROMPTS": CONCEPT_EXTRACTION_N_PROMPTS,
            "DEFAULT_CONCEPT_EXTRACTION_PROMPT": DEFAULT_CONCEPT_EXTRACTION_PROMPT,
        
        "META_PROMPTING_MODEL": META_PROMPTING_MODEL,
        "META_PROMPTING_TEMPERATURE": META_PROMPTING_TEMPERATURE,
        "META_PROMPTING_ENABLED": META_PROMPTING_ENABLED,
        "META_PROMPTING_VERBOSE": META_PROMPTING_VERBOSE,
        "DYNAMIC_PROMPTING_ENABLED": DYNAMIC_PROMPTING_ENABLED,
        "DYNAMIC_PROMPTING_USE_CACHE": DYNAMIC_PROMPTING_USE_CACHE,
        "DYNAMIC_PROMPTING_USE_REFLECTION": DYNAMIC_PROMPTING_USE_REFLECTION,
        "DYNAMIC_PROMPTING_TRACK_PERFORMANCE": DYNAMIC_PROMPTING_TRACK_PERFORMANCE,
        
        "CLUSTERING_METHOD": CLUSTERING_METHOD,
        "CLUSTERING_ALGORITHM": CLUSTERING_ALGORITHM,
        "CLUSTERING_EMBEDDING_MODEL": CLUSTERING_EMBEDDING_MODEL,
        "DIMENSIONALITY_REDUCTION": DIMENSIONALITY_REDUCTION,
        "USE_LLM_CLUSTERING": USE_LLM_CLUSTERING,
        "LLM_CLUSTERING_MODEL": LLM_CLUSTERING_MODEL,
        "CLUSTER_NAMING_BATCH_SIZE": CLUSTER_NAMING_BATCH_SIZE,
        
        "HDBSCAN_MIN_CLUSTER_SIZE": HDBSCAN_MIN_CLUSTER_SIZE,
        "HDBSCAN_MIN_SAMPLES": HDBSCAN_MIN_SAMPLES,
        
        "UMAP_N_NEIGHBORS": UMAP_N_NEIGHBORS,
        "UMAP_MIN_DIST": UMAP_MIN_DIST,
        "UMAP_N_COMPONENTS": UMAP_N_COMPONENTS,
        
        "TSNE_PERPLEXITY": TSNE_PERPLEXITY,
        "TSNE_EARLY_EXAGGERATION": TSNE_EARLY_EXAGGERATION,
        "TSNE_LEARNING_RATE": TSNE_LEARNING_RATE,
        "TSNE_N_ITER": TSNE_N_ITER,
        
        "AGGLOMERATIVE_MAX_CLUSTERS": AGGLOMERATIVE_MAX_CLUSTERS,
        "AGGLOMERATIVE_USE_ELBOW_METHOD": AGGLOMERATIVE_USE_ELBOW_METHOD,
        "AGGLOMERATIVE_USE_DISTANCE_THRESHOLD": AGGLOMERATIVE_USE_DISTANCE_THRESHOLD,
        "AGGLOMERATIVE_DISTANCE_THRESHOLD": AGGLOMERATIVE_DISTANCE_THRESHOLD,
        
        "EDGE_INFERENCE_MODEL": EDGE_INFERENCE_MODEL,
        "EDGE_INFERENCE_TEMPERATURE": EDGE_INFERENCE_TEMPERATURE,
        "EDGE_INFERENCE_BATCH_SIZE": EDGE_INFERENCE_BATCH_SIZE,
        "CLUSTER_EDGE_BATCH_SIZE": CLUSTER_EDGE_BATCH_SIZE,
        "ENABLE_INTRA_CLUSTER_EDGES": ENABLE_INTRA_CLUSTER_EDGES,
        "USE_CONFIDENCE_FILTERING": USE_CONFIDENCE_FILTERING,
        "EDGE_CONFIDENCE_THRESHOLD": EDGE_CONFIDENCE_THRESHOLD,
        "MAX_EDGE_INFERENCE_TEXT_LENGTH": MAX_EDGE_INFERENCE_TEXT_LENGTH,
        "DEFAULT_EDGE_INFERENCE_PROMPT": DEFAULT_EDGE_INFERENCE_PROMPT,
        "DEFAULT_INTER_CLUSTER_EDGE_PROMPT": DEFAULT_INTER_CLUSTER_EDGE_PROMPT,
        "DEFAULT_INTRA_CLUSTER_EDGE_PROMPT": DEFAULT_INTRA_CLUSTER_EDGE_PROMPT,
        
        "EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES": EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES,
        "EVALUATION_INCLUDE_INTRA_CLUSTER_NODES": EVALUATION_INCLUDE_INTRA_CLUSTER_NODES,
        
        "ENABLE_POST_CLUSTERING": ENABLE_POST_CLUSTERING,
        "POST_CLUSTERING_SIMILARITY_THRESHOLD": POST_CLUSTERING_SIMILARITY_THRESHOLD,
        "POST_CLUSTERING_EMBEDDING_MODEL": POST_CLUSTERING_EMBEDDING_MODEL,
        "POST_CLUSTERING_MAX_MERGES_PER_CLUSTER": POST_CLUSTERING_MAX_MERGES_PER_CLUSTER,
        "POST_CLUSTERING_REQUIRE_MINIMUM_SIMILARITY": POST_CLUSTERING_REQUIRE_MINIMUM_SIMILARITY,
        
        "ACO_MAX_ITERATIONS": ACO_MAX_ITERATIONS,
        "ACO_SAMPLES_PER_ITERATION": ACO_SAMPLES_PER_ITERATION,
        "ACO_EVAPORATION_RATE": ACO_EVAPORATION_RATE,
        "ACO_INITIAL_PHEROMONE": ACO_INITIAL_PHEROMONE,
        "ACO_CONVERGENCE_THRESHOLD": ACO_CONVERGENCE_THRESHOLD,
        "ACO_GUARANTEE_COVERAGE": ACO_GUARANTEE_COVERAGE,
        
        "LLM_CLUSTERING_PROMPT_TEMPLATE": LLM_CLUSTERING_PROMPT_TEMPLATE,
        "LLM_CLUSTER_REFINEMENT_PROMPT_TEMPLATE": LLM_CLUSTER_REFINEMENT_PROMPT_TEMPLATE,
        
        "DEFAULT_INTERVIEW_FILE": DEFAULT_INTERVIEW_FILE,
        "PROCESS_ALL_FILES": PROCESS_ALL_FILES,
        "INTERVIEWS_DIRECTORY": INTERVIEWS_DIRECTORY,
        "OUTPUT_DIRECTORY": OUTPUT_DIRECTORY,
        
        "ENABLE_FILE_LOGGING": ENABLE_FILE_LOGGING,
        "LOG_DIRECTORY": LOG_DIRECTORY,
        "SEPARATE_LOG_PER_DOCUMENT": SEPARATE_LOG_PER_DOCUMENT,
        "INCLUDE_TIMESTAMP_IN_LOGS": INCLUDE_TIMESTAMP_IN_LOGS,
        "LOG_LEVEL": LOG_LEVEL
        }
        param_file = os.path.join(output_dir, f"{base_name}_fcm_params.json")
        with open(param_file, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved all run parameters to {param_file}")
    except Exception as e:
        print(f"Error saving run parameters: {e}")
        import traceback
        traceback.print_exc()

def process_single_document(file_path: str, output_dir: str = OUTPUT_DIRECTORY) -> Dict:
    print(f"\nProcessing: {file_path}")
    base_name = Path(file_path).stem
    doc_output_dir = Path(output_dir) / base_name
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    
    document_log_path = None
    if ENABLE_FILE_LOGGING and SEPARATE_LOG_PER_DOCUMENT:
        document_log_filename = f"{base_name}_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        document_log_path = setup_logging(
            log_directory=LOG_DIRECTORY,
            log_filename=document_log_filename,
            enable_file_logging=True,
            include_timestamp=INCLUDE_TIMESTAMP_IN_LOGS,
            log_level=LOG_LEVEL
        )
        print(f"Document-specific log: {document_log_path}")
    
    try:
        results = {
            'file_path': file_path,
            'document_name': base_name,
            'stage': 'started',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        api_call_tracker = {
            'concept_extraction': 0,
            'cluster_naming': 0,
            'inter_cluster_edges': 0,
            'intra_cluster_edges': 0,
            'total_calls': 0
        }
        
        print('\n=== 1. Reading Document & Extracting Concepts ===')
        text = read_document(file_path)
        if text is None:
            raise Exception(f"Failed to read document: {file_path}")
        results['text_length'] = len(text)
        results['stage'] = 'document_read'
        
        print(f"Document length: {len(text)} characters")
        # No longer saving intermediate temp files
        
        concepts, concept_metadata = extract_concepts_with_metadata(
            text, 
            n_prompts=CONCEPT_EXTRACTION_N_PROMPTS, 
            model=CONCEPT_EXTRACTION_MODEL, 
            temperature=CONCEPT_EXTRACTION_TEMPERATURE
        )
        api_call_tracker['concept_extraction'] = CONCEPT_EXTRACTION_N_PROMPTS
        
        print(f"Extracted {len(concepts)} concepts with metadata")
        results['concepts_count'] = len(concepts)
        results['stage'] = 'concepts_extracted'
        # No longer saving intermediate temp files
        
        print('\n=== 2. Concept Clustering with Metadata ===')
        cluster_manager = cluster_concepts_with_metadata(concepts, concept_metadata)
        
        num_clusters = len(cluster_manager.clusters)
        multi_concept_clusters = sum(1 for cluster in cluster_manager.clusters.values() if len(cluster.concepts) > 1)
        estimated_naming_calls = (multi_concept_clusters + CLUSTER_NAMING_BATCH_SIZE - 1) // CLUSTER_NAMING_BATCH_SIZE
        api_call_tracker['cluster_naming'] = estimated_naming_calls
        
        cluster_metadata_path = doc_output_dir / f"{base_name}_cluster_metadata.json"
        cluster_manager.save_to_file(str(cluster_metadata_path))
        
        simple_clusters = cluster_manager.to_simple_clusters()
        results['clusters_count'] = len(simple_clusters)
        results['cluster_metadata'] = str(cluster_metadata_path)
        results['stage'] = 'clustering_complete'
        # No longer saving intermediate temp files
        
        print(f"Generated {num_clusters} clusters with rich metadata")

        print('\n=== 3. Enhanced Edge Inference ===')
        print(f"Processing edge inference for {len(simple_clusters)} clusters...")
        
        total_clusters = len(simple_clusters)
        inter_cluster_pairs = (total_clusters * (total_clusters - 1)) // 2
        estimated_inter_calls = (inter_cluster_pairs + CLUSTER_EDGE_BATCH_SIZE - 1) // CLUSTER_EDGE_BATCH_SIZE
        
        print(f"  Cost Estimation:")
        print(f"    - Inter-cluster pairs: {inter_cluster_pairs}")
        print(f"    - Estimated API calls (standard): {estimated_inter_calls}")
        
        USE_ACO = True
        
        if USE_ACO:
            print(f"  Using ACO (Ant Colony Optimization) edge inference...")
            aco_inference = ACOEdgeInference()
            inter_cluster_edges, intra_cluster_edges = aco_inference.infer_edges(
                simple_clusters, text, cluster_metadata_manager=cluster_manager
            )
        else:
            print(f"  Using standard batch edge inference...")
            inter_cluster_edges, intra_cluster_edges = infer_edges(
                simple_clusters, text, cluster_metadata_manager=cluster_manager
            )
        
        api_call_tracker['inter_cluster_edges'] = estimated_inter_calls
        
        if ENABLE_INTRA_CLUSTER_EDGES:
            total_intra_pairs = sum(
                len(list(combinations(concepts, 2))) 
                for concepts in simple_clusters.values() 
                if len(concepts) >= 2
            )
            estimated_intra_calls = (total_intra_pairs + EDGE_INFERENCE_BATCH_SIZE - 1) // EDGE_INFERENCE_BATCH_SIZE
            api_call_tracker['intra_cluster_edges'] = estimated_intra_calls
        else:
            api_call_tracker['intra_cluster_edges'] = 0
            
        api_call_tracker['total_calls'] = sum(api_call_tracker.values())
        
        results['stage'] = 'edge_inference_complete'
        # No longer saving intermediate temp files
        
        print('\n=== 4. Post-clustering (Synonym Grouping) ===')
        if ENABLE_POST_CLUSTERING:
            print(f"Post-clustering enabled with similarity threshold: {POST_CLUSTERING_SIMILARITY_THRESHOLD}")
            
            initial_graph = build_fcm_graph(simple_clusters, inter_cluster_edges, intra_cluster_edges)
            
            _, updated_clusters, merge_mapping = apply_post_clustering(initial_graph, simple_clusters)
            
            simple_clusters = updated_clusters
            G = build_fcm_graph(simple_clusters, inter_cluster_edges, intra_cluster_edges)
            
            # Count final edges after post-clustering
            final_inter_edges = [d for _, _, d in G.edges(data=True) if d.get('type') == 'inter_cluster']
            final_intra_edges = [d for _, _, d in G.edges(data=True) if d.get('type') == 'intra_cluster']
            results['inter_cluster_edges_count'] = len(final_inter_edges)
            results['intra_cluster_edges_count'] = len(final_intra_edges)
            
            if merge_mapping:
                print(f"Post-clustering complete: Merged {len(merge_mapping)} unconnected clusters")
                for old_name, new_name in merge_mapping.items():
                    print(f"  - '{old_name}' merged into '{new_name}'")
            else:
                print("No clusters were merged during post-clustering")
        else:
            print("Post-clustering is disabled in configuration")
            G = build_fcm_graph(simple_clusters, inter_cluster_edges, intra_cluster_edges)
            
            # Count final edges when post-clustering is disabled
            final_inter_edges = [d for _, _, d in G.edges(data=True) if d.get('type') == 'inter_cluster']
            final_intra_edges = [d for _, _, d in G.edges(data=True) if d.get('type') == 'intra_cluster']
            results['inter_cluster_edges_count'] = len(final_inter_edges)
            results['intra_cluster_edges_count'] = len(final_intra_edges)
        
        results['clusters_count'] = len(simple_clusters)
        results['stage'] = 'post_clustering_complete'
        # No longer saving intermediate temp files

        print('\n=== 5. Final Graph Export ===')
        
        output_file = os.path.join(doc_output_dir, f"{base_name}_fcm.json")
        
        # Get constants configuration to include in output
        constants_config = get_constants_config()
        
        export_graph_to_json(G, output_file, constants_config)
        print(f'Graph exported to {output_file}')
        
        print('\n=== 6. Create Visualizations ===')
        interactive_viz_path = os.path.join(doc_output_dir, f"{base_name}_fcm_interactive.html")
        create_interactive_visualization(G, interactive_viz_path)
        
        results['stage'] = 'completed'
        results['output_files'] = {
            'fcm_json': output_file,
            'interactive_viz': interactive_viz_path,
            'cluster_metadata': str(cluster_metadata_path)
        }
        
        # No longer saving intermediate temp files
        
        results.update({
            'fcm_json': output_file,
            'interactive_viz': interactive_viz_path,
            'cluster_metadata': str(cluster_metadata_path),
            'api_calls': api_call_tracker
        })
        
        print(f"\n=== COST EFFICIENCY SUMMARY ===")
        print(f"Total API Calls: {api_call_tracker['total_calls']}")
        print(f"   ├─ Concept Extraction: {api_call_tracker['concept_extraction']}")
        print(f"   ├─ Cluster Naming: {api_call_tracker['cluster_naming']}")
        print(f"   ├─ Inter-cluster Edges: {api_call_tracker['inter_cluster_edges']}")
        print(f"   └─ Intra-cluster Edges: {api_call_tracker['intra_cluster_edges']}")
        
        individual_naming_calls = multi_concept_clusters
        individual_inter_calls = inter_cluster_pairs
        individual_intra_calls = 0
        individual_total = (api_call_tracker['concept_extraction'] + 
                           individual_naming_calls + 
                           individual_inter_calls + 
                           individual_intra_calls)
        
        cost_savings = individual_total - api_call_tracker['total_calls']
        efficiency_percent = (cost_savings / individual_total * 100) if individual_total > 0 else 0
        
        print(f"\nCost Optimization:")
        print(f"   Without batching: {individual_total} API calls")
        print(f"   With batching: {api_call_tracker['total_calls']} API calls")
        print(f"   Saved: {cost_savings} calls ({efficiency_percent:.1f}% reduction)")
        
        print(f"\nBatch Efficiency:")
        print(f"   Cluster naming: {CLUSTER_NAMING_BATCH_SIZE} clusters/call")
        print(f"   Inter-cluster edges: {CLUSTER_EDGE_BATCH_SIZE} pairs/call") 
        print(f"   Intra-cluster edges: {EDGE_INFERENCE_BATCH_SIZE} pairs/call")
        
        save_run_parameters(str(doc_output_dir), base_name)
        
        print(f'\nProcessing completed successfully!')
        print(f'Results: {len(concepts)} concepts → {len(cluster_manager.clusters)} clusters → {len(inter_cluster_edges)} edges')
        print(f'Output directory: {doc_output_dir}')
        if document_log_path:
            print(f'Document log saved to: {document_log_path}')
        
        if ENABLE_FILE_LOGGING and SEPARATE_LOG_PER_DOCUMENT:
            finalize_logging()
        
        return results
        
    except Exception as e:
        print(f'\nError processing {file_path}: {str(e)}')
        print("Full traceback:")
        import traceback
        traceback.print_exc()
        
        results['stage'] = 'error'
        results['error'] = str(e)
        # No longer saving intermediate temp files
        
        if ENABLE_FILE_LOGGING and SEPARATE_LOG_PER_DOCUMENT:
            finalize_logging()
        
        return results

def process_interviews(interviews_dir: str = INTERVIEWS_DIRECTORY, 
                      output_dir: str = OUTPUT_DIRECTORY,
                      specific_file: str = DEFAULT_INTERVIEW_FILE) -> List[Dict]:
    
    if not os.path.exists(interviews_dir):
        print(f"Interviews directory '{interviews_dir}' not found.")
        print("Please create the directory and add your documents (.docx, .doc, or .txt files).")
        return []
    
    if specific_file and specific_file != "None" and not PROCESS_ALL_FILES:
        file_path = os.path.join(interviews_dir, specific_file)
        if not os.path.exists(file_path):
            print(f"Specified file '{specific_file}' not found in '{interviews_dir}' directory.")
            print(f"Please check the DEFAULT_INTERVIEW_FILE setting in constants.py or ensure the file exists.")
            return []
        
        document_files = [file_path]
        print(f"Processing specific file: {specific_file}")
        
    else:
        document_files = glob.glob(os.path.join(interviews_dir, "*.docx"))
        document_files.extend(glob.glob(os.path.join(interviews_dir, "*.doc")))
        document_files.extend(glob.glob(os.path.join(interviews_dir, "*.txt")))
        
        if not document_files:
            print(f"No documents found in '{interviews_dir}' directory.")
            print("Please add .docx, .doc, or .txt files to the interviews folder.")
            return []
        
        if PROCESS_ALL_FILES:
            print(f"Processing all {len(document_files)} document(s) (PROCESS_ALL_FILES=True):")
        else:
            print(f"Processing all {len(document_files)} document(s) (requested via command line):")
        for file in document_files:
            print(f"  - {os.path.basename(file)}")
    
    results = []
    for i, file_path in enumerate(document_files, 1):
        print(f"\nProcessing document {i}/{len(document_files)}")
        result = process_single_document(file_path, output_dir)
        if result:
            results.append(result)
    
    print_summary(results)
    
    return results

def print_summary(results: List[Dict]):
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")  
    print("="*60)
    
    for result in results:
        doc_name = result.get('document_name', 'Unknown')
        file_path = result.get('file_path', '')
        file_extension = os.path.splitext(file_path)[1] if file_path else '.docx'
        print(f"\n{doc_name}{file_extension}:")
        
        if 'text_length' in result:
            print(f"  - Text length: {result['text_length']} characters")
        
        if 'concepts_count' in result:
            print(f"  - Concepts extracted: {result['concepts_count']}")
        
        if 'clusters_count' in result:
            print(f"  - Clusters formed: {result['clusters_count']}")
        
        if 'inter_cluster_edges_count' in result:
            print(f"  - Inter-cluster edges: {result['inter_cluster_edges_count']}")
        
        if 'intra_cluster_edges_count' in result:
            print(f"  - Intra-cluster edges: {result['intra_cluster_edges_count']}")
        
        if 'api_calls' in result:
            api_calls = result['api_calls']
            total_calls = api_calls.get('total_calls', 0)
            print(f"  - Total API calls: {total_calls}")
        
        stage = result.get('stage', 'unknown')
        if stage != 'completed':
            print(f"  - Processing stopped at: {stage}")
        
        if 'error' in result:
            print(f"  - Error: {result['error']}")
    
    if ENABLE_FILE_LOGGING:
        print(f"\nLogging Information:")
        if SEPARATE_LOG_PER_DOCUMENT:
            print(f"  - Per-document logs saved to: {LOG_DIRECTORY}/")
        else:
            current_log = get_log_file_path()
            if current_log:
                print(f"  - Session log saved to: {current_log}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    results = process_interviews()
    
    if results:
        print(f"\nSuccessfully processed {len(results)} document(s).")
        print(f"Check the '{OUTPUT_DIRECTORY}' directory for outputs.")
    else:
        print("\nNo documents were processed successfully.") 