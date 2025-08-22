"""
FCM Extraction Configuration Constants

This file contains all configurable parameters for the FCM extraction pipeline.
Modify these values to customize the behavior of concept extraction, clustering,
edge inference, and post-processing steps.
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# model = "gemini-1.5-flash"
# model = "gemini-2.0-flash"
# model = "gpt-4.1-2025-04-14"
# model = "gpt-4.1-mini-2025-04-14"
# model = "gpt-4o-2024-05-13"
model = "gpt-5-chat-latest"

CONCEPT_EXTRACTION_MODEL = model
META_PROMPTING_MODEL = model
EDGE_INFERENCE_MODEL = model
LLM_CLUSTERING_MODEL = model

CONCEPT_EXTRACTION_TEMPERATURE = 0.0    # Lower = more deterministic extraction
EDGE_INFERENCE_TEMPERATURE = 0.0        # Lower = more consistent edge inference
META_PROMPTING_TEMPERATURE = 0.5        # Higher = more creative meta-prompting

# =============================================================================
# CLUSTERING CONFIGURATION
# =============================================================================

# Primary clustering method selection
CLUSTERING_METHOD = "no_clustering" 
# Options:
# - "llm_only": Use LLM for all clustering decisions
# - "hybrid": Combine embedding-based and LLM-based clustering
# - "embedding_enhanced": Use only embeddings (zero LLM API calls)
# - "no_clustering": Skip clustering, work with individual concepts

# Embedding model for clustering (when using embedding-based methods)
CLUSTERING_EMBEDDING_MODEL = "sentence-transformers/allenai-specter"

# Alternative options:
# - "all-mpnet-base-v2": Good baseline
# - "all-MiniLM-L12-v2": Faster, good quality
# - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": Multilingual
# - "sentence-transformers/all-distilroberta-v1": Lighter version
# - "sentence-transformers/allenai-specter": For scientific text
# - "sentence-transformers/msmarco-distilbert-base-v4": For search/retrieval
# "sentence-transformers/all-roberta-large-v1"
# "intfloat/e5-large-v2"
# "BAAI/bge-large-en-v1.5"


CLUSTERING_ALGORITHM = "hdbscan"
# Options: "hdbscan", "kmeans", "agglomerative", "spectral"

# Dimensionality reduction before clustering
DIMENSIONALITY_REDUCTION = "umap"
# Options: "tsne", "umap", "pca", "none"

# LLM-based clustering settings
USE_LLM_CLUSTERING = False              # Whether to use LLM for semantic clustering
CLUSTER_NAMING_BATCH_SIZE = 5           # Concepts per batch for cluster naming

# =============================================================================
# CLUSTERING ALGORITHM PARAMETERS
# =============================================================================

# HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE = 2           # Minimum concepts per cluster (lower = more clusters)
HDBSCAN_MIN_SAMPLES = 5               # Minimum samples for core points (lower = more clusters)

# UMAP parameters
UMAP_N_NEIGHBORS = 5                    # Local vs global structure (1-50)
UMAP_MIN_DIST = 0.1                     # Cluster tightness (0.0-1.0)
UMAP_N_COMPONENTS = 2                   # Dimensions after reduction

# t-SNE parameters
TSNE_PERPLEXITY = 10                    # Local vs global structure (5-50)
TSNE_EARLY_EXAGGERATION = 20.0          # Cluster separation (1.0-50.0)
TSNE_LEARNING_RATE = 200.0              # Convergence speed (10.0-1000.0)
TSNE_N_ITER = 1000                      # Number of iterations

# Agglomerative clustering parameters
AGGLOMERATIVE_MAX_CLUSTERS = 50         # Maximum number of clusters
AGGLOMERATIVE_USE_ELBOW_METHOD = True   # Use elbow method to find optimal clusters
AGGLOMERATIVE_USE_DISTANCE_THRESHOLD = False  # Use distance threshold instead of cluster count
AGGLOMERATIVE_DISTANCE_THRESHOLD = 0.5  # Distance threshold for merging

# =============================================================================
# POST-CLUSTERING CONFIGURATION
# =============================================================================

# Post-clustering: Merge unconnected nodes with connected clusters based on similarity
ENABLE_POST_CLUSTERING = False   

# Similarity threshold for merging unconnected clusters with connected ones
POST_CLUSTERING_SIMILARITY_THRESHOLD = 0.5  # Range: 0.0-1.0 (if lower may merge unrelated concepts)


# Embedding model specifically for post-clustering similarity computation
POST_CLUSTERING_EMBEDDING_MODEL = "sentence-transformers/allenai-specter"
# Alternative options:
# - "sentence-transformers/all-MiniLM-L6-v2": Faster but lower quality
# - "sentence-transformers/all-mpnet-base-v2": Good balance of speed/quality
# - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": Multilingual support

# Post-clustering behavior settings
POST_CLUSTERING_MAX_MERGES_PER_CLUSTER = None  # Limit merges per connected cluster (None = unlimited)
POST_CLUSTERING_REQUIRE_MINIMUM_SIMILARITY = True  # Require threshold to be met for any merge

# =============================================================================
# EDGE INFERENCE CONFIGURATION
# =============================================================================

# Edge inference settings
EDGE_CONFIDENCE_THRESHOLD = 0.5        # Minimum confidence for edges (0.0-1.0)
USE_CONFIDENCE_FILTERING = False        # Whether to filter edges by confidence
ENABLE_INTRA_CLUSTER_EDGES = False     # Whether to infer edges within clusters

# Batch sizes for API efficiency
EDGE_INFERENCE_BATCH_SIZE = 5          # Batch size for intra-cluster edges
CLUSTER_EDGE_BATCH_SIZE = 2            # Batch size for inter-cluster edges
MAX_EDGE_INFERENCE_TEXT_LENGTH = 24000  # Maximum text length for edge inference

# =============================================================================
# CONCEPT EXTRACTION CONFIGURATION
# =============================================================================

CONCEPT_EXTRACTION_N_PROMPTS = 2        # Number of extraction prompts to run

# =============================================================================
# ACO (ANT COLONY OPTIMIZATION) PARAMETERS
# =============================================================================

ACO_MAX_ITERATIONS = 5    #3              # Number of ACO iterations
ACO_SAMPLES_PER_ITERATION = 50     #30     # Number of edges to sample per iteration
ACO_EVAPORATION_RATE = 0.15             # Pheromone evaporation rate (0.0-1.0)
ACO_INITIAL_PHEROMONE = 0.01            # Initial pheromone level for all edges
ACO_CONVERGENCE_THRESHOLD = 0.03        # Convergence threshold for early stopping
ACO_GUARANTEE_COVERAGE = True           # Ensure all edges tested at least once

# =============================================================================
# FILE PROCESSING SETTINGS
# =============================================================================

DEFAULT_INTERVIEW_FILE = "BD006.docx"   # Default file when no specific file given
PROCESS_ALL_FILES = False            
INTERVIEWS_DIRECTORY = "../interviews"  # (.docx, .doc, .txt)
OUTPUT_DIRECTORY = "../fcm_outputs"   

# Backward compatibility
INTERVIEW_FILE_NAME = DEFAULT_INTERVIEW_FILE

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

ENABLE_FILE_LOGGING = True            
LOG_DIRECTORY = "../logs"             
LOG_LEVEL = "INFO"                      # Options: DEBUG, INFO, WARNING, ERROR
INCLUDE_TIMESTAMP_IN_LOGS = True     
SEPARATE_LOG_PER_DOCUMENT = True      

# Verbosity settings
META_PROMPTING_VERBOSE = False          # Show meta-prompting decisions

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES = False  # Include intra-cluster edges in evaluation
EVALUATION_INCLUDE_INTRA_CLUSTER_NODES = False  # Include concept nodes in evaluation

# =============================================================================
# META-PROMPTING SETTINGS
# =============================================================================

META_PROMPTING_ENABLED = True          # Use dynamic prompts vs fixed prompts

# Dynamic prompting settings (advanced meta-prompting)
DYNAMIC_PROMPTING_ENABLED = True       # Enable dynamic prompt generation (requires META_PROMPTING_ENABLED)
DYNAMIC_PROMPTING_USE_CACHE = True     # Cache generated prompts for similar contexts
DYNAMIC_PROMPTING_USE_REFLECTION = True # Use self-reflection to refine prompts
DYNAMIC_PROMPTING_TRACK_PERFORMANCE = True # Track prompt performance for learning

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

DEFAULT_CONCEPT_EXTRACTION_PROMPT = (
    "Extract only the most important and salient key concepts or entities from the following text. "
    "Be selective and avoid minor, redundant, or overly specific concepts. "
    "Return ONLY a comma-separated list with no bullets, numbers, asterisks, markdown, explanations, or long phrases. "
    "Each concept should be normalized to lowercase and contain no special characters or formatting."
)

DEFAULT_EDGE_INFERENCE_PROMPT = (
    "Analyze causal relationships between concept pairs based on the text. "
    "Look for words like: causes, leads to, results in, affects, influences, correlates, associated with, depends on, drives, impacts.\n\n"
    "Text: {text}\n\n"
    "Pairs: {pairs}\n\n"
    "For each pair, respond ONLY in this format:\n"
    "Pair X: concept1 -> concept2 (positive/negative, confidence: 0.XX)\n"
    "OR\n"
    "Pair X: no relationship\n\n"
    "Guidelines:\n"
    "- positive: when concept1 increases/improves/enables concept2\n"
    "- negative: when concept1 decreases/hinders/prevents concept2\n"
    "- confidence: how certain you are (0.0-1.00)\n"
    "- Consider indirect relationships and implications\n"
    "- Look for words like: causes, leads to, results in, affects, influences, correlates, associated with, depends on, drives, impacts"
)

DEFAULT_INTER_CLUSTER_EDGE_PROMPT = (
    "You are an expert analyst building a Fuzzy Cognitive Map. Your task is to analyze the causal relationships "
    "between the following pairs of concept clusters, based on the provided text.\n\n"
    "Look for words like: causes, leads to, results in, affects, influences, correlates, associated with, depends on, drives, impacts.\n\n"
    "--- TEXT ---\n"
    "{text}\n"
    "--- END TEXT ---\n\n"
    "For each pair below, analyze the causal relationship.\n"
    "{pairs}\n"
    "--- RESPONSE FORMAT ---\n"
    "For each pair, respond with exactly one line using this format:\n"
    "Pair X: concept1 -> concept2 (positive/negative, confidence: 0.XX)\n"
    "OR\n"
    "Pair X: no relationship\n\n"
    "Guidelines:\n"
    "- positive: when concept1 increases/improves/enables concept2\n"
    "- negative: when concept1 decreases/hinders/prevents concept2\n"
    "- confidence: how certain you are (0.0-1.00)\n"
    "- Focus on direct causal links mentioned in the text\n\n"
    "--- EXAMPLE ---\n"
    "Text: 'Increased stress at work has negatively impacted team productivity.'\n"
    "Pair 1: 'stress' and 'productivity'\n"
    "Response:\n"
    "Pair 1: stress -> productivity (negative, confidence: 0.85)\n"
    "--- END EXAMPLE ---\n\n"
    "Begin your response now. Provide one line per pair."
)

DEFAULT_INTRA_CLUSTER_EDGE_PROMPT = (
    "You are an expert analyst building a Fuzzy Cognitive Map from an interview transcript.\n"
    "Your task is to identify direct causal relationships **WITHIN** a single concept cluster based on the following text:\n"
    "Look for words like: causes, leads to, results in, affects, influences, correlates, associated with, depends on, drives, impacts.\n\n"
    "--- TEXT ---\n"
    "{text}\n"
    "--- END TEXT ---\n\n"
    "Here are the concept pairs to analyze:\n"
    "{pairs}\n"
    "--- EXAMPLE ---\n"
    "Text: '...a lack of sleep really affects my ability to concentrate the next day. I just can't focus when I'm tired.'\n"
    "Pair 1: 'sleep' and 'concentration'\n"
    "Response:\n"
    "Pair 1: sleep -> concentration (positive, confidence: 0.95)\n"
    "--- END EXAMPLE ---\n\n"
    "INSTRUCTIONS:\n"
    "- Focus on direct, specific causal links mentioned in the text.\n"
    "- Respond with exactly one line per pair using this format: 'Pair X: [source concept] -> [target concept] (positive/negative, confidence: 0.XX)'\n"
    "- If no direct causal link is discussed, respond with 'Pair X: no relationship'."
)

LLM_CLUSTERING_PROMPT_TEMPLATE = """Group the following concepts into semantically related clusters. Concepts in the same cluster should be closely related in meaning or domain.

Concepts:
{concepts_str}

Instructions:
1. Create clusters where concepts are semantically similar
2. Each concept should belong to exactly one cluster
3. Aim for 2-{max_clusters} clusters
4. Provide output as: Cluster X: concept_number, concept_number, ...

Example format:
Cluster 1: 1, 3, 5
Cluster 2: 2, 4
Cluster 3: 6, 7, 8

Your clustering:"""

LLM_CLUSTER_REFINEMENT_PROMPT_TEMPLATE = """Refine the following concept groups by splitting them into more focused sub-clusters if necessary. In this case, the sub-clusters must be semantically or grammatically unified.

{cluster_descriptions}

Instructions:
1. For each group, create a few focused sub-clusters if necessary
2. Each concept should belong to exactly one sub-cluster within its group
3. Provide output as: Group X.Y: concept_number, concept_number, ...

Example format:
Group 1.1: 1, 3
Group 1.2: 2, 4, 5
Group 2.1: 1, 2
Group 2.2: 3

Your refined clustering:"""

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_configuration():
    """Validate that configuration parameters are within acceptable ranges."""
    errors = []
    
    # Validate post-clustering parameters
    if not 0.0 <= POST_CLUSTERING_SIMILARITY_THRESHOLD <= 1.0:
        errors.append("POST_CLUSTERING_SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
    
    # Validate other thresholds
    if not 0.0 <= EDGE_CONFIDENCE_THRESHOLD <= 1.0:
        errors.append("EDGE_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
    
    # Validate temperature settings
    for temp_name, temp_value in [
        ("CONCEPT_EXTRACTION_TEMPERATURE", CONCEPT_EXTRACTION_TEMPERATURE),
        ("EDGE_INFERENCE_TEMPERATURE", EDGE_INFERENCE_TEMPERATURE),
        ("META_PROMPTING_TEMPERATURE", META_PROMPTING_TEMPERATURE)
    ]:
        if not 0.0 <= temp_value <= 2.0:
            errors.append(f"{temp_name} should be between 0.0 and 2.0")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))

# Validate configuration on import
if __name__ != "__main__":
    validate_configuration()