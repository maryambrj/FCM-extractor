# FCM Extractor

A comprehensive Python package for extracting Fuzzy Cognitive Maps (FCMs) from interview transcripts using advanced NLP, clustering techniques, and semantic similarity analysis. This tool automates the conversion of qualitative interview data into structured cognitive maps for research and analysis.

## 📁 Project Structure

```
fcm_extractor/
├── run_extraction.py       # Main entry point
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files
│   ├── constants.py       # All configuration constants
│   └── prompt_templates.json  # LLM prompt templates
├── src/                   # Source code
│   ├── core/             # Core functionality
│   │   ├── extract_concepts.py  # Concept extraction
│   │   └── build_graph.py       # Graph construction
│   ├── models/           # Data models and clients
│   │   ├── cluster_metadata.py  # Cluster metadata
│   │   ├── llm_client.py       # LLM API client
│   │   └── meta_prompting_agent.py  # Meta-prompting
│   ├── clustering/       # Clustering algorithms
│   │   ├── embed_and_cluster.py  # Embedding-based
│   │   └── improved_clustering.py # Advanced clustering
│   ├── edge_inference/   # Edge inference
│   │   └── edge_inference.py    # Causal relationships
│   └── pipeline/         # Processing pipelines
│       ├── process_interviews.py # Interview processing
│       └── resume_processing.py  # Resume processing
├── utils/                # Utilities
│   ├── logging_utils.py  # Logging helpers
│   └── visualize_fcm.py  # FCM visualization
└── tests/               # Unit tests
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- GPU recommended for faster embedding computation
- OpenAI API key and/or Google AI Studio API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd new-FCM-extraction
```

2. Install dependencies:
```bash
pip install -r fcm_extractor/requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-gemini-api-key"
```

### Basic Usage

**Extract FCMs from interviews:**
```bash
cd fcm_extractor
# Process all documents
python run_extraction.py

# Process specific document
python run_extraction.py BD007.docx
```


### Configuration

Edit `config/constants.py` to customize:

**Model Settings:**
- `MODEL_PROVIDER`: Choose between "openai" or "google"
- `MODEL_NAME`: Specific model (e.g., "gpt-4o", "gemini-1.5-pro")
- `EMBEDDING_MODEL`: Model for semantic similarity (default: "Qwen/Qwen3-Embedding-0.6B")

**Clustering Parameters:**
- `DEFAULT_NUM_CLUSTERS`: Number of concept clusters
- `MIN_CLUSTER_SIZE`: Minimum concepts per cluster
- `USE_IMPROVED_CLUSTERING`: Advanced clustering algorithm

**Evaluation Settings:**
- `EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES`: Include intra-cluster edges in evaluation
- `EDGE_CONFIDENCE_THRESHOLD`: Minimum confidence for edge inference

**I/O Directories:**
- `BASE_INPUT_DIR`: Input documents directory
- `BASE_OUTPUT_DIR`: Output files directory

## 📊 Output Files

### FCM Extraction Output:
- `*_fcm.json` - FCM graph data (nodes, edges, weights)
- `*_fcm_interactive.html` - Interactive D3.js visualization
- `*_fcm_static.png` - Static NetworkX graph image
- `*_cluster_metadata.json` - Detailed cluster information

### Processing Logs:
- `*_extraction_{timestamp}.log` - Complete processing log for each document
- `logs/` - Directory containing all processing logs

## 🔧 Advanced Usage

### FCM Extraction Pipeline

```python
from fcm_extractor.src.pipeline import process_single_document

# Process single document with custom settings
result = process_single_document(
    file_path="path/to/document.docx",
    output_dir="custom_output/",
    use_improved_clustering=True
)
```

### Custom Processing Settings

```python
from fcm_extractor.config.constants import *

# Modify clustering behavior
CLUSTERING_METHOD = "hybrid"  # or "llm_only", "embedding_enhanced"
HDBSCAN_MIN_CLUSTER_SIZE = 3  # Adjust cluster granularity
EDGE_CONFIDENCE_THRESHOLD = 0.8  # Higher = more conservative edges

# Process with custom settings
from fcm_extractor.src.pipeline import process_single_document
result = process_single_document("interview.docx", custom_config=True)
```

### Component-Level Usage

```python
# Extract concepts from text
from fcm_extractor.src.core import extract_concepts
concepts = extract_concepts(interview_text)

# Cluster concepts with metadata
from fcm_extractor.src.clustering import cluster_concepts_with_metadata
clusters = cluster_concepts_with_metadata(concepts, metadata)

# Infer causal relationships
from fcm_extractor.src.edge_inference import infer_edges
edges = infer_edges(clusters, interview_text)

# Visualize results
from fcm_extractor.utils.visualize_fcm import create_interactive_visualization
create_interactive_visualization(fcm_data, "output.html")
```

## 🎯 Key Features

- **Automated FCM Extraction**: Convert interview transcripts to structured cognitive maps
- **Advanced Clustering**: Semantic clustering of concepts using embeddings
- **Causal Inference**: LLM-powered edge detection with confidence scoring
- **Interactive Visualization**: Web-based D3.js visualizations
- **Confidence Scoring**: Edge relationships include confidence scores for reliability assessment
- **Post-Clustering Optimization**: Merges similar unconnected clusters automatically
- **Multi-Model Support**: OpenAI GPT-4, Google Gemini, and various embedding models
- **Flexible Pipeline**: Modular architecture for custom workflows

## 🔧 Pipeline Components

### Core Processing Steps
The FCM extraction pipeline consists of several key components:
- **Concept Extraction**: Uses LLMs to identify key concepts from interview text
- **Semantic Clustering**: Groups related concepts using embeddings and/or LLM analysis
- **Edge Inference**: Determines causal relationships between concept clusters
- **Post-Processing**: Merges similar clusters and refines the final FCM structure
- **Visualization**: Generates interactive and static visualizations of the FCM

## 🚨 Requirements

- Python 3.8+
- PyTorch (for embeddings)
- Transformers (Hugging Face)
- NetworkX, Matplotlib, Plotly
- scikit-learn
- pandas, numpy
- OpenAI API key (if using GPT models)
- Google API key (if using Gemini models)

## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@software{fcm_extractor,
  title={FCM Extractor: Automated Fuzzy Cognitive Map Extraction from Interview Data},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

[Specify your license here]