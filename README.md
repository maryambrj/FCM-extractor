# FCM Extractor

A Python package for extracting Fuzzy Cognitive Maps (FCMs) from interview transcripts using advanced NLP, clustering techniques, and semantic similarity analysis. This tool automates the conversion of qualitative interview data into structured cognitive maps for research and analysis.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/berijani/fcm-extractor.git
cd fcm-extractor
```

2. **Install dependencies:**
```bash
pip install -r fcm_extractor/requirements.txt
```

3. **Set up API keys:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-gemini-api-key"
```

### Basic Usage

**Process all interview documents:**
```bash
cd fcm_extractor
python run_extraction.py --all
```

**Process a specific document:**
```bash
python run_extraction.py BD007.docx
```

**Get help:**
```bash
python run_extraction.py --help
```

## 📁 Project Structure

```
fcm_extractor/
├── config/
│   ├── constants.py          # Configuration settings
│   └── prompt_templates.json # LLM prompt templates
├── src/
│   ├── core/                 # Core extraction logic
│   ├── clustering/           # Concept clustering algorithms
│   ├── edge_inference/       # Causal relationship detection
│   ├── models/              # LLM clients and metadata
│   └── pipeline/            # Main processing pipeline
├── utils/                   # Visualization and scoring tools
├── interviews/              # Input interview documents
├── fcm_outputs/            # Generated FCM results
└── logs/                   # Processing logs
```

## 🎯 Key Features

- **🤖 Automated FCM Extraction**: Convert interview transcripts to structured cognitive maps
- **🧠 Advanced Clustering**: Semantic clustering using embeddings and LLMs
- **🔗 Causal Inference**: LLM-powered edge detection with confidence scoring
- **📊 Interactive Visualization**: Web-based D3.js visualizations
- **📈 FCM Evaluation**: Semantic similarity-based scoring against ground truth
- **⚡ Post-Clustering Optimization**: Automatic merging of similar clusters
- **🔧 Multi-Model Support**: OpenAI GPT, Google Gemini, and embedding models
- **📝 Comprehensive Logging**: Detailed processing logs for debugging

## 🔧 Pipeline Components

### 1. Concept Extraction
Uses LLMs to identify key concepts from interview text with metadata tracking.

### 2. Semantic Clustering
Groups related concepts using:
- **Embedding-based clustering** (HDBSCAN, UMAP)
- **LLM-based clustering** for semantic understanding
- **Hybrid approaches** combining both methods

### 3. Edge Inference
Determines causal relationships between concept clusters using:
- **Ant Colony Optimization (ACO)** for efficient sampling
- **LLM-based inference** with confidence scoring
- **Batch processing** for cost optimization

### 4. Post-Processing
- Merges similar unconnected clusters
- Refines cluster names and relationships
- Optimizes final FCM structure

### 5. Visualization
Generates interactive HTML visualizations with:
- Hierarchical cluster exploration
- Confidence filtering
- Detailed edge and node information

## 📊 Output Files

### FCM Data Files
- `*_fcm.json` - Complete FCM graph data (nodes, edges, weights)
- `*_cluster_metadata.json` - Detailed cluster information and metadata
- `*_fcm_params.json` - Processing parameters and configuration

### Visualizations
- `*_fcm_interactive.html` - Interactive D3.js visualization
- `*_fcm_static.png` - Static graph visualization

### Evaluation Files
- `*_generated_matrix.csv` - FCM adjacency matrix
- `*_scoring_results.csv` - Evaluation metrics against ground truth

### Logs
- `*_extraction_{timestamp}.log` - Complete processing log for each document
- `logs/` - Directory containing all processing logs

## 🎨 Interactive Visualization

### Features
- **Hierarchical View**: Explore clusters and their internal concepts
- **Confidence Filtering**: Adjust edge visibility based on confidence scores
- **Interactive Controls**: Hover, drag, and click for detailed information
- **Visual Elements**:
  - 🔵 **Blue nodes**: Clusters (click to explore)
  - 🟠 **Orange nodes**: Individual concepts
  - 🟢 **Green lines**: Positive relationships
  - 🔴 **Red lines**: Negative relationships
  - ➖ **Solid lines**: Inter-cluster relationships
  - ⬜ **Dashed lines**: Intra-cluster relationships

### Opening Visualizations
```bash
# macOS
open fcm_outputs/BD007/BD007_fcm_interactive.html

# Linux
xdg-open fcm_outputs/BD007/BD007_fcm_interactive.html

# Windows
start fcm_outputs\BD007\BD007_fcm_interactive.html
```

## 📈 FCM Evaluation

### Score Against Ground Truth
```bash
cd fcm_extractor
python utils/score_fcm.py \
  --gt-path ../ground_truth/BD007.csv \
  --gen-path ../fcm_outputs/BD007/BD007_fcm.json
```

### Create Visualizations from Existing Data
```bash
python utils/visualize_fcm.py \
  --gen-path ../fcm_outputs/BD007/BD007_fcm.json \
  --interactive

python utils/visualize_fcm.py \
  --gen-path ../fcm_outputs/BD007/BD007_fcm.json \
  --summary
```

## ⚙️ Configuration

### Key Settings in `config/constants.py`

**Model Configuration:**
```python
CONCEPT_EXTRACTION_MODEL = "gpt-5-2025-08-07"
EDGE_INFERENCE_MODEL = "gpt-5-2025-08-07"
CLUSTERING_EMBEDDING_MODEL = "sentence-transformers/allenai-specter"
```

**Clustering Settings:**
```python
CLUSTERING_METHOD = "hybrid"  # Options: llm_only, hybrid, embedding_enhanced
CLUSTERING_ALGORITHM = "hdbscan"  # Options: hdbscan, kmeans, agglomerative
HDBSCAN_MIN_CLUSTER_SIZE = 2
```

**Edge Inference:**
```python
EDGE_CONFIDENCE_THRESHOLD = 0.7
ENABLE_INTRA_CLUSTER_EDGES = False
ACO_MAX_ITERATIONS = 5
```

**Post-Clustering:**
```python
ENABLE_POST_CLUSTERING = True
POST_CLUSTERING_SIMILARITY_THRESHOLD = 0.5
```

## 🔧 Advanced Usage

### Programmatic Usage
```python
from fcm_extractor.src.pipeline import process_single_document

# Process single document
result = process_single_document(
    file_path="interviews/BD007.docx",
    output_dir="custom_output/"
)
```

### Component-Level Usage
```python
# Extract concepts from text
from fcm_extractor.src.core import extract_concepts_with_metadata
concepts, metadata = extract_concepts_with_metadata(interview_text)

# Cluster concepts with metadata
from fcm_extractor.src.clustering import cluster_concepts_with_metadata
cluster_manager = cluster_concepts_with_metadata(concepts, metadata)

# Infer causal relationships
from fcm_extractor.src.edge_inference.aco_edge_inference import ACOEdgeInference
aco = ACOEdgeInference()
inter_edges, intra_edges = aco.infer_edges(clusters, interview_text)

# Visualize results
from fcm_extractor.utils.visualize_fcm import create_interactive_visualization
create_interactive_visualization(fcm_graph, "output.html")
```

### Custom Processing Settings
```python
from fcm_extractor.config import constants

# Modify settings at runtime
constants.CLUSTERING_METHOD = "embedding_enhanced"
constants.EDGE_CONFIDENCE_THRESHOLD = 0.8
constants.ENABLE_POST_CLUSTERING = True

# Process with custom settings
result = process_single_document("interview.docx")
```

## 🚨 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM recommended (16GB for large documents)
- **Storage**: 2GB free space for models and outputs

### Python Dependencies
- **Core ML**: numpy, pandas, scikit-learn
- **Deep Learning**: torch, transformers, sentence-transformers
- **Clustering**: umap-learn, hdbscan, numba
- **Visualization**: matplotlib, plotly, networkx, pyvis
- **LLM APIs**: google-generativeai, openai, langchain-*
- **Document Processing**: python-docx
- **Utilities**: python-dotenv, setuptools, protobuf

### API Keys Required
- **OpenAI API Key**: For GPT models (concept extraction, edge inference)
- **Google API Key**: For Gemini models (alternative LLM provider)

## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@software{fcm_extractor,
  title={FCM Extractor: Automated Fuzzy Cognitive Map Extraction from Interview Data},
  author={Berijani, Maryam},
  year={2024},
  url={https://github.com/berijani/fcm-extractor},
  note={A Python package for extracting Fuzzy Cognitive Maps from interview transcripts using NLP and LLMs}
}
```

## 🤝 Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black fcm_extractor/

# Lint code
flake8 fcm_extractor/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/berijani/fcm-extractor/issues)
- **Documentation**: [Wiki](https://github.com/berijani/fcm-extractor/wiki)
- **Email**: berijani@msu.edu

## 🙏 Acknowledgments

- Built with [OpenAI GPT](https://openai.com/) and [Google Gemini](https://ai.google.dev/) APIs
- Uses [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
- Visualization powered by [D3.js](https://d3js.org/) and [Pyvis](https://pyvis.readthedocs.io/)
- Clustering algorithms from [scikit-learn](https://scikit-learn.org/) and [HDBSCAN](https://hdbscan.readthedocs.io/)