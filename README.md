# FCM Extractor

A comprehensive Python toolkit for extracting Fuzzy Cognitive Maps (FCMs) from interview transcripts using advanced NLP, clustering techniques, and semantic similarity analysis. This research tool automates the conversion of qualitative interview data into structured cognitive maps, with extensive evaluation capabilities across multiple datasets and LLM models.

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
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

### Basic Usage

**Process all interview documents:**
```bash
cd fcm_extractor
python run_extraction.py --all
```

**Process a specific document:**
```bash
cd fcm_extractor
python run_extraction.py <file_name>
```

**Run batch evaluation:**
```bash
python run_evaluation_batch.py
```

**Analyze F1 scores and generate reports:**
```bash
python analyze_f1_scores.py --input results/your-dataset/fcm_outputs_analysis
```

**Tutorial and examples:**
```bash
# Open the tutorial notebook for step-by-step guidance
jupyter notebook fcm_tutorial.ipynb

# View structured output examples  
python examples/structured_output_examples.py
```


## 🎯 Key Features

- **🤖 Automated FCM Extraction**: Convert interview transcripts to structured cognitive maps
- **🧠 Advanced Clustering**: Semantic clustering using embeddings and LLMs with optional clustering modes
- **🔗 Causal Inference**: ACO-optimized LLM-powered edge detection with confidence scoring
- **📊 Interactive Visualization**: Web-based D3.js visualizations with hierarchical cluster exploration
- **📈 Comprehensive Evaluation**: Multi-dataset evaluation with F1 scoring, statistical analysis, and visualization
- **⚡ Post-Clustering Optimization**: Automatic merging of similar clusters and unconnected node removal
- **🔧 Multi-Model Support**: OpenAI GPT (4o, 4.1, 4.1-mini), Google Gemini (1.5-flash, 2.0-flash), Anthropic Claude
- **📊 Research-Ready Analysis**: Statistical comparisons, bootstrap confidence intervals, correlation analysis
- **📝 Comprehensive Logging**: Detailed processing logs with timestamps for debugging and reproducibility

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

## 📊 Output Structure

### Individual Document Outputs (fcm_outputs/)
```
fcm_outputs/
├── model_name/                    # e.g., gpt-4o, gemini-1.5-flash
│   └── document_id/               # e.g., BD006, IEA-Wind-CM-AR
│       ├── *_fcm.json            # Complete FCM graph data
│       ├── *_cluster_metadata.json # Cluster information and metadata
│       ├── *_fcm_params.json     # Processing parameters
│       ├── *_fcm_interactive.html # Interactive D3.js visualization
│       ├── *_generated_matrix.csv # FCM adjacency matrix
│       └── *_scoring_results.csv  # Evaluation metrics vs ground truth
```

### Research Results (results/)
```
results/
├── dataset-name/                  # e.g., biodiversity-data, gulf-osw-data
│   ├── interviews/               # Input interview documents
│   ├── ground-truth/             # Reference FCM adjacency matrices
│   ├── with-clustering/          # Results using clustering pipeline
│   │   ├── fcm_outputs/          # Raw model outputs
│   │   └── fcm_outputs_analysis/ # Statistical analysis and plots
│   └── without-clustering/       # Results without clustering
│       ├── fcm_outputs/
│       └── fcm_outputs_analysis/
```

### Analysis Outputs
- `all_f1_scores.csv` - Complete F1 scores across all models and datasets
- `per_model_stats.csv` - Statistical summaries by model
- `per_dataset_stats.csv` - Statistical summaries by dataset
- `best_model_per_dataset.csv` - Top performing model for each dataset
- Various visualization plots (bar charts, boxplots, heatmaps, correlation matrices)

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
open fcm_outputs/path_to_fcm_interactive.html

# Linux
xdg-open fcm_outputs/path_to_fcm_interactive.html

# Windows
start fcm_outputs\path_to_fcm_interactive.html
```

## 📈 Research Evaluation & Analysis

### Batch Processing & Evaluation
```bash
# Run extraction for all models and datasets
python run_evaluation_batch.py
```

### Individual FCM Evaluation
```bash
# Score single FCM against ground truth
python fcm_extractor/utils/score_fcm.py --gt-path ground_truth/BD006.csv --gen-path fcm_outputs/gpt-4o/BD006/BD006_fcm.json

# Create visualizations from existing FCM data
python fcm_extractor/utils/visualize_fcm.py --gen-path fcm_outputs/gpt-4o/BD006/BD006_fcm.json --interactive
```

### Post-Process Visualizations
```bash
# Remove unconnected nodes from HTML visualizations
python edit_visualizations.py -i fcm_outputs
python edit_visualizations.py -i results/biodiversity-data/with-clustering/fcm_outputs -b  # with backup
```

<!-- ## 🔧 Advanced Usage

### Programmatic Usage
```python
from fcm_extractor.src.pipeline import process_single_document

# Process single document
result = process_single_document(
    file_path="interviews/BD007.docx",
    output_dir="custom_output/"
) -->
<!-- ```

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
``` -->

<!-- ### Custom Processing Settings
```python
from fcm_extractor.config import constants

# Modify settings at runtime
constants.CLUSTERING_METHOD = "embedding_enhanced"
constants.EDGE_CONFIDENCE_THRESHOLD = 0.8
constants.ENABLE_POST_CLUSTERING = True

# Process with custom settings
result = process_single_document("interview.docx")
``` -->

<!-- ## 🚨 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM recommended (16GB for large documents)
- **Storage**: 2GB free space for models and outputs -->


## 📋 Supported Datasets

The project includes evaluation on multiple research datasets:

- **Biodiversity Data**: 50+ interview transcripts (BD001-BD089) with ground truth FCMs
- **Caribbean Data**: Stakeholder interviews from Puerto Rico and US Virgin Islands  
- **Gulf OSW Data**: Offshore wind development stakeholder interviews
- **CTF Data**: Additional validation dataset

Each dataset supports both clustering and non-clustering evaluation modes.

## 🧪 Research Capabilities

### Model Comparison
- Systematic evaluation across 6+ LLM models
- Statistical significance testing with bootstrap confidence intervals
- Performance visualization with bar charts, boxplots, and heatmaps

### Clustering Analysis  
- Compare clustering vs non-clustering approaches
- Evaluate semantic clustering effectiveness on FCM quality
- Analyze clustering impact on different stakeholder groups



<!-- ## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@software{fcm_extractor,
  title={FCM Extractor: Automated Fuzzy Cognitive Map Extraction from Interview Data},
  author={Berijanian, Maryam},
  year={2024},
  url={https://github.com/berijani/fcm-extractor},
  note={A comprehensive Python toolkit for extracting Fuzzy Cognitive Maps from interview transcripts using NLP and LLMs with extensive evaluation capabilities}
}
``` -->

<!-- ## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/berijani/fcm-extractor/issues)
- **Email**: berijani@msu.edu
- **Institution**: Michigan State University -->
<!-- 
## 🙏 Acknowledgments

- **LLM APIs**: [OpenAI GPT](https://openai.com/), [Google Gemini](https://ai.google.dev/), [Anthropic Claude](https://anthropic.com/)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) for semantic similarity
- **Clustering**: [HDBSCAN](https://hdbscan.readthedocs.io/), [UMAP](https://umap-learn.readthedocs.io/)
- **Visualization**: [D3.js](https://d3js.org/), [Pyvis](https://pyvis.readthedocs.io/), [Matplotlib](https://matplotlib.org/)
- **Optimization**: Ant Colony Optimization for efficient edge sampling -->