# FCM Extractor

A Python package for extracting Fuzzy Cognitive Maps (FCMs) from interview transcripts using advanced NLP, clustering techniques, and semantic similarity analysis. This tool automates the conversion of qualitative interview data into structured cognitive maps for research and analysis.

 Install dependencies:
```bash
pip install -r fcm_extractor/requirements.txt
```

. Set up environment variables:
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

# Process specific document (supports .docx, .doc, .txt)
python run_extraction.py <file_name>
```


### Configuration

Edit `config/constants.py` 


### FCM Extraction Output:
- `*_fcm.json` - FCM graph data (nodes, edges, weights)
- `*_fcm_interactive.html` - Interactive D3.js visualization
- `*_cluster_metadata.json` - Detailed cluster information
- `*_fcm_params.json`
- `*_generated_matrix.csv`
`*_scoring_results.csv`

### Processing Logs:
- `*_extraction_{timestamp}.log` - Complete processing log for each document
- `logs/` - Directory containing all processing logs

## 📊 FCM Evaluation and Scoring

### Score FCMs Against Ground Truth

The scoring system evaluates generated FCMs against ground truth using semantic similarity:

**Evaluate single FCM:**
```bash
cd fcm_extractor
python utils/score_fcm.py --gt-path ../ground_truth/BD007.csv --gen-path ../fcm_outputs/BD007-1/BD007_fcm.json
python utils/score_fcm.py --gt-path ../ground_truth/P3_MentalModeler_csv.csv --gen-path ../fcm_outputs/P3_Day1_FCM_closed_caption/P3_Day1_FCM_closed_caption_fcm.json
```

## 🎨 FCM Visualization

### Create Interactive HTML Visualizations

The FCM Extractor generates interactive HTML visualizations that allow you to explore the cognitive maps in detail.

**Create visualization from existing FCM JSON:**
```bash
cd fcm_extractor
python utils/visualize_fcm.py --gen-path ../fcm_outputs/BD007-1/BD007_fcm.json --interactive
python utils/visualize_fcm.py --gen-path ../fcm_outputs/P1_Day1_FCM_closed_caption/P1_Day1_FCM_closed_caption_fcm.json --interactive
P1_Day1_FCM_closed_caption_fcm_interactive
```

**Print graph summary only:**
```bash
python utils/visualize_fcm.py --gen-path ../fcm_outputs/BD007-1/BD007_fcm.json --summary
```

```bash
# Open with default browser
open fcm_outputs/BD007-1/BD007_fcm_interactive.html

# Or specify browser
open -a "Google Chrome" fcm_outputs/BD007-1/BD007_fcm_interactive.html
```

**Method 5: Command line (Windows):**
```bash
# Open with default browser
start fcm_outputs\BD007-1\BD007_fcm_interactive.html
```

### Interactive Features

The HTML visualization includes:

- **Hierarchical View**: 
  - Top level shows clusters and inter-cluster relationships
  - Click any cluster to explore internal concepts and relationships
  - Use "Back to Clusters" button to return to overview

- **Interactive Controls**:
  - Confidence filter slider to show/hide edges based on confidence
  - Hover over nodes and edges for detailed information
  - Drag nodes to rearrange the layout

- **Visual Elements**:
  - **Blue nodes**: Clusters (click to explore)
  - **Orange nodes**: Individual concepts
  - **Green lines**: Positive relationships
  - **Red lines**: Negative relationships
  - **Solid lines**: Inter-cluster relationships
  - **Dashed lines**: Intra-cluster relationships

- **Information Display**:
  - Edge weights and confidence scores on hover
  - Node details and cluster information
  - Current view indicator (Cluster Overview or specific cluster)

## 🔧 Advanced Usage

### FCM Extraction Pipeline

```python
from fcm_extractor.src.pipeline import process_single_document

# Process single document with custom settings
result = process_single_document(
    file_path="path/to/document.docx",
    output_dir="custom_output/",
    use_clustering=True
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
- **FCM Evaluation**: Semantic similarity-based scoring against ground truth using Qwen embeddings
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