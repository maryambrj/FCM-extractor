import json
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional
import argparse
import os
from pyvis.network import Network

def load_fcm_from_json(json_path: str) -> nx.DiGraph:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    if 'nodes' in data:
        for node in data['nodes']:
            node_id = node.get('id', 'unknown_node')
            concepts = node.get('concepts', [])
            G.add_node(node_id, 
                      concepts=concepts,
                      concept_list=concepts.split(', ') if isinstance(concepts, str) else concepts,
                      type='cluster',
                      label=node_id)
    
    concept_to_cluster = {}
    
    if 'edges' in data:
        for edge in data['edges']:
            source = edge.get('source')
            target = edge.get('target')
            weight = edge.get('weight', 0)
            confidence = edge.get('confidence', 1.0)
            edge_type = edge.get('type', 'inter_cluster')
            
            if source and target:
              
                if edge_type == 'intra_cluster':
                    source_cluster = None
                    target_cluster = None
                    
                    for cluster_node in G.nodes():
                        if G.nodes[cluster_node].get('type') == 'cluster':
                            cluster_concepts = G.nodes[cluster_node].get('concept_list', [])
                            if source in cluster_concepts:
                                source_cluster = cluster_node
                            if target in cluster_concepts:
                                target_cluster = cluster_node
                    
                    if source not in G:
                        G.add_node(source,
                                  concepts=source,
                                  type='concept',
                                  cluster=source_cluster or 'unknown',
                                  label=source)
                    
                    if target not in G:
                        G.add_node(target,
                                  concepts=target,
                                  type='concept', 
                                  cluster=target_cluster or 'unknown',
                                  label=target)
                
                G.add_edge(source, target, 
                          weight=weight, 
                          confidence=confidence, 
                          type=edge_type)
    
    return G

def create_interactive_visualization(G: nx.DiGraph, output_file: str, min_confidence: float = 0.3):
    import json
    
    cluster_nodes = {node: data for node, data in G.nodes(data=True) if data.get('type') == 'cluster'}
    concept_nodes = {node: data for node, data in G.nodes(data=True) if data.get('type') == 'concept'}
    
    inter_cluster_edges = []
    intra_cluster_edges = []
    
    for u, v, edge_data in G.edges(data=True):
        edge_type = edge_data.get('type', 'inter_cluster')
        if edge_type == 'inter_cluster':
            inter_cluster_edges.append([u, v, dict(edge_data)])
        else:
            intra_cluster_edges.append([u, v, dict(edge_data)])
    
    cluster_concepts = {}
    cluster_edges = {}
    
    for concept, data in concept_nodes.items():
        cluster_name = data.get('cluster', 'Unknown')
        if cluster_name not in cluster_concepts:
            cluster_concepts[cluster_name] = []
            cluster_edges[cluster_name] = []
        cluster_concepts[cluster_name].append([concept, dict(data)])
    
    for u, v, edge_data in intra_cluster_edges:
        u_cluster = concept_nodes.get(u, {}).get('cluster', 'Unknown')
        v_cluster = concept_nodes.get(v, {}).get('cluster', 'Unknown')
        
        if u_cluster == v_cluster and u_cluster in cluster_edges:
            cluster_edges[u_cluster].append([u, v, dict(edge_data)])

    cluster_nodes_json = json.dumps({k: dict(v) for k, v in cluster_nodes.items()})
    concept_nodes_json = json.dumps({k: dict(v) for k, v in concept_nodes.items()})
    inter_cluster_edges_json = json.dumps(inter_cluster_edges)
    cluster_concepts_json = json.dumps(cluster_concepts)
    cluster_edges_json = json.dumps(cluster_edges)

   
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hierarchical FCM Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px;
            background: #f5f5f5;
        }}
        
        #network {{ 
            width: 100%; 
            height: 600px; 
            border: 2px solid #ccc; 
            background: white;
            border-radius: 8px;
        }}
        
        #controls {{
            margin-bottom: 15px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .btn {{
            padding: 8px 16px;
            margin: 5px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }}
        
        .btn-primary {{ background: #4A90E2; color: white; }}
        .btn-primary:hover {{ background: #357ABD; }}
        
        .btn-secondary {{ background: #6c757d; color: white; }}
        .btn-secondary:hover {{ background: #545b62; }}
        
        #current-view {{
            font-weight: bold;
            color: #333;
            margin-left: 10px;
        }}
        
        #legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            border: 1px solid #ccc;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 12px;
            max-width: 200px;
        }}
        
        .legend-item {{
            margin: 5px 0;
            display: flex;
            align-items: center;
        }}
        
        .legend-color {{
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .legend-line {{
            width: 20px;
            height: 2px;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <h1>Fuzzy Cognitive Map - Hierarchical View</h1>
    
    <div id="controls">
        <button class="btn btn-secondary" id="back-btn" onclick="showClusterView()" style="display: none;">
            ← Back to Clusters
        </button>
        <span id="current-view">Cluster Overview</span>
        <div style="float: right;">
            <label>Min Confidence: </label>
            <input type="range" id="confidence-slider" min="0" max="1" step="0.1" value="{min_confidence}" 
                   onchange="updateConfidenceFilter(this.value)">
            <span id="confidence-value">{min_confidence}</span>
        </div>
    </div>
    
    <div id="network"></div>
    
    <div id="legend">
        <h4>Legend</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: #4A90E2;"></div>
            <span>Clusters</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #FF9500;"></div>
            <span>Concepts</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: #28A745;"></div>
            <span>Positive relationship</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: #DC3545;"></div>
            <span>Negative relationship</span>
        </div>
        <div style="margin-top: 10px; font-style: italic; color: #666;">
            Click clusters to explore internal concepts
        </div>
    </div>

    <script>
        // Embedded data
        const clusterData = {cluster_nodes_json};
        const conceptData = {concept_nodes_json};
        const interClusterEdges = {inter_cluster_edges_json};
        const clusterConcepts = {cluster_concepts_json};
        const clusterEdges = {cluster_edges_json};
        
        let network;
        let currentView = 'clusters';
        let currentCluster = null;
        let minConfidence = {min_confidence};
        
        // Network options
        const options = {{
            nodes: {{
                font: {{ color: 'white', size: 14 }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                arrows: {{ to: {{ enabled: true, scaleFactor: 1.2 }} }},
                shadow: true,
                smooth: {{ type: 'continuous' }}
            }},
            physics: {{
                stabilization: {{ iterations: 100 }},
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springConstant: 0.001,
                    springLength: 200
                }}
            }},
            interaction: {{
                hover: true,
                selectConnectedEdges: false
            }}
        }};
        
        function initNetwork() {{
            const container = document.getElementById('network');
            const data = getClusterViewData();
            network = new vis.Network(container, data, options);
            
            // Handle cluster clicks for drill-down
            network.on("click", function(params) {{
                if (params.nodes.length > 0 && currentView === 'clusters') {{
                    const clickedNode = params.nodes[0];
                    if (clusterConcepts[clickedNode]) {{
                        showClusterDetail(clickedNode);
                    }}
                }}
            }});
        }}
        
        function getClusterViewData() {{
            const nodes = [];
            const edges = [];
            
            // Add cluster nodes
            Object.entries(clusterData).forEach(([clusterId, data]) => {{
                const concepts = data.concepts || [];
                const conceptsStr = Array.isArray(concepts) ? concepts.join(', ') : concepts;
                
                nodes.push({{
                    id: clusterId,
                    label: clusterId,
                    title: `Cluster: ${{clusterId}}\\nConcepts: ${{conceptsStr}}\\nClick to explore internal relationships`,
                    color: '#4A90E2',
                    size: 25 + (conceptsStr.split(',').length * 2),
                    font: {{ color: 'white' }}
                }});
            }});
            
            // Add inter-cluster edges
            interClusterEdges.forEach(([source, target, edgeData]) => {{
                const confidence = edgeData.confidence || 1.0;
                const weight = edgeData.weight || 0;
                
                if (confidence >= minConfidence) {{
                    edges.push({{
                        from: source,
                        to: target,
                        color: weight > 0 ? '#28A745' : '#DC3545',
                        width: Math.abs(weight) * 3 + 1,
                        title: `${{source}} → ${{target}}\\nWeight: ${{weight}}\\nConfidence: ${{confidence.toFixed(2)}}`,
                        dashes: false
                    }});
                }}
            }});
            
            return {{ nodes: nodes, edges: edges }};
        }}
        
        function getClusterDetailData(clusterId) {{
            const nodes = [];
            const edges = [];
            
            // Add concept nodes for this cluster
            if (clusterConcepts[clusterId]) {{
                clusterConcepts[clusterId].forEach(([conceptId, data]) => {{
                    nodes.push({{
                        id: conceptId,
                        label: conceptId,
                        title: `Concept: ${{conceptId}}\\nFrom cluster: ${{clusterId}}`,
                        color: '#FF9500',
                        size: 20,
                        font: {{ color: 'white' }}
                    }});
                }});
            }}
            
            // Add intra-cluster edges
            if (clusterEdges[clusterId]) {{
                clusterEdges[clusterId].forEach(([source, target, edgeData]) => {{
                    const confidence = edgeData.confidence || 1.0;
                    const weight = edgeData.weight || 0;
                    
                    if (confidence >= minConfidence) {{
                        edges.push({{
                            from: source,
                            to: target,
                            color: weight > 0 ? '#28A745' : '#DC3545',
                            width: Math.abs(weight) * 3 + 1,
                            title: `${{source}} → ${{target}}\\nWeight: ${{weight}}\\nConfidence: ${{confidence.toFixed(2)}}`,
                            dashes: true
                        }});
                    }}
                }});
            }}
            
            return {{ nodes: nodes, edges: edges }};
        }}
        
        function showClusterView() {{
            currentView = 'clusters';
            currentCluster = null;
            const data = getClusterViewData();
            network.setData(data);
            
            document.getElementById('back-btn').style.display = 'none';
            document.getElementById('current-view').textContent = 'Cluster Overview';
        }}
        
        function showClusterDetail(clusterId) {{
            currentView = 'concepts';
            currentCluster = clusterId;
            const data = getClusterDetailData(clusterId);
            network.setData(data);
            
            document.getElementById('back-btn').style.display = 'inline-block';
            document.getElementById('current-view').textContent = `Inside Cluster: ${{clusterId}}`;
        }}
        
        function updateConfidenceFilter(value) {{
            minConfidence = parseFloat(value);
            document.getElementById('confidence-value').textContent = value;
            
            // Refresh current view
            if (currentView === 'clusters') {{
                showClusterView();
            }} else {{
                showClusterDetail(currentCluster);
            }}
        }}
        
        // Initialize the network when page loads
        window.onload = function() {{
            initNetwork();
        }};
    </script>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f'Hierarchical interactive visualization saved to {output_file}')
    print('Features:')
    print('   • Top level: Clusters and inter-cluster relationships')
    print('   • Click any cluster to explore internal concepts and relationships') 
    print('   • Use "Back to Clusters" button to return to overview')
    print('   • Adjust confidence filter to show/hide edges')
    print('Tip: Open the file in your browser for interactive exploration!')

def print_graph_summary(G: nx.DiGraph):
    print("\n" + "="*50)
    print("FCM GRAPH SUMMARY")
    print("="*50)
    
    print(f"Number of clusters (nodes): {G.number_of_nodes()}")
    print(f"Number of relationships (edges): {G.number_of_edges()}")
    
    print("\nCLUSTERS:")
    for node in G.nodes():
        concept_list = G.nodes[node].get('concept_list')
        if concept_list and isinstance(concept_list, list):
            concepts_display = ', '.join(concept_list)
        else:
            concepts_display = G.nodes[node]['concepts']
        print(f"  Cluster {node}: {concepts_display}")
    
    print("\nRELATIONSHIPS:")
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        confidence = data['confidence']
        direction = "→" if weight > 0 else "←"
        print(f"  Cluster {u} {direction} Cluster {v} (weight: {weight:.2f}, confidence: {confidence:.2f})")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Visualize FCM graphs from JSON files')
    parser.add_argument('--gen-path', required=True, help='Path to generated FCM JSON')
    parser.add_argument('--interactive', action='store_true', help='Create interactive visualization')
    parser.add_argument('--summary', action='store_true', help='Print graph summary')
    
    args = parser.parse_args()
    
    data_name = os.path.splitext(os.path.basename(args.gen_path))[0].replace('_fcm', '')
    output_dir = os.path.dirname(args.gen_path)

    if not os.path.exists(args.gen_path):
        print(f"Error: File {args.gen_path} not found!")
        return
    
    G = load_fcm_from_json(args.gen_path)
    
    if args.summary or not args.interactive:
        print_graph_summary(G)
    
    if args.interactive or (not args.summary):
        create_interactive_visualization(G, os.path.join(output_dir, f"{data_name}_fcm_interactive.html"))

if __name__ == "__main__":
    main() 