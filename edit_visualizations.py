#!/usr/bin/env python3
"""
Standalone script to post-process existing FCM HTML visualization files
to remove unconnected nodes from the visualization only.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Set, List, Any


def extract_data_from_html(html_content: str) -> Dict[str, Any]:
    """Extract embedded JSON data from HTML content."""
    data = {}
    
    # Extract cluster data
    cluster_match = re.search(r'const clusterData = ({.*?});', html_content, re.DOTALL)
    if cluster_match:
        try:
            data['clusterData'] = json.loads(cluster_match.group(1))
        except json.JSONDecodeError as e:
            print(f"Error parsing cluster data: {e}")
            data['clusterData'] = {}
    
    # Extract concept data
    concept_match = re.search(r'const conceptData = ({.*?});', html_content, re.DOTALL)
    if concept_match:
        try:
            data['conceptData'] = json.loads(concept_match.group(1))
        except json.JSONDecodeError as e:
            print(f"Error parsing concept data: {e}")
            data['conceptData'] = {}
    
    # Extract inter-cluster edges
    inter_edges_match = re.search(r'const interClusterEdges = (\[.*?\]);', html_content, re.DOTALL)
    if inter_edges_match:
        try:
            data['interClusterEdges'] = json.loads(inter_edges_match.group(1))
        except json.JSONDecodeError as e:
            print(f"Error parsing inter-cluster edges: {e}")
            data['interClusterEdges'] = []
    
    # Extract cluster concepts
    cluster_concepts_match = re.search(r'const clusterConcepts = ({.*?});', html_content, re.DOTALL)
    if cluster_concepts_match:
        try:
            data['clusterConcepts'] = json.loads(cluster_concepts_match.group(1))
        except json.JSONDecodeError as e:
            print(f"Error parsing cluster concepts: {e}")
            data['clusterConcepts'] = {}
    
    # Extract cluster edges
    cluster_edges_match = re.search(r'const clusterEdges = ({.*?});', html_content, re.DOTALL)
    if cluster_edges_match:
        try:
            data['clusterEdges'] = json.loads(cluster_edges_match.group(1))
        except json.JSONDecodeError as e:
            print(f"Error parsing cluster edges: {e}")
            data['clusterEdges'] = {}
    
    return data


def find_connected_nodes(data: Dict[str, Any]) -> Set[str]:
    """Find all nodes that are connected via edges."""
    connected_nodes = set()
    
    # Add nodes from inter-cluster edges
    for edge in data.get('interClusterEdges', []):
        if len(edge) >= 2:
            connected_nodes.add(edge[0])  # source
            connected_nodes.add(edge[1])  # target
    
    # Add nodes from intra-cluster edges
    for cluster_edges in data.get('clusterEdges', {}).values():
        for edge in cluster_edges:
            if len(edge) >= 2:
                connected_nodes.add(edge[0])  # source
                connected_nodes.add(edge[1])  # target
    
    return connected_nodes


def filter_unconnected_nodes(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove unconnected nodes from all data structures."""
    connected_nodes = find_connected_nodes(data)
    
    filtered_data = {}
    
    # Filter cluster data
    filtered_data['clusterData'] = {
        node_id: node_data for node_id, node_data in data.get('clusterData', {}).items()
        if node_id in connected_nodes
    }
    
    # Filter concept data
    filtered_data['conceptData'] = {
        node_id: node_data for node_id, node_data in data.get('conceptData', {}).items()
        if node_id in connected_nodes
    }
    
    # Keep all edges (they only contain connected nodes by definition)
    filtered_data['interClusterEdges'] = data.get('interClusterEdges', [])
    
    # Filter cluster concepts
    filtered_data['clusterConcepts'] = {}
    for cluster_id, concepts in data.get('clusterConcepts', {}).items():
        if cluster_id in connected_nodes:
            # Filter individual concepts within the cluster
            filtered_concepts = [
                concept for concept in concepts
                if len(concept) >= 2 and concept[0] in connected_nodes
            ]
            if filtered_concepts:  # Only include cluster if it has connected concepts
                filtered_data['clusterConcepts'][cluster_id] = filtered_concepts
    
    # Keep cluster edges as is (they only contain connected nodes)
    filtered_data['clusterEdges'] = data.get('clusterEdges', {})
    
    return filtered_data


def update_html_with_filtered_data(html_content: str, filtered_data: Dict[str, Any]) -> str:
    """Replace the embedded data in HTML with filtered data."""
    
    # Replace cluster data
    cluster_json = json.dumps(filtered_data['clusterData'])
    html_content = re.sub(
        r'const clusterData = {.*?};',
        f'const clusterData = {cluster_json};',
        html_content,
        flags=re.DOTALL
    )
    
    # Replace concept data
    concept_json = json.dumps(filtered_data['conceptData'])
    html_content = re.sub(
        r'const conceptData = {.*?};',
        f'const conceptData = {concept_json};',
        html_content,
        flags=re.DOTALL
    )
    
    # Replace inter-cluster edges
    inter_edges_json = json.dumps(filtered_data['interClusterEdges'])
    html_content = re.sub(
        r'const interClusterEdges = \[.*?\];',
        f'const interClusterEdges = {inter_edges_json};',
        html_content,
        flags=re.DOTALL
    )
    
    # Replace cluster concepts
    cluster_concepts_json = json.dumps(filtered_data['clusterConcepts'])
    html_content = re.sub(
        r'const clusterConcepts = {.*?};',
        f'const clusterConcepts = {cluster_concepts_json};',
        html_content,
        flags=re.DOTALL
    )
    
    # Replace cluster edges
    cluster_edges_json = json.dumps(filtered_data['clusterEdges'])
    html_content = re.sub(
        r'const clusterEdges = {.*?};',
        f'const clusterEdges = {cluster_edges_json};',
        html_content,
        flags=re.DOTALL
    )
    
    return html_content


def process_html_file(input_path: str, output_path: str = None) -> bool:
    """Process a single HTML file to remove unconnected nodes."""
    try:
        # Read the original HTML
        with open(input_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract data from HTML
        print(f"Extracting data from {input_path}...")
        data = extract_data_from_html(html_content)
        
        if not data.get('clusterData') and not data.get('conceptData'):
            print(f"Warning: No FCM data found in {input_path}")
            return False
        
        # Count original nodes
        original_cluster_count = len(data.get('clusterData', {}))
        original_concept_count = len(data.get('conceptData', {}))
        
        # Filter unconnected nodes
        print("Filtering unconnected nodes...")
        filtered_data = filter_unconnected_nodes(data)
        
        # Count filtered nodes
        filtered_cluster_count = len(filtered_data['clusterData'])
        filtered_concept_count = len(filtered_data['conceptData'])
        
        print(f"  Clusters: {original_cluster_count} → {filtered_cluster_count} "
              f"({original_cluster_count - filtered_cluster_count} removed)")
        print(f"  Concepts: {original_concept_count} → {filtered_concept_count} "
              f"({original_concept_count - filtered_concept_count} removed)")
        
        # Update HTML with filtered data
        updated_html = update_html_with_filtered_data(html_content, filtered_data)
        
        # Add a comment to indicate processing
        updated_html = updated_html.replace(
            '<title>Hierarchical FCM Visualization</title>',
            '<title>Hierarchical FCM Visualization (Unconnected nodes removed)</title>'
        )
        
        # Write the updated HTML
        output_file = output_path or input_path
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(updated_html)
        
        print(f"Updated HTML saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def find_html_files(directory: str) -> List[str]:
    """Find all FCM interactive HTML files in the directory."""
    html_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('_fcm_interactive.html'):
                html_files.append(os.path.join(root, file))
    return html_files


def main():
    parser = argparse.ArgumentParser(
        description='Remove unconnected nodes from existing FCM HTML visualizations'
    )
    parser.add_argument(
        '--input', '-i',
        help='Input HTML file or directory containing HTML files'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file (for single file) or directory (optional, defaults to overwrite input)'
    )
    parser.add_argument(
        '--backup', '-b',
        action='store_true',
        help='Create backup copies before modifying files'
    )
    
    args = parser.parse_args()
    
    if not args.input:
        # Default to fcm_outputs directory
        args.input = 'fcm_outputs'
        if not os.path.exists(args.input):
            print("Error: Please specify --input file or directory")
            return
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        if args.backup:
            backup_path = str(input_path) + '.backup'
            print(f"Creating backup: {backup_path}")
            import shutil
            shutil.copy2(input_path, backup_path)
        
        success = process_html_file(str(input_path), args.output)
        if success:
            print("✓ File processed successfully")
        else:
            print("✗ Failed to process file")
            
    elif input_path.is_dir():
        # Process directory
        html_files = find_html_files(str(input_path))
        
        if not html_files:
            print(f"No *_fcm_interactive.html files found in {input_path}")
            return
        
        print(f"Found {len(html_files)} HTML files to process...")
        
        success_count = 0
        for html_file in html_files:
            print(f"\nProcessing: {html_file}")
            
            if args.backup:
                backup_path = html_file + '.backup'
                print(f"Creating backup: {backup_path}")
                import shutil
                shutil.copy2(html_file, backup_path)
            
            if process_html_file(html_file):
                success_count += 1
        
        print(f"\n✓ Successfully processed {success_count}/{len(html_files)} files")
        
    else:
        print(f"Error: {input_path} is neither a file nor a directory")


if __name__ == "__main__":
    main()