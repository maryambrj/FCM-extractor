#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the necessary functions from the existing modules
from utils.visualize_fcm import load_fcm_from_json, create_interactive_visualization
from config.constants import OUTPUT_DIRECTORY

def check_document_status(doc_name: str, output_dir: str = OUTPUT_DIRECTORY):
    """Check the status of a document's processing."""
    
    doc_output_dir = os.path.join(output_dir, doc_name)
    base_name = doc_name
    
    # Check for various output files
    fcm_json = os.path.join(doc_output_dir, f"{base_name}_fcm.json")
    interactive_viz = os.path.join(doc_output_dir, f"{base_name}_fcm_interactive.html")
    temp_file = os.path.join(doc_output_dir, f"{base_name}_temp.json")
    
    status = {
        'fcm_json': os.path.exists(fcm_json),
        'interactive_viz': os.path.exists(interactive_viz),
        'temp_file': os.path.exists(temp_file)
    }
    
    print(f"\n📋 Status for {doc_name}:")
    print(f"   FCM JSON: {'✅' if status['fcm_json'] else '❌'}")
    print(f"   Interactive visualization: {'✅' if status['interactive_viz'] else '❌'}")
    
    if status['temp_file']:
        try:
            with open(temp_file, 'r') as f:
                temp_data = json.load(f)
            print(f"   Stage: {temp_data.get('stage', 'unknown')}")
            print(f"   Completed: {'✅' if temp_data.get('completed', False) else '❌'}")
        except:
            print(f"   Temp file: ⚠️ (corrupted)")
    
    return status

def regenerate_visualizations(doc_name: str, output_dir: str = OUTPUT_DIRECTORY):
    """Regenerate visualizations for a document."""
    
    doc_output_dir = os.path.join(output_dir, doc_name)
    fcm_file = os.path.join(doc_output_dir, f"{doc_name}_fcm.json")
    
    if not os.path.exists(fcm_file):
        print(f"❌ No FCM JSON file found for {doc_name}")
        return False
    
    try:
        print(f"🎨 Regenerating visualizations for {doc_name}...")
        
        # Load the existing FCM JSON
        G = load_fcm_from_json(fcm_file)
        
        # Create output paths
        interactive_viz_path = os.path.join(doc_output_dir, f"{doc_name}_fcm_interactive.html")
        
        # Generate visualizations
        print("   Creating interactive visualization...")
        create_interactive_visualization(G, interactive_viz_path)
        
        # Update temp file to mark as completed
        temp_file = os.path.join(doc_output_dir, f"{doc_name}_temp.json")
        if os.path.exists(temp_file):
            try:
                with open(temp_file, 'r') as f:
                    results = json.load(f)
                results['completed'] = True
                results['stage'] = 'completed'
                results['output_file'] = fcm_file
                with open(temp_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print("   Updated temp file status")
            except Exception as e:
                print(f"   Warning: Could not update temp file: {e}")
        
        print(f"✅ Visualizations completed for {doc_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating visualizations for {doc_name}: {e}")
        return False

def process_all_documents(output_dir: str = OUTPUT_DIRECTORY):
    """Check and process all documents in the output directory."""
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory {output_dir} does not exist")
        return
    
    completed_count = 0
    processed_count = 0
    
    # Find all subdirectories (document folders)
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            doc_name = item
            status = check_document_status(doc_name, output_dir)
            
            if status['fcm_json']:
                if not status['interactive_viz']:
                    print(f"\n🔄 Missing visualizations for {doc_name}, regenerating...")
                    if regenerate_visualizations(doc_name, output_dir):
                        processed_count += 1
                else:
                    print(f"✅ {doc_name} is fully completed")
                    completed_count += 1
            else:
                print(f"⚠️  {doc_name} needs full processing (no FCM JSON)")
    
    print(f"\n📊 Summary:")
    print(f"   Already completed: {completed_count}")
    print(f"   Processed (visualizations): {processed_count}")

def show_document_info(doc_name: str, output_dir: str = OUTPUT_DIRECTORY):
    """Show detailed information about a document's FCM."""
    
    doc_output_dir = os.path.join(output_dir, doc_name)
    fcm_file = os.path.join(doc_output_dir, f"{doc_name}_fcm.json")
    
    if not os.path.exists(fcm_file):
        print(f"❌ No FCM JSON file found for {doc_name}")
        return
    
    try:
        with open(fcm_file, 'r') as f:
            data = json.load(f)
        
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        
        print(f"\n📊 FCM Information for {doc_name}:")
        print(f"   Nodes (clusters): {len(nodes)}")
        print(f"   Edges: {len(edges)}")
        
        if len(nodes) > 0:
            print(f"\n📋 Clusters:")
            for i, node in enumerate(nodes[:10]):  # Show first 10
                concepts = node.get('concepts', '')
                if isinstance(concepts, list):
                    concepts = ', '.join(concepts)
                print(f"   {i+1}. {node.get('id', 'Unknown')}: {concepts[:60]}...")
            if len(nodes) > 10:
                print(f"   ... and {len(nodes)-10} more clusters")
        
        if len(edges) > 0:
            positive_edges = [e for e in edges if e.get('weight', 0) > 0]
            negative_edges = [e for e in edges if e.get('weight', 0) < 0]
            print(f"\n🔗 Edges:")
            print(f"   Positive relationships: {len(positive_edges)}")
            print(f"   Negative relationships: {len(negative_edges)}")
            
            print(f"\n   Sample edges:")
            for i, edge in enumerate(edges[:5]):
                weight = edge.get('weight', 0)
                confidence = edge.get('confidence', 0)
                symbol = "→" if weight > 0 else "⊸" if weight < 0 else "~"
                print(f"   {edge.get('source', '')} {symbol} {edge.get('target', '')} (conf: {confidence:.2f})")
        
    except Exception as e:
        print(f"❌ Error reading FCM file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume and manage FCM processing")
    parser.add_argument("--doc", type=str, help="Document name to process (e.g., BD007)")
    parser.add_argument("--all", action="store_true", help="Process all documents")
    parser.add_argument("--info", type=str, help="Show detailed info for a document")
    parser.add_argument("--status", type=str, help="Check status of a document")
    
    args = parser.parse_args()
    
    if args.doc:
        regenerate_visualizations(args.doc)
    elif args.all:
        process_all_documents()
    elif args.info:
        show_document_info(args.info)
    elif args.status:
        check_document_status(args.status)
    else:
        print("🚀 FCM Resume Processing Tool")
        print("\nUsage:")
        print("  python resume_processing.py --doc BD007          # Regenerate visualizations for BD007")
        print("  python resume_processing.py --all               # Process all documents")
        print("  python resume_processing.py --info BD007        # Show FCM details for BD007")
        print("  python resume_processing.py --status BD007      # Check processing status")
        print("\nThis tool can:")
        print("  ✅ Regenerate missing visualizations from existing FCM JSON")
        print("  ✅ Check processing status of documents")
        print("  ✅ Show detailed FCM information") 