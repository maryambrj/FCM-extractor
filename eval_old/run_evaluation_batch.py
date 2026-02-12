#!/usr/bin/env python3
"""
Batch evaluation script for FCM outputs.
Finds all folders and runs evaluation for each one.
"""

import os
import subprocess
import sys
from pathlib import Path

def find_folders(fcm_outputs_dir="fcm_outputs"):
    """Find all folders in the fcm_outputs directory."""
    # Try multiple possible paths
    possible_paths = [
        Path(fcm_outputs_dir),
        Path("codes") / fcm_outputs_dir,
        Path("../fcm_outputs"),
        Path("fcm_outputs")
    ]
    
    fcm_path = None
    for path in possible_paths:
        if path.exists():
            fcm_path = path
            break
    
    if fcm_path is None:
        print(f"Error: FCM outputs directory not found. Tried: {[str(p) for p in possible_paths]}")
        return []
    
    folders = []
    for folder in fcm_path.iterdir():
        if folder.is_dir():
            folders.append(folder.name)
    
    return sorted(folders)

def run_evaluation(folder_name, ground_truth_dir="ground_truth", fcm_outputs_dir="fcm_outputs"):
    """Run evaluation for a specific folder."""
    
    # Find the correct base paths
    gt_base_paths = [Path(ground_truth_dir), Path("codes") / ground_truth_dir, Path("../ground-truth"), Path("ground-truth")]
    fcm_base_paths = [Path(fcm_outputs_dir), Path("codes") / fcm_outputs_dir, Path("../fcm_outputs"), Path("fcm_outputs")]
    
    gt_base = None
    for path in gt_base_paths:
        if path.exists():
            gt_base = path
            break
    
    fcm_base = None
    for path in fcm_base_paths:
        if path.exists():
            fcm_base = path
            break
    
    if gt_base is None:
        gt_path = f"{ground_truth_dir}/{folder_name}.csv"  # fallback
    else:
        gt_path = str(gt_base / f"{folder_name}.csv")
        
    if fcm_base is None:
        gen_path = f"{fcm_outputs_dir}/{folder_name}/{folder_name}_fcm.json"  # fallback
    else:
        gen_path = str(fcm_base / folder_name / f"{folder_name}_fcm.json")
    
    # Check if files exist
    if not os.path.exists(gt_path):
        print(f"  ⚠ Skipping {folder_name}: Ground truth not found at {gt_path}")
        return False
        
    if not os.path.exists(gen_path):
        print(f"  ⚠ Skipping {folder_name}: Generated FCM not found at {gen_path}")
        return False
    
    # Construct command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "utils/score_fcm.py",
        "--gt-path", gt_path,
        "--gen-path", gen_path
    ]
    
    print(f"  📊 Evaluating {folder_name}...")
    print(f"     Command: {' '.join(cmd)}")
    
    try:
        # Change to fcm_extractor directory if needed
        if os.path.exists("fcm_extractor"):
            original_dir = os.getcwd()
            os.chdir("fcm_extractor")
            
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"  ✅ {folder_name}: Evaluation completed successfully")
            if result.stdout.strip():
                # Print last few lines of output (usually contains the scores)
                lines = result.stdout.strip().split('\n')
                for line in lines[-3:]:  # Show last 3 lines
                    if line.strip():
                        print(f"     {line}")
            return True
        else:
            print(f"  ❌ {folder_name}: Evaluation failed")
            if result.stderr:
                print(f"     Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {folder_name}: Evaluation timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"  ❌ {folder_name}: Error running evaluation: {e}")
        return False
    finally:
        # Return to original directory if we changed it
        if 'original_dir' in locals():
            os.chdir(original_dir)

def main():
    print("🔍 FCM Batch Evaluation Script")
    print("=" * 50)
    
    # Find all folders
    folders = find_folders()
    
    if not folders:
        print("❌ No folders found in ../fcm_outputs/")
        return
    
    print(f"📁 Found {len(folders)} folders: {', '.join(folders)}")
    print()
    
    # Run evaluation for each folder
    successful = 0
    failed = 0
    
    for folder in folders:
        if run_evaluation(folder):
            successful += 1
        else:
            failed += 1
        print()  # Add spacing between evaluations
    
    # Summary
    print("=" * 50)
    print("📈 EVALUATION SUMMARY")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(folders)}")
    
    if successful > 0:
        print(f"\n💡 Check the individual folders in ../fcm_outputs/ for detailed results")
        print(f"💡 Look for *_scoring_results.csv files for numerical scores")

if __name__ == "__main__":
    main()