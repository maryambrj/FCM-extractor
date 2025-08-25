#!/usr/bin/env python3
"""
Simplified FCM Pipeline
For each file in /interviews: run extraction, then scoring
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Directories
    interviews_dir = Path("../interviews")
    ground_truth_dir = Path("../ground_truth") 
    fcm_outputs_dir = Path("../fcm_outputs")
    
    if not interviews_dir.exists():
        print(f"Error: {interviews_dir} not found")
        return 1
    
    # Get all .docx and .txt files
    docx_files = list(interviews_dir.glob("*.docx"))
    txt_files = list(interviews_dir.glob("*.txt"))
    interview_files = docx_files + txt_files
    
    if not interview_files:
        print(f"No .docx or .txt files found in {interviews_dir}")
        return 1
    
    print(f"Processing {len(interview_files)} interview files...")
    
    for interview_file in interview_files:
        filename = interview_file.name
        # Remove file extension (.docx or .txt) to get base name
        base_name = interview_file.stem
        
        print(f"\n=== Processing {filename} ===")
        
        # Step 1: Run extraction
        print(f"1. Running extraction...")
        extraction_cmd = [sys.executable, "run_extraction.py", filename]
        try:
            result = subprocess.run(extraction_cmd, check=True)
            print(f"   ✓ Extraction completed")
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Extraction failed: {e}")
            continue
        
        print(f"   ✓ Extraction completed for {base_name}")
        
        # Note: Scoring can be run separately using run_evaluation_batch.py
    
    print("\n=== Pipeline completed ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())