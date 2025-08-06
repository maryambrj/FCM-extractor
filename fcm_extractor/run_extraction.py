#!/usr/bin/env python
"""
Main entry point for FCM extraction pipeline.

Usage:
    python run_extraction.py                      # Process default file (set in constants.py)
    python run_extraction.py BD007.docx           # Process specific interview
    python run_extraction.py --all                # Process all interviews
    python run_extraction.py --help               # Show help

Configuration:
    - Edit DEFAULT_INTERVIEW_FILE in config/constants.py to change default document
    - Set PROCESS_ALL_FILES=True in constants.py to process all files by default
"""

import sys
import os
from pathlib import Path

# Suppress noisy numerical computation logs BEFORE any other imports
from utils.suppress_numba_logs import setup_clean_logging
setup_clean_logging()

# Add fcm_extractor to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import process_interviews
from config.constants import (
    INTERVIEWS_DIRECTORY, OUTPUT_DIRECTORY, DEFAULT_INTERVIEW_FILE,
    ENABLE_FILE_LOGGING, LOG_DIRECTORY, LOG_LEVEL, INCLUDE_TIMESTAMP_IN_LOGS
)
from utils.logging_utils import setup_logging, finalize_logging


def main():
    """Main entry point for FCM extraction."""
    
    # Set up logging first (before any print statements)
    log_file_path = None
    if ENABLE_FILE_LOGGING:
        log_file_path = setup_logging(
            log_directory=LOG_DIRECTORY,
            enable_file_logging=ENABLE_FILE_LOGGING,
            include_timestamp=INCLUDE_TIMESTAMP_IN_LOGS,
            log_level=LOG_LEVEL
        )
        if log_file_path:
            print(f"📝 Logging enabled - all output will be saved to: {log_file_path}")
    
    try:
        # Simple argument parsing
        if len(sys.argv) > 1:
            if sys.argv[1] in ['--help', '-h']:
                print(__doc__)
                return
            elif sys.argv[1] in ['--all', '-a']:
                # Process all files
                specific_file = None
                print("Processing all interview files...")
            else:
                # Process specific file
                specific_file = sys.argv[1]
                print(f"Processing specific file: {specific_file}")
        else:
            # Use default file from configuration
            specific_file = DEFAULT_INTERVIEW_FILE
            print(f"Processing default file: {specific_file} (configured in constants.py)")
        
        # Run the pipeline
        results = process_interviews(
            interviews_dir=INTERVIEWS_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            specific_file=specific_file
        )
        
        if results:
            print(f"\n✅ Successfully processed {len(results)} document(s)")
            print(f"📁 Results saved to: {OUTPUT_DIRECTORY}")
            if log_file_path:
                print(f"📝 Session log saved to: {log_file_path}")
        else:
            print("❌ No documents were processed successfully.")
            print("Please check the configuration and input files.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred during processing: {e}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return 1
        
    finally:
        # Always clean up logging
        if ENABLE_FILE_LOGGING:
            finalize_logging()


if __name__ == "__main__":
    main() 