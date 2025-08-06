import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import builtins

# Global variables
_original_print = builtins.print
_log_file_path = None
_console_and_file_logger = None

def setup_logging(log_directory: str = "../logs", 
                 log_filename: str = None,
                 enable_file_logging: bool = True,
                 include_timestamp: bool = True,
                 log_level: str = "INFO") -> str:
    """
    Set up comprehensive logging that captures all output to both console and file.
    
    Args:
        log_directory: Directory to store log files
        log_filename: Specific log filename (if None, auto-generates based on timestamp)
        enable_file_logging: Whether to enable file logging
        include_timestamp: Whether to include timestamps in log entries
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Path to the log file created
    """
    global _log_file_path, _console_and_file_logger
    
    if not enable_file_logging:
        return None
    
    # Create log directory if it doesn't exist
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"fcm_extraction_{timestamp}.log"
    
    _log_file_path = log_dir / log_filename
    
    # Set up logger
    logger = logging.getLogger('fcm_extractor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if include_timestamp:
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter('%(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(_log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    _console_and_file_logger = logger
    
    # Replace built-in print function
    builtins.print = logged_print
    
    # Log the setup
    logger.info(f"=== FCM Extraction Session Started ===")
    logger.info(f"Log file: {_log_file_path}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    return str(_log_file_path)

def logged_print(*args, **kwargs):
    """
    Replacement for built-in print that logs to both console and file.
    Maintains all the same arguments and behavior as the original print function.
    """
    # Convert arguments to string like print does
    output = " ".join(str(arg) for arg in args)
    
    # Handle special print arguments
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    
    if len(args) > 1:
        output = sep.join(str(arg) for arg in args)
    
    # Add the end character
    if end != '\n':
        output += end
    else:
        # Don't add extra newline since logger.info already adds one
        pass
    
    # Log through our logger if it exists
    if _console_and_file_logger:
        # Use info level for regular print statements
        _console_and_file_logger.info(output.rstrip('\n'))
    else:
        # Fallback to original print if logger not set up
        _original_print(*args, **kwargs)

def log_to_console_and_file(message: str, log_file: str = None):
    """
    Legacy function for backward compatibility.
    Now uses the global logging system.
    """
    if _console_and_file_logger:
        _console_and_file_logger.info(message)
    else:
        print(message)

def log_clusters(clusters: Dict):
    """Log cluster information using the global logging system."""
    print("CLUSTERS:")
    for cluster_id, concepts in clusters.items():
        if isinstance(concepts, list):
            formatted = ', '.join(concepts)
        else:
            formatted = str(concepts)
        print(f"  Cluster {cluster_id}: {formatted}")

def log_edges(edges: list, log_file: str = None):
    """Log edge information using the global logging system."""
    print(f"Edge Decisions: {edges}")

def finalize_logging():
    """
    Clean up logging and restore original print function.
    Call this at the end of processing.
    """
    global _console_and_file_logger
    
    if _console_and_file_logger:
        _console_and_file_logger.info("=" * 50)
        _console_and_file_logger.info(f"=== FCM Extraction Session Ended ===")
        _console_and_file_logger.info(f"End timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Close handlers
        for handler in _console_and_file_logger.handlers:
            handler.close()
        
        _console_and_file_logger = None
    
    # Restore original print function
    builtins.print = _original_print

def get_log_file_path() -> Optional[str]:
    """Get the current log file path."""
    return str(_log_file_path) if _log_file_path else None

def log_error(message: str):
    """Log an error message."""
    if _console_and_file_logger:
        _console_and_file_logger.error(f"ERROR: {message}")
    else:
        print(f"ERROR: {message}")

def log_warning(message: str):
    """Log a warning message.""" 
    if _console_and_file_logger:
        _console_and_file_logger.warning(f"WARNING: {message}")
    else:
        print(f"WARNING: {message}")

def log_debug(message: str):
    """Log a debug message."""
    if _console_and_file_logger:
        _console_and_file_logger.debug(f"DEBUG: {message}")
    else:
        print(f"DEBUG: {message}") 