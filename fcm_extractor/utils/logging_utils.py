import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import builtins

_original_print = builtins.print
_original_stderr = sys.stderr
_log_file_path = None
_console_and_file_logger = None

class LoggingStderr:
    
    def __init__(self, logger):
        self.logger = logger
        self.original_stderr = _original_stderr
        
    def write(self, message):
        if message.strip(): 
            if self.logger:
                self.logger.error(message.rstrip('\n'))
            self.original_stderr.write(message)
            
    def flush(self):
        self.original_stderr.flush()

def setup_logging(log_directory: str = "../logs", 
                 log_filename: str = None,
                 enable_file_logging: bool = True,
                 include_timestamp: bool = True,
                 log_level: str = "INFO") -> str:

    global _log_file_path, _console_and_file_logger
    
    if not enable_file_logging:
        return None
    
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"fcm_extraction_{timestamp}.log"
    
    _log_file_path = log_dir / log_filename
    
    logger = logging.getLogger('fcm_extractor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    logger.handlers.clear()
    
    print("📝 Note: Numba compilation details will appear in logs (system works better this way)")
    
    if include_timestamp:
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter('%(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(_log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    _console_and_file_logger = logger
    
    # Replace built-in print functio
    builtins.print = logged_print
    
    sys.stderr = LoggingStderr(_console_and_file_logger)
    
    logger.info(f"=== FCM Extraction Session Started ===")
    logger.info(f"Log file: {_log_file_path}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    return str(_log_file_path)

def logged_print(*args, **kwargs):

    output = " ".join(str(arg) for arg in args)
    
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    
    if len(args) > 1:
        output = sep.join(str(arg) for arg in args)
    
    if end != '\n':
        output += end
    else:
        pass
    
    if _console_and_file_logger:
        _console_and_file_logger.info(output.rstrip('\n'))
    else:
        _original_print(*args, **kwargs)

def log_to_console_and_file(message: str, log_file: str = None):

    if _console_and_file_logger:
        _console_and_file_logger.info(message)
    else:
        print(message)

def log_clusters(clusters: Dict):

    print("CLUSTERS:")
    for cluster_id, concepts in clusters.items():
        if isinstance(concepts, list):
            formatted = ', '.join(concepts)
        else:
            formatted = str(concepts)
        print(f"  Cluster {cluster_id}: {formatted}")

def log_edges(edges: list, log_file: str = None):

    print(f"Edge Decisions: {edges}")

def finalize_logging():

    global _console_and_file_logger
    
    if _console_and_file_logger:
        _console_and_file_logger.info("=" * 50)
        _console_and_file_logger.info(f"=== FCM Extraction Session Ended ===")
        _console_and_file_logger.info(f"End timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for handler in _console_and_file_logger.handlers:
            handler.close()
        
        _console_and_file_logger = None
    
  
    builtins.print = _original_print
    sys.stderr = _original_stderr

def get_log_file_path() -> Optional[str]:
    """Get the current log file path."""
    return str(_log_file_path) if _log_file_path else None

def log_error(message: str):
    if _console_and_file_logger:
        _console_and_file_logger.error(f"ERROR: {message}")
    else:
        print(f"ERROR: {message}")

def log_warning(message: str):
    if _console_and_file_logger:
        _console_and_file_logger.warning(f"WARNING: {message}")
    else:
        print(f"WARNING: {message}")

def log_debug(message: str):
    if _console_and_file_logger:
        _console_and_file_logger.debug(f"DEBUG: {message}")
    else:
        print(f"DEBUG: {message}") 