import psutil
import logging
import gc

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def force_garbage_collection():
    """Force garbage collection and log memory usage."""
    gc.collect()
    log_memory_usage() 