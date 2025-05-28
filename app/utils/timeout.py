import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    """Context manager for timeout handling."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Register the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0) 