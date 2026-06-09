from pathlib import Path

def get_cache_dir() -> Path:
    """Get the cache directory for storing temporary files.

    Returns:
        Path to the cache directory.
    """
    dir = Path("./cache")
    dir.mkdir(parents=True, exist_ok=True)
    return dir