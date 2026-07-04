import os
from pathlib import Path
from typing import Optional


def get_model_cache_dir() -> Optional[str]:
    return os.environ.get("ULTRADIFFEDIT_MODEL_CACHE")


def configure_model_cache() -> Optional[str]:
    cache_root = get_model_cache_dir()
    if not cache_root:
        return None

    cache_dir = Path(cache_root).expanduser()
    os.environ["HF_HOME"] = str(cache_dir.parent)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    return str(cache_dir)


def ensure_model_cache_dir() -> Optional[str]:
    cache_dir = configure_model_cache()
    if cache_dir is None:
        return None

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


MODEL_CACHE_DIR = configure_model_cache()
