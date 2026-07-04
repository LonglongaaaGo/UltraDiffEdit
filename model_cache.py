import os
from pathlib import Path


DEFAULT_MODEL_CACHE = "/Volumes/Long_ssd/ultradiffedit_cache/huggingface/hub"


def get_model_cache_dir() -> str:
    return os.environ.get("ULTRADIFFEDIT_MODEL_CACHE", DEFAULT_MODEL_CACHE)


def configure_model_cache() -> str:
    cache_dir = Path(get_model_cache_dir()).expanduser()
    os.environ["HF_HOME"] = str(cache_dir.parent)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    return str(cache_dir)


def ensure_model_cache_dir() -> str:
    cache_dir = Path(configure_model_cache())
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


MODEL_CACHE_DIR = configure_model_cache()
