import os

def init_hf_path():
    cache_path = open("HF_CACHE_PATH", "r").read().strip()
    assert os.path.exists(cache_path)
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_CACHE"] = cache_path