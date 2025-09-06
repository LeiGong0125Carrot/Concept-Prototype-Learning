"""
Configuration file for model paths and cache directories
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path("/bigtemp/nkw3mr/concept-prototype-learing")

# Cache directories
CACHE_DIR = BASE_DIR / "huggingface_cache"
TORCH_CACHE_DIR = BASE_DIR / "torch_cache"

# Set environment variables
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)

# Model names
MODELS = {
    "bioclinical": "thomas-sounack/BioClinical-ModernBERT-base",
    "clinical": "Simonlee711/Clinical_ModernBERT"
}

# Create cache directories if they don't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "datasets").mkdir(parents=True, exist_ok=True)

def get_model_path(model_key):
    """Get model name by key"""
    return MODELS.get(model_key, model_key)

def print_config():
    """Print current configuration"""
    print("Current Configuration:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
    print(f"  HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}")
    print(f"  TORCH_HOME: {os.environ.get('TORCH_HOME')}")
    print("\nAvailable models:")
    for key, value in MODELS.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    print_config()