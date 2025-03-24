"""Environment configuration utilities for P3GPT."""

import os
from pathlib import Path
from typing import Optional

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Load from .env file in the project root
    env_path = Path(__file__).parents[2] / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # If dotenv is not installed, just continue without it
    pass

def get_env_var(key: str, default: Optional[str] = None) -> str:
    """Get environment variable with fallback to default value.
    
    Args:
        key: Environment variable name
        default: Default value if environment variable is not set
        
    Returns:
        Value of environment variable or default
    """
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set and no default provided")
    return value

# Model paths
BASE_MODEL_PATH = get_env_var("P3GPT_BASE_MODEL_PATH", "insilicomedicine/precious3-gpt-multi-modal")
print("BASE_MODEL_PATH:", BASE_MODEL_PATH)
NACH0_MODEL_PATH = get_env_var("P3GPT_NACH0_MODEL_PATH", "insilicomedicine/nach0_base")
print("NACH0_MODEL_PATH:", NACH0_MODEL_PATH)
# Default to the same as BASE_MODEL_PATH if not explicitly set
SMILES_MODEL_PATH = get_env_var("P3GPT_SMILES_MODEL_PATH", "AAAAAAAAAA")
print("SMILES_MODEL_PATH:", SMILES_MODEL_PATH)

# Data URLs
ENTITIES_CSV_URL = get_env_var(
    "P3GPT_ENTITIES_CSV_URL", 
    "https://huggingface.co/insilicomedicine/precious3-gpt/raw/main/all_entities_with_type.csv"
)
GPT_GENES_EMBEDDINGS_URL = get_env_var(
    "P3GPT_GPT_GENES_EMBEDDINGS_URL",
    "https://huggingface.co/insilicomedicine/precious3-gpt-multi-modal/resolve/main/multi-modal-data/emb_gpt_genes.pickle"
)
HGT_GENES_EMBEDDINGS_URL = get_env_var(
    "P3GPT_HGT_GENES_EMBEDDINGS_URL",
    "https://huggingface.co/insilicomedicine/precious3-gpt-multi-modal/resolve/main/multi-modal-data/emb_hgt_genes.pickle"
)
SMILES_EMBEDDINGS_PATH = get_env_var("P3GPT_SMILES_EMBEDDINGS_PATH", "enter_actual_path_here")

def get_model_path(model_type: str) -> str:
    """Get the appropriate model path based on type and environment configuration.
    
    Args:
        model_type: Type of model ('base', 'smiles', 'nach0')
        
    Returns:
        Path to the model
    """
    model_paths = {
        'base': BASE_MODEL_PATH,
        'smiles': SMILES_MODEL_PATH,
        'nach0': NACH0_MODEL_PATH
    }
    
    if model_type not in model_paths:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model_paths[model_type]
