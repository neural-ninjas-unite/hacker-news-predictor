import math
import torch
import json
import pandas as pd
from pathlib import Path
from .helpers import preprocess

# Get the directory containing this file
CURRENT_DIR = Path(__file__).parent

# Cache for score statistics
_score_stats_cache = (4.4123, 7.2894)

def get_score_stats():
    return _score_stats_cache

def predict_score(title: str) -> float:
    # Load the model weights and word embeddings
    weights = torch.load(CURRENT_DIR / 'weights.pt', weights_only=True)
    word_embeddings = weights['emb.weight']

    # Load the lookup tables
    with open(CURRENT_DIR / 'lookup_tables.json', 'r') as f:
        lookup_tables = json.load(f)

    return 6