import math
import torch
import json
import pandas as pd
# from helpers import preprocess

# Cache for score statistics
_score_stats_cache = (4.4123, 7.2894)

def get_score_stats():
    global _score_stats_cache
    
    # Return cached values if available
    if _score_stats_cache is not None:
        return _score_stats_cache
    
    # Calculate stats if not cached
    df = pd.read_csv('../data-1737988940684.csv')
    scores = torch.tensor(df['score'].values, dtype=torch.float32)
    mean_score = scores.mean()
    std_score = scores.std()
    
    # Cache the results
    _score_stats_cache = (mean_score, std_score)
    return _score_stats_cache

def predict_score(title: str) -> float:
    return 6