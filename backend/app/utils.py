import math
import torch
import json
import pandas as pd
from pathlib import Path
from .helpers import preprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

# Get the directory containing this file
CURRENT_DIR = Path(__file__).parent

# Cache for score statistics (mean_score, std_score)
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

    # Preprocess the input title
    preprocessed_title = preprocess(title, 0)
    if len(preprocessed_title) == 0:
        preprocessed_title = ['<UNK>']

    # Get word IDs, using <UNK> for unknown words
    title_ids = [lookup_tables['words_to_ids'].get(word, 
                lookup_tables['words_to_ids']['<UNK>']) 
                for word in preprocessed_title]

    # Get embeddings and average them
    title_embeddings = word_embeddings[title_ids]
    title_embedding_avg = torch.mean(title_embeddings, dim=0)

    # Add batch dimension for BatchNorm
    title_embedding_avg = title_embedding_avg.unsqueeze(0)

    # Load the score predictor model
    model = ScorePredictor(embedding_dim=64, hidden_dims=[128, 64])
    model.load_state_dict(torch.load(CURRENT_DIR / 'score_predictor.pt', weights_only=True))
    model.eval()

    # Make prediction
    with torch.no_grad():
        prediction = model(title_embedding_avg)

    # Denormalize prediction
    mean_score, std_score = get_score_stats()
    denormalized_prediction = math.ceil((prediction.item() * std_score) + mean_score)

    # Ensure prediction is non-negative
    return max(0, denormalized_prediction)

class ScorePredictor(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dims):
        super().__init__()
        
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.Dropout(0.2))
            input_dim = hidden_dim
            
        self.hidden_network = torch.nn.Sequential(*layers)
        self.final_layer = torch.nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        x = self.hidden_network(x)
        x = self.final_layer(x)
        return x
