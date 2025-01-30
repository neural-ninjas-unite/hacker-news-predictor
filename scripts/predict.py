# import math
# import torch
# import json
# import pandas as pd
# from helpers import preprocess

# # Cache for score statistics
# _score_stats_cache = (4.4123, 7.2894)

# def get_score_stats():
#     global _score_stats_cache
    
#     # Return cached values if available
#     if _score_stats_cache is not None:
#         return _score_stats_cache
    
#     # Calculate stats if not cached
#     df = pd.read_csv('../data-1737988940684.csv')
#     scores = torch.tensor(df['score'].values, dtype=torch.float32)
#     mean_score = scores.mean()
#     std_score = scores.std()
    
#     # Cache the results
#     _score_stats_cache = (mean_score, std_score)
#     return _score_stats_cache

def predict_score(title: str) -> float:
    return 6
    # # Load the model weights and word embeddings
    # weights = torch.load('weights.pt', weights_only=True)
    # word_embeddings = weights['emb.weight']

    # # Load the lookup tables
    # with open('lookup_tables.json', 'r') as f:
    #     lookup_tables = json.load(f)

    # # Preprocess the input title
    # preprocessed_title = preprocess(title, 0)
    # if len(preprocessed_title) == 0:
    #     preprocessed_title = ['<UNK>']

    # # Get word IDs, using <UNK> for unknown words
    # title_ids = [lookup_tables['words_to_ids'].get(word, 
    #             lookup_tables['words_to_ids']['<UNK>']) 
    #             for word in preprocessed_title]

    # # Get embeddings and average them
    # title_embeddings = word_embeddings[title_ids]
    # title_embedding_avg = torch.mean(title_embeddings, dim=0)

    # # Add batch dimension for BatchNorm
    # title_embedding_avg = title_embedding_avg.unsqueeze(0)

    # # Load the score predictor model
    # model = ScorePredictor(embedding_dim=64, hidden_dims=[128, 64])
    # model.load_state_dict(torch.load('score_predictor.pt', weights_only=True))
    # model.eval()

    # # Make prediction
    # with torch.no_grad():
    #     prediction = model(title_embedding_avg)

    # # Denormalize prediction
    # mean_score, std_score = get_score_stats()
    # denormalized_prediction = math.ceil((prediction.item() * std_score) + mean_score)

    # # Ensure prediction is non-negative
    # return max(0, denormalized_prediction)

# class ScorePredictor(torch.nn.Module):
#     def __init__(self, embedding_dim, hidden_dims):
#         super().__init__()
        
#         layers = []
#         input_dim = embedding_dim
#         for hidden_dim in hidden_dims:
#             layers.append(torch.nn.Linear(input_dim, hidden_dim))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.BatchNorm1d(hidden_dim))
#             layers.append(torch.nn.Dropout(0.2))
#             input_dim = hidden_dim
            
#         self.hidden_network = torch.nn.Sequential(*layers)
#         self.final_layer = torch.nn.Linear(hidden_dims[-1], 1)
    
#     def forward(self, x):
#         x = self.hidden_network(x)
#         x = self.final_layer(x)
#         return x

# if __name__ == "__main__":
#     # Example titles to test
#     test_titles = [
#         "Ask HN: How to improve my personal website?",
#         "Show HN: I built a new programming language",
#         "Why Rust is the future of systems programming",
#         "Launch HN: My startup (YC W24)",
#     ]
    
#     print("\nPredicting Hacker News scores...")
#     print("-" * 50)
#     for title in test_titles:
#         predicted_score = predict_score(title)
#         print(f"\nTitle: {title}")
#         print(f"Predicted score: {predicted_score:.1f}")

