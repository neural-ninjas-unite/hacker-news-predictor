import torch
import json
from helpers import preprocess

weights = torch.load('weights.pt', weights_only=True)

word_embeddings = weights['emb.weight']

hacker_news_title = "Ask HN: How to improve my personal website?"

# Get the word embeddings for the words in the title
preprocessed_title = preprocess(hacker_news_title, 0)
print(preprocessed_title)

# Load the lookup tables
with open('lookup_tables.json', 'r') as f:
    lookup_tables = json.load(f)

# lookup_tables['words_to_ids']
print(lookup_tables['words_to_ids']['ask'])

# Get word IDs, using <UNK> for unknown words
title_ids = [lookup_tables['words_to_ids'].get(word, lookup_tables['words_to_ids']['<UNK>']) for word in preprocessed_title]
print('title_ids: ', title_ids)

# Get the word embeddings for the words in the title
title_embeddings = word_embeddings[title_ids]

# Average pooling of title embeddings
title_embedding_avg = torch.mean(title_embeddings, dim=0)
print('Average title embedding shape:', title_embedding_avg.shape)
print('Average title embedding:', title_embedding_avg)

# Define hidden layer dimensions
hidden_dims = [32, 16, 8]

# Create sequential layers with ReLU activations
layers = []
input_dim = title_embedding_avg.shape[0]  # 64 from embedding_dim
for hidden_dim in hidden_dims:
    layers.append(torch.nn.Linear(input_dim, hidden_dim))
    layers.append(torch.nn.ReLU())
    input_dim = hidden_dim

hidden_network = torch.nn.Sequential(*layers)

# Pass the averaged embedding through the layers
output = hidden_network(title_embedding_avg)
print('Final output shape:', output.shape)  # Should be [8]
print('Final output:', output)


