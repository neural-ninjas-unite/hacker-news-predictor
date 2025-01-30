import torch
from helpers import test_help

weights = torch.load('weights.pt', weights_only=True)

word_embeddings = weights['emb.weight']

hacker_news_title = "Ask HN: How to improve my personal website?"

test_help()
