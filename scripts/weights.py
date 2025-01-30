import torch
from helpers import preprocess, create_lookup_tables

weights = torch.load('weights.pt', weights_only=True)

word_embeddings = weights['emb.weight']

hacker_news_title = "Ask HN: How to improve my personal website?"

# Get the word embeddings for the words in the title
preprocessed_title = preprocess(hacker_news_title, 0)
print(preprocessed_title)
# word_ids = [weights['words_to_ids'][word] for word in preprocessed_title]

