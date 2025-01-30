import tqdm
import more_itertools
import wandb
import torch
from helpers import preprocess, download_wikipedia_text8, create_lookup_tables
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get W&B toggle from environment variable
USE_WANDB = os.getenv('USE_WANDB', 'false').lower() == 'true'

# STEP 0: Set hyperparameters
learning_rate = 0.003
embedding_dim = 64
batch_size=512


# STEP 1: Set the random seed
torch.manual_seed(42)


# STEP 2: Download the text8 dataset
if not os.path.exists('text8'):
    download_wikipedia_text8()

with open('text8') as f:
    text8: str = f.read()


# STEP 3: Preprocess the text8 dataset
min_words = 5
corpus: list[str] = preprocess(text8, min_words)
print(type(corpus)) # <class 'list'>
print(len(corpus))  # 16,680,599
print(corpus[:7])   # ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse']

# STEP 4: Create lookup tables
words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokens = [words_to_ids[word] for word in corpus]
print(type(tokens)) # <class 'list'>
print(len(tokens))  # 16,680,599
print(tokens[:7])   # [5234, 3081, 12, 6, 195, 2, 3134]

print(ids_to_words[5234])        # anarchism
print(words_to_ids['anarchism']) # 5234
print(words_to_ids['have'])      # 3081
print(len(words_to_ids))         # 63,642


# STEP 5: Define the SkipGram model
class SkipGram(torch.nn.Module):
  def __init__(self, voc, emb, _):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
    self.sig = torch.nn.Sigmoid()

  def forward(self, inpt, trgs, rand):
    emb = self.emb(inpt)
    ctx = self.ffw.weight[trgs]
    rnd = self.ffw.weight[rand]
    out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
    rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()
    out = self.sig(out)
    rnd = self.sig(rnd)
    pst = -out.log().mean()
    ngt = -(1 - rnd + 10**(-3)).log().mean()
    return pst + ngt


# STEP 6: Create the SkipGram model, optimizer and device
args = (len(words_to_ids), embedding_dim, 2)
mFoo = SkipGram(*args)
print('mFoo', sum(p.numel() for p in mFoo.parameters()))
opFoo = torch.optim.Adam(mFoo.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # STEP 7: Create the sliding window, dataset and dataloader
windows = list(more_itertools.windowed(tokens, 3))
inputs = [w[1] for w in windows]
targets = [[w[0], w[2]] for w in windows]
input_tensor = torch.LongTensor(inputs)
target_tensor = torch.LongTensor(targets)
dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# STEP 8: Initialize W&B if enabled
if USE_WANDB:
    wandb.init(
        project='hacker-news-word2vec',
        name='attempt-1',
        config={
            'learning_rate': learning_rate,
            'embedding_dim': embedding_dim,
            'vocab_size': len(words_to_ids),
            'batch_size': batch_size
        }
    )

# STEP 9: Train the SkipGram model
mFoo.to(device)
for epoch in range(1):
  prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
  for inpt, trgs in prgs:
    inpt, trgs = inpt.to(device), trgs.to(device)
    rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2)).to(device)
    opFoo.zero_grad()
    loss = mFoo(inpt, trgs, rand)
    loss.backward()
    opFoo.step()
    if USE_WANDB:
        wandb.log({'loss': loss.item()})


# Step 9: Save the model weights and upload to W&B if enabled
print('Saving...')
torch.save(mFoo.state_dict(), './weights.pt')

if USE_WANDB:
    print('Uploading to W&B...')
    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file('./weights.pt')
    wandb.log_artifact(artifact)
    wandb.finish()

print('Done!')
