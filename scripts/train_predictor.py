import torch
import json
from torch.utils.data import Dataset, DataLoader
from helpers import preprocess
import wandb
import pandas as pd

def load_from_database():
    # Load data from CSV
    df = pd.read_csv('../data-1737988940684.csv')
    return df['title'].tolist(), df['score'].tolist()

# 1. Create a Dataset class for your HackerNews data
class HackerNewsDataset(Dataset):
    def __init__(self, titles, scores, word_embeddings, lookup_tables):
        self.titles = titles
        self.scores = scores
        self.word_embeddings = word_embeddings
        self.lookup_tables = lookup_tables
    
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        title = self.titles[idx]
        score = self.scores[idx]
        
        # Preprocess title and get word IDs
        preprocessed_title = preprocess(title, 0)
        if len(preprocessed_title) == 0:
            preprocessed_title = ['<UNK>']
            
        title_ids = [self.lookup_tables['words_to_ids'].get(word, 
                    self.lookup_tables['words_to_ids']['<UNK>']) 
                    for word in preprocessed_title]
        
        # Get embeddings and average them
        title_embeddings = self.word_embeddings[title_ids]
        title_embedding_avg = torch.mean(title_embeddings, dim=0)
        
        return title_embedding_avg, torch.tensor(score, dtype=torch.float32)

# 2. Define the complete model
class ScorePredictor(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dims):
        super().__init__()
        
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
            
        self.hidden_network = torch.nn.Sequential(*layers)
        self.final_layer = torch.nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        x = self.hidden_network(x)
        x = self.final_layer(x)
        return torch.relu(x)  # Ensure predictions are non-negative

# 3. Training setup
def train_model():
    # Hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 1000
    HIDDEN_DIMS = [32, 16]
    
    # Load your data
    titles, scores = load_from_database()
    print("Data loaded:")
    print(f"Number of samples: {len(titles)}")
    print(f"Score range: {min(scores)} to {max(scores)}")
    print(f"Sample title: {titles[0]}")
    print(f"Sample score: {scores[0]}")
    
    # Load word embeddings and lookup tables
    weights = torch.load('weights.pt')
    word_embeddings = weights['emb.weight']
    print(f"\nWord embeddings shape: {word_embeddings.shape}")
    print(f"Word embeddings range: {word_embeddings.min().item():.2f} to {word_embeddings.max().item():.2f}")
    
    with open('lookup_tables.json', 'r') as f:
        lookup_tables = json.load(f)
    print(f"Vocabulary size: {len(lookup_tables['words_to_ids'])}")
    
    # Create dataset and dataloader
    dataset = HackerNewsDataset(titles, scores, word_embeddings, lookup_tables)
    
    # Check first item in dataset
    first_embedding, first_score = dataset[0]
    print(f"\nFirst item check:")
    print(f"Embedding shape: {first_embedding.shape}")
    print(f"Embedding range: {first_embedding.min().item():.2f} to {first_embedding.max().item():.2f}")
    print(f"Score: {first_score}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model, loss function, and optimizer
    model = ScorePredictor(embedding_dim=64, hidden_dims=HIDDEN_DIMS)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize wandb
    # wandb.init(
    #     project='hacker-news-predictor',
    #     name='score-predictor',
    #     config={
    #     'learning_rate': LEARNING_RATE,
    #     'batch_size': BATCH_SIZE,
    #     'hidden_dims': HIDDEN_DIMS
    # })
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for i, (batch_embeddings, batch_scores) in enumerate(train_loader):
            if i == 0 and epoch == 0:
                print(f"\nFirst batch check:")
                print(f"Batch embeddings shape: {batch_embeddings.shape}")
                print(f"Batch embeddings range: {batch_embeddings.min().item():.2f} to {batch_embeddings.max().item():.2f}")
                print(f"Batch scores shape: {batch_scores.shape}")
                print(f"Batch scores range: {batch_scores.min().item():.2f} to {batch_scores.max().item():.2f}")
            
            optimizer.zero_grad()
            predictions = model(batch_embeddings)
            
            if i == 0 and epoch == 0:
                print(f"Predictions shape: {predictions.shape}")
                print(f"Predictions range: {predictions.min().item():.2f} to {predictions.max().item():.2f}")
            
            loss = criterion(predictions, batch_scores.unsqueeze(1))
            
            if torch.isnan(loss):
                print(f"\nNaN detected in loss!")
                print(f"Predictions: {predictions}")
                print(f"Targets: {batch_scores}")
                raise ValueError("NaN loss detected")
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_embeddings, batch_scores in val_loader:
                predictions = model(batch_embeddings)
                val_loss += criterion(predictions, batch_scores.unsqueeze(1)).item()
        
        # Log metrics
        # wandb.log({
        #     'train_loss': train_loss / len(train_loader),
        #     'val_loss': val_loss / len(val_loader),
        #     'epoch': epoch
        # })
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'score_predictor.pt')
    # wandb.save('score_predictor.pt')

if __name__ == "__main__":
    train_model() 