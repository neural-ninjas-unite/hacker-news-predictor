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
    EPOCHS = 10
    HIDDEN_DIMS = [32, 16]
    
    # Load your data
    # Assuming you have a function to load from your database
    titles, scores = load_from_database()  
    
    # Load word embeddings and lookup tables
    weights = torch.load('weights.pt')
    word_embeddings = weights['emb.weight']
    with open('lookup_tables.json', 'r') as f:
        lookup_tables = json.load(f)
    
    # Create dataset and dataloader
    dataset = HackerNewsDataset(titles, scores, word_embeddings, lookup_tables)
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
    wandb.init(
        project='hacker-news-predictor',
        name='score-predictor',
        config={
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'hidden_dims': HIDDEN_DIMS
    })
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_embeddings, batch_scores in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_embeddings)
            loss = criterion(predictions, batch_scores.unsqueeze(1))
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
        wandb.log({
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'epoch': epoch
        })
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'score_predictor.pt')
    wandb.save('score_predictor.pt')

if __name__ == "__main__":
    train_model() 