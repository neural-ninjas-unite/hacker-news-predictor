import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
import logging
from collections import Counter
import re
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Word2VecDataset(Dataset):
    def __init__(self, texts: List[str], window_size: int = 5, min_count: int = 5):
        """
        Initialize Word2Vec dataset.
        
        Args:
            texts: List of text documents
            window_size: Context window size
            min_count: Minimum word frequency
        """
        start_time = time.time()
        logger.info("Initializing Word2Vec dataset...")
        self.window_size = window_size
        
        # Preprocess texts and build vocabulary
        logger.info("Building vocabulary...")
        self.word_counts = Counter()
        self.processed_texts = []
        
        for text in tqdm(texts, desc="Processing texts"):
            words = self._preprocess_text(text)
            self.word_counts.update(words)
            self.processed_texts.append(words)
        
        # Filter vocabulary by minimum count
        logger.info("Filtering vocabulary...")
        self.vocab = {word: idx + 1 for idx, (word, count) in 
                     enumerate([item for item in self.word_counts.items() 
                              if item[1] >= min_count])}
        self.vocab['<UNK>'] = 0
        
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Create training pairs
        logger.info("Creating training pairs...")
        self.pairs = self._create_pairs()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Dataset initialization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Vocabulary size: {self.vocab_size:,} words")
        logger.info(f"Number of training pairs: {len(self.pairs):,}")
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text."""
        if text is None:
            return []  # Return empty list for None values
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()
    
    def _create_pairs(self) -> List[Tuple[int, int]]:
        """Create (target, context) pairs for training."""
        pairs = []
        for words in tqdm(self.processed_texts, desc="Creating word pairs"):
            word_ids = [self.vocab.get(word, 0) for word in words]
            
            for i in range(len(word_ids)):
                target = word_ids[i]
                # Generate context words within window
                for j in range(max(0, i - self.window_size), 
                             min(len(word_ids), i + self.window_size + 1)):
                    if i != j:
                        context = word_ids[j]
                        pairs.append((target, context))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target, context = self.pairs[idx]
        return torch.tensor(target), torch.tensor(context)

class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize Word2Vec model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super(Word2Vec, self).__init__()
        
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self.target_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.context_embeddings.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        target_embeds = self.target_embeddings(target)
        context_embeds = self.context_embeddings(context)
        
        # Compute dot product between target and context embeddings
        output = torch.sum(target_embeds * context_embeds, dim=1)
        return output
    
    def get_word_vector(self, word_idx: int) -> torch.Tensor:
        """Get the embedding vector for a word."""
        return self.target_embeddings.weight[word_idx].detach()

def train_word2vec(texts: List[str], 
                  embedding_dim: int = 100,
                  window_size: int = 5,
                  min_count: int = 5,
                  batch_size: int = 1024,
                  num_epochs: int = 5,
                  learning_rate: float = 0.025) -> Tuple[Word2Vec, Word2VecDataset]:
    """
    Train Word2Vec model on input texts.
    
    Args:
        texts: List of input texts
        embedding_dim: Dimension of word embeddings
        window_size: Context window size
        min_count: Minimum word frequency
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Trained model and dataset
    """
    start_time = time.time()
    logger.info("\n=== Starting Word2Vec Training ===")
    logger.info(f"Configuration:")
    logger.info(f"- Embedding dimension: {embedding_dim}")
    logger.info(f"- Window size: {window_size}")
    logger.info(f"- Minimum word count: {min_count}")
    logger.info(f"- Batch size: {batch_size}")
    logger.info(f"- Number of epochs: {num_epochs}")
    logger.info(f"- Learning rate: {learning_rate}")
    
    # Create dataset
    dataset = Word2VecDataset(texts, window_size, min_count)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = Word2Vec(dataset.vocab_size, embedding_dim).to(device)
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    logger.info("\nStarting training...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (target, context) in enumerate(progress_bar):
            target = target.to(device)
            context = context.to(device)
            
            # Forward pass
            output = model(target, context)
            loss = criterion(output, torch.ones_like(output))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch statistics
        logger.info(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        logger.info(f"- Average loss: {avg_loss:.4f}")
        logger.info(f"- Time taken: {epoch_time:.2f} seconds")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.info(f"- New best loss achieved!")
    
    total_time = time.time() - start_time
    logger.info("\n=== Training Complete ===")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Best loss achieved: {best_loss:.4f}")
    logger.info("========================\n")
    
    return model, dataset 