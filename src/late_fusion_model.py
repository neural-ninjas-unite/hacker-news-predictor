import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import pandas as pd
import torch.optim as optim
import math
import re
import torch.nn.functional as F
import copy
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

def safe_title_length(title: str) -> int:
    """Safely get title length."""
    return len(title) if title and isinstance(title, str) else 0
    
def safe_word_count(title: str) -> int:
    """Safely count words in title."""
    return len(title.split()) if title and isinstance(title, str) else 0
    
def safe_char_count(title: str, char: str) -> int:
    """Safely count occurrences of a character in title."""
    return title.count(char) if title and isinstance(title, str) else 0
    
def has_numbers(title: str) -> bool:
    """Safely check if title contains numbers."""
    return bool(re.search(r'\d', title)) if title and isinstance(title, str) else False

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Self attention with skip connection and layer norm
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out).squeeze(1)

class EnhancedNumericalProcessor(nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_features)
        self.expand = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        expanded = self.expand(x)
        gate_values = self.gate(x)
        return expanded * gate_values

class FeatureExtractor:
    def __init__(self, word2vec_model, word2vec_dataset):
        """Initialize feature extractor with trained word2vec model."""
        self.word2vec_model = word2vec_model
        self.word2vec_dataset = word2vec_dataset
        self.numerical_scaler = StandardScaler()
        self.scaler_is_fit = False
        self.train_numerical_stats = None
        
    def _preprocess_title(self, title: str) -> str:
        """Enhanced title preprocessing with null check."""
        # Handle None or empty title
        if title is None or not isinstance(title, str):
            return ""
            
        # Convert to lowercase
        title = title.lower()
        
        # Remove special characters but keep important punctuation
        title = re.sub(r'[^a-z0-9\s\'\"\-\(\)]', ' ', title)
        
        # Handle common abbreviations
        title = title.replace("'s", " is")
        title = title.replace("'ve", " have")
        title = title.replace("'re", " are")
        title = title.replace("'ll", " will")
        
        # Normalize whitespace
        title = ' '.join(title.split())
        return title
    
    def extract_title_features(self, titles: List[str]) -> torch.Tensor:
        """Extract enhanced features from article titles using word2vec."""
        title_vectors = []
        
        for title in titles:
            # Enhanced preprocessing with null check
            processed_title = self._preprocess_title(title)
            words = self.word2vec_dataset._preprocess_text(processed_title) if processed_title else []
            
            # Get word indices with position information
            word_indices = [self.word2vec_dataset.vocab.get(word, 0) for word in words]
            
            if not word_indices:
                title_vectors.append(torch.zeros(self.word2vec_model.target_embeddings.weight.shape[1]))
                continue
            
            # Get word vectors
            vectors = [self.word2vec_model.get_word_vector(idx) for idx in word_indices]
            vectors = torch.stack(vectors)
            
            # Apply position-based weighting
            positions = torch.arange(len(vectors), dtype=torch.float32)
            position_weights = 1.0 / (1.0 + positions)  # Earlier words get higher weights
            weighted_vectors = vectors * position_weights.unsqueeze(1)
            
            # Compute weighted average
            title_vector = weighted_vectors.mean(dim=0)
            title_vectors.append(title_vector)
        
        return torch.stack(title_vectors)
    
    def extract_numerical_features(self, data: Dict[str, List], is_training: bool = False) -> np.ndarray:
        """Extract and normalize numerical features with safeguards."""
        # Basic features with clipping for stability
        num_comments = np.clip(
            np.array(data['num_comments']).reshape(-1, 1),
            0,
            np.percentile(data['num_comments'], 99)  # Clip at 99th percentile
        )
        
        # Derived features
        time_features = self._extract_time_features(data)
        title_features = self._extract_title_numerical_features(data)
        
        # Combine all features
        numerical_features = np.hstack([
            num_comments,
            time_features,
            title_features
        ])
        
        # Handle NaN/Inf values
        numerical_features = np.nan_to_num(numerical_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if is_training:
            # Fit scaler on training data and save statistics
            self.numerical_scaler.fit(numerical_features)
            self.scaler_is_fit = True
            self.train_numerical_stats = {
                'mean': numerical_features.mean(axis=0),
                'std': numerical_features.std(axis=0),
                'min': numerical_features.min(axis=0),
                'max': numerical_features.max(axis=0)
            }
            
        if not self.scaler_is_fit:
            raise ValueError("Scaler must be fit on training data first")
            
        # Transform features with clipping
        scaled_features = self.numerical_scaler.transform(numerical_features)
        return np.clip(scaled_features, -5, 5)  # Clip to reasonable range
        
    def _extract_time_features(self, data: Dict[str, List]) -> np.ndarray:
        """Extract time-based features with enhanced stability."""
        batch_size = len(data['num_comments'])
        
        if 'time' not in data:
            return np.zeros((batch_size, 4))
            
        try:
            timestamps = pd.to_datetime(data['time'])
            
            # Extract time components with bounds
            hour_of_day = np.clip(timestamps.dt.hour.values, 0, 23).reshape(-1, 1)
            day_of_week = np.clip(timestamps.dt.dayofweek.values, 0, 6).reshape(-1, 1)
            is_weekend = (day_of_week >= 5).astype(int).reshape(-1, 1)
            
            # Convert hour to bounded cyclical features
            hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
            hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
            
            features = np.hstack([hour_sin, hour_cos, day_of_week, is_weekend])
            return np.nan_to_num(features, nan=0.0)
            
        except Exception as e:
            logger.warning(f"Error in time feature extraction: {str(e)}")
            return np.zeros((batch_size, 4))
            
    def _extract_title_numerical_features(self, data: Dict[str, List]) -> np.ndarray:
        """Extract numerical features from titles with enhanced stability."""
        titles = data['title']
        batch_size = len(titles)
        
        try:
            # Safe feature extraction with bounds
            title_lengths = np.clip([safe_title_length(t) for t in titles], 0, 500).reshape(-1, 1)
            word_counts = np.clip([safe_word_count(t) for t in titles], 0, 100).reshape(-1, 1)
            question_marks = np.clip([safe_char_count(t, '?') for t in titles], 0, 10).reshape(-1, 1)
            exclamation_marks = np.clip([safe_char_count(t, '!') for t in titles], 0, 10).reshape(-1, 1)
            has_numbers_array = np.array([has_numbers(t) for t in titles], dtype=np.float32).reshape(-1, 1)
            
            features = np.hstack([
                title_lengths,
                word_counts,
                question_marks,
                exclamation_marks,
                has_numbers_array
            ])
            return np.nan_to_num(features, nan=0.0)
            
        except Exception as e:
            logger.warning(f"Error in title feature extraction: {str(e)}")
            return np.zeros((batch_size, 5))

class LateFusionModel(nn.Module):
    def __init__(self, text_embedding_dim, num_numerical_features):
        super().__init__()
        
        # Initialize weights with small values
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                torch.nn.init.zeros_(m.bias)
        
        # Text processing branch (ultra simple)
        self.text_norm = nn.LayerNorm(text_embedding_dim)
        self.text_fc = nn.Linear(text_embedding_dim, 32)
        self.text_bn = nn.BatchNorm1d(32)
        
        # Numerical processing branch (ultra simple)
        self.numerical_norm = nn.LayerNorm(num_numerical_features)
        self.numerical_fc = nn.Linear(num_numerical_features, 16)
        self.numerical_bn = nn.BatchNorm1d(16)
        
        # Combined processing (ultra simple)
        combined_dim = 32 + 16
        self.combined_fc = nn.Linear(combined_dim, 1)
        
        # Apply weight initialization
        self.apply(init_weights)
        
    def forward(self, text_features, numerical_features):
        # Text processing with strong normalization
        text = self.text_norm(text_features)
        text = torch.tanh(self.text_fc(text))
        text = self.text_bn(text)
        
        # Numerical processing with strong normalization
        num = self.numerical_norm(numerical_features)
        num = torch.tanh(self.numerical_fc(num))
        num = self.numerical_bn(num)
        
        # Combine features
        combined = torch.cat((text, num), dim=1)
        
        # Final processing with bounded output
        return torch.tanh(self.combined_fc(combined))

def train_late_fusion(model, feature_extractor, train_data, num_epochs, batch_size, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Ultra conservative optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.00001,  # Minimal weight decay
        betas=(0.9, 0.99),
        eps=1e-7
    )
    
    # Simple MSE loss
    criterion = nn.MSELoss()
    
    # Early stopping setup
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    # Extract and normalize features
    title_features = feature_extractor.extract_title_features(train_data['title'])
    numerical_features = feature_extractor.extract_numerical_features(train_data, is_training=True)  # Set training flag
    
    # Log feature statistics during training
    logger.info("\nTraining Feature Statistics:")
    logger.info(f"Title features shape: {title_features.shape}")
    logger.info(f"Numerical features shape: {numerical_features.shape}")
    logger.info(f"Numerical features stats - Mean: {numerical_features.mean():.4f}, Std: {numerical_features.std():.4f}")
    
    # Normalize and clip targets more aggressively
    targets = torch.FloatTensor(train_data['score'].values)
    target_mean = targets.mean()
    target_std = targets.std()
    normalized_targets = torch.clamp((targets - target_mean) / target_std, min=-1, max=1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(title_features, torch.FloatTensor(numerical_features), normalized_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info("Starting Late Fusion model training...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        valid_batches = 0
        model.train()
        
        for batch_title, batch_numerical, batch_targets in dataloader:
            batch_title = batch_title.to(device)
            batch_numerical = batch_numerical.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass with gradient clipping
                predictions = model(batch_title, batch_numerical)
                loss = criterion(predictions.squeeze(), batch_targets)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    valid_batches += 1
            except Exception as e:
                logger.warning(f"Error in batch: {str(e)}")
                continue
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Valid batches: {valid_batches}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered!")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        else:
            logger.warning(f"No valid batches in epoch {epoch+1}")
    
    return model 