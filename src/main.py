import logging
import torch
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import gc
from tqdm import tqdm

from data_collection import get_training_data
from word2vec_model import train_word2vec
from late_fusion_model import LateFusionModel, FeatureExtractor, train_late_fusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
WIKIPEDIA_CATEGORIES = [
    'Technology',
    'Computer_science',
    'Software_engineering'  # Reduced categories
]
DB_URL = 'postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki'
WORD2VEC_PARAMS = {
    'embedding_dim': 50,     # Reduced from 100
    'window_size': 4,        # Reduced from 5
    'min_count': 10,         # Increased from 5
    'batch_size': 1024,      # Reduced for stability
    'num_epochs': 3,         # Reduced from 5
    'learning_rate': 0.025
}
LATE_FUSION_PARAMS = {
    'num_epochs': 30,        # More epochs for smoother convergence
    'batch_size': 8,         # Ultra small batch size
    'learning_rate': 0.000001 # Ultra small learning rate
}

def save_checkpoint(state, filename='checkpoint.pt'):
    """Save training checkpoint."""
    logger.info(f"Saving checkpoint to {filename}")
    os.makedirs('models', exist_ok=True)
    torch.save(state, os.path.join('models', filename))

def load_checkpoint(filename='checkpoint.pt'):
    """Load training checkpoint."""
    path = os.path.join('models', filename)
    if os.path.exists(path):
        logger.info(f"Loading checkpoint from {path}")
        return torch.load(path)
    return None

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        # Step 1: Load training data
        logger.info("Loading training data...")
        data = get_training_data(DB_URL, WIKIPEDIA_CATEGORIES)
        
        # Step 2: Train word2vec on Wikipedia data
        logger.info("Training word2vec model on Wikipedia data...")
        word2vec_model, word2vec_dataset = train_word2vec(
            texts=data['wikipedia']['text'].tolist(),
            **WORD2VEC_PARAMS
        )
        
        # Save word2vec checkpoint
        save_checkpoint({
            'word2vec_state_dict': word2vec_model.state_dict(),
            'word2vec_params': WORD2VEC_PARAMS
        }, 'word2vec_checkpoint.pt')
        
        # Clear some memory
        del data['wikipedia']
        gc.collect()
        
        # Step 3: Prepare Hacker News data
        hn_data = data['hacker_news']
        del data
        gc.collect()
        
        # Split data into train and test sets
        train_data, test_data = train_test_split(hn_data, test_size=0.2, random_state=42)
        del hn_data
        gc.collect()
        
        # Step 4: Initialize feature extractor
        feature_extractor = FeatureExtractor(word2vec_model, word2vec_dataset)
        
        # Step 5: Initialize Late Fusion model
        late_fusion_model = LateFusionModel(
            text_embedding_dim=WORD2VEC_PARAMS['embedding_dim'],
            num_numerical_features=10  # num_comments + 4 time features + 5 title numerical features
        )
        
        # Step 6: Train Late Fusion model
        logger.info("Training Late Fusion model...")
        trained_model = train_late_fusion(
            model=late_fusion_model,
            feature_extractor=feature_extractor,
            train_data=train_data,
            **LATE_FUSION_PARAMS
        )
        
        # Save late fusion checkpoint
        save_checkpoint({
            'late_fusion_state_dict': trained_model.state_dict(),
            'late_fusion_params': LATE_FUSION_PARAMS
        }, 'late_fusion_checkpoint.pt')
        
        # Step 7: Evaluate on test set
        logger.info("Evaluating model on test set...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract test features
        test_title_features = feature_extractor.extract_title_features(test_data['title'])
        test_numerical_features = feature_extractor.extract_numerical_features(test_data)
        
        # Normalize test targets the same way as training
        test_targets = torch.FloatTensor(test_data['score'].values)
        target_mean = test_targets.mean()
        target_std = test_targets.std()
        test_targets = torch.clamp((test_targets - target_mean) / target_std, min=-1, max=1)
        
        # Convert to tensors
        test_title_features = test_title_features.to(device)
        test_numerical_features = torch.FloatTensor(test_numerical_features).to(device)
        test_targets = test_targets.to(device)
        
        # Get predictions
        trained_model.eval()
        with torch.no_grad():
            try:
                predictions = trained_model(test_title_features, test_numerical_features)
                # Denormalize predictions for true MSE
                predictions = predictions.squeeze() * target_std + target_mean
                test_targets_denorm = test_targets * target_std + target_mean
                test_loss = torch.nn.MSELoss()(predictions, test_targets_denorm)
                logger.info(f"Test MSE Loss: {test_loss.item():.4f}")
                
                # Also log some basic statistics
                logger.info(f"Predictions - Mean: {predictions.mean():.2f}, Std: {predictions.std():.2f}")
                logger.info(f"Actual - Mean: {test_targets_denorm.mean():.2f}, Std: {test_targets_denorm.std():.2f}")
            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
                test_loss = torch.tensor(float('nan'))
        
        # Step 8: Save final models
        logger.info("Saving final models...")
        save_checkpoint({
            'word2vec_state_dict': word2vec_model.state_dict(),
            'late_fusion_state_dict': trained_model.state_dict(),
            'word2vec_params': WORD2VEC_PARAMS,
            'late_fusion_params': LATE_FUSION_PARAMS,
            'target_mean': target_mean,
            'target_std': target_std
        }, 'trained_models.pt')
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 