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
            'late_fusion_params': LATE_FUSION_PARAMS,
            'feature_extractor_state': {
                'scaler_state': feature_extractor.numerical_scaler.__dict__,
                'train_stats': feature_extractor.train_numerical_stats
            }
        }, 'late_fusion_checkpoint.pt')
        
        # Step 7: Evaluate on test set
        logger.info("Evaluating model on test set...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Extract test features using the same scaler fit on training data
            test_title_features = feature_extractor.extract_title_features(test_data['title'])
            test_numerical_features = feature_extractor.extract_numerical_features(test_data, is_training=False)
            
            # Log feature statistics
            logger.info("\nFeature Statistics:")
            logger.info(f"Title features shape: {test_title_features.shape}")
            logger.info(f"Numerical features shape: {test_numerical_features.shape}")
            logger.info(f"Numerical features stats - Mean: {test_numerical_features.mean():.4f}, Std: {test_numerical_features.std():.4f}")
            
            # Get training data statistics for normalization
            train_targets = torch.FloatTensor(train_data['score'].values)
            train_mean = train_targets.mean()
            train_std = train_targets.std()
            logger.info(f"\nTraining data statistics:")
            logger.info(f"- Mean: {train_mean:.2f}")
            logger.info(f"- Std: {train_std:.2f}")
            logger.info(f"- Min: {train_targets.min().item():.2f}")
            logger.info(f"- Max: {train_targets.max().item():.2f}")
            
            # Normalize test targets using training statistics
            test_targets = torch.FloatTensor(test_data['score'].values)
            normalized_test_targets = torch.clamp((test_targets - train_mean) / train_std, min=-1, max=1)
            
            # Convert to tensors and move to device
            test_title_features = test_title_features.to(device)
            test_numerical_features = torch.FloatTensor(test_numerical_features).to(device)
            normalized_test_targets = normalized_test_targets.to(device)
            
            # Set model to eval mode
            trained_model = trained_model.to(device)
            trained_model.eval()
            
            with torch.no_grad():
                # Get predictions in smaller batches
                batch_size = 128
                all_predictions = []
                
                for i in range(0, len(test_title_features), batch_size):
                    batch_title = test_title_features[i:i + batch_size]
                    batch_numerical = test_numerical_features[i:i + batch_size]
                    
                    # Get normalized predictions for batch
                    batch_predictions = trained_model(batch_title, batch_numerical).squeeze()
                    
                    # Check for NaN in batch predictions
                    if torch.isnan(batch_predictions).any():
                        logger.warning(f"NaN detected in batch {i//batch_size + 1}")
                        logger.info(f"Batch title features stats - Mean: {batch_title.mean():.4f}, Std: {batch_title.std():.4f}")
                        logger.info(f"Batch numerical features stats - Mean: {batch_numerical.mean():.4f}, Std: {batch_numerical.std():.4f}")
                    
                    all_predictions.append(batch_predictions)
                
                # Combine all predictions
                normalized_predictions = torch.cat(all_predictions)
                
                # Log normalized prediction statistics
                logger.info("\nNormalized Prediction Statistics:")
                logger.info(f"- Mean: {normalized_predictions.mean():.4f}")
                logger.info(f"- Std: {normalized_predictions.std():.4f}")
                logger.info(f"- Min: {normalized_predictions.min():.4f}")
                logger.info(f"- Max: {normalized_predictions.max():.4f}")
                
                # Denormalize predictions
                predictions = normalized_predictions * train_std + train_mean
                
                # Ensure predictions are non-negative
                predictions = torch.clamp(predictions, min=0)
                
                # Calculate MSE on denormalized values
                test_loss = torch.nn.MSELoss()(predictions, test_targets.to(device))
                logger.info(f"\nTest MSE Loss: {test_loss.item():.4f}")
                
                # Log final statistics
                logger.info("\nFinal Statistics:")
                logger.info("Predictions:")
                logger.info(f"- Mean: {predictions.mean():.2f}")
                logger.info(f"- Std: {predictions.std():.2f}")
                logger.info(f"- Min: {predictions.min():.2f}")
                logger.info(f"- Max: {predictions.max():.2f}")
                logger.info("Actual Targets:")
                logger.info(f"- Mean: {test_targets.mean():.2f}")
                logger.info(f"- Std: {test_targets.std():.2f}")
                logger.info(f"- Min: {test_targets.min():.2f}")
                logger.info(f"- Max: {test_targets.max():.2f}")
                
                # Save all statistics
                save_checkpoint({
                    'word2vec_state_dict': word2vec_model.state_dict(),
                    'late_fusion_state_dict': trained_model.state_dict(),
                    'word2vec_params': WORD2VEC_PARAMS,
                    'late_fusion_params': LATE_FUSION_PARAMS,
                    'train_mean': train_mean.item(),
                    'train_std': train_std.item(),
                    'test_mse': test_loss.item(),
                    'prediction_stats': {
                        'mean': predictions.mean().item(),
                        'std': predictions.std().item(),
                        'min': predictions.min().item(),
                        'max': predictions.max().item()
                    }
                }, 'trained_models.pt')
                
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Step 8: Save final models
        logger.info("Saving final models...")
        save_checkpoint({
            'word2vec_state_dict': word2vec_model.state_dict(),
            'late_fusion_state_dict': trained_model.state_dict(),
            'word2vec_params': WORD2VEC_PARAMS,
            'late_fusion_params': LATE_FUSION_PARAMS,
            'target_mean': train_mean.item(),
            'target_std': train_std.item()
        }, 'trained_models.pt')
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 