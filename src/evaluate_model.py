import os
import sys
import torch
import logging
import numpy as np

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print(f"Added {current_dir} to Python path")

from word2vec_model import Word2Vec, Word2VecDataset
from late_fusion_model import LateFusionModel, FeatureExtractor
from data_collection import get_training_data

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - INFO - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, models_dir='models'):
        """Initialize evaluator with trained models."""
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the saved model parameters
        model_path = os.path.join(self.models_dir, 'trained_models.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")
        
        self.saved_data = torch.load(model_path, map_location=self.device)
        
        # Get some sample data to initialize vocabulary and scaling parameters
        DB_URL = 'postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki'
        WIKIPEDIA_CATEGORIES = ['Technology', 'Computer_science', 'Software_engineering']
        data = get_training_data(DB_URL, WIKIPEDIA_CATEGORIES)
        
        # Calculate score statistics for denormalization
        scores = data['hacker_news']['score'].values
        self.q1 = np.percentile(scores, 25)
        self.q3 = np.percentile(scores, 75)
        self.iqr = self.q3 - self.q1
        logger.info(f"Score statistics - Q1: {self.q1:.2f}, Q3: {self.q3:.2f}, IQR: {self.iqr:.2f}")
        
        # Initialize Word2Vec model and dataset with actual data
        word2vec_params = self.saved_data.get('word2vec_params', {
            'embedding_dim': 50,
            'window_size': 4,
            'min_count': 10
        })
        
        self.word2vec_model = Word2Vec(vocab_size=1848, embedding_dim=word2vec_params['embedding_dim'])
        self.word2vec_dataset = Word2VecDataset(
            texts=data['wikipedia']['text'].tolist(),
            window_size=word2vec_params['window_size'],
            min_count=word2vec_params['min_count']
        )
        self.load_word2vec()
        
        # Initialize and load Late Fusion model
        self.late_fusion_model = LateFusionModel(text_embedding_dim=word2vec_params['embedding_dim'], 
                                               num_numerical_features=1)
        self.load_late_fusion()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.word2vec_model, self.word2vec_dataset)
    
    def load_word2vec(self):
        """Load trained Word2Vec model."""
        if 'word2vec_state_dict' in self.saved_data:
            self.word2vec_model.load_state_dict(self.saved_data['word2vec_state_dict'])
            self.word2vec_model.eval()
            logger.info("Loaded Word2Vec model successfully")
        else:
            raise KeyError("Could not find word2vec_state_dict in saved model file")
    
    def load_late_fusion(self):
        """Load trained Late Fusion model."""
        if 'late_fusion_state_dict' in self.saved_data:
            self.late_fusion_model.load_state_dict(self.saved_data['late_fusion_state_dict'])
            self.late_fusion_model.eval()
            logger.info("Loaded Late Fusion model successfully")
        else:
            raise KeyError("Could not find late_fusion_state_dict in saved model file")
    
    def predict_score(self, title: str, num_comments: int = 0) -> float:
        """Predict the score for a given headline."""
        # Extract features
        title_features = self.feature_extractor.extract_title_features([title])
        numerical_features = self.feature_extractor.extract_numerical_features({
            'num_comments': [num_comments]
        })
        
        # Convert to tensors
        title_features = title_features.to(self.device)
        numerical_features = torch.FloatTensor(numerical_features).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.late_fusion_model(title_features, numerical_features)
            # Denormalize prediction (reverse the IQR normalization)
            normalized_score = prediction.item()
            denormalized_score = (normalized_score * self.iqr) + self.q1
            return max(0, denormalized_score)  # Ensure non-negative score

def main():
    """Test the model with sample headlines."""
    evaluator = ModelEvaluator()
    
    # Test headlines with varying characteristics
    test_headlines = [
        ("OpenAI Releases GPT-4 with Multimodal Capabilities", 150),
        ("Show HN: I built a free alternative to expensive data visualization tools", 45),
        ("Understanding Transformers: A Deep Dive into Attention Mechanisms", 80),
        ("Tutorial: Building a Real-time Chat Application with WebSockets", 25),
        ("Ask HN: What's your preferred tech stack for side projects?", 200)
    ]
    
    logger.info("\nTesting model predictions...")
    logger.info("-" * 80)
    for title, num_comments in test_headlines:
        predicted_score = evaluator.predict_score(title, num_comments)
        logger.info(f"\nTitle: {title}")
        logger.info(f"Number of Comments: {num_comments}")
        logger.info(f"Predicted Score: {predicted_score:.1f}")
    logger.info("-" * 80)

if __name__ == "__main__":
    main() 