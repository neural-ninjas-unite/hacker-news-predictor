# Hacker News Score Predictor

This project implements a late fusion model to predict Hacker News post scores using various features including:
- Title text (processed using Word2Vec pre-trained on Wikipedia)
- Number of comments
- Author information

## Project Structure

```
.
├── README.md
├── requirements.txt
└── src/
    ├── model.py      # Contains model architecture and utility functions
    └── main.py       # Main script to run the training
```

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Model

To train the model:

```bash
python src/main.py
```

This will:
1. Load the pre-trained Word2Vec model
2. Fetch data from the Hacker News database
3. Process features (text, numerical, and author features)
4. Train the late fusion model
5. Save the trained model as 'trained_model.pth'

## Model Architecture

The late fusion model consists of three branches:
1. Text Processing Branch:
   - Processes title text using Word2Vec embeddings
   - Two fully connected layers with ReLU activation and dropout

2. Numerical Processing Branch:
   - Processes numerical features (number of comments)
   - Two fully connected layers with ReLU activation

3. Author Processing Branch:
   - Processes author information using one-hot encoding
   - Two fully connected layers with ReLU activation

These branches are combined in a fusion layer that produces the final score prediction.

## Data

The model uses data from a PostgreSQL database containing Hacker News posts. The following features are used:
- Title: The title of the post
- Score: The number of upvotes (target variable)
- Author: The post author
- Number of Comments: The number of comments on the post

## Requirements

See `requirements.txt` for a complete list of dependencies. Key requirements include:
- PyTorch
- Gensim (for Word2Vec)
- SQLAlchemy
- NumPy
- Pandas
- scikit-learn
