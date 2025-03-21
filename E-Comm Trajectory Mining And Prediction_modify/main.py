import pandas as pd
import torch
from src.preprocess import preprocess_data
from src.clustering import cluster_trajectories
from src.prediction import train_hmm, train_lstm, train_transformer_hmm
from src.evaluation import evaluate_predictions
from src.utils import create_directories, log_message
from src.config import HMM_N_STATES

# Ensure directories exist
create_directories()

# Load dataset
data = pd.read_csv("data/user_behavior_data.csv")

# Preprocess data
log_message("Starting data preprocessing...")
user_sessions = preprocess_data(data)

# Perform clustering
log_message("Starting trajectory clustering...")
cluster_labels = cluster_trajectories(user_sessions)

# Train models
log_message("Training HMM model...")
hmm_model = train_hmm(user_sessions)

log_message("Training LSTM model...")
train_data = torch.randn(100, 10, 50)  # Simulated data
train_labels = torch.randint(0, 10, (100,))  # Simulated labels
lstm_model = train_lstm(train_data, train_labels)

log_message("Training Transformer-HMM model...")
transformer_model = train_transformer_hmm(train_data, train_labels)

# Evaluate model performance
log_message("Evaluating models...")
p1, p6, mrr = evaluate_predictions([], [])  # Placeholder
log_message(f"P@1: {p1}, P@6: {p6}, MRR: {mrr}")
