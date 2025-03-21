import pandas as pd
from src.preprocess import preprocess_data
from src.clustering import cluster_trajectories
from src.prediction import train_hmm, compute_transition_matrix
from src.evaluation import evaluate_predictions

# Load dataset
data = pd.read_csv("data/user_behavior_data.csv")

# Data preprocessing
user_sessions = preprocess_data(data)

# Clustering
cluster_labels = cluster_trajectories(user_sessions)

# Train HMM
hmm_model = train_hmm(user_sessions)

# Evaluation
p1, p6, mrr = evaluate_predictions([], [])  # Add actual values here
print(f"P@1: {p1}, P@6: {p6}, MRR: {mrr}")
