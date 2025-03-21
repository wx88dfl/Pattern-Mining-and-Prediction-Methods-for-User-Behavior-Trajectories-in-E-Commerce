Features
1. Data Preprocessing (preprocess.py)
Cleans and processes user clickstream data.
Extracts time-based behavior features (morning, afternoon, evening).
Filters abnormal dwell times and limits trajectory length.
2. Trajectory Clustering (clustering.py)
Uses HDBSCAN for density-based clustering.
Computes DTW (Dynamic Time Warping) + LCS (Longest Common Subsequence) similarity.
Groups users based on trajectory similarity.
3. Trajectory Prediction (prediction.py)
Computes an HMM transition matrix with Bayesian smoothing.
Implements Transformer-HMM for deep feature extraction.
Uses LSTM for long-sequence trajectory prediction.
Supports model training, saving, and loading.
4. Frequent Trajectory Mining (frequent_patterns.py)
Uses FP-Growth to mine frequent trajectory patterns.
5. Evaluation (evaluation.py)
Computes P@1, P@6, and MRR for prediction accuracy assessment.
Compares HMM, Transformer-HMM, and LSTM models.
6. Utility Functions (utils.py)
Handles logging, error handling, and directory management.
7. Configuration File (config.py)
Centralized hyperparameter tuning for clustering, prediction, and evaluation.