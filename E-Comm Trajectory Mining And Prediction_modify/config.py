# Configuration file for hyperparameters and file paths

# Data preprocessing
MIN_DWELL_TIME = 15
MAX_DWELL_TIME = 1800
MAX_SEQUENCE_LENGTH = 30

# Clustering
CLUSTERING_METHOD = "hdbscan"  # Options: "dbscan" or "hdbscan"
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5

# HMM Parameters
HMM_N_STATES = 10
HMM_N_ITER = 100

# LSTM Parameters
LSTM_INPUT_DIM = 50
LSTM_HIDDEN_DIM = 128
LSTM_OUTPUT_DIM = 10
LSTM_NUM_LAYERS = 2

# Transformer-HMM Parameters
TRANSFORMER_HIDDEN_DIM = 10

# Frequent Pattern Mining
FP_GROWTH_MIN_SUPPORT = 0.05
