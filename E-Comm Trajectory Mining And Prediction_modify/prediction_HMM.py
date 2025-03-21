import numpy as np
import pickle
from hmmlearn import hmm
from src.utils import log_message
from src.config import HMM_N_STATES, HMM_N_ITER

def train_hmm(user_sessions):
    """
    Train an HMM model for trajectory prediction.

    :param user_sessions: User trajectory data
    :return: Trained HMM model
    """
    # Convert trajectories into numerical format
    page_to_idx = {}
    observations = []
    lengths = []

    for seq in user_sessions.values():
        obs_seq = [page_to_idx.setdefault(p[0], len(page_to_idx)) for p in seq]
        observations.extend(obs_seq)
        lengths.append(len(obs_seq))

    model = hmm.MultinomialHMM(n_components=HMM_N_STATES, n_iter=HMM_N_ITER, verbose=True)
    model.fit(np.array(observations).reshape(-1, 1), lengths)

    # Save trained HMM model
    with open("models/hmm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    log_message("HMM model training completed and saved as 'hmm_model.pkl'.")
    return model

def load_hmm():
    """Load the trained HMM model from disk."""
    with open("models/hmm_model.pkl", "rb") as f:
        model = pickle.load(f)
    log_message("Loaded HMM model from 'hmm_model.pkl'.")
    return model

