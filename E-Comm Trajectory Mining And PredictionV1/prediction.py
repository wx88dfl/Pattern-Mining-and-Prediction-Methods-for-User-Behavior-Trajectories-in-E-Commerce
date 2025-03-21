import numpy as np
from hmmlearn import hmm
import torch
import torch.nn as nn
from transformers import BertModel

class TransformerHMM(nn.Module):
    """
    Transformer-HMM hybrid model for trajectory prediction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransformerHMM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hmm = hmm.MultinomialHMM(n_components=hidden_dim)
        self.fc = nn.Linear(768, output_dim)
    
    def forward(self, x):
        x = self.bert(x)[0][:, 0, :]  # Extract CLS token features
        x = self.fc(x)
        return self.hmm.predict(x.detach().numpy())

def compute_transition_matrix(user_sessions, n_states):
    """
    Compute transition matrix for HMM using Bayesian smoothing.

    :param user_sessions: User trajectory sequences
    :param n_states: Number of hidden states
    :return: Transition matrix
    """
    state_counts = np.ones((n_states, n_states))  # Laplace smoothing

    for seq in user_sessions.values():
        for i in range(len(seq) - 1):
            from_state, to_state = seq[i][0], seq[i + 1][0]
            state_counts[from_state, to_state] += 1

    return state_counts / np.maximum(state_counts.sum(axis=1, keepdims=True), 1)

def train_hmm(user_sessions, n_states=10, n_iter=100):
    """
    Train HMM for trajectory prediction.

    :param user_sessions: User trajectory data
    :param n_states: Number of hidden states
    :param n_iter: Number of iterations
    :return: Trained HMM model
    """
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=n_iter, verbose=True)
    observations = [list(map(lambda x: x[0], seq)) for seq in user_sessions.values()]
    model.fit(np.array(observations).reshape(-1, 1))

    return model
