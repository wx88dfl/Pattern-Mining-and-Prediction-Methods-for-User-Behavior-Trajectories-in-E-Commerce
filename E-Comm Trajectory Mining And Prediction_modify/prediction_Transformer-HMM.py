import torch
import torch.nn as nn
from transformers import BertModel
from src.utils import log_message
from src.config import TRANSFORMER_HIDDEN_DIM

class TransformerHMM(nn.Module):
    """Transformer-HMM hybrid model for trajectory prediction."""
    def __init__(self):
        super(TransformerHMM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, TRANSFORMER_HIDDEN_DIM)

    def forward(self, x):
        x = self.bert(x)[0][:, 0, :]  # Extract CLS token features
        x = self.fc(x)
        return x

def train_transformer_hmm(train_data, train_labels, epochs=10, learning_rate=0.001):
    """
    Train a Transformer-HMM model for trajectory prediction.

    :param train_data: Input training sequences
    :param train_labels: Expected output labels
    :param epochs: Number of training iterations
    :param learning_rate: Learning rate
    :return: Trained Transformer-HMM model
    """
    model = TransformerHMM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        log_message(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save trained Transformer-HMM model
    torch.save(model.state_dict(), "models/transformer_hmm.pth")
    log_message("Transformer-HMM model training completed and saved as 'transformer_hmm.pth'.")
    return model

def load_transformer_hmm():
    """Load the trained Transformer-HMM model from disk."""
    model = TransformerHMM()
    model.load_state_dict(torch.load("models/transformer_hmm.pth"))
    model.eval()
    log_message("Loaded Transformer-HMM model from 'transformer_hmm.pth'.")
    return model
