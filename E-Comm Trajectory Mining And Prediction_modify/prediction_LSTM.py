import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import log_message
from src.config import LSTM_INPUT_DIM, LSTM_HIDDEN_DIM, LSTM_OUTPUT_DIM, LSTM_NUM_LAYERS

class LSTMModel(nn.Module):
    """LSTM model for trajectory prediction."""
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(LSTM_INPUT_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(LSTM_HIDDEN_DIM, LSTM_OUTPUT_DIM)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm(train_data, train_labels, epochs=10, learning_rate=0.001):
    """
    Train an LSTM model for trajectory prediction.

    :param train_data: Input training sequences
    :param train_labels: Expected output labels
    :param epochs: Number of training iterations
    :param learning_rate: Learning rate
    :return: Trained LSTM model
    """
    model = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        log_message(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save trained LSTM model
    torch.save(model.state_dict(), "models/lstm_model.pth")
    log_message("LSTM model training completed and saved as 'lstm_model.pth'.")
    return model

def load_lstm():
    """Load the trained LSTM model from disk."""
    model = LSTMModel()
    model.load_state_dict(torch.load("models/lstm_model.pth"))
    model.eval()
    log_message("Loaded LSTM model from 'lstm_model.pth'.")
    return model
