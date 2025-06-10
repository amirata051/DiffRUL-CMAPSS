import torch
import torch.nn as nn

class RNNRULPredictor(nn.Module):
    def __init__(self, input_size=14, hidden_size=64):
        super(RNNRULPredictor, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1)

        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)

        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        return out