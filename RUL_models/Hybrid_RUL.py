import torch
import torch.nn as nn

class HybridRULPredictor(nn.Module):
    def __init__(self, in_channels=14, seq_len=50):
        super(HybridRULPredictor, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=18, kernel_size=2, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv1d(in_channels=18, out_channels=36, kernel_size=2, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv1d(in_channels=36, out_channels=72, kernel_size=2, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv_output_size = self.get_conv_output_size((in_channels, seq_len))
        print(f"HybridRULPredictor: conv_output_size = {self.conv_output_size}")
        self.fc1 = nn.Linear(self.conv_output_size, 420)
        self.dropout1 = nn.Dropout(0.1)

        self.lstm1 = nn.LSTM(input_size=420, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)

        self.fc2 = nn.Linear(64, 16)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(16, 1)

    def get_conv_output_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        return x.shape[1] * x.shape[2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = x.view(x.size(0), 1, -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x