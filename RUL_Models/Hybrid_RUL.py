import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import BearingDataset

class HybridRULPredictor(nn.Module):
    def __init__(self, in_channels=14):
        super(HybridRULPredictor, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=18, kernel_size=2, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.conv2 = nn.Conv1d(in_channels=18, out_channels=36, kernel_size=2, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv1d(in_channels=36, out_channels=72, kernel_size=2, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        # Calculate conv output size
        self.conv_output_size = self.get_conv_output_size((in_channels, 50))
        self.fc1 = nn.Linear(self.conv_output_size, 420)
        self.dropout1 = nn.Dropout(0.1)

        self.lstm1 = nn.LSTM(input_size=420, hidden_size=32, batch_first=True, dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True, dropout=0.1)

        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(16, 1)

    def get_conv_output_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        return x.shape[1] * x.shape[2]

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Reshape for LSTM
        x = x.view(x.size(0), 1, -1)  # (batch_size, seq_len=1, 420)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]  # Take last time step
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

# Path to the C-MAPSS dataset
data_dir = "/workspace/BearingGroup/nasa-cmaps/CMaps"
dataset = BearingDataset(data_dir, window_size=50, return_pairs=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = HybridRULPredictor(in_channels=14)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.permute(0, 2, 1).to(device)  # Shape: (batch_size, 14, 50)
        batch_y = batch_y.to(device)

        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.permute(0, 2, 1).to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x).squeeze()
        print("Predicted RUL:", outputs[:5].cpu().numpy())
        print("Actual RUL:", batch_y[:5].cpu().numpy())
        break