import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import BearingDataset

class RNNRULPredictor(nn.Module):
    def __init__(self, input_size=14, hidden_size=64):
        super(RNNRULPredictor, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, dropout=0.1)

        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)

        # Take the last time step
        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        return out

# Path to the C-MAPSS dataset
data_dir = "/workspace/BearingGroup/nasa-cmaps/CMaps"
dataset = BearingDataset(data_dir, window_size=30, return_pairs=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = RNNRULPredictor(input_size=14)
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
        batch_x = batch_x.to(device)  # Shape: (batch_size, 30, 14)
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
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x).squeeze()
        print("Predicted RUL:", outputs[:5].cpu().numpy())
        print("Actual RUL:", batch_y[:5].cpu().numpy())
        break