import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import BearingDataset

class CNNRULPredictor(nn.Module):
    def __init__(self):
        super(CNNRULPredictor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(14, 3), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=2, padding=0)

        # Calculate feature map size dynamically
        self.feature_map_size = self.get_feature_map_size((1, 30, 14))
        self.fc1 = nn.Linear(self.feature_map_size, 16)
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(8, 1)

    def get_feature_map_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

# Path to the C-MAPSS dataset
data_dir = "/workspace/BearingGroup/nasa-cmaps/CMaps"
dataset = BearingDataset(data_dir, window_size=30, return_pairs=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = CNNRULPredictor()
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
        batch_x = batch_x.unsqueeze(1).to(device)  # Add channel dimension (batch_size, 1, 30, 14)
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
        batch_x = batch_x.unsqueeze(1).to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x).squeeze()
        print("Predicted RUL:", outputs[:5].cpu().numpy())
        print("Actual RUL:", batch_y[:5].cpu().numpy())
        break