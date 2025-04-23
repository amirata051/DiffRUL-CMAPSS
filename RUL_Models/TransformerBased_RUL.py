import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import BearingDataset
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerRULPredictor(nn.Module):
    def __init__(self, input_dim=14, seq_len=30, d_model=64, num_heads=8, num_layers=2, ff_dim=512, dropout=0.1):
        super(TransformerRULPredictor, self).__init__()

        self.linear1 = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.dropout1 = nn.Dropout(dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 16)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(16, 8)
        self.dropout3 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.linear1(x)  # (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        x = self.dropout1(x)

        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # Take last time step

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.fc3(x)
        return x

# Path to the C-MAPSS dataset
data_dir = "/workspace/BearingGroup/nasa-cmaps/CMaps"
dataset = BearingDataset(data_dir, window_size=30, return_pairs=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = TransformerRULPredictor(input_dim=14, seq_len=30)
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