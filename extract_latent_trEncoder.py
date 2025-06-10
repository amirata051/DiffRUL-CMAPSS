import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import umap  # pip install umap-learn

from config import config
from dataset import CMAPSDataset
from DTE_model.Transformer_Encoder import TransformerEncoder, TransformerDecoder, TSHAE
from utils.loss import TotalLoss
from utils.utils import set_seed

# Step 1: Setup
set_seed(2023)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Load dataset
train_dataset = CMAPSDataset(
    config['data_dir'], mode="train",
    window_size=config['window_size'], return_pairs=True
)
test_dataset = CMAPSDataset(
    config['data_dir'], mode="test",
    window_size=config['window_size'], return_pairs=False
)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# Step 3: Initialize model
encoder = TransformerEncoder(config)
decoder = TransformerDecoder(config)
model = TSHAE(config, encoder, decoder).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
criterion = TotalLoss(config)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config['lr_scheduler']['step_size'],
    gamma=config['lr_scheduler']['gamma']
)

# Step 4: Train TSHAE for a few epochs
model.train()
for epoch in range(10):  # You can increase epochs
    for batch in train_loader:
        x, pos_x, neg_x, y = batch
        x, y = x.to(device), y.to(device)
        pos_x, neg_x = pos_x.to(device), neg_x.to(device)

        optimizer.zero_grad()
        y_hat, z, mean, log_var, x_hat, z_pos, z_neg = model(x)

        # Optional: get z_pos and z_neg directly from model (sanity check)
        pos_out = model(pos_x)
        neg_out = model(neg_x)
        _, z_pos, *_ = pos_out
        _, z_neg, *_ = neg_out

        loss_dict = criterion(
            mean=mean, log_var=log_var, y=y, y_hat=y_hat,
            x=x, x_hat=x_hat, z=z, z_pos=z_pos, z_neg=z_neg
        )
        loss = loss_dict["TotalLoss"]
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"[Epoch {epoch+1}] Training complete.")

# Step 5: Extract latent vectors from test set
model.eval()
all_z = []
all_labels = []

with torch.no_grad():
    for data in test_loader:
        x, labels = data
        x = x.to(device)
        y_hat, z, mean, log_var, x_hat, _, _ = model(x)
        all_z.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

all_z = np.vstack(all_z)
all_labels = np.concatenate(all_labels)

# Step 6: Plot latent dims 1, 2, and 3 vs. RUL
plt.figure(figsize=(10, 6))
plt.scatter(all_z[:, 0], all_labels, c='blue', alpha=0.6, label='Latent dim 1')
plt.scatter(all_z[:, 1], all_labels, c='orange', alpha=0.6, label='Latent dim 2')
if all_z.shape[1] > 2:
    plt.scatter(all_z[:, 2], all_labels, c='green', alpha=0.6, label='Latent dim 3')
plt.xlabel('Latent Dimension Value')
plt.ylabel('RUL')
plt.title('Latent Dimensions vs. RUL')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("latent_dims_vs_rul.png", dpi=300)
plt.close()

# Step 7: UMAP visualization
reducer = umap.UMAP(n_components=2, random_state=42)
z_umap = reducer.fit_transform(all_z)

plt.figure(figsize=(8, 6))
sc = plt.scatter(z_umap[:, 0], z_umap[:, 1], c=all_labels, cmap='viridis', s=10)
plt.colorbar(sc, label='RUL')
plt.title('UMAP Visualization of Latent Space')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.savefig("latent_space_umap.png", dpi=300)
plt.close()

import seaborn as sns
import pandas as pd

umap_df = pd.DataFrame({
    'UMAP1': z_umap[:, 0],
    'UMAP2': z_umap[:, 1],
    'RUL': all_labels
})

grid_size = 50 
umap_df['x_bin'] = pd.cut(umap_df['UMAP1'], bins=grid_size, labels=False)
umap_df['y_bin'] = pd.cut(umap_df['UMAP2'], bins=grid_size, labels=False)

heatmap_data = umap_df.groupby(['y_bin', 'x_bin'])['RUL'].mean().unstack()

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Mean RUL'})
plt.title("Heatmap of RUL in UMAP Space")
plt.xlabel("UMAP1 Grid")
plt.ylabel("UMAP2 Grid")
plt.tight_layout()
plt.savefig("umap_heatmap_rul.png", dpi=300)
plt.close()
