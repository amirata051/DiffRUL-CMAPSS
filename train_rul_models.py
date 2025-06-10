import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import config
from dataset import PreprocessedDataset, AugmentedDataset
from RUL_Models.CNNBased_RUL import CNNRULPredictor
from RUL_Models.RNNBased_RUL import RNNRULPredictor
from RUL_Models.Hybrid_RUL import HybridRULPredictor
from RUL_Models.TransformerBased_RUL import TransformerRULPredictor
from utils.utils import set_seed, create_dirs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define evaluation metrics (RMSE and Score as per C-MAPSS)
def calculate_metrics(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    
    score = 0
    for pred, true in zip(predictions, targets):
        diff = pred - true
        if diff < 0:
            score += np.exp(-diff / 13) - 1
        else:
            score += np.exp(diff / 10) - 1
    
    rmse = rmse.item() if isinstance(rmse, np.ndarray) and rmse.size == 1 else float(rmse)
    score = score.item() if isinstance(score, np.ndarray) and score.size == 1 else float(score)
    
    return rmse, score

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, model_name, output_dir):
    best_rmse = float('inf')
    best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")
    
    for epoch in tqdm(range(num_epochs), desc=f"Training {model_name}"):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze()
            
            # Pad batch_x for Hybrid model to ensure seq_len=50
            if isinstance(model, HybridRULPredictor):
                target_seq_len = 50
                current_seq_len = batch_x.size(1)  # (batch_size, seq_len, input_dim)
                if current_seq_len < target_seq_len:
                    padding = torch.zeros(batch_x.size(0), target_seq_len - current_seq_len, batch_x.size(2)).to(device)
                    batch_x = torch.cat([batch_x, padding], dim=1)
                batch_x = batch_x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
            
            if isinstance(model, CNNRULPredictor):
                batch_x = batch_x.unsqueeze(1)
            
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], {model_name} Loss: {avg_loss:.4f}")
        
        rmse, score = evaluate_model(model, test_loader, model_name)
        print(f"Epoch [{epoch+1}/{num_epochs}], {model_name} Test RMSE: {rmse:.4f}, Score: {score:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), best_model_path)
    
    return best_rmse, score

# Evaluation function
def evaluate_model(model, dataloader, model_name):
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze()
            
            # Pad batch_x for Hybrid model to ensure seq_len=50
            if isinstance(model, HybridRULPredictor):
                target_seq_len = 50
                current_seq_len = batch_x.size(1)
                if current_seq_len < target_seq_len:
                    padding = torch.zeros(batch_x.size(0), target_seq_len - current_seq_len, batch_x.size(2)).to(device)
                    batch_x = torch.cat([batch_x, padding], dim=1)
                batch_x = batch_x.permute(0, 2, 1)
            
            if isinstance(model, CNNRULPredictor):
                batch_x = batch_x.unsqueeze(1)
            
            outputs = model(batch_x).squeeze()
            predictions.extend(outputs.cpu().numpy().tolist())
            targets.extend(batch_y.cpu().numpy().tolist())
    
    rmse, score = calculate_metrics(predictions, targets)
    return rmse, score

def main():
    set_seed(2023)
    
    output_dir = os.path.join(config['output_dir'], 'rul_models')
    create_dirs([output_dir])
    
    preprocessed_dir = os.path.join('output', 'preprocessed')
    train_data_path = os.path.join(preprocessed_dir, 'preprocessed_data_train.npy')
    train_ruls_path = os.path.join(preprocessed_dir, 'preprocessed_ruls_train.npy')
    test_data_path = os.path.join(preprocessed_dir, 'preprocessed_data_test.npy')
    test_ruls_path = os.path.join(preprocessed_dir, 'preprocessed_ruls_test.npy')
    train_full_runs_path = os.path.join(preprocessed_dir, 'full_runs_train.npy')
    train_full_ruls_path = os.path.join(preprocessed_dir, 'full_ruls_train.npy')
    test_full_runs_path = os.path.join(preprocessed_dir, 'full_runs_test.npy')
    test_full_ruls_path = os.path.join(preprocessed_dir, 'full_ruls_test.npy')
    
    for path in [train_data_path, train_ruls_path, test_data_path, test_ruls_path,
                 train_full_runs_path, train_full_ruls_path, test_full_runs_path, test_full_ruls_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessed file not found: {path}. Please run CMAPSDataset first to generate preprocessed data.")
    
    train_dataset = PreprocessedDataset(
        train_data_path, train_ruls_path, train_full_runs_path, train_full_ruls_path,
        window_size=config['window_size'], return_pairs=False
    )
    test_dataset = PreprocessedDataset(
        test_data_path, test_ruls_path, test_full_runs_path, test_full_ruls_path,
        window_size=config['window_size'], return_pairs=False
    )
    
    hybrid_window_size = 50
    train_dataset_hybrid = PreprocessedDataset(
        train_data_path, train_ruls_path, train_full_runs_path, train_full_ruls_path,
        window_size=hybrid_window_size, return_pairs=False
    )
    test_dataset_hybrid = PreprocessedDataset(
        test_data_path, test_ruls_path, test_full_runs_path, test_full_ruls_path,
        window_size=hybrid_window_size, return_pairs=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    train_loader_hybrid = DataLoader(train_dataset_hybrid, batch_size=config['batch_size'], shuffle=True)
    test_loader_hybrid = DataLoader(test_dataset_hybrid, batch_size=config['batch_size'], shuffle=False)
    
    models = [
        ('CNN-based', CNNRULPredictor(), train_loader, test_loader),
        ('RNN-based', RNNRULPredictor(input_size=14, hidden_size=64), train_loader, test_loader),
        ('Hybrid', HybridRULPredictor(in_channels=14, seq_len=hybrid_window_size), train_loader_hybrid, test_loader_hybrid),
        ('Transformer-based', TransformerRULPredictor(input_dim=14, seq_len=config['window_size']), train_loader, test_loader)
    ]
    
    results = []
    num_epochs = config.get('max_epochs', 50)
    
    for model_name, model, train_loader, test_loader in models:
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        
        print(f"\nTraining {model_name}...")
        best_rmse, score = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, model_name, output_dir)
        
        results.append({
            'Model': model_name,
            'Dataset': 'C-MAPSS FD001',
            'RMSE': best_rmse,
            'Score': score
        })
    
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'rul_results.csv')
    results_df.to_csv(results_path, index=False)
    print("\nResults saved to:", results_path)
    print("Results:")
    print(results_df)


def main():
    set_seed(2023)
    
    output_dir = os.path.join(config['output_dir'], 'rul_models_augmented')
    create_dirs([output_dir])

    augmented_path = os.path.join('output', 'augmented_data.pkl')

    if not os.path.exists(augmented_path):
        raise FileNotFoundError(f"Augmented data not found at {augmented_path}")

    # Load augmented data
    train_dataset = AugmentedDataset(augmented_path, window_size=config['window_size'])
    test_dataset = AugmentedDataset(augmented_path, window_size=config['window_size'])  # Same for test

    hybrid_window_size = 50
    train_dataset_hybrid = AugmentedDataset(augmented_path, window_size=hybrid_window_size)
    test_dataset_hybrid = AugmentedDataset(augmented_path, window_size=hybrid_window_size)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    train_loader_hybrid = DataLoader(train_dataset_hybrid, batch_size=config['batch_size'], shuffle=True)
    test_loader_hybrid = DataLoader(test_dataset_hybrid, batch_size=config['batch_size'], shuffle=False)

    models = [
        ('CNN-based', CNNRULPredictor(), train_loader, test_loader),
        ('RNN-based', RNNRULPredictor(input_size=14, hidden_size=64), train_loader, test_loader),
        ('Hybrid', HybridRULPredictor(in_channels=14, seq_len=hybrid_window_size), train_loader_hybrid, test_loader_hybrid),
        ('Transformer-based', TransformerRULPredictor(input_dim=14, seq_len=config['window_size']), train_loader, test_loader)
    ]

    results = []
    num_epochs = config.get('max_epochs', 50)

    for model_name, model, train_loader, test_loader in models:
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        
        print(f"\nTraining {model_name} on AUGMENTED DATA...")
        best_rmse, score = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, model_name, output_dir)
        
        results.append({
            'Model': model_name,
            'Dataset': 'Augmented C-MAPSS',
            'RMSE': best_rmse,
            'Score': score
        })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'rul_results_augmented.csv')
    results_df.to_csv(results_path, index=False)
    print("\nResults saved to:", results_path)
    print("Results:")
    print(results_df)

if __name__ == "__main__":
    main()