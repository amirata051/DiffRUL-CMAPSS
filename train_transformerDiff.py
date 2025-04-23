# train.py
import os
import torch
from torch.utils.data import DataLoader
from config import config
from dataset import CMAPSDataset, PreprocessedDataset
from DTE_main import model_train as dte_model_train
from Diffusion_transformer_main import model_train as diff_model_train, model_test as diff_model_test
from DTE_model.DTE_network import TSHAE, Encoder, Decoder
from utils.utils import create_dirs, set_seed

def main():
    # Set random seed for reproducibility
    set_seed(2023)

    # Create output directories
    create_dirs([config['output_dir'], os.path.join(config['output_dir'], 'preprocessed')])

    # Load and preprocess dataset
    train_dataset = CMAPSDataset(config['data_dir'], mode="train", window_size=config['window_size'], return_pairs=True)
    test_dataset = CMAPSDataset(config['data_dir'], mode="test", window_size=config['window_size'], return_pairs=False)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Train DTE model
    encoder = Encoder(config)
    decoder = Decoder(config)
    dte_model = TSHAE(config, encoder, decoder)
    dte_model_train(config, train_loader, test_loader)

    # Load preprocessed data for diffusion
    train_data_path = os.path.join(config['output_dir'], 'preprocessed', 'preprocessed_data_train.npy')
    train_ruls_path = os.path.join(config['output_dir'], 'preprocessed', 'preprocessed_ruls_train.npy')
    full_runs_path = os.path.join(config['output_dir'], 'preprocessed', 'full_runs_train.npy')
    full_ruls_path = os.path.join(config['output_dir'], 'preprocessed', 'full_ruls_train.npy')
    preprocessed_train_dataset = PreprocessedDataset(
        train_data_path,
        train_ruls_path,
        full_runs_path,
        full_ruls_path,
        window_size=config['window_size'],
        return_pairs=True
    )
    preprocessed_train_loader = DataLoader(preprocessed_train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Train diffusion model (uses DiffTransformer)
    diff_model_train(config, preprocessed_train_loader)

    # Test diffusion model and generate augmented data
    best_diff_model_path = os.path.join(config['output_dir'], 'best_diff_model_50.pt')  # Adjust if needed
    output_path = os.path.join(config['output_dir'], 'augmented_data.pkl')
    diff_model_test(config, preprocessed_train_loader, best_diff_model_path, output_path)

    print("Training and augmentation completed successfully!")

if __name__ == "__main__":
    main()
