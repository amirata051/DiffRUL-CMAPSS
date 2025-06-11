# train_transformerDiff.py
import os
import glob
import time
import torch
import numpy as np
import pandas as pd
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
    
    # Define C-MAPSS subsets
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    timing_results = []
    
    for subset in subsets:
        # Update config for current subset
        config["subset"] = subset
        config["output_dir"] = f"./output/{subset}"
        config["vae_model_path"] = f"./output/{subset}/best_vae_model.pt"
        config["diff_model_path"] = f"./output/{subset}/best_diff_model.pt"
        
        # Create output directories
        create_dirs([config["output_dir"], os.path.join(config["output_dir"], "preprocessed"), os.path.join(config["output_dir"], "diff_models")])
        
        # Load and preprocess dataset
        train_dataset = CMAPSDataset(config["data_dir"], mode="train", window_size=config["window_size"], return_pairs=True, subset=subset)
        test_dataset = CMAPSDataset(config["data_dir"], mode="test", window_size=config["window_size"], return_pairs=False, subset=subset)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Train DTE model
        encoder = Encoder(config)
        decoder = Decoder(config)
        dte_model = TSHAE(config, encoder, decoder)
        dte_model_train(config, train_loader, test_loader)
        
        # Preprocess train data using DTE
        preprocessed_train_data = []
        preprocessed_train_labels = []
        with torch.no_grad():
            for x, pos_x, neg_x, y in train_loader:
                x = x.to(config["device"])
                mu, _ = dte_model.encode(x)
                preprocessed_train_data.append(mu.cpu().numpy())
                preprocessed_train_labels.append(y.cpu().numpy())
        
        preprocessed_train_data = np.concatenate(preprocessed_train_data, axis=0)
        preprocessed_train_labels = np.concatenate(preprocessed_train_labels, axis=0)
        
        # Save preprocessed data
        np.save(os.path.join(config["output_dir"], "preprocessed", "preprocessed_data_train.npy"), preprocessed_train_data)
        np.save(os.path.join(config["output_dir"], "preprocessed", "preprocessed_labels_train.npy"), preprocessed_train_labels)
        
        # Load preprocessed dataset for diffusion training
        train_data_path = os.path.join(config["output_dir"], "preprocessed", "preprocessed_data_train.npy")
        train_ruls_path = os.path.join(config["output_dir"], "preprocessed", "preprocessed_labels_train.npy")
        full_runs_path = os.path.join(config["output_dir"], "preprocessed", "full_runs_train.npy")
        full_ruls_path = os.path.join(config["output_dir"], "preprocessed", "full_ruls_train.npy")
        preprocessed_train_dataset = PreprocessedDataset(
            train_data_path,
            train_ruls_path,
            full_runs_path,
            full_ruls_path,
            window_size=config["window_size"],
            return_pairs=True
        )
        preprocessed_train_loader = DataLoader(preprocessed_train_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Train diffusion model and measure training time
        train_start_time = time.time()
        diff_model_train(config, preprocessed_train_loader)
        train_time = time.time() - train_start_time
        
        # Store timing results
        timing_results.append({
            "Subset": subset,
            "Model": "TransformerDiffWave",
            "Train Time (s)": train_time
        })
        
        # Find the latest diffusion model
        model_files = glob.glob(os.path.join(config["output_dir"], "diff_models", "best_diff_model_*.pt"))
        if model_files:
            best_diff_model_path = max(model_files, key=os.path.getctime)
        else:
            raise FileNotFoundError(f"No diffusion model found for {subset}!")
        
        # Test diffusion model and generate augmented data
        output_path = os.path.join(config["output_dir"], "augmented_data.pkl")
        diff_model_test(config, preprocessed_train_loader, best_diff_model_path, output_path)
        
        print(f"Training and augmentation completed for {subset}!")

    # Save timing results to CSV
    pd.DataFrame(timing_results).to_csv("./output/timing_transformer.csv", index=False)
    print("All subsets processed successfully!")

if __name__ == "__main__":
    main()