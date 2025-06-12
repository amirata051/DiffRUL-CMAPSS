import os
import torch
import time
import pandas as pd
from torch.utils.data import DataLoader
from config import get_subset_config
from dataset import CMAPSDataset, PreprocessedDataset
from DTE_main import model_train as dte_model_train
from Diffusion_transformer_main import model_train as diff_model_train, model_test as diff_model_test
from DTE_model.DTE_network import TSHAE, Encoder, Decoder
from utils.utils import create_dirs, set_seed

def main():
    set_seed(2023)
    
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    training_times = []
    
    for subset_id in subsets:
        print(f"\nProcessing subset: {subset_id}")
        
        config = get_subset_config(subset_id)
        create_dirs([config['output_dir'], os.path.join(config['output_dir'], 'preprocessed')])
        
        train_dataset = CMAPSDataset(
            config['data_dir'], 
            subset_id=subset_id, 
            mode="train", 
            window_size=config['window_size'], 
            return_pairs=True
        )
        test_dataset = CMAPSDataset(
            config['data_dir'], 
            subset_id=subset_id, 
            mode="test", 
            window_size=config['window_size'], 
            return_pairs=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        encoder = Encoder(config)
        decoder = Decoder(config)
        dte_model = TSHAE(config, encoder, decoder)
        dte_model_train(config, train_loader, test_loader)
        
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
        
        start_time = time.time()
        diff_model_train(config, preprocessed_train_loader)
        end_time = time.time()
        training_time = end_time - start_time
        
        training_times.append({"Subset": subset_id, "Training_Time_Seconds": training_time})
        print(f"Training time for {subset_id}: {training_time:.2f} seconds")
        
        best_diff_model_path = os.path.join(config['output_dir'], 'best_diff_model_50.pt')
        output_path = os.path.join(config['output_dir'], 'augmented_data.pkl')
        diff_model_test(config, preprocessed_train_loader, best_diff_model_path, output_path)
        
        print(f"Completed processing for {subset_id}")
    
    times_df = pd.DataFrame(training_times)
    times_df.to_csv("./output/training_times_transformer.csv", index=False)
    print("Training times saved to ./output/training_times_transformer.csv")

if __name__ == "__main__":
    main()