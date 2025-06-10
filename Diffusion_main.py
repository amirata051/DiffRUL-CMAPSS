# Diffusion_main.py
import os
import json
import pickle
import torch
import logging
import argparse
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import utils
from Diffusion_model.Diff_network import DiffWave, EMA
from Diffusion_model.ddpm import Diffusion as DDPMDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_train(config, train_loader):
    # load DTE Model
    model_vae = torch.load(config['vae_model_path'])
    model_vae.to(device)
    model_vae.eval()
    for param in model_vae.parameters():
        param.requires_grad = False

    # Diff model initialization
    model = DiffWave(config)
    model.to(device)
    diffusion = DDPMDiffusion(config['noise_steps'], config['beta_start'], config['beta_end'], config['schedule_name'], device)

    # Exponential Moving Average (EMA)
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)  # EMA model

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()

    # start training
    model.train()
    best_epoch = 0
    best_loss = float('inf')

    epoch_loss = []
    for epoch in tqdm(range(config['max_epochs']), desc='Training'):
        batch_loss = []
        for batch_idx, data in enumerate(train_loader):
            pairs_mode = train_loader.dataset.return_pairs

            if pairs_mode:
                x, pos_x, neg_x, true_rul = data
            else:
                x, true_rul = data

            x = x.to(device)
            with torch.no_grad():
                predicted_rul, z = model_vae(x)[:2]
                conditioner = z.to(device)

            time = diffusion.sample_time_steps(x.shape[0]).to(device)
            noisy_x, noise = diffusion.noise_images(x=x, time=time)

            predicted_noise = model(noisy_x, time, conditioner)
            loss = criterion(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA
            ema.step_ema(ema_model=ema_model, model=model)

            batch_loss.append(loss.item())

        epoch_loss.append(np.mean(batch_loss))
        logging.info("Epoch:{}/{}, Train Loss:{:.4f}".format(epoch, config['max_epochs'], np.mean(batch_loss)))
        print("Epoch:{}/{}, Train Loss:{:.4f}".format(epoch, config['max_epochs'], np.mean(batch_loss)))

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config['output_dir'], 'best_diff_model_{}.pt'.format(epoch + 1))
            ema_save_path = os.path.join(config['output_dir'], 'ema_best_diff_model_{}.pt'.format(epoch + 1))

            torch.save({'state_dict': model.state_dict()}, save_path)
            torch.save({'state_dict': ema_model.state_dict()}, ema_save_path)

    return epoch_loss

def model_test(config, train_loader, best_diff_model_path, output_path):
    # Load DTE Model
    model_vae = torch.load(config['vae_model_path'])
    model_vae.to(device)
    model_vae.eval()

    # Load diffusion model
    model_diff = DiffWave(config)
    checkpoint = utils.load_model(best_diff_model_path)
    model_diff.load_state_dict(checkpoint['state_dict'])
    model_diff.to(device)
    model_diff.eval()

    diffusion = DDPMDiffusion(config['noise_steps'], config['beta_start'], config['beta_end'], config['schedule_name'], device)

    sample_result = {}
    # Get unique unit IDs (e.g., unit_1, unit_2, ...)
    unit_ids = sorted(set(id_.split('_window')[0] for id_ in train_loader.dataset.ids))
    print(f"Unique unit IDs: {unit_ids}")

    for unit_id in tqdm(unit_ids, desc='Sample'):
        with torch.no_grad():
            # Get full data for this unit
            x, y = train_loader.dataset.get_full_run(unit_id.split('_')[1])
            print(f"Full data shape for {unit_id}: {x.shape}")

            # Calculate the number of windows needed to match the exact number of cycles
            total_cycles = x.shape[0]  # e.g., 192 for unit_1
            chunk_size = config['window_size']  # 30
            num_windows = (total_cycles + chunk_size - 1) // chunk_size  # Ceiling division to cover all cycles
            print(f"Total cycles: {total_cycles}, Number of windows: {num_windows}")

            full_sample_x_chunks = []
            for i in range(0, total_cycles, chunk_size):
                # Extract the chunk
                chunk_x = x[i:i+chunk_size].to(device)
                # Calculate the actual size of this chunk
                actual_chunk_size = chunk_x.shape[0]  # Might be less than chunk_size for the last chunk
                if actual_chunk_size < chunk_size:
                    # Pad the last chunk if necessary
                    padding = torch.zeros(chunk_size - actual_chunk_size, chunk_x.shape[1]).to(device)
                    chunk_x = torch.cat([chunk_x, padding], dim=0)

                # Get conditioner (latent representation) for this chunk
                predicted_rul, z = model_vae(chunk_x.unsqueeze(0))[:2]  # Add batch dimension
                conditioner = z.to(device)

                # Generate samples for this chunk
                sample_x = diffusion.sample(config, model_diff, conditioner)
                # Remove batch dimension and take only the actual number of cycles
                sample_x = sample_x.squeeze(0)[:actual_chunk_size]  # [actual_chunk_size, 14]
                full_sample_x_chunks.append(sample_x.cpu())

            # Concatenate all chunks
            full_sample_x = torch.cat(full_sample_x_chunks, dim=0)
            print(f"Sampled data shape for {unit_id}: {full_sample_x.shape}")

            # Store the result
            sample_result[unit_id] = (x.cpu(), full_sample_x.cpu())

    utils.save_to_pickle(output_path, sample_result)
    print("Augmentation completed successfully!")