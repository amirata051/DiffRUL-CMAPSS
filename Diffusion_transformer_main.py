import os
import torch
import logging
import argparse
import copy
import numpy as np
from tqdm import tqdm

from utils import utils
from Diffusion_model.Diff_transformer_network import TransformerDiffWave
from Diffusion_model.ddpm import Diffusion as DDPMDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is set to: {device}")


def model_train(config, train_loader):
    model_vae = torch.load(config['vae_model_path'])
    model_vae.to(device)
    model_vae.eval()
    for param in model_vae.parameters():
        param.requires_grad = False

    # model = HybridDiffWave(config)
    """model = HybridDiffWave(
    in_channels=1,      # input tensor has 1 channel
    cond_channels=1,    # conditioning input has 1 channel (e.g., latent z from VAE)
    residual_channels=64,
    residual_layers=6,
    transformer_layers=2,
    nhead=4
)"""
    model_diff = TransformerDiffWave(config)


    model_diff.to(device)
    diffusion = DDPMDiffusion(config['noise_steps'], config['beta_start'], config['beta_end'], config['schedule_name'], device)

    from Diffusion_model.Diff_network import EMA
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model_diff).eval().requires_grad_(False)

    optimizer = torch.optim.Adam(model_diff.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()

    model_diff.train()
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

            predicted_noise = model_diff(noisy_x, time, conditioner)
            loss = criterion(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.step_ema(ema_model=ema_model, model=model_diff)
            batch_loss.append(loss.item())

        epoch_avg_loss = np.mean(batch_loss)
        epoch_loss.append(epoch_avg_loss)
        logging.info(f"Epoch:{epoch + 1}/{config['max_epochs']}, Train Loss:{epoch_avg_loss:.4f}")
        print(f"Epoch:{epoch + 1}/{config['max_epochs']}, Train Loss:{epoch_avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config['output_dir'], f'best_diff_model_{epoch + 1}.pt')
            ema_save_path = os.path.join(config['output_dir'], f'ema_best_diff_model_{epoch + 1}.pt')
            torch.save({'state_dict': model_diff.state_dict()}, save_path)
            torch.save({'state_dict': ema_model.state_dict()}, ema_save_path)

    return epoch_loss


def model_test(config, train_loader, best_diff_model_path, output_path):
    model_vae = torch.load(config['vae_model_path'])
    model_vae.to(device)
    model_vae.eval()

    # model_diff = HybridDiffWave(config)
    """model = HybridDiffWave(
    in_channels=1,      # input tensor has 1 channel
    cond_channels=1,    # conditioning input has 1 channel (e.g., latent z from VAE)
    residual_channels=64,
    residual_layers=6,
    transformer_layers=2,
    nhead=4
)"""
    model_diff = TransformerDiffWave(config)
    checkpoint = utils.load_model(best_diff_model_path)
    model_diff.load_state_dict(checkpoint['state_dict'])
    model_diff.to(device)
    model_diff.eval()

    diffusion = DDPMDiffusion(config['noise_steps'], config['beta_start'], config['beta_end'], config['schedule_name'], device)

    sample_result = {}
    unit_ids = sorted(set(id_.split('_window')[0] for id_ in train_loader.dataset.ids))
    print(f"Unique unit IDs: {unit_ids}")

    for unit_id in tqdm(unit_ids, desc='Sample'):
        with torch.no_grad():
            x, y = train_loader.dataset.get_full_run(unit_id.split('_')[1])
            print(f"Full data shape for {unit_id}: {x.shape}")

            total_cycles = x.shape[0]
            chunk_size = config['window_size']
            num_windows = (total_cycles + chunk_size - 1) // chunk_size
            print(f"Total cycles: {total_cycles}, Number of windows: {num_windows}")

            full_sample_x_chunks = []
            for i in range(0, total_cycles, chunk_size):
                chunk_x = x[i:i+chunk_size].to(device)
                actual_chunk_size = chunk_x.shape[0]
                if actual_chunk_size < chunk_size:
                    padding = torch.zeros(chunk_size - actual_chunk_size, chunk_x.shape[1]).to(device)
                    chunk_x = torch.cat([chunk_x, padding], dim=0)

                z = model_vae(chunk_x.unsqueeze(0))[1]
                conditioner = z.to(device)

                sample_x = diffusion.sample(config, model_diff, conditioner)
                sample_x = sample_x.squeeze(0)[:actual_chunk_size]
                full_sample_x_chunks.append(sample_x.cpu())

            full_sample_x = torch.cat(full_sample_x_chunks, dim=0)
            print(f"Sampled data shape for {unit_id}: {full_sample_x.shape}")

            sample_result[unit_id] = (x.cpu(), full_sample_x.cpu())

    utils.save_to_pickle(output_path, sample_result)
    print("Augmentation completed successfully!")
