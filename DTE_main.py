# DTE_main.py
"""
DTE train
"""

import os
import json
import pickle
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from utils import utils
from utils.loss import TotalLoss
from DTE_model.DTE_network import Encoder, Decoder, TSHAE
from DTE_running import train_epoch, valid_epoch, get_dataset_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_train(config, train_loader, valid_loader):
    # Model initialization
    encoder = Encoder(config)
    decoder = Decoder(config)
    model = TSHAE(config, encoder, decoder)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = TotalLoss(config)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_scheduler']['step_size'], gamma=config['lr_scheduler']['gamma'])

    # Start training
    best_epoch = 0
    best_rmse = float('inf')
    history = defaultdict(list)

    for epoch in tqdm(range(config['max_epochs']), desc="Training"):
        # Train
        train_loss_dict = train_epoch(config, epoch, model, optimizer, criterion, train_loader, history)
        
        # Valid
        valid_loss_dict = valid_epoch(config, epoch, model, criterion, valid_loader, history)
        
        # Get valid score
        valid_score, valid_rmse = get_dataset_score(config, model, valid_loader, history)

        # Step the scheduler
        scheduler.step()

        # Save the epoch with the best validation loss
        if valid_rmse < best_rmse:
            best_epoch = epoch
            best_rmse = valid_rmse
            torch.save(model, os.path.join(config['output_dir'], 'best_vae_model.pt'))

        # Logging
        for key in history:
            logging.info(f"Epoch: {epoch}/{config['max_epochs']}, {key}: {history[key][-1] :.4f}")
            print(f"Epoch: {epoch}/{config['max_epochs']}, {key}: {history[key][-1] :.4f}")

        logging.info("Epoch: {}/{}, Valid Score: {:.4f}, Valid Rmse: {:.4f}, Beat Valid Rmse: {:.4f}"
                     .format(epoch, config['max_epochs'], valid_score, valid_rmse, best_rmse))
        print("Epoch: {}/{}, Valid Score: {:.4f}, Valid Rmse: {:.4f}, Beat Valid Rmse: {:.4f}"
              .format(epoch, config['max_epochs'], valid_score, valid_rmse, best_rmse))

        if (epoch + 1) % 5 == 0:
            torch.save(model, os.path.join(config['output_dir'], 'vae_model_epoch_{}.pt'.format(epoch+1)))

    torch.save(model, os.path.join(config['output_dir'], 'final_vae_model.pt'))