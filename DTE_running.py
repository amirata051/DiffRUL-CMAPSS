# DTE_running.py
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_score_fun(y, y_hat):
    score = 0
    y = y.cpu()
    y_hat = y_hat.cpu()
    for i in range(len(y_hat)):
        if y[i] <= y_hat[i]:
            score += np.exp(-(y[i] - y_hat[i]) / 10.0) - 1
        else:
            score += np.exp((y[i] - y_hat[i]) / 13.0) - 1
    return score

def train_epoch(config, epoch, model, optimizer, criterion, train_loader, history):
    model.train()
    epoch_loss = defaultdict(list)
    
    for batch_idx, data in enumerate(train_loader):
        pairs_mode = train_loader.dataset.return_pairs
        optimizer.zero_grad()

        if pairs_mode:
            x, pos_x, neg_x, true_rul = data
            x, true_rul = x.to(device), true_rul.to(device)
            pos_x, neg_x = pos_x.to(device), neg_x.to(device)

            predicted_rul, z, mean, log_var, x_hat, z_pos, z_neg = model(x)
            pos_outputs = model(pos_x)
            neg_outputs = model(neg_x)

            # Debugging: Check model outputs
            print(f"Batch {batch_idx}: pos_outputs={pos_outputs}, neg_outputs={neg_outputs}")

            # Validate pos_outputs
            if isinstance(pos_outputs, (list, tuple)) and len(pos_outputs) >= 2 and isinstance(pos_outputs[1], torch.Tensor):
                _, z_pos, *_ = pos_outputs
            else:
                print(f"Warning: pos_outputs invalid in batch {batch_idx}, setting z_pos to zero tensor")
                z_pos = torch.zeros_like(z)

            # Validate neg_outputs
            if isinstance(neg_outputs, (list, tuple)) and len(neg_outputs) >= 2 and isinstance(neg_outputs[1], torch.Tensor):
                _, z_neg, *_ = neg_outputs
            else:
                print(f"Warning: neg_outputs invalid in batch {batch_idx}, setting z_neg to zero tensor")
                z_neg = torch.zeros_like(z)

            # Final debug print
            print(f"Batch {batch_idx}: z={z.shape}, z_pos={z_pos.shape}, z_neg={z_neg.shape}")

            loss_dict = criterion(
                mean=mean,
                log_var=log_var,
                y=true_rul,
                y_hat=predicted_rul,
                x=x,
                x_hat=x_hat,
                z=z,
                z_pos=z_pos,
                z_neg=z_neg
            )

        else:
            x, true_rul = data
            x, true_rul = x.to(device), true_rul.to(device)
            predicted_rul, z, mean, log_var, x_hat = model(x)

            loss_dict = criterion(
                mean=mean,
                log_var=log_var,
                y=true_rul,
                y_hat=predicted_rul,
                x=x,
                x_hat=x_hat,
                z=z
            )

        loss = loss_dict["TotalLoss"]
        loss.backward()
        optimizer.step()

        for key in loss_dict:
            epoch_loss[key].append(loss_dict[key].item())

    for key in loss_dict:
        history["Train_" + key].append(np.mean(epoch_loss[key]))

def valid_epoch(config, epoch, model, criterion, valid_loader, history):
    model.eval()
    epoch_loss = defaultdict(list)

    for batch_idx, data in enumerate(valid_loader):
        pairs_mode = valid_loader.dataset.return_pairs

        with torch.no_grad():
            if pairs_mode:
                x, pos_x, neg_x, y = data
                x, y = x.to(device), y.to(device)
                pos_x, neg_x = pos_x.to(device), neg_x.to(device)

                y_hat, z, mean, log_var, x_hat = model(x)
                pos_outputs = model(pos_x)
                neg_outputs = model(neg_x)

                print(f"Validation Batch {batch_idx}: pos_outputs={pos_outputs}, neg_outputs={neg_outputs}")

                # Validate pos_outputs
                if isinstance(pos_outputs, (list, tuple)) and len(pos_outputs) >= 2 and isinstance(pos_outputs[1], torch.Tensor):
                    _, z_pos, *_ = pos_outputs
                else:
                    print(f"Warning: pos_outputs invalid in validation batch {batch_idx}, setting z_pos to zero tensor")
                    z_pos = torch.zeros_like(z)

                # Validate neg_outputs
                if isinstance(neg_outputs, (list, tuple)) and len(neg_outputs) >= 2 and isinstance(neg_outputs[1], torch.Tensor):
                    _, z_neg, *_ = neg_outputs
                else:
                    print(f"Warning: neg_outputs invalid in validation batch {batch_idx}, setting z_neg to zero tensor")
                    z_neg = torch.zeros_like(z)

                print(f"Validation Batch {batch_idx}: z={z.shape}, z_pos={z_pos.shape}, z_neg={z_neg.shape}")

                loss_dict = criterion(
                    mean=mean,
                    log_var=log_var,
                    y=y,
                    y_hat=y_hat,
                    x=x,
                    x_hat=x_hat,
                    z=z,
                    z_pos=z_pos,
                    z_neg=z_neg
                )

            else:
                x, y = data
                x, y = x.to(device), y.to(device)
                y_hat, z, mean, log_var, x_hat, z_pos, z_neg = model(x)

                loss_dict = criterion(
                    mean=mean,
                    log_var=log_var,
                    y=y,
                    y_hat=y_hat,
                    x=x,
                    x_hat=x_hat,
                    z=z,
                    z_pos=z_pos,
                    z_neg=z_neg
                )

            for key in loss_dict:
                epoch_loss[key].append(loss_dict[key].item())

    for key in loss_dict:
        history["Val_" + key].append(np.mean(epoch_loss[key]))

def get_dataset_score(config, model, dataloader, history):
    """
    Calculates score and RMSE on the dataset.
    """
    model.eval()
    rmse = 0
    score = 0
    pairs_mode = dataloader.dataset.return_pairs

    for batch_idx, data in enumerate(dataloader):
        with torch.no_grad():
            if pairs_mode:
                x, _, _, y = data
            else:
                x, y = data
            x, y = x.to(device), y.to(device)

            y_hat, *_ = model(x)

            loss = nn.MSELoss()(y_hat, y)

            rmse += loss.item() * len(y)
            score += cal_score_fun(y, y_hat).item()

    rmse = (rmse / len(dataloader.dataset)) ** 0.5

    history["Val_Score"].append(score)
    history["Val_RMSE"].append(rmse)

    return score, rmse
