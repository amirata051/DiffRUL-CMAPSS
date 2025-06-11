
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss:
    def __init__(self, weight):
        self.name = "KLLoss"
        self.weight = weight
    def __call__(self, mean, log_var):
        loss = (-0.5 * (1 + log_var - mean ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        return loss


class RegLoss:
    def __init__(self, weight):
        self.name = "RegLoss"
        self.weight = weight
        self.criterion = nn.MSELoss()
    def __call__(self, y, y_hat):
        return self.criterion(y, y_hat)


class ReconLoss:
    def __init__(self, weight):
        self.name = "ReconLoss"
        self.weight = weight
    def __call__(self, x, x_hat):
        batch_size = x.shape[0]
        loss = F.mse_loss(x, x_hat, reduction='none')
        loss = loss.view(batch_size, -1).sum(axis=1)
        loss = loss.mean()
        return loss


class TripletLoss:
    def __init__(self, weight, margin, p):
        self.name = "TripletLoss"
        self.weight = weight
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p)
    def __call__(self, z, z_pos, z_neg):
        return self.criterion(z, z_pos, z_neg)


class TotalLoss:
    def __init__(self, config):
        self.losses = []
        self.losses.append(KLLoss(config['KLLoss_weight']))
        self.losses.append(RegLoss(config['RegLoss_weight']))
        self.losses.append(ReconLoss(config['ReconLoss_weight']))
        self.losses.append(TripletLoss(config['TripletLoss_weight'], config['TripletLoss_margin'], config['TripletLoss_p']))

    def __call__(self, mean=None, log_var=None, y=None, y_hat=None, x=None, x_hat=None, z=None, z_pos=None, z_neg=None):
        losses_dict = {"TotalLoss": 0}
        for loss in self.losses:
            name = loss.name
            if name == "KLLoss":
                losses_dict[name] = loss(mean, log_var) * loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            elif name == "RegLoss":
                losses_dict[name] = loss(y, y_hat) * loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            elif name == "ReconLoss":
                losses_dict[name] = loss(x, x_hat) * loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            elif name == "TripletLoss":
                losses_dict[name] = loss(z, z_pos, z_neg) * loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            else:
                raise Exception(f"No such loss: {name}")
        return losses_dict