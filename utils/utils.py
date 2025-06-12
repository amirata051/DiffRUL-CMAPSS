import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os

def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_to_pickle(data_path, data):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(data_path):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    return data

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def create_dirs(dirs):
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

def load_model(ckpt):
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)