import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class CMAPSDataset(Dataset):
    def __init__(self, data_dir, subset_id="FD001", mode="train", window_size=30, return_pairs=False):
        self.data_dir = data_dir
        self.subset_id = subset_id  # e.g., FD001, FD002, FD003, FD004
        self.mode = mode
        self.window_size = window_size
        self.return_pairs = return_pairs
        self.data = []
        self.ruls = []
        self.ids = []
        self.full_runs = {}
        self.full_ruls = {}

        if self.mode == "train":
            data_file = os.path.join(data_dir, f"train_{subset_id}.txt")
        else:
            data_file = os.path.join(data_dir, f"test_{subset_id}.txt")
            rul_file = os.path.join(data_dir, f"RUL_{subset_id}.txt")

        df = pd.read_csv(data_file, delim_whitespace=True, header=None)
        df.columns = ["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]

        if self.mode == "test":
            rul_df = pd.read_csv(rul_file, header=None)
            rul_values = rul_df[0].values

        sensor_cols = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
                       "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]
        units = df["unit"].unique()

        for unit in units:
            unit_data = df[df["unit"] == unit][sensor_cols].values
            cycles = df[df["unit"] == unit]["cycle"].values

            unit_data_min = unit_data.min(axis=0)
            unit_data_max = unit_data.max(axis=0)
            unit_data = (unit_data - unit_data_min) / (unit_data_max - unit_data_min + 1e-10)

            if self.mode == "train":
                max_cycle = cycles.max()
                unit_ruls = max_cycle - cycles
            else:
                unit_rul = rul_values[unit - 1]
                unit_ruls = unit_rul + (cycles.max() - cycles)

            unit_ruls = np.minimum(unit_ruls, 130)

            self.full_runs[f"unit_{unit}"] = unit_data
            self.full_ruls[f"unit_{unit}"] = unit_ruls

            for i in range(0, len(unit_data) - window_size + 1, window_size):
                window = unit_data[i:i + window_size]
                if window.shape[0] != window_size:
                    continue
                self.data.append(window)
                self.ruls.append(unit_ruls[i])
                self.ids.append(f"unit_{unit}_window_{i}")

        self.data = np.array(self.data)
        self.ruls = np.array(self.ruls)

        preprocessed_dir = os.path.join('output', self.subset_id, 'preprocessed')
        os.makedirs(preprocessed_dir, exist_ok=True)
        np.save(os.path.join(preprocessed_dir, f'preprocessed_data_{self.mode}.npy'), self.data)
        np.save(os.path.join(preprocessed_dir, f'preprocessed_ruls_{self.mode}.npy'), self.ruls)
        np.save(os.path.join(preprocessed_dir, f'full_runs_{self.mode}.npy'), self.full_runs)
        np.save(os.path.join(preprocessed_dir, f'full_ruls_{self.mode}.npy'), self.full_ruls)

        self.raw_data = df
        self.sensor_cols = sensor_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor([self.ruls[idx]])
        if self.return_pairs:
            pos_idx, neg_idx = self._get_pairs(idx)
            pos_x = torch.FloatTensor(self.data[pos_idx])
            neg_x = torch.FloatTensor(self.data[neg_idx])
            return x, pos_x, neg_x, y
        return x, y

    def _get_pairs(self, idx):
        rul = self.ruls[idx]
        similar_ruls = np.where(np.abs(self.ruls - rul) <= 5)[0]
        dissimilar_ruls = np.where((np.abs(self.ruls - rul) > 5) & (np.abs(self.ruls - rul) <= 15))[0]
        pos_idx = np.random.choice(similar_ruls) if len(similar_ruls) > 0 else idx
        neg_idx = np.random.choice(dissimilar_ruls) if len(dissimilar_ruls) > 0 else idx
        return pos_idx, neg_idx

    def get_run(self, engine_id):
        idx = self.ids.index(f"unit_{engine_id}_window_0")
        return self.data[idx], self.ruls[idx]

    def get_full_run(self, unit_id):
        unit_data = self.full_runs[f"unit_{unit_id}"]
        unit_ruls = self.full_ruls[f"unit_{unit_id}"]
        return torch.FloatTensor(unit_data), torch.FloatTensor(unit_ruls)

class PreprocessedDataset(Dataset):
    def __init__(self, preprocessed_data_path, preprocessed_ruls_path, full_runs_path, full_ruls_path, window_size=30, return_pairs=False):
        self.data = np.load(preprocessed_data_path)
        self.ruls = np.load(preprocessed_ruls_path)
        self.full_runs = np.load(full_runs_path, allow_pickle=True).item()
        self.full_ruls = np.load(full_ruls_path, allow_pickle=True).item()

        self.ruls = np.minimum(self.ruls, 130)
        
        self.window_size = window_size
        self.return_pairs = return_pairs

        self.ids = []
        for unit in self.full_runs.keys():
            unit_data = self.full_runs[unit]
            for i in range(0, len(unit_data) - window_size + 1, window_size):
                window = unit_data[i:i + window_size]
                if window.shape[0] != window_size:
                    continue
                self.ids.append(f"{unit}_window_{i}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor([self.ruls[idx]])
        if self.return_pairs:
            pos_idx, neg_idx = self._get_pairs(idx)
            pos_x = torch.FloatTensor(self.data[pos_idx])
            neg_x = torch.FloatTensor(self.data[neg_idx])
            return x, pos_x, neg_x, y
        return x, y

    def _get_pairs(self, idx):
        rul = self.ruls[idx]
        similar_ruls = np.where(np.abs(self.ruls - rul) <= 5)[0]
        dissimilar_ruls = np.where((np.abs(self.ruls - rul) > 5) & (np.abs(self.ruls - rul) <= 15))[0]
        pos_idx = np.random.choice(similar_ruls) if len(similar_ruls) > 0 else idx
        neg_idx = np.random.choice(dissimilar_ruls) if len(dissimilar_ruls) > 0 else idx
        return pos_idx, neg_idx

    def get_run(self, engine_id):
        idx = self.ids.index(f"unit_{engine_id}_window_0")
        return self.data[idx], self.ruls[idx]

    def get_full_run(self, unit_id):
        unit_data = self.full_runs[f"unit_{unit_id}"]
        unit_ruls = self.full_ruls[f"unit_{unit_id}"]
        return torch.FloatTensor(unit_data), torch.FloatTensor(unit_ruls)

class AugmentedDataset(Dataset):
    def __init__(self, data_path, window_size=30):
        import pickle
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.data = torch.tensor(data_dict['data'], dtype=torch.float32)
        self.ruls = torch.tensor(data_dict['rul'], dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.ruls[idx]
        return x, y

if __name__ == "__main__":
    from config import get_subset_config
    config = get_subset_config("FD001")
    dataset = CMAPSDataset(config['data_dir'], subset_id="FD001", mode="train", window_size=config['window_size'], return_pairs=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    for batch in dataloader:
        x, pos_x, neg_x, y = batch
        print("Batch shape:", x.shape, pos_x.shape, neg_x.shape, y.shape)
        break