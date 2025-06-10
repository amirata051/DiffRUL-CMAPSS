import pickle
import pandas as pd
import numpy as np
from config import config
import os

# Load the augmented data
augmented_data = pickle.load(open(config['output_dir'] + '/augmented_data.pkl', 'rb'))

# Load real data for unit_1
df = pd.read_csv(os.path.join(config['data_dir'], "train_FD001.txt"), delim_whitespace=True, header=None)
df.columns = ["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
unit_1 = df[df["unit"] == 1]
sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
           "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
           "sensor_17", "sensor_20", "sensor_21"]
real_data_unit_1 = unit_1[sensors].values  # All cycles for unit_1
# Normalize real_data_unit_1
real_data_unit_1_normalized = (real_data_unit_1 - real_data_unit_1.min(axis=0)) / (real_data_unit_1.max(axis=0) - real_data_unit_1.min(axis=0) + 1e-10)

# Print all keys and shapes
print("Total number of keys:", len(augmented_data))
unit_1_key = 'unit_1'
if unit_1_key in augmented_data:
    x, sample_x = augmented_data[unit_1_key]
    print(f"Key: {unit_1_key}, Real data shape: {x.shape}, Sampled data shape: {sample_x.shape}")
    # Compare the first 30 cycles for verification
    x_normalized = (x.numpy() if hasattr(x, 'numpy') else x)
    x_normalized = (x_normalized - x_normalized.min(axis=0)) / (x_normalized.max(axis=0) - x_normalized.min(axis=0) + 1e-10)
    if np.allclose(x_normalized[:30], real_data_unit_1_normalized[:30], atol=1e-1):
        print(f"Key: {unit_1_key} matches unit_1!")
else:
    print(f"Key {unit_1_key} not found in augmented_data. Available keys: {list(augmented_data.keys())}")