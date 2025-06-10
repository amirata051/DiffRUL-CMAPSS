# plot_comparison
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from config import config
from utils.utils import load_from_pickle

def compare_data(real, sampled):
    mse = np.mean((real - sampled) ** 2)
    mean_diff = np.mean(np.abs(real - sampled))
    var_real = np.var(real)
    var_sampled = np.var(sampled)
    var_ratio = var_sampled / (var_real + 1e-10)
    print(f"MSE: {mse:.4f}, Mean Diff: {mean_diff:.4f}, Var Ratio: {var_ratio:.4f}")

def plot_comparison():
    # Load real data
    df = pd.read_csv(os.path.join(config['data_dir'], "train_FD001.txt"), delim_whitespace=True, header=None)
    df.columns = ["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    unit_1 = df[df["unit"] == 1]
    print(f"Number of cycles in unit_1: {len(unit_1)}")
    if len(unit_1) < 2:
        print("Unit_1 has too few cycles, trying Unit_2...")
        unit_1 = df[df["unit"] == 2]
        print(f"Number of cycles in unit_2: {len(unit_1)}")
    sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
               "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
               "sensor_17", "sensor_20", "sensor_21"]
    real_data = unit_1[sensors].values
    real_data_normalized = (real_data - real_data.min(axis=0)) / (real_data.max(axis=0) - real_data.min(axis=0) + 1e-10)

    # Load augmented data
    augmented_data = load_from_pickle(os.path.join(config['output_dir'], 'augmented_data.pkl'))

    # Get data for unit_1
    unit_id = "unit_1"
    if unit_id not in augmented_data:
        raise ValueError(f"Unit {unit_id} not found in augmented data! Available keys: {list(augmented_data.keys())}")
    x, unit_1_samples = augmented_data[unit_id]
    unit_1_samples = unit_1_samples.numpy() if hasattr(unit_1_samples, 'numpy') else unit_1_samples
    x = x.numpy() if hasattr(x, 'numpy') else x

    # Match lengths
    min_len = min(len(real_data_normalized), len(unit_1_samples))
    real_data_normalized = real_data_normalized[:min_len]
    unit_1_samples = unit_1_samples[:min_len]

    # Debug: Print data stats
    print(f"Real data shape: {real_data_normalized.shape}, min: {real_data_normalized.min()}, max: {real_data_normalized.max()}")
    print(f"Sampled data shape: {unit_1_samples.shape}, min: {unit_1_samples.min()}, max: {unit_1_samples.max()}")

    # Plot
    sensor_names = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32"]
    cycles = unit_1["cycle"].values[:min_len]

    # Create a figure with subplots (7x2 grid to accommodate all 14 sensors)
    fig, axes = plt.subplots(7, 2, figsize=(12, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot all 14 sensors
    for idx, sensor_idx in enumerate(range(len(sensors))):
        ax = axes[idx]
        ax.plot(cycles, real_data_normalized[:, sensor_idx], label="Real Data", color="blue")
        ax.plot(cycles, unit_1_samples[:, sensor_idx], label="Sampled Data", color="orange")
        ax.set_title(sensor_names[sensor_idx], fontsize=12, pad=10)
        ax.grid(True)
        if idx % 2 == 0:  # Left column
            ax.set_ylabel("Value", fontsize=10)
        if idx >= 12:  # Bottom row
            ax.set_xlabel("Cycle", fontsize=10)
            ax.set_xticks(np.arange(0, max(cycles) + 1, 50))  # Add cycle numbers on x-axis

    # Add a legend to the last subplot
    axes[-1].legend(loc="upper right", fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("figure_7x2.png", dpi=300)
    plt.close()

    # Print comparison metrics for each sensor
    for i, sensor in enumerate(sensors):
        compare_data(real_data_normalized[:, i], unit_1_samples[:, i])

if __name__ == "__main__":
    plot_comparison()