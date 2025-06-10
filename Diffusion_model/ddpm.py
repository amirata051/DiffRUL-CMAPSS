# Diffusion_model/ddpm.py
import torch
from tqdm import tqdm
from .base import BaseDiffusion

class Diffusion(BaseDiffusion):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, schedule_name="linear", device="cpu"):
        super().__init__(noise_steps, beta_start, beta_end, schedule_name, device)

    def sample(self, config, model, conditioner):
        with torch.no_grad():
            # Debug: Print input shapes
            print(f"Sampling with conditioner shape: {conditioner.shape}")
            x = torch.randn(conditioner.shape[0], config['window_size'], config['input_size']).to(self.device)  # [B, L, feature_dim]
            print(f"Initial noise shape: {x.shape}")

            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, conditioner)

                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]

                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                else:
                    noise = torch.zeros_like(x).to(self.device)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            print(f"Sampled data shape: {x.shape}")
        return x