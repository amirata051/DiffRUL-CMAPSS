
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d(*args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

class ConditionerEmbedding(nn.Module):
    def __init__(self, seq_len, input_dim, emb_dim):
        super(ConditionerEmbedding, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim

        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.Embedding = nn.Sequential(*layers)

        # self.Embedding = Conv1d(1, emb_dim, 1)

    def forward(self, x):
        """
        :param x: [B, 2]
        :return: [B, seq_len, emb_dim]
        """

        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, seq_len, 2]
        # x = x.permute(0, 2, 1)  # [B, 2, seq_len]

        return self.Embedding(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)
    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, seq_len, residual_channels, dilation):
        '''
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        '''
        super().__init__()

        self.dilated_conv = Conv2d(residual_channels, 2 * residual_channels, kernel_size=[3, 3], padding=dilation, dilation=dilation)

        self.diffusion_projection = nn.Linear(512, residual_channels)

        # self.conditioner_projection = Conv1d(1, 2 * residual_channels, 1)
        self.conditioner_projection = ConditionerEmbedding(seq_len, 2, 2 * residual_channels)

        self.output_projection = Conv2d(residual_channels, 2 * residual_channels, kernel_size=[3, 3], padding='same')

    def forward(self, x, diffusion_step, conditioner):
        """
        :param x: [B, 64, fea_dim, seq_len]
        :param diffusion_step: [B, 512]
        :param conditioner: [B, 2]
        :return:
        """

        diffusion_step = self.diffusion_projection(diffusion_step)[:, :, None, None]  # [B, 64, 1, 1]
        y = x + diffusion_step                                                        # [B, 64, fea_dim, seq_len]

        # conditioner = self.conditioner_projection(conditioner).unsqueeze(2)  # [B, 2 * 64, 1, seq_len]

        conditioner = self.conditioner_projection(conditioner)   # [B, seq_len, 2 * 64]
        conditioner = conditioner.permute(0, 2, 1).unsqueeze(2)  # [B, 2 * 64, 1, seq_len]

        y = self.dilated_conv(y) + conditioner        # [B, 2 * 64, fea_dim, seq_len]

        gate, filter = torch.chunk(y, 2, dim=1)       # [B, 64, fea_dim, seq_len], [B, 64, fea_dim, seq_len]
        y = torch.sigmoid(gate) * torch.tanh(filter)  # [B, 64, fea_dim, seq_len]

        y = self.output_projection(y)                 # [B, 2 * 64, fea_dim, seq_len]
        residual, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_projection = nn.Sequential(
            Conv2d(1, config['residual_channels'], kernel_size=1, stride=1),
            nn.BatchNorm2d(config['residual_channels']),
            nn.GELU()
        )

        self.diffusion_embedding = DiffusionEmbedding(config['noise_steps'])

        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                config['window_size'],
                config['residual_channels'],
                2**(i % config['dilation_cycle_length'])
            )
            for i in range(config['residual_layers'])
        ])

        self.skip_projection = nn.Sequential(
            Conv2d(config['residual_channels'], config['residual_channels'], kernel_size=[3, 3], padding='same'),
            nn.BatchNorm2d(config['residual_channels']),
            nn.GELU()
        )

        self.output_projection = Conv2d(config['residual_channels'], 1, kernel_size=1, stride=1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, diffusion_step, conditioner):
        """
        :param x: [B, seq_len, fea_dim]
        :param diffusion_step: [B]
        :param conditioner: [B, 2]
        :return:
        """

        x = x.permute(0, 2, 1).unsqueeze(1)  # [B, 1, fea_dim, seq_len]
        x = self.input_projection(x)         # [B, 64, fea_dim, seq_len]
        # x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)  # [B, 512]

        # conditioner = conditioner.permute(0, 2, 1)  # [B, 1, seq_len]

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, conditioner)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))  # [B, 64, fea_dim, seq_len]

        x = self.skip_projection(x)    # [B, 64, fea_dim, seq_len]
        # x = F.relu(x)

        x = self.output_projection(x)      # [B, 1, fea_dim, seq_len]
        x = x.squeeze(1).permute(0, 2, 1)  # [B, seq_len, fea_dim]

        return x


class EMA:
    """
    Exponential Moving Average
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, up_weight = ema_params, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old_weight, new_weight):
        if old_weight is None:
            return new_weight
        return old_weight * self.beta + (1 - self.beta) * new_weight

    def step_ema(self, ema_model, model, step_start_ema=5000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())