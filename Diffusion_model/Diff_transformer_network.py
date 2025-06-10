import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
# from Diff_network import DiffWave
from Diffusion_model.Diff_network import Conv2d, DiffusionEmbedding, ConditionerEmbedding, ResidualBlock


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


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerResidualBlock(nn.Module):
    def __init__(self, seq_len, residual_channels, dilation):
        super().__init__()
        self.diffusion_projection = nn.Linear(512, residual_channels)
        self.conditioner_projection = ConditionerEmbedding(seq_len, 2, residual_channels)
        self.transformer = TransformerBlock(dim=residual_channels)
        self.output_projection = nn.Linear(residual_channels, residual_channels * 2)

    def forward(self, x, diffusion_step, conditioner):
        # x: [B, C, F, L] -> reshape to [B*F, L, C]
        B, C, F, L = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * F, L, C)

        cond = self.conditioner_projection(conditioner)  # [B, L, C]
        cond = cond.unsqueeze(1).repeat(1, F, 1, 1).reshape(B * F, L, C)

        diff_proj = self.diffusion_projection(diffusion_step).unsqueeze(1)  # [B, 1, C]
        diff_proj = diff_proj.repeat(F, L, 1).reshape(B * F, L, C)

        x = x + cond + diff_proj
        x = self.transformer(x)

        y = self.output_projection(x)  # [B*F, L, 2C]
        y = y.view(B, F, L, 2 * C).permute(0, 3, 1, 2)  # [B, 2C, F, L]
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.view(B, F, L, C).permute(0, 3, 1, 2)

        return (x + residual) / sqrt(2.0), skip


class TransformerDiffWave(DiffWave):
    def __init__(self, config):
        super().__init__(config)
        self.residual_layers = nn.ModuleList([
            TransformerResidualBlock(
                config['window_size'],
                config['residual_channels'],
                2 ** (i % config['dilation_cycle_length'])
            )
            for i in range(config['residual_layers'])
        ])

        self.attn_post_conditioning = TransformerBlock(config['residual_channels'])

    def forward(self, x, diffusion_step, conditioner):
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.input_projection(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, conditioner)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))

        # attention after conditioning
        B, C, F, L = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * F, L, C)
        x = self.attn_post_conditioning(x)
        x = x.view(B, F, L, C).permute(0, 3, 1, 2)

        x = self.skip_projection(x)
        x = self.output_projection(x)
        x = x.squeeze(1).permute(0, 2, 1)

        return x




print("this is test!")

"""import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class DiffTransformer(nn.Module):
    def __init__(self, seq_len, feature_dim, cond_dim, diffusion_dim=512, 
                 nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.diffusion_dim = diffusion_dim

        self.input_proj = nn.Linear(feature_dim, diffusion_dim)
        self.diffusion_embedding = DiffusionEmbedding(1000)  # یا همون max_noise_steps

        # برای اضافه کردن diffusion timestep و conditioner z
        self.time_proj = nn.Linear(diffusion_dim, diffusion_dim)
        self.cond_proj = nn.Linear(cond_dim, diffusion_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=diffusion_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(diffusion_dim, feature_dim)

    def forward(self, x, t, cond):
        B, L, _ = x.shape
        x = self.input_proj(x)  # [B, seq_len, diffusion_dim]

        t_emb = self.diffusion_embedding(t)           # [B, 512]
        t_emb = self.time_proj(t_emb).unsqueeze(1)    # [B, 1, diffusion_dim]

        cond_emb = self.cond_proj(cond).unsqueeze(1)  # [B, 1, diffusion_dim]

        x = x + t_emb + cond_emb  # broadcasting

        x = self.transformer_encoder(x)  # [B, seq_len, diffusion_dim]

        x = self.output_proj(x)  # [B, seq_len, feature_dim]

        return x


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


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)"""

"""import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps=1000, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer('embedding', self._build_embedding(max_steps, embedding_dim))
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def _build_embedding(self, max_steps, dim):
        half_dim = dim // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        pos = torch.arange(max_steps, dtype=torch.float32).unsqueeze(1)
        emb = pos * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        return self.proj(x)

class ConditionalEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, embedding_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(in_channels, 2 * in_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(in_channels, in_channels)
        self.condition_projection = nn.Conv1d(cond_channels, 2 * in_channels, 1)
        self.output_projection = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x, diffusion_emb, cond):
        diffusion_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = self.dilated_conv(x) + self.condition_projection(cond) + diffusion_proj
        gate, filter = y.chunk(2, dim=1)
        out = torch.tanh(filter) * torch.sigmoid(gate)
        out = self.output_projection(out)
        return (x + out) / torch.sqrt(torch.tensor(2.0))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )

    def forward(self, x):
        return self.transformer_layer(x)

class HybridDiffWave(nn.Module):
    def __init__(self, in_channels=1, cond_channels=1, residual_channels=64,
                 residual_layers=6, transformer_layers=2, nhead=4):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(embedding_dim=residual_channels)
        self.cond_embedding = ConditionalEmbedding(cond_channels, residual_channels)

        dilations = [2 ** i for i in range(residual_layers)]
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, residual_channels, d)
            for d in dilations
        ])

        self.transformer_input_proj = nn.Conv1d(residual_channels, residual_channels, 1)
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(residual_channels, nhead)
            for _ in range(transformer_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(residual_channels, residual_channels, 1),
            nn.SiLU(),
            nn.Conv1d(residual_channels, in_channels, 1)
        )

    def forward(self, x, diffusion_step, cond):
        x = self.input_proj(x)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        cond = self.cond_embedding(cond)

        for layer in self.residual_layers:
            x = layer(x, diffusion_emb, cond)

        # transformer expects (B, T, C) -> input is (B, C, T)
        x = self.transformer_input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.transformer_blocks(x)
        x = x.permute(0, 2, 1)

        x = self.output_proj(x)
        return x"""
