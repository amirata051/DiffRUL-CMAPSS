import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.input_size = config['input_size']
        self.latent_dim = config['latent_dim']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.num_heads = config.get('nhead', 4)
        self.p = config['dropout_layer_encoder']
        self.window_size = config['window_size']

        self.positional_encoding = nn.Parameter(torch.randn(1, self.window_size, self.hidden_size))

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dropout=self.p,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.latent_proj = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(self.hidden_size, 2 * self.latent_dim)
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        return mean + var * epsilon

    def forward(self, x):
        x_proj = self.input_proj(x) + self.positional_encoding[:, :x.size(1)]  # Add positional encoding
        encoded = self.transformer_encoder(x_proj)
        pooled = torch.mean(encoded, dim=1)
        latent_out = self.latent_proj(pooled)
        mean, log_var = latent_out.chunk(2, dim=-1)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return z, mean, log_var


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config.get('nhead', 4)
        self.window_size = config['window_size']
        self.p = config['dropout_layer_decoder']

        self.latent_to_embed = nn.Linear(self.latent_dim, self.hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.window_size, self.hidden_size))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dropout=self.p,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, z):
        tgt = self.latent_to_embed(z).unsqueeze(1).repeat(1, self.window_size, 1) + self.positional_encoding[:, :self.window_size]
        memory = torch.zeros_like(tgt)  # Dummy memory for structure compatibility (can be modified)
        decoded = self.transformer_decoder(tgt, memory)
        return self.output_layer(decoded)


class TSHAE(nn.Module):
    def __init__(self, config, encoder, decoder):
        super(TSHAE, self).__init__()
        self.p = config['dropout_regressor']
        self.regression_dims = config['regression_dims']
        self.decode_mode = config['reconstruct']

        self.encoder = encoder
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should pass a valid decoder"
            self.decoder = decoder

        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.regression_dims),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(self.regression_dims, 1)
        )

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        z_pos = z
        z_neg = torch.randn_like(z)
        y_hat = self.regressor(z)

        if self.decode_mode:
            x_hat = self.decoder(z)
            return y_hat, z, mean, log_var, x_hat, z_pos, z_neg

        return y_hat, z, mean, log_var, z_pos, z_neg
