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

        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dropout=self.p,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.fc_mean = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(self.hidden_size, self.latent_dim)
        )
        self.fc_log_var = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(self.hidden_size, self.latent_dim)
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        return mean + var * epsilon

    def forward(self, x):
        """
        :param x: [B, seq_len, fea_dim]
        :return: z, mean, log_var [B, latent_dim]
        """
        print(f"TransformerEncoder input x shape: {x.shape}")
        x_proj = self.input_proj(x)  # [B, seq_len, hidden_size]
        encoded = self.transformer_encoder(x_proj)  # [B, seq_len, hidden_size]

        pooled = torch.mean(encoded, dim=1)  # [B, hidden_size]
        mean = self.fc_mean(pooled)
        log_var = self.fc_log_var(pooled)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        print(f"TransformerEncoder output z shape: {z.shape}")
        return z, mean, log_var
