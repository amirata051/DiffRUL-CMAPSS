import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size'] 
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.num_directions = 2 if self.bidirectional else 1
        self.p_lstm = config['dropout_lstm_encoder']
        self.p = config['dropout_layer_encoder']

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.fc_mean = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * self.hidden_size,
                out_features=self.latent_dim)
        )

        self.fc_log_var = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * self.hidden_size,
                out_features=self.latent_dim)
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        """
        :param x: [B, seq_len, fea_dim]
        :return:
            z: [B, 2]
            mean: [B, 2]
            log_var: [B, 2]
        """
        print(f"Encoder input x shape: {x.shape}, has_nan: {torch.isnan(x).any()}")
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            print(f"Encoder input x reshaped: {x.shape}")

        batch_size = x.shape[0]
        _, (h_n, _) = self.lstm(x)
        print(f"Encoder LSTM output h_n shape: {h_n.shape}")
        print('---------------------------------------------------------------')
        print(f'lstm : {self.lstm(x)}')
        print('---------------------------------------------------------------')
        print('---------------------------------------------------------------')
        print(f'h_n : {h_n}')
        print('---------------------------------------------------------------')

        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        if self.bidirectional:
            h = torch.cat((h_n[-1, -2, :, :], h_n[-1, -1, :, :]), dim=1)
        else:
            h = h_n[-1, -1, :, :]
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        print(f"Encoder output z shape: {z.shape}, mean shape: {mean.shape}, log_var shape: {log_var.shape}")
        return z, mean, log_var


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.window_size = config['window_size']
        self.p_lstm = config['dropout_lstm_decoder']
        self.p_dropout_layer = config['dropout_layer_decoder']
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm_to_hidden = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.dropout_layer = nn.Dropout(self.p_dropout_layer)

        self.lstm_to_output = nn.LSTM(
            input_size=self.num_directions * self.hidden_size,
            hidden_size=self.input_size,
            batch_first=True
        )

    def forward(self, z):
        """
        :param z: [B, 2]
        :return: [B, seq_len, fea_dim]
        """
        print(f"Decoder input z shape: {z.shape}")
        latent_z = z.unsqueeze(1).repeat(1, self.window_size, 1)  # [B, seq_len, 2]
        out, _ = self.lstm_to_hidden(latent_z)
        out = self.dropout_layer(out)
        out, _ = self.lstm_to_output(out)
        print(f"Decoder output x_hat shape: {out.shape}")
        return out

class TSHAE(nn.Module):
    def __init__(self, config, encoder, decoder):
        super(TSHAE, self).__init__()

        self.p = config['dropout_regressor']
        self.regression_dims = config['regression_dims']

        self.decode_mode = config['reconstruct']
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder"
            self.decoder = decoder

        self.encoder = encoder

        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.regression_dims),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(self.regression_dims, 1)
        )

    def forward(self, x):
        """
        :param x: [B, seq_len, fea_dim]
        :return:
            y_hat: [B, 1]
            z: [B, 2]
            mean: [B, 2]
            log_var: [B, 2]
            x_hat: [B, seq_len, fea_dim]
            z_pos: [B, 2] (add this)
            z_neg: [B, 2] (add this)
        """
        # Pass through encoder to get latent space
        z, mean, log_var = self.encoder(x)

        # Example logic for positive and negative samples (you may need to modify this based on your needs)
        z_pos = z  # In your case, you might want to apply logic to select positive samples
        z_neg = torch.randn_like(z)  # Example negative samples (modify as necessary)

        # Apply regressor to latent space
        y_hat = self.regressor(z)

        # Reconstruct the input using the decoder (if decoding is enabled)
        if self.decode_mode:
            x_hat = self.decoder(z)
            return y_hat, z, mean, log_var, x_hat, z_pos, z_neg
        
        return y_hat, z, mean, log_var, z_pos, z_neg

""" 
class TSHAE(nn.Module):
    def __init__(self, config, encoder, decoder):
        super(TSHAE, self).__init__()

        self.p = config['dropout_regressor']
        self.regression_dims = config['regression_dims']

        self.decode_mode = config['reconstruct']
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder"
            self.decoder = decoder

        self.encoder = encoder

        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.regression_dims),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(self.regression_dims, 1)
        )
    def forward(self, x):
        """""""
        :param x: [B, seq_len, fea_dim]
        :return:
            y_hat: [B, 1]
            z: [B, 2]
            mean: [B, 2]
            log_var: [B, 2]
            x_hat: [B, seq_len, fea_dim]
        """""""
        print(f"TSHAE input x shape: {x.shape}, has_nan: {torch.isnan(x).any()}")
        z, mean, log_var = self.encoder(x)
        y_hat = self.regressor(z)
        print(f"TSHAE regressor output y_hat shape: {y_hat.shape}, z shape: {z.shape}")
        if self.decode_mode:
            x_hat = self.decoder(z)
            print(f"TSHAE decoder output x_hat shape: {x_hat.shape}")
            return y_hat, z, mean, log_var, x_hat
        return y_hat, z, mean, log_var"""