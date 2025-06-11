# config.py
# Configuration file for the DiffRUL project
config = {
    "input_size": 14,  # Number of selected sensor features for C-MAPSS dataset
    "hidden_size": 64,  # Hidden layer size for LSTM/VAE
    "latent_dim": 2,   # Latent dimension for VAE
    "window_size": 30,  # Size of the time window for data processing
    "num_layers": 2,   # Number of LSTM layers
    "bidirectional": True,  # Whether to use bidirectional LSTM
    "dropout_lstm_encoder": 0.1,  # Dropout rate for LSTM encoder
    "dropout_lstm_decoder": 0.1,  # Dropout rate for LSTM decoder
    "dropout_layer_encoder": 0.1,  # Dropout rate for encoder layers
    "dropout_layer_decoder": 0.1,  # Dropout rate for decoder layers
    "regression_dims": 64,  # Dimensions for regression layer
    "dropout_regressor": 0.1,  # Dropout rate for regressor
    "reconstruct": True,  # Whether to enable reconstruction in the model
    "lr": 0.001,  # Learning rate
    "max_epochs": 50,  # Maximum number of training epochs
    "batch_size": 16,  # Batch size for training
    "output_dir": "./output",  # Directory to save output files
    "vae_model_path": "./output/best_vae_model.pt",  # Path to save the best VAE model
    "diff_model_path": "./output/best_diff_model.pt",  # Path to save the best diffusion model
    "noise_steps": 50,  # Change to 50 as per paper (T=50)
    "beta_start": 0.0004,  # Change to 0.0004 as per paper
    "beta_end": 0.05,  # Change to 0.05 as per paper
    "schedule_name": "linear",  # Type of noise schedule
    "residual_channels": 64,  # Number of channels in residual layers (Nc=64)
    "residual_layers": 30,  # Number of residual layers (Nd=30)
    "dilation_cycle_length": 3,  # Change to 3 as per paper (m=3)
    "KLLoss_weight": 1,  # Weight for KL divergence loss (lambda_1=1)
    "RegLoss_weight": 1,  # Weight for regression loss (lambda_2=1)
    "ReconLoss_weight": 1,  # Weight for reconstruction loss (lambda_3=1)
    "TripletLoss_weight": 10,  # Weight for triplet loss (lambda_4=10)
    "TripletLoss_margin": 0.4,  # Margin for triplet loss (alpha=0.4)
    "TripletLoss_p": 2,  # Norm for triplet loss
    "data_dir": "/workspace/TurboFan/data/CMAPSS",  # Directory of C-MAPSS dataset
    # Add learning rate scheduler
    "lr_scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.98}  # Decay factor 0.98
}