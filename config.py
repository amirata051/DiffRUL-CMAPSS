import os

def get_subset_config(subset_id):
    base_config = {
        "input_size": 14,
        "hidden_size": 64,
        "latent_dim": 2,
        "window_size": 30,
        "num_layers": 2,
        "bidirectional": True,
        "dropout_lstm_encoder": 0.1,
        "dropout_lstm_decoder": 0.1,
        "dropout_layer_encoder": 0.1,
        "dropout_layer_decoder": 0.1,
        "regression_dims": 64,
        "dropout_regressor": 0.1,
        "reconstruct": True,
        "lr": 0.001,
        "max_epochs": 50,
        "batch_size": 16,
        "noise_steps": 50,
        "beta_start": 0.0004,
        "beta_end": 0.05,
        "schedule_name": "linear",
        "residual_channels": 64,
        "residual_layers": 30,
        "dilation_cycle_length": 3,
        "KLLoss_weight": 1,
        "RegLoss_weight": 1,
        "ReconLoss_weight": 1,
        "TripletLoss_weight": 10,
        "TripletLoss_margin": 0.4,
        "TripletLoss_p": 2,
        "data_dir": "/workspace/TurboFan/data/CMAPSS",
        "lr_scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.98}
    }
    
    base_config["output_dir"] = os.path.join("./output", subset_id)
    base_config["vae_model_path"] = os.path.join(base_config["output_dir"], "best_vae_model.pt")
    base_config["diff_model_path"] = os.path.join(base_config["output_dir"], "best_diff_model.pt")
    
    return base_config