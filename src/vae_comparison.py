import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.custom_model import CustomAnomalyModel
from utils.preprocessing import TimeSeriesPreprocessor, temporal_split
from utils.plotting import plot_prediction_samples
from utils.anomaly_score import composite_score

# --- Minimal VAE Implementation ---
class StandaloneVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, window=100):
        super(StandaloneVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * window, latent_dim)
        self.fc_logvar = nn.Linear(64 * window, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 64 * window)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, window)),
            nn.ConvTranspose1d(64, 32, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, input_dim, 5, padding=2)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.decoder_fc(z)), mu, logvar

def vae_loss_fn(recon, x, mu, logvar):
    mse = torch.mean((recon - x) ** 2)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + 0.01 * kld

def main():
    WINDOW, CHANNELS, PRED_STEPS = 100, 25, 20
    RESULTS_DIR = Path("results/comparison").absolute()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading data...")
    data = np.load("data/raw/smap_train.npy")
    train, _, test = temporal_split(data)
    prep = TimeSeriesPreprocessor(WINDOW)
    prep.fit(train)
    
    train_norm = prep.transform(train)
    X_train, _ = prep.create_predictive_windows(train_norm, PRED_STEPS)
    test_norm = prep.transform(test)
    X_test, Y_test = prep.create_predictive_windows(test_norm, PRED_STEPS)

    # 1. Train VAE quickly
    print("Training baseline VAE (Fast)...")
    vae = StandaloneVAE(CHANNELS, window=WINDOW).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_train[:5000], dtype=torch.float32)), batch_size=128, shuffle=True)
    
    vae.train()
    for _ in range(3): # 3 Epochs on subset is enough for a baseline check
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            loss = vae_loss_fn(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

    # 2. Load Custom Model
    print("Loading Custom Model...")
    custom = CustomAnomalyModel(CHANNELS, latent_dim=64, window=WINDOW).to(device)
    custom.load_state_dict(torch.load("outputs/models/custom_model.pt", map_location=device))
    custom.eval()
    vae.eval()

    # 3. Generate Comparative results for 5 random samples
    indices = random.sample(range(len(X_test)), 5)
    for i, idx in enumerate(indices):
        print(f"Comparing index {idx}...")
        x_samp = X_test[idx:idx+1]
        x_t = torch.tensor(x_samp, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            mu_c, lv_c, fut_c, _, _ = custom(x_t)
            mu_v, _, _ = vae(x_t)
            
            # --- Plot Comparison ---
            plt.figure(figsize=(12, 6))
            t = np.arange(WINDOW)
            plt.plot(t, x_samp[0, 0], 'k-', label='Ground Truth', alpha=0.3)
            plt.plot(t, mu_c[0, 0].cpu().numpy(), 'b-', label='Custom Model Recon', linewidth=2)
            plt.plot(t, mu_v[0, 0].cpu().numpy(), 'r--', label='VAE Baseline Recon', linewidth=1.5)
            
            # Add confidence interval for custom model
            std_c = np.sqrt(torch.exp(lv_c[0, 0]).cpu().numpy())
            plt.fill_between(t, mu_c[0, 0].cpu().numpy() - std_c, mu_c[0, 0].cpu().numpy() + std_c, color='blue', alpha=0.1, label='Custom Uncertainty')
            
            plt.title(f"Comparison Sample {i+1} (Index {idx})")
            plt.legend()
            plt.grid(alpha=0.2)
            
            save_path = RESULTS_DIR / f"comparison_sample_{i+1}.png"
            plt.savefig(save_path)
            plt.close()
            print(f"[SUCCESS] Saved comparison to {save_path}")

    print(f"\nStandalone Comparison Complete. Plots in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
