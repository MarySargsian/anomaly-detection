import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import os

from src.models.custom_model import CustomAnomalyModel
from src.models.baseline_ae import BaselineAE
from src.losses.temporal_contrastive import temporal_contrastive_loss
from src.losses.gaussian_nll import gaussian_nll
from utils.preprocessing import TimeSeriesPreprocessor, temporal_split
from utils.augmentations import TimeSeriesAugmentation

# ======================
# Hyperparameters
# ======================
WINDOW = 100
PRED_STEPS = 20
BATCH_SIZE = 64
EPOCHS = 20 # Returned to 20 for stability as requested
LR = 1e-4   # Reduced LR for stability
CHANNELS = 25
LATENT_DIM = 64

LAMBDA_CONTRAST = 0.1
LAMBDA_PRED = 1.0


def train():
    # ======================
    # 1. Load Data
    # ======================
    data_path = Path("data/raw/smap_train.npy")
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run scripts/load_smap.py first.")
        return

    raw_data = np.load(data_path)
    if np.isnan(raw_data).any():
        print("Warning: NaN found in raw data. Cleaning...")
        raw_data = np.nan_to_num(raw_data)

    train_data, val_data, _ = temporal_split(raw_data)

    # ======================
    # 2. Preprocessing
    # ======================
    prep = TimeSeriesPreprocessor(WINDOW)
    prep.fit(train_data)

    train_norm = prep.transform(train_data)
    val_norm = prep.transform(val_data)

    X_train, Y_train = prep.create_predictive_windows(train_norm, PRED_STEPS)
    X_val, Y_val = prep.create_predictive_windows(val_norm, PRED_STEPS)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                      torch.tensor(Y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                      torch.tensor(Y_val, dtype=torch.float32)),
        batch_size=BATCH_SIZE
    )

    # ======================
    # 3. Models & Optimizers
    # ======================
    from src.losses.contrastive import info_nce
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom_model = CustomAnomalyModel(CHANNELS, LATENT_DIM, WINDOW).to(device)
    baseline_model = BaselineAE(CHANNELS, LATENT_DIM, WINDOW).to(device)

    optimizer_c = optim.Adam(custom_model.parameters(), lr=LR)
    optimizer_b = optim.Adam(baseline_model.parameters(), lr=LR)

    os.makedirs("outputs/models", exist_ok=True)

    # ======================
    # 4. Training Loop
    # ======================
    print(f"Starting training on {device}...")

    for epoch in range(EPOCHS):
        custom_model.train()
        baseline_model.train()

        total_loss_c = 0
        total_loss_b = 0

        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)

            # ======================
            # Custom Model
            # ======================
            optimizer_c.zero_grad()

            # Contrastive augmentation
            x1, x2 = TimeSeriesAugmentation.make_pair(x)

            mu1, logvar1, future1, proj1, _ = custom_model(x1)
            _, _, _, proj2, _ = custom_model(x2)

            # Gaussian reconstruction loss
            loss_recon = gaussian_nll(x1, mu1, logvar1)
            
            # Prediction loss (MSE)
            loss_pred = torch.mean((y - future1) ** 2)

            # Contrastive loss (Using pair of augmentations)
            loss_cont = info_nce(proj1, proj2)

            loss_c = loss_recon + LAMBDA_PRED * loss_pred + LAMBDA_CONTRAST * loss_cont
            
            if torch.isnan(loss_c):
                print(f"NaN detected at Epoch {epoch+1}. Aborting training.")
                return

            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(custom_model.parameters(), max_norm=1.0)
            optimizer_c.step()

            total_loss_c += loss_c.item()

            # ======================
            # Baseline AE
            # ======================
            optimizer_b.zero_grad()
            recon_b = baseline_model(x)
            loss_b = torch.mean((x - recon_b) ** 2)
            loss_b.backward()
            optimizer_b.step()

            total_loss_b += loss_b.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Custom Loss: {total_loss_c/len(train_loader):.4f} | Baseline Loss: {total_loss_b/len(train_loader):.4f}")

    # ======================
    # 5. Save Models
    # ======================
    torch.save(custom_model.state_dict(), "outputs/models/custom_model.pt")
    torch.save(baseline_model.state_dict(), "outputs/models/baseline_ae.pt")

    print("Models saved in outputs/models/")


if __name__ == "__main__":
    train()
