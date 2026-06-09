import torch
import numpy as np
from src.models.custom_model import CustomAnomalyModel
from src.models.baseline_ae import BaselineAE
from utils.preprocessing import TimeSeriesPreprocessor, temporal_split
from utils.plotting import plot_time_series_with_anomalies

WINDOW = 100
CHANNELS = 25

# Load data
raw = np.load("data/raw/smap_train.npy")
train, _, test = temporal_split(raw)

prep = TimeSeriesPreprocessor(WINDOW)
prep.fit(train)

X_test = prep.create_windows(prep.transform(test))
X_test = torch.tensor(X_test, dtype=torch.float32)

# Load models
custom = CustomAnomalyModel(CHANNELS, latent_dim=64)
baseline = BaselineAE(CHANNELS, latent_dim=64)

custom.load_state_dict(torch.load("outputs/models/custom_model.pt"))
baseline.load_state_dict(torch.load("outputs/models/baseline_ae.pt"))

custom.eval()
baseline.eval()

# Compare 5 samples
for i in range(5):
    x = X_test[i:i+1]
    with torch.no_grad():
        recon_c, _, _ = custom(x)
        recon_b = baseline(x)

    err_c = ((x - recon_c)**2).mean(dim=(1,2)).item()
    err_b = ((x - recon_b)**2).mean(dim=(1,2)).item()

    print(f"Sample {i}: Custom={err_c:.4f}, Baseline={err_b:.4f}")
