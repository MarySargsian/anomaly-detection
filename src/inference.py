import torch
import numpy as np
import os
import json
import random
import argparse
from pathlib import Path

from src.models.custom_model import CustomAnomalyModel
from utils.preprocessing import TimeSeriesPreprocessor, temporal_split
from utils.plotting import (
    plot_time_series_with_anomalies, 
    plot_prediction_samples
)
from utils.anomaly_score import composite_score
from utils.metrics import calculate_metrics, point_adjust

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- CONFIG ---
WINDOW = 100
CHANNELS = 25
PRED_STEPS = 20
BATCH_SIZE = 512
RESULTS_DIR = Path("results").absolute()
RESULTS_DIR.mkdir(exist_ok=True)

def run_inference():
    parser = argparse.ArgumentParser(description="SMAP Anomaly Detection Inference")
    parser.add_argument("--n", type=int, default=1, help="Number of random samples to plot")
    args = parser.parse_args()

    print(f"--- Starting Inference ---")
    print(f"Results will be saved to: {RESULTS_DIR}")

    # 1. Prepare Data
    data = np.load("data/raw/smap_train.npy")
    labels_all = np.load("data/raw/smap_labels.npy")

    train, _, test = temporal_split(data)
    _, _, test_labels = temporal_split(labels_all)

    prep = TimeSeriesPreprocessor(WINDOW)
    prep.fit(train)

    test_norm = prep.transform(test)
    X_test, Y_test = prep.create_predictive_windows(test_norm, PRED_STEPS)

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomAnomalyModel(CHANNELS, latent_dim=64, window=WINDOW).to(device)
    model_path = "outputs/models/custom_model.pt"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Please train first.")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 3. Batch Processing (Fast)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                      torch.tensor(Y_test, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    all_scores = []
    print(f"Processing dataset on {device}...")
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Inference"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mu, logvar, future_pred, proj, _ = model(x_batch)
            
            # Components for score
            mse_win = torch.mean((x_batch - mu) ** 2, dim=(1, 2))
            fut_mse = torch.mean((y_batch - future_pred) ** 2, dim=(1, 2))
            drift = torch.norm(proj, dim=1)
            var = torch.exp(logvar).mean(dim=(1, 2))
            
            # Combine into composite score
            score = composite_score(mse_win, fut_mse, drift, var)
            all_scores.extend(score.cpu().numpy())

    scores = np.array(all_scores)
    threshold = np.percentile(scores, 95)
    preds = (scores > threshold).astype(int)
    y_true = test_labels[WINDOW : WINDOW + len(scores)]

    # 4. Save Metrics (results_summary.json)
    base_metrics = calculate_metrics(y_true, preds)
    pa_metrics = calculate_metrics(y_true, point_adjust(y_true, preds))

    summary = {
        "threshold": float(threshold),
        "standard_metrics": base_metrics,
        "point_adjusted_metrics": pa_metrics
    }
    
    summary_path = RESULTS_DIR / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[SUCCESS] Metrics saved to: {summary_path}")

    # 5. Save Global Plot (anomaly_detection.png)
    signal = test[WINDOW : WINDOW + len(scores), 0]
    plot_path = RESULTS_DIR / "anomaly_detection.png"
    plot_time_series_with_anomalies(
        signal, scores, threshold, "Main Performance", 
        save_path=plot_path, show=False
    )
    print(f"[SUCCESS] Global plot saved to: {plot_path}")

    # 6. Save N Sample Results (Multiple files per sample)
    if args.n > 0:
        print(f"Selecting {args.n} random samples for granular visualization...")
        indices = random.sample(range(len(X_test)), args.n)
        
        # We also need the per-window scores for the local anomaly plots
        # scores[i] corresponds to signal[WINDOW + i]
        
        for i, idx in enumerate(indices):
            print(f"Generating results for Sample {i+1} (Index: {idx})...")
            
            # --- A. Detailed Reconstruction/Prediction Plot ---
            x_samp = X_test[idx]
            y_samp = Y_test[idx]
            
            with torch.no_grad():
                x_t = torch.tensor(x_samp, dtype=torch.float32).unsqueeze(0).to(device)
                mu, logvar, future, proj, _ = model(x_t)
                
                # Plot 1: Prediction/Reconstruction with uncertainty
                sample_plot_path = RESULTS_DIR / f"sample_{i+1}.png"
                plot_prediction_samples(
                    [x_samp], [mu.squeeze(0).cpu().numpy()], 
                    [future.squeeze(0).cpu().numpy()], [y_samp], 
                    [torch.exp(logvar).squeeze(0).cpu().numpy()],
                    save_path=sample_plot_path, show=False
                )
                print(f"  [SUCCESS] Detailed Plot: {sample_plot_path}")

                # --- B. Zoome-in Anomaly Detection Plot ---
                # Context: idx-250 to idx+250
                context_size = 250
                start_idx = max(0, idx - context_size)
                end_idx = min(len(scores), idx + context_size)
                
                local_signal = test[WINDOW + start_idx : WINDOW + end_idx, 0]
                local_scores = scores[start_idx : end_idx]
                
                local_anomaly_path = RESULTS_DIR / f"anomaly_detection_sample_{i+1}.png"
                plot_time_series_with_anomalies(
                    local_signal, local_scores, threshold, 
                    f"Anomaly Context for Sample {i+1} (Idx: {idx})",
                    save_path=local_anomaly_path, show=False
                )
                print(f"  [SUCCESS] Context Anomaly Plot: {local_anomaly_path}")

                # --- C. Local Results Summary JSON ---
                # We can calculate local window components here
                mse_win = torch.mean((x_t - mu) ** 2).item()
                fut_mse = torch.mean((torch.tensor(y_samp).to(device) - future) ** 2).item()
                uncertainty = torch.exp(logvar).mean().item()
                drift = torch.norm(proj).item()
                total_score = scores[idx]

                local_summary = {
                    "sample_index": int(idx),
                    "total_anomaly_score": float(total_score),
                    "is_anomaly": bool(total_score > threshold),
                    "threshold_used": float(threshold),
                    "components": {
                        "reconstruction_error": float(mse_win),
                        "prediction_error": float(fut_mse),
                        "uncertainty_variance": float(uncertainty),
                        "latent_drift": float(drift)
                    }
                }
                
                local_json_path = RESULTS_DIR / f"results_summary_sample_{i+1}.json"
                with open(local_json_path, "w") as f:
                    json.dump(local_summary, f, indent=4)
                print(f"  [SUCCESS] Local Summary JSON: {local_json_path}")

    print(f"\n--- ALL REQUESTED SAMPLES COMPLETED ---")
    print(f"Global results available at: {RESULTS_DIR / 'results_summary.json'}")

if __name__ == "__main__":
    run_inference()
