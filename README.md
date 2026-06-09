# anomaly-detection
Robust Multivariate Anomaly Detection in Satellite Telemetry via Multi-Task Contrastive Learning

This repository contains the core implementation modules for CustomAnomalyModel, an advanced, multi-task deep representation learning framework optimized for multivariate anomaly detection in satellite telemetry (specifically validated on the Soil Moisture Active Passive (SMAP) dataset).

The system integrates multi-scale temporal modeling, uncertainty-aware probabilistic reconstruction, future forecasting, and self-supervised contrastive learning to robustly isolate transient spikes, long-term sensor drifts, and regime changes in 25-dimensional telemetry streams.

Key Architectural Pillars

Our architecture addresses the limitations of standard autoencoders by leveraging three unified training objectives:

1. Multi-Scale Temporal Encoder (MSTE)

Captures diverse temporal dynamics using three parallel 1D convolutional branches:

Short-term branch (Kernel Size = 3) to capture pointwise anomalies/spikes.

Mid-term branch (Kernel Size = 7) for localized micro-trends.

Long-term branch (Kernel Size = 15) for macro-seasonal variations.

2. Probabilistic Reconstruction & Uncertainty Estimation

Rather than modeling deterministic reconstructions, the decoder estimates a Gaussian distribution:


$$p(x | z) = \mathcal{N}(x; \mu(z), \sigma^2(z))$$


We model the variance log-space $\log \sigma^2$ constrained by a $\tanh$ activation to ensure numerical stability and prevent $\text{NaN}$ losses during training.

3. Multi-Task Self-Supervised Learning

The model optimizes a joint loss function:


$$\mathcal{L} = \mathcal{L}_{\text{prob\_recon}} + \lambda_{\text{pred}} \mathcal{L}_{\text{future}} + \lambda_{\text{cont}} \mathcal{L}_{\text{infoNCE}}$$

Gaussian Negative Log-Likelihood ($\mathcal{L}_{\text{prob\_recon}}$): Accounts for heteroscedastic noise in sensor readings.

Future Prediction Loss ($\mathcal{L}_{\text{future}}$): Recovers predictive temporal sequences.

InfoNCE Contrastive Loss ($\mathcal{L}_{\text{infoNCE}}$): Formulated over jittered, scaled, and masked versions of the same window to enforce semantic invariant representations.

Repository Structure

The codebase is modularized as follows:

File

Description

config.py

Global hyperparameters, window sizing ($W=100$), input dimensions, training batch configurations, and execution hardware settings.

preprocessing.py

TimeSeriesPreprocessor class for robust z-score standardization, rolling window generation, and predictive sequence splitting.

augmentations.py

Real-time, lightweight time-series augmentations (jittering, scaling, masking) to feed the InfoNCE contrastive optimization branch.

anomaly_score.py

Computes a composite metric $S$ combining forecasting error, reconstruction error, latent representation drift, and reconstruction variance.

metrics.py

Standard classification metrics (Precision, Recall, $F_1$) paired with the standard Point-Adjustment (PA) evaluation strategy.

plotting.py

Comprehensive visualization suite covering signal anomalies, error decomposition, latent PCA trajectories, and heatmaps.

report.tex

Fully compiled scientific manuscript documenting technical formulations, background, and experimental findings.


Pipeline Details & Setup

Mathematical Evaluation Workflow

To isolate anomalous regimes, the system calculates a weighted Composite Anomaly Score $S$ at each sliding window index:


$$S = \delta_{\text{recon}} + \omega \delta_{\text{pred}} + \phi \| z \|_{\text{drift}} + \theta \overline{\sigma^2}$$

Where:

$\delta_{\text{recon}}$ is the reconstruction MSE.

$\delta_{\text{pred}}$ is the future prediction MSE.

$\| z \|_{\text{drift}}$ is the latent feature space offset.

$\overline{\sigma^2}$ is the average probabilistic model variance (uncertainty indicator).

During evaluation, the Point-Adjustment (PA) protocol is activated: if any single timestamp in a continuous ground-truth anomaly segment exceeds the threshold, the entire sequence is classified as a True Positive. This matches real-world satellite operations where early warnings on continuous failures are highly valued.


Sample Execution Workflow

To tie all the modules together in a training and evaluation script:

import numpy as np
import torch
from config import Config
from preprocessing import TimeSeriesPreprocessor
from augmentations import TimeSeriesAugmentation
from anomaly_score import composite_score
from metrics import calculate_metrics, point_adjust
from plotting import plot_time_series_with_anomalies

# 1. Load telemetry data (Mock 25-channel data)
data = np.random.randn(5000, Config.INPUT_CHANNELS)
labels = np.zeros(5000)
labels[1200:1350] = 1  # Simulated anomaly segment

# 2. Preprocess and create sliding windows
preprocessor = TimeSeriesPreprocessor(window_size=Config.WINDOW_SIZE)
preprocessor.fit(data)
scaled_data = preprocessor.transform(data)
x_windows, y_windows = preprocessor.create_predictive_windows(scaled_data, pred_steps=20)

# 3. Simulate contrastive augmentation pair
x_tensor = torch.tensor(x_windows, dtype=torch.float32)
augmented_view_1, augmented_view_2 = TimeSeriesAugmentation.make_pair(x_tensor)

print(f"Original Window Shape: {x_windows.shape}")
print(f"Augmented View Shape: {augmented_view_1.shape}")

# 4. Score & Evaluate (Example downstream pipeline)
# Assuming mock arrays generated by model forward passes
recon_err = np.random.exponential(scale=0.5, size=len(x_windows))
pred_err = np.random.exponential(scale=0.5, size=len(x_windows))
latent_dist = np.random.normal(loc=0.1, scale=0.05, size=len(x_windows))
uncertainty = np.random.normal(loc=0.2, scale=0.02, size=len(x_windows))

# Calculate final composite anomaly scores
scores = composite_score(recon_err, pred_err, latent_dist, uncertainty)

# Set global anomaly threshold (e.g., 95th percentile)
threshold = np.percentile(scores, 95)
predictions = (scores > threshold).astype(int)

# Apply point adjustment protocol
aligned_labels = labels[Config.WINDOW_SIZE + 20 - 1 : len(labels)] # align shapes
adjusted_preds = point_adjust(aligned_labels[:len(predictions)], predictions)

# Compute performance metrics
results = calculate_metrics(aligned_labels[:len(predictions)], adjusted_preds)
print("\nEvaluation Results (Point-Adjusted):")
print(f"Precision: {results['precision']:.4f} | Recall: {results['recall']:.4f} | F1-Score: {results['f1']:.4f}")



Visualizations Included

The plotting.py module exposes rich functions to capture training metrics and explain model decisions:

plot_time_series_with_anomalies: Overlays anomalies in red dots on top of the raw multi-dimensional signal trajectory.

plot_error_decomposition: Generates a 4-panel subplot splitting the contribution of $\delta_{\text{recon}}$, $\delta_{\text{pred}}$, uncertainty, and composite score.

plot_latent_trajectory: Evaluates model representation stability by mapping $z$ vector transitions over time via Principal Component Analysis (PCA).
