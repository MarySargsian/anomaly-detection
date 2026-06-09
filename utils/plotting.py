import matplotlib.pyplot as plt
import numpy as np

def plot_time_series_with_anomalies(signal, scores, threshold, title, save_path=None, show=True):
    plt.figure(figsize=(14, 4))
    plt.plot(signal, label="Signal", alpha=0.8)
    anomalies = scores > threshold
    plt.scatter(np.where(anomalies), signal[anomalies],
                c="red", s=15, label="Detected anomalies")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_loss(train_losses, val_losses=None):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    if val_losses:
        plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_error_decomposition(recon_errors, pred_errors, uncertainties, scores, threshold, save_path=None, show=True):
    """
    Multi-panel plot showing core anomaly factors
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    axes[0].plot(recon_errors, color='blue', label='Reconstruction Error')
    axes[0].set_title('Reconstruction Error')
    axes[1].plot(pred_errors, color='green', label='Prediction Error')
    axes[1].set_title('Prediction Error')
    axes[2].plot(uncertainties, color='purple', label='Uncertainty')
    axes[2].set_title('Uncertainty')
    
    axes[3].plot(scores, color='black', label='Composite Score')
    axes[3].axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    axes[3].set_title('Composite Anomaly Score')
    
    for ax in axes:
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_latent_trajectory(latents, save_path=None, show=True):
    """
    PCA trajectory of latent vectors
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(latents)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(z_pca[:, 0], z_pca[:, 1], c=np.arange(len(z_pca)), cmap='viridis', s=5, alpha=0.5)
    plt.plot(z_pca[:, 0], z_pca[:, 1], color='black', alpha=0.1) # connection line
    plt.colorbar(label='Time Step')
    plt.title('Latent Space Trajectory (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_window_time_heatmap(scores_matrix, title="Window-to-Time Anomaly Heatmap", save_path=None, show=True):
    """
    Heatmap where Rows = sliding windows, Columns = time
    Actually, a more common interpretation: color = score, x = time, y = window index
    """
    plt.figure(figsize=(15, 5))
    plt.imshow(scores_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Anomaly Score')
    plt.title(title)
    plt.xlabel('Time inside window')
    plt.ylabel('Window Index')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_prediction_samples(samples, recons, preds, truths, uncertainties=None, save_path=None, show=True):
    """
    Plot 5 samples: past window (truth & recon) and future window (truth & pred)
    """
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        window_size = samples[i].shape[1]
        pred_steps = preds[i].shape[1]
        
        # Time axes
        t_past = np.arange(window_size)
        t_future = np.arange(window_size, window_size + pred_steps)
        
        # Plotting first channel for visualization
        axes[i].plot(t_past, samples[i][0], 'k-', label='Input (GT)')
        axes[i].plot(t_past, recons[i][0], 'b--', label='Reconstruction')
        
        axes[i].plot(t_future, truths[i][0], 'k-', alpha=0.5, label='Future (GT)')
        axes[i].plot(t_future, preds[i][0], 'g-', label='Prediction')
        
        if uncertainties is not None:
            # Simple visualization of uncertainty as range
            std = np.sqrt(uncertainties[i][0])
            axes[i].fill_between(t_past, recons[i][0] - std, recons[i][0] + std, 
                                 color='blue', alpha=0.1, label='Uncertainty')

        axes[i].set_title(f"Instance {i+1}")
        axes[i].legend(loc='upper right')
        axes[i].grid(alpha=0.2)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
