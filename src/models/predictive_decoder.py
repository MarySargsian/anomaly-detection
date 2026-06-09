import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveDecoder(nn.Module):
    """
    Reconstructs past window (mu, logvar) and predicts future k steps
    """

    def __init__(
        self,
        latent_dim: int,
        channels: int,
        window_size: int,
        pred_steps: int = 20
    ):
        super().__init__()

        self.window_size = window_size
        self.pred_steps = pred_steps

        # Expand latent vector into temporal feature map
        self.fc = nn.Linear(latent_dim, 128 * (window_size // 4))

        # Distributional heads
        self.decoder_mu = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, channels, 4, stride=2, padding=1)
        )
        
        self.decoder_logvar = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, channels, 4, stride=2, padding=1),
            nn.Tanh() # Constrain logvar to [-1, 1]
        )

        # Future prediction head
        self.future_head = nn.Linear(latent_dim, channels * pred_steps)

    def forward(self, z):
        """
        z: (B, latent_dim)

        returns:
        - mu: (B, C, T)
        - logvar: (B, C, T)
        - future: (B, C, pred_steps)
        """

        B = z.size(0)

        # Latent → temporal feature map
        x = self.fc(z)
        x = x.view(B, 128, self.window_size // 4)

        # Gaussian reconstruction
        mu = self.decoder_mu(x)
        logvar = self.decoder_logvar(x) * 5.0 # Stable but flexible range

        # Future prediction
        future = self.future_head(z)
        future = future.view(B, -1, self.pred_steps)

        return mu, logvar, future
