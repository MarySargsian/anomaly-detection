import torch.nn as nn

class BaselineAE(nn.Module):
    def __init__(self, channels, latent_dim=64, window=100):
        super().__init__()
        reduced = window // 4
        
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * reduced),
            nn.ReLU(),
            nn.Unflatten(1, (128, reduced)),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, channels, 4, 2, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
