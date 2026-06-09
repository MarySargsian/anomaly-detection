import torch.nn as nn
from .temporal_attention import TemporalGatedAttention

class Encoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()

        self.attn = TemporalGatedAttention(channels)

        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        x = self.attn(x)
        return self.net(x)
