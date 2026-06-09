import torch
import torch.nn as nn

class TemporalGatedAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, C, T)
        """
        gates = self.net(x)
        return x * gates
