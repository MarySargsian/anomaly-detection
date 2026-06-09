import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleEncoder(nn.Module):
    """
    Parallel encoders capturing short / mid / long temporal patterns
    """
    def __init__(self, channels, latent_dim):
        super().__init__()

        self.short = TemporalEncoderBlock(channels, 64, kernel=3, stride=1)
        self.mid   = TemporalEncoderBlock(channels, 64, kernel=7, stride=2)
        self.long  = TemporalEncoderBlock(channels, 64, kernel=15, stride=4)

        self.project = nn.Sequential(
            nn.Linear(64 * 3, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, C, T)
        """
        s = F.adaptive_avg_pool1d(self.short(x), 1).squeeze(-1)
        m = F.adaptive_avg_pool1d(self.mid(x), 1).squeeze(-1)
        l = F.adaptive_avg_pool1d(self.long(x), 1).squeeze(-1)

        z = torch.cat([s, m, l], dim=1)
        return self.project(z)
