import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, channels, latent_dim, window):
        super().__init__()
        reduced = window // 4

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128 * reduced),
            nn.ReLU(),
            nn.Unflatten(1, (128, reduced)),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, channels, 4, 2, 1)
        )

    def forward(self, z):
        return self.net(z)
