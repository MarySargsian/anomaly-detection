import torch.nn as nn
from .contrastive_head import ProjectionHead
from .multiscale_encoder import MultiScaleEncoder
from .predictive_decoder import PredictiveDecoder


class CustomAnomalyModel(nn.Module):
    def __init__(self, channels, latent_dim=64, window=100):
        super().__init__()
        self.encoder = MultiScaleEncoder(channels, latent_dim)
        self.decoder = PredictiveDecoder(latent_dim=latent_dim, channels=channels, window_size=window, pred_steps=20)
        self.projector = ProjectionHead(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        mu, logvar, future = self.decoder(z)
        proj = self.projector(z)
        return mu, logvar, future, proj, z
