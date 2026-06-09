import torch
import random

class TimeSeriesAugmentation:
    """
    Lightweight time-series augmentations for contrastive learning
    """

    @staticmethod
    def jitter(x, sigma=0.03):
        return x + sigma * torch.randn_like(x)

    @staticmethod
    def scaling(x, sigma=0.1):
        scale = 1 + sigma * torch.randn(x.size(0), 1, 1, device=x.device)
        return x * scale

    @staticmethod
    def masking(x, ratio=0.1):
        B, C, T = x.shape
        mask = torch.rand(B, 1, T, device=x.device) > ratio
        return x * mask

    @staticmethod
    def augment(x):
        if random.random() < 0.5:
            x = TimeSeriesAugmentation.jitter(x)
        if random.random() < 0.5:
            x = TimeSeriesAugmentation.scaling(x)
        if random.random() < 0.5:
            x = TimeSeriesAugmentation.masking(x)
        return x

    @staticmethod
    def make_pair(x):
        return TimeSeriesAugmentation.augment(x.clone()), \
               TimeSeriesAugmentation.augment(x.clone())
