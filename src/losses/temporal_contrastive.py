import torch
import torch.nn.functional as F

def temporal_contrastive_loss(z, temperature=0.1):
    """
    Positive: neighboring windows
    Negative: distant windows
    """
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / (temperature + 1e-8)
    labels = torch.arange(z.size(0)).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss
