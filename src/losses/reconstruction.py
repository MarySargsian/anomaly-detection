import torch.nn.functional as F

def reconstruction_loss(x, recon):
    return F.mse_loss(recon, x)
