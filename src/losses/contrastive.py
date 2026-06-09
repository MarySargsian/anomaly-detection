import torch
import torch.nn.functional as F

def info_nce(z1, z2, temperature=0.2):
    sim = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(sim, labels)
