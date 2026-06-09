import torch

def gaussian_nll(x, mu, log_var):
    """
    Negative log-likelihood loss with stability clipping
    """
    # Clip log_var to prevent NaN (exp explosion or div by zero)
    log_var = torch.clamp(log_var, min=-7.0, max=7.0)
    
    return torch.mean(
        0.5 * (log_var + (x - mu) ** 2 / (torch.exp(log_var) + 1e-6))
    )
