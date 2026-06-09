import numpy as np

def composite_score(recon_err, pred_err, latent_dist, uncertainty,
                    weights=(1.0, 0.8, 0.5, 0.3)):
    return (
        weights[0] * recon_err +
        weights[1] * pred_err +
        weights[2] * latent_dist +
        weights[3] * uncertainty
    )
