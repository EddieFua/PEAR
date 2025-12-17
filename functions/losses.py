import torch
import torch.nn.functional as F
from scipy.stats import norm
import math
def focal_loss(
    logits, 
    y, 
    alpha_pos=0.25, 
    alpha_neg=0.75, 
    gamma=2.0, 
    reduction="mean"
):
    """
    logits: [N] or [N, 1]
    y:      [N] (0/1)
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, y.float(), reduction="none"
    ) 
    pt = torch.exp(-bce)
    alpha_t = y * alpha_pos + (1 - y) * alpha_neg

    loss = alpha_t * (1 - pt) ** gamma * bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def cosine_align(z1, z2, eps=1e-8):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    return (1.0 - (z1 * z2).sum(dim=-1)).mean()

def cosine_orth(z1, z2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    cos = (z1 * z2).sum(dim=-1)
    return (cos * cos).mean()

def cross_covariance_penalty(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-6):
    N = A.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=A.device, dtype=A.dtype)
    A0 = A - A.mean(dim=0, keepdim=True)
    B0 = B - B.mean(dim=0, keepdim=True)
    C = (A0.t() @ B0) / (N - 1)
    return (C.pow(2).mean())


def stg_regularizer(mu: torch.Tensor,
                    sigma: torch.Tensor,
                    lam: float = 1e-3) -> torch.Tensor:
    while sigma.dim() < mu.dim():
        sigma = sigma.unsqueeze(0)
    t = mu / (sigma + 1e-8)
    p = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
    pen = p.sum()
    return lam * pen
