"""Focal, alignment, cross-covariance, and stochastic-gate losses for PEAR."""

import math

import torch
from torch.nn import functional as F


def focal_loss(logits, y, alpha_pos=0.25, alpha_neg=0.75, gamma=2.0, reduction="mean"):
    logits = logits.reshape(-1)
    y = y.to(dtype=logits.dtype).reshape(-1)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    loss = (y * alpha_pos + (1 - y) * alpha_neg) * (-torch.expm1(-bce)) ** gamma * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def cosine_align(z1, z2, eps=1e-8):
    return (
        1
        - (F.normalize(z1, dim=-1, eps=eps) * F.normalize(z2, dim=-1, eps=eps)).sum(-1)
    ).mean()


def cross_covariance_penalty(A, B):
    if len(A) <= 1:
        return (A.sum() + B.sum()) * 0
    C = (A - A.mean(0)).T @ (B - B.mean(0)) / (len(A) - 1)
    return C.square().mean()


def stg_regularizer(mu, sigma, lam=1e-3):
    return lam * (0.5 * (1 + torch.erf(mu / sigma / math.sqrt(2)))).sum()


def pear_loss(
    out,
    y,
    sigma,
    alpha_pos,
    alpha_neg,
    gamma=2.0,
    lam_pred=10.0,
    lam_align=0.1,
    lam_orth=200.0,
    lam_stg=5e-5,
):
    def focal(logits):
        return focal_loss(logits, y, alpha_pos, alpha_neg, gamma)

    terms = {"task": lam_pred * focal(out["logit"])}
    terms["aux"] = (
        lam_pred
        * sum(
            focal(out[k]) for k in ("logit_shared", "logit_dist_prs", "logit_dist_ehr")
        )
        / 3
    )
    terms["align"] = lam_align * cosine_align(out["z_prs_shared"], out["z_ehr_shared"])
    terms["orth"] = lam_orth * (
        cross_covariance_penalty(out["z_prs_shared"], out["z_prs_dist"])
        + cross_covariance_penalty(out["z_ehr_shared"], out["z_ehr_dist"])
    )
    terms["stg"] = stg_regularizer(out["mu"], sigma, lam_stg)
    return sum(terms.values()), terms
