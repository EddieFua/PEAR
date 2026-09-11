"""PEAR shared-distinct architecture from Supplementary Table 10."""

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F


def mlp_layers(dimensions, dropout=0.1):
    layers = []
    for i, (din, dout) in enumerate(pairwise(dimensions)):
        layers.append(nn.Linear(din, dout))
        if i < len(dimensions) - 2:
            layers.extend([nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(dout)])
    return nn.Sequential(*layers)


@torch.no_grad()
def _chol_from_semantics(S_sem, alpha=1.0, lam=0.05, jitter=0.0):
    S = F.normalize(S_sem, dim=1)
    K = S @ S.T
    eye = torch.eye(len(S), device=S.device, dtype=S.dtype)
    return torch.linalg.cholesky(alpha * K + (lam + jitter) * eye)


class OrthoFusion(nn.Module):
    def __init__(
        self,
        d_prs,
        d_ehr,
        d_sem,
        S_sem,
        d_shared=16,
        d_dist=16,
        prs_hidden=(512, 256),
        ehr_hidden=(32, 32),
        final_hidden=(64, 32),
        p_drop=0.1,
        alpha=1.0,
        lam=0.05,
        jitter=0.0,
        cov_dim=0,
    ):
        super().__init__()
        S_sem = torch.as_tensor(S_sem, dtype=torch.float32).clone()
        self.d_prs, self.d_ehr, self.cov_dim = d_prs, d_ehr, cov_dim
        self.u0 = nn.Parameter(torch.full((d_ehr,), 0.5))
        self.register_buffer("S_sem", S_sem)
        self.u_mlp = mlp_layers([d_sem, 32, 1], p_drop)
        nn.init.zeros_(self.u_mlp[-1].weight)
        nn.init.zeros_(self.u_mlp[-1].bias)
        scale = _chol_from_semantics(S_sem, alpha, lam, jitter)
        self.register_buffer("scale_tril", scale)
        self.register_buffer("sigma", torch.sqrt((scale**2).sum(dim=1)))
        self.prs_dist_enc = mlp_layers([d_prs, *prs_hidden, d_dist], p_drop)
        self.ehr_dist_enc = mlp_layers([d_ehr, *ehr_hidden, d_dist], p_drop)
        self.prs_shared_enc = mlp_layers([d_prs, *prs_hidden, d_shared], p_drop)
        self.ehr_shared_enc = mlp_layers([d_ehr, *ehr_hidden, d_shared], p_drop)
        self.fusion_enc = mlp_layers([2 * d_shared, d_shared, d_shared], p_drop)
        self.shared_head = mlp_layers([d_shared, d_shared, d_shared, 1], p_drop)
        self.dist_head_prs = mlp_layers([d_dist, d_dist, d_dist, 1], p_drop)
        self.dist_head_ehr = mlp_layers([d_dist, d_dist, d_dist, 1], p_drop)
        joint_dim = 2 * d_dist + cov_dim + d_shared
        self.final_head = mlp_layers([joint_dim, *final_hidden, 1], p_drop)

    def gate_parameters(self):
        offset = self.u_mlp(self.S_sem).squeeze(-1)
        return self.u0 + offset

    @torch.no_grad()
    def deterministic_mask(self):
        training = self.u_mlp.training
        self.u_mlp.eval()
        try:
            return self.gate_parameters().clamp(0, 1)
        finally:
            self.u_mlp.train(training)

    def forward(self, x_prs, x_ehr, x_covariates=None):
        mu = self.gate_parameters()
        eps = self.scale_tril @ torch.randn_like(mu) if self.training else 0
        mask = (mu + eps).clamp(0, 1)
        prs_shared = self.prs_shared_enc(x_prs)
        prs_dist = self.prs_dist_enc(x_prs)
        ehr_shared = self.ehr_shared_enc(x_ehr * (1 - mask))
        ehr_dist = self.ehr_dist_enc(x_ehr * mask)
        fused = self.fusion_enc(torch.cat([prs_shared, ehr_shared], dim=-1))
        joint = [fused, ehr_dist, prs_dist]
        if self.cov_dim:
            joint.append(x_covariates)
        return {
            "logit": self.final_head(torch.cat(joint, dim=-1)).squeeze(-1),
            "logit_shared": self.shared_head(fused).squeeze(-1),
            "logit_dist_prs": self.dist_head_prs(prs_dist).squeeze(-1),
            "logit_dist_ehr": self.dist_head_ehr(ehr_dist).squeeze(-1),
            "z_prs_shared": prs_shared,
            "z_prs_dist": prs_dist,
            "z_ehr_shared": ehr_shared,
            "z_ehr_dist": ehr_dist,
            "z_fused": fused,
            "mu": mu,
            "mask": mask.expand(len(x_ehr), -1),
        }
