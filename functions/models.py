# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(in_dim, hid, out_dim, p=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hid),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
        nn.LayerNorm(hid),
        nn.Linear(hid, hid),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
        nn.LayerNorm(hid),
        nn.Linear(hid, out_dim),
    )

@torch.no_grad()
def _chol_from_semantics(S_sem: torch.Tensor, alpha: float, lam: float, jitter: float) -> torch.Tensor:
    S = F.normalize(S_sem, dim=1)                       # [F, d] 单位化
    K = S @ S.T                                         # [F, F] 余弦 Gram，PSD
    Fnum = K.shape[0]
    Sigma = alpha * K + lam * torch.eye(Fnum, device=K.device, dtype=K.dtype)
    Sigma = 0.5 * (Sigma + Sigma.T)                     # 数值对称
    diag_mean = torch.mean(torch.diag(Sigma))
    Sigma = Sigma + (jitter * diag_mean + 1e-8) * torch.eye(Fnum, device=K.device, dtype=K.dtype)
    scale_tril = torch.linalg.cholesky(Sigma)           # [F, F]
    return scale_tril

class OrthoFusion(nn.Module):
    def __init__(self, d_prs, d_ehr, d_sem, S_sem,
                 d_latent=64, d_shared=64, d_dist=64,
                 p_drop=0.1,
                 alpha=1.0, lam=0.05, jitter=1e-3,
                 gamma_init=1.0,
                 hard_gate=False,
                 cov_dim=22,
                 delta_scale=5e-3):
        super().__init__()
        assert S_sem.shape == (d_ehr, d_sem), f"S_sem expect [{d_ehr}, {d_sem}] but got {tuple(S_sem.shape)}"
        self.d_ehr = d_ehr
        self.hard_gate = hard_gate
        self.delta_scale = float(delta_scale)

        # ---------- Encoders & heads ----------
        self.prs_shared_enc = mlp(d_prs,  d_latent, d_shared, p=p_drop)
        self.prs_dist_enc = mlp(d_prs,  d_latent, d_shared, p=p_drop)
        self.ehr_shared_enc = mlp(d_ehr,  d_latent, d_shared, p=p_drop)
        self.ehr_dist_enc   = mlp(d_ehr,  d_latent, d_dist,   p=p_drop)
        self.shared_head    = mlp(d_shared, d_latent, 1, p=p_drop)
        self.dist_head_prs      = mlp(d_dist,   d_latent, 1, p=p_drop)
        self.dist_head_ehr      = mlp(d_dist,   d_latent, 1, p=p_drop)
        self.use_covariates = (cov_dim is not None) and (cov_dim > 0)
        if self.use_covariates:
            self.covariate_head = mlp(cov_dim, 12, 1, p=p_drop)

        self.u0 = nn.Parameter(torch.full((d_ehr,), 0.5))
        self.u_mlp = nn.Sequential( 
            nn.LayerNorm(d_sem),
            nn.Linear(d_sem, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        nn.init.zeros_(self.u_mlp[1].weight)
        nn.init.zeros_(self.u_mlp[-1].bias)
        self.register_buffer("gamma", torch.tensor(float(gamma_init)))
        S_sem = torch.as_tensor(S_sem, dtype=torch.float32)
        self.register_buffer("S_sem", S_sem)
        scale_tril = _chol_from_semantics(S_sem, alpha=alpha, lam=lam, jitter=jitter)
        self.register_buffer("scale_tril", scale_tril)  # [F, F]
        diag_var = torch.sum(self.scale_tril**2, dim=1)     # diag(Σ)
        self.register_buffer("sigma_base", torch.sqrt(diag_var + 1e-8))  # [F]

    @property
    def sigma(self) -> torch.Tensor:
        return self.gamma * self.sigma_base

    @torch.no_grad()
    def set_gamma(self, val: float):
        self.gamma.fill_(float(val))

    def _sample_eps(self) -> torch.Tensor:
        z = torch.randn(self.d_ehr, device=self.scale_tril.device, dtype=self.scale_tril.dtype)
        return self.scale_tril @ z  # [F]

    def forward(self, x_prs, x_ehr, x_covariates=None):
        B = x_ehr.size(0)
        delta_u = torch.tanh(self.u_mlp(self.S_sem)).squeeze(-1) * self.delta_scale 
        u_eff = self.u0 + delta_u              

        if self.training:
            eps = self._sample_eps()                          # [F]
            z_pre = u_eff + float(self.gamma.item()) * eps
        else:
            z_pre = u_eff                           
        z = torch.clamp(z_pre, 0.0, 1.0)
        # z = torch.sigmoid(z_pre)

        mB = z.view(1, -1).expand(B, -1)
        z_prs_shared = self.prs_shared_enc(x_prs)
        x_shared = x_ehr * (1.0 - mB)
        x_dist   = x_ehr * mB
        z_ehr_shared = self.ehr_shared_enc(x_shared)
        z_ehr_dist   = self.ehr_dist_enc(x_dist)
        z_fused = z_prs_shared + z_ehr_shared
        z_prs_dist = self.prs_dist_enc(x_prs)

        logit_shared = self.shared_head(z_fused).squeeze(-1)
        logit_dist_ehr  = self.dist_head_ehr(z_ehr_dist).squeeze(-1)
        logit_dist_prs = self.dist_head_prs(z_prs_dist).squeeze(-1)
        if self.use_covariates and (x_covariates is not None):
            logit_covariates = self.covariate_head(x_covariates).squeeze(-1)
        else:
            logit_covariates = 0.0
        logit = logit_shared + logit_covariates + logit_dist_ehr + logit_dist_prs

        return {
            "logit": logit,
            "logit_shared": logit_shared,
            "logit_dist_ehr": logit_dist_ehr,
            "logit_dist_prs": logit_dist_prs,
            "logit_covariates": logit_covariates,
            "z_prs_shared": z_prs_shared,
            "z_prs_dist": z_prs_dist,
            "z_ehr_shared": z_ehr_shared,
            "z_ehr_dist": z_ehr_dist,
            "z_fused": z_fused,
            "mask": mB,
            "mu": u_eff,
            "mu_sig": torch.sigmoid(u_eff),
        }
