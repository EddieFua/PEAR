"""Fit PRS scaling and EHR prevalence filtering on the training partition."""

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class FoldPreprocessor:
    def __init__(self, prevalence=0.01):
        self.prevalence = prevalence

    def fit(self, prs, ehr, cov, indices):
        indices = np.asarray(indices)
        self.prs = MinMaxScaler()
        counts = np.zeros(ehr.shape[1], dtype=np.float64)
        for start in range(0, len(indices), 4096):
            rows = indices[start : start + 4096]
            self.prs.partial_fit(np.asarray(prs[rows], dtype=np.float32))
            counts += ehr[rows].sum(axis=0)
        self.keep = np.flatnonzero(counts / len(indices) >= self.prevalence)
        self.d_prs, self.d_ehr = prs.shape[1], ehr.shape[1]
        self.d_cov = 0 if cov is None else cov.shape[1]
        return self

    def transform(self, prs, ehr, cov=None):
        xp = np.asarray(prs, dtype=np.float32) * self.prs.scale_ + self.prs.min_
        xe = np.asarray(ehr[..., self.keep], dtype=np.float32)
        xc = (
            np.asarray(cov, dtype=np.float32)
            if self.d_cov
            else np.empty((*prs.shape[:-1], 0), dtype=np.float32)
        )
        return xp, xe, xc

    def state_dict(self):
        return {
            "prevalence": self.prevalence,
            "d_prs": self.d_prs,
            "d_ehr": self.d_ehr,
            "d_cov": self.d_cov,
            "keep": torch.tensor(self.keep),
            "prs_scale": torch.tensor(self.prs.scale_),
            "prs_min": torch.tensor(self.prs.min_),
        }

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state["prevalence"])
        obj.d_prs, obj.d_ehr = state["d_prs"], state["d_ehr"]
        obj.d_cov = state["d_cov"]
        obj.keep = state["keep"].cpu().numpy()
        obj.prs = MinMaxScaler()
        obj.prs.scale_ = state["prs_scale"].cpu().numpy()
        obj.prs.min_ = state["prs_min"].cpu().numpy()
        return obj


class IndexedDataset(Dataset):
    def __init__(self, arrays, indices, preprocessor):
        self.prs, self.ehr, self.y, self.cov = arrays
        self.indices = np.asarray(indices)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        xp, xe, xc = self.preprocessor.transform(
            self.prs[i], self.ehr[i], None if self.cov is None else self.cov[i]
        )
        return xp, xe, np.float32(self.y[i]), xc
