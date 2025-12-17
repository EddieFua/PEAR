import numpy as np
import torch
from torch.utils.data import Dataset

class ArrayDataset(Dataset):
    def __init__(self, X_prs, X_ehr, y, covariates):
        assert len(X_prs) == len(X_ehr) == len(y)
        self.X_prs = torch.from_numpy(X_prs).float()
        self.X_ehr = torch.from_numpy(X_ehr).float()
        self.y = torch.from_numpy(y).float()
        self.covariates = torch.from_numpy(covariates).float() if covariates is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_prs[i], self.X_ehr[i], self.y[i], self.covariates[i] if self.covariates is not None else None

def load_arrays(prs_path, ehr_path, y_path, covariates_path=None):
    def load_any(p):
        kwargs = {}
        if not p.endswith(".npz"):
            kwargs['mmap_mode'] = 'r'
        arr = np.load(p, **kwargs)
        if isinstance(arr, np.lib.npyio.NpzFile):  # .npz
            return arr["arr_0"]
        return arr

    X_prs = load_any(prs_path)
    X_ehr = load_any(ehr_path)
    X_covariates = load_any(covariates_path) if covariates_path is not None else None
    y = load_any(y_path).reshape(-1)
    return X_prs, X_ehr, y, X_covariates

def load_semantic_embeddings(path):
    arr = np.load(path, mmap_mode='r')
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr["arr_0"]
    return arr
