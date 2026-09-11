"""Load aligned PRS, EHR, labels, and feature descriptions."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_array(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return archive["arr_0"]
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_arrays(prs_path, ehr_path, y_path, covariates_path=None):
    prs, ehr, y = [load_array(p) for p in (prs_path, ehr_path, y_path)]
    cov = load_array(covariates_path) if covariates_path else None
    return prs, ehr, y.reshape(-1), cov


def load_features(path, n):
    return (
        pd.read_csv(path, dtype=str, keep_default_na=False)
        if path
        else pd.DataFrame({"feature": [str(i) for i in range(n)]})
    )


def load_ids(path):
    ids = (
        np.load(path, allow_pickle=False)
        if Path(path).suffix == ".npy"
        else np.loadtxt(path, dtype=str, ndmin=1, comments=None)
    )
    return ids.reshape(-1).astype(str)
