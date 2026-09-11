"""Reproducibility, metrics, and strict JSON output."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def compute_metrics(y_true, y_prob):
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    both = len(np.unique(y)) == 2
    roc = float(roc_auc_score(y, p)) if both else None
    ap = float(average_precision_score(y, p)) if both else None
    return {"roc_auc": roc, "pr_auc": ap, "n": len(y), "cases": int(y.sum())}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
