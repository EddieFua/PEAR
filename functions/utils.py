import os, random
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    P = y_pred.sum()
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()

    precision = TP / max(P, 1)
    recall = TP / max((y_true == 1).sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
    best_f1 = np.nanmax(f1s) if f1s.size > 0 else float("nan")

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "precision@0.5": float(precision),
        "recall@0.5": float(recall),
        "f1@0.5": float(f1),
        "best_f1": float(best_f1),
    }
