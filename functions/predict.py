"""Apply one saved PEAR fold to new samples with training-time preprocessing."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions.dataset import load_array, load_ids
from functions.models import OrthoFusion
from functions.preprocessing import FoldPreprocessor


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = OrthoFusion(
        S_sem=checkpoint["state_dict"]["S_sem"], **checkpoint["model_config"]
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return (
        model.to(device).eval(),
        FoldPreprocessor.from_state_dict(checkpoint["preprocessor"]),
        checkpoint,
    )


@torch.no_grad()
def predict_arrays(model, pre, prs, ehr, cov=None, batch_size=4096, device="cpu"):
    model.eval()
    result = []
    for i in range(0, len(prs), batch_size):
        rows = slice(i, i + batch_size)
        xp, xe, xc = pre.transform(
            prs[rows], ehr[rows], None if cov is None else cov[rows]
        )
        output = model(
            *(
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (xp, xe, xc)
            )
        )
        result.append(torch.sigmoid(output["logit"]).cpu().numpy())
    return np.concatenate(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "prs", "ehr", "EID", "out"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--covariates")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    model, pre, _ = load_checkpoint(args.checkpoint, args.device)
    prs, ehr = [load_array(x) for x in (args.prs, args.ehr)]
    cov = load_array(args.covariates) if args.covariates else None
    ids = load_ids(args.EID)
    out = Path(args.out)
    probs = predict_arrays(model, pre, prs, ehr, cov, args.batch_size, args.device)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"EID": ids, "y_prob": probs}).to_csv(out, index=False)


if __name__ == "__main__":
    main()
