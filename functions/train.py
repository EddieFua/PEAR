"""Train PEAR with inner validation and stratified outer cross-validation."""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions.dataset import (
    load_array,
    load_arrays,
    load_features,
    load_ids,
)
from functions.losses import pear_loss
from functions.models import OrthoFusion
from functions.preprocessing import FoldPreprocessor, IndexedDataset
from functions.utils import compute_metrics, set_seed, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("prs", "ehr", "y", "sem", "EID"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--semantic", help="Feature CSV, one row per original EHR column")
    p.add_argument("--split_info", help="Reuse an EID,y,fold CSV in input row order")
    p.add_argument("--covariates")
    p.add_argument("--out_dir", default="outputs/pear")
    for key, default in (
        ("seed", 1),
        ("epochs", 100),
        ("batch_size", 4096),
        ("kfolds", 5),
        ("patience", 5),
        ("threads", 1),
    ):
        p.add_argument("--" + key, type=int, default=default)
    for key, default in (
        ("dropout", 0.1),
        ("base_lr", 1e-3),
        ("weight_decay", 1e-3),
        ("lam_pred", 10.0),
        ("lam_align", 0.1),
        ("lam_orth", 200.0),
        ("lam_stg", 5e-5),
        ("prevalence", 0.01),
        ("inner_fraction", 0.2),
    ):
        p.add_argument("--" + key, type=float, default=default)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def make_loader(arrays, indices, pre, args, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed + 1)
    return DataLoader(
        IndexedDataset(arrays, indices, pre),
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    return np.concatenate(
        [
            torch.sigmoid(model(xp.to(device), xe.to(device), xc.to(device))["logit"])
            .cpu()
            .numpy()
            for xp, xe, _, xc in loader
        ]
    )


def model_config(arrays, sem, pre, args):
    return {
        "d_prs": arrays[0].shape[1],
        "d_ehr": len(pre.keep),
        "d_sem": sem.shape[1],
        "cov_dim": 0 if arrays[3] is None else arrays[3].shape[1],
        "p_drop": args.dropout,
    }


def fit_model(arrays, sem, train, validation, pre, args, seed):
    set_seed(seed)
    config = model_config(arrays, sem, pre, args)
    model = OrthoFusion(S_sem=sem[pre.keep], **config).to(args.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = make_loader(arrays, train, pre, args, shuffle=True)
    loader.generator.manual_seed(seed + 1)
    val_loader = make_loader(arrays, validation, pre, args)
    positive_fraction = float(arrays[2][train].mean())
    ap, an = 1 - positive_fraction, positive_fraction
    history, best_score, best_epoch, stale = [], -np.inf, 0, 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xp, xe, y, xc in loader:
            opt.zero_grad(set_to_none=True)
            output = model(xp.to(args.device), xe.to(args.device), xc.to(args.device))
            loss, _ = pear_loss(
                output,
                y.to(args.device),
                model.sigma,
                ap,
                an,
                lam_pred=args.lam_pred,
                lam_align=args.lam_align,
                lam_orth=args.lam_orth,
                lam_stg=args.lam_stg,
            )
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        scheduler.step()
        row = {"epoch": epoch, "loss": total / len(train)}
        score = compute_metrics(
            arrays[2][validation], predict_loader(model, val_loader, args.device)
        )["roc_auc"]
        row["inner_roc_auc"] = score
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
        history.append(row)
        print(row, flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), config, best_epoch, history


def make_splits(ids, y, args):
    if args.split_info:
        fold_id = pd.read_csv(args.split_info)["fold"].to_numpy()
    else:
        fold_id = np.zeros(len(y), dtype=int)
        cv = StratifiedKFold(args.kfolds, shuffle=True, random_state=args.seed)
        for fold, (_, test) in enumerate(cv.split(ids, y), 1):
            fold_id[test] = fold
    partitions = []
    for fold in range(1, args.kfolds + 1):
        outer_train = np.flatnonzero(fold_id != fold)
        test = np.flatnonzero(fold_id == fold)
        train, validation = train_test_split(
            outer_train,
            test_size=args.inner_fraction,
            stratify=y[outer_train],
            random_state=args.seed + fold,
        )
        partitions.append((train, validation, test))
    return pd.DataFrame({"EID": ids, "y": y, "fold": fold_id.astype(int)}), partitions


def run(args):
    torch.set_num_threads(args.threads)
    arrays = load_arrays(args.prs, args.ehr, args.y, args.covariates)
    prs, ehr, y, cov = arrays
    ids = load_ids(args.EID)
    sem = load_array(args.sem)
    semantic = load_features(args.semantic, ehr.shape[1])
    splits, partitions = make_splits(ids, y, args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out / "split_info.csv", index=False)
    semantic.to_csv(out / "ehr_feature_semantic.csv", index=False)
    write_json(out / "config.json", vars(args))
    oof = np.full(len(y), np.nan)
    masks, metrics = [], []
    for fold, (train, validation, test) in enumerate(partitions, 1):
        seed = args.seed + 10000 * fold
        print(f"Fold {fold}/{args.kfolds}", flush=True)
        pre = FoldPreprocessor(args.prevalence).fit(prs, ehr, cov, train)
        model, config, selected, history = fit_model(
            arrays,
            sem,
            train,
            validation,
            pre,
            args,
            seed,
        )
        oof[test] = predict_loader(
            model, make_loader(arrays, test, pre, args), args.device
        )
        mask = np.full(ehr.shape[1], np.nan)
        mask[pre.keep] = model.deterministic_mask().detach().cpu().numpy()
        masks.append(mask)
        feature_table = semantic.copy()
        feature_table["eligible"] = np.isfinite(mask)
        feature_table["gate"] = mask
        feature_table.to_csv(out / f"fold{fold}_features.csv", index=False)
        torch.save(
            {
                "format_version": 2,
                "model_config": config,
                "state_dict": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "preprocessor": pre.state_dict(),
                "features": semantic.feature.tolist(),
                "selected_epochs": selected,
                "fold": fold,
                "seed": seed,
            },
            out / f"fold{fold}_model.pt",
        )
        np.savez(
            out / f"fold{fold}_indices.npz",
            train=train,
            test=test,
            validation=validation,
        )
        write_json(out / f"fold{fold}_history.json", history)
        metrics.append(
            {
                "fold": fold,
                "selected_epochs": selected,
                **compute_metrics(y[test], oof[test]),
            }
        )
        del model
        gc.collect()
    frame = splits.rename(columns={"y": "y_true"}).copy()
    frame["y_prob"] = oof
    frame.to_csv(out / "oof_pred.csv", index=False)
    mask_array = np.stack(masks)
    count = np.isfinite(mask_array).sum(0)
    mean = np.divide(
        np.nansum(mask_array, axis=0),
        count,
        out=np.full(ehr.shape[1], np.nan),
        where=count > 0,
    )
    table = semantic.copy()
    table["mean_gate"], table["eligible_folds"] = mean, count
    table.to_csv(out / "feature_summary.csv", index=False)
    result = compute_metrics(y, oof)
    write_json(out / "metrics.json", {"overall": result, "folds": metrics})
    print(result, flush=True)
    return result


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
