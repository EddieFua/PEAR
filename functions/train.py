import os, argparse, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from dataset import ArrayDataset, load_arrays, load_semantic_embeddings
from models import OrthoFusion
from losses import focal_loss, cosine_align, cross_covariance_penalty, stg_regularizer
from utils import set_seed, compute_metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
sc_prs = MinMaxScaler()
sc_ehr = MinMaxScaler()
sc_cov = MinMaxScaler()

def infer_focal_params(y_tr: np.ndarray):
    y_tr = y_tr.reshape(-1)
    pos_frac = float((y_tr == 1).mean() if set(np.unique(y_tr)) <= {0, 1} else y_tr.mean())
    neg_frac = 1.0 - pos_frac
    alpha_pos = neg_frac
    alpha_neg = pos_frac

    imb = min(pos_frac, neg_frac)
    if imb < 0.01:
        gamma = 3.5
    elif imb < 0.05:
        gamma = 3.0
    elif imb < 0.10:
        gamma = 2.5
    else:
        gamma = 2.0
    return alpha_pos, alpha_neg, gamma, pos_frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prs", required=True)
    parser.add_argument("--ehr", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--covariates", required=False)
    parser.add_argument("--semantic", required=False)
    parser.add_argument("--sem", required=True)
    parser.add_argument("--EID", required=True, help="Path to sample ID file (same order as y)")
    parser.add_argument("--out_dir", default="./outputs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--kfolds", type=int, default=5)
    # Model dims
    parser.add_argument("--d_latent", type=int, default=128)
    parser.add_argument("--d_shared", type=int, default=32)
    parser.add_argument("--d_dist", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Regularization
    parser.add_argument("--lam_pred", type=float, default=10)
    parser.add_argument("--lam_align", type=float, default=1e-1)
    parser.add_argument("--lam_orth", type=float, default=2e2)
    parser.add_argument("--lam_stg", type=float, default=5e-5)
    parser.add_argument("--lam_counterfactual", type=float, default=1)
    parser.add_argument("--alpha_pos", type=float, default=0.25)
    parser.add_argument("--alpha_neg", type=float, default=0.75)
    parser.add_argument("--gamma_focal", type=float, default=2.0)
    parser.add_argument("--base_lr", type=float, default=1e-3)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = args.device
    X_prs, X_ehr, y, X_cov = load_arrays(args.prs, args.ehr, args.y, args.covariates)
    S_sem = load_semantic_embeddings(args.sem)
    semantic = pd.read_csv(args.semantic)
    if args.EID.endswith(".npy"):
        sample_ids = np.load(args.EID)
    else:
        sample_ids = np.loadtxt(args.EID, dtype=str)
    sample_ids = np.asarray(sample_ids).reshape(-1)
    idx = (np.sum(X_ehr, axis=0) / X_ehr.shape[0] >= 0.01)
    X_ehr = X_ehr[:, idx]
    S_sem = S_sem[idx, :]
    semantic = semantic[idx]
    semantic.to_csv(os.path.join(args.out_dir, "ehr_feature_semantic.csv"), index=False)

    N, d_prs = X_prs.shape
    _, d_ehr = X_ehr.shape
    d_sem = S_sem.shape[1]
    print(f"PRS dim: {d_prs}, EHR dim: {d_ehr}, Semantic dim: {d_sem}")
    X_prs = sc_prs.fit_transform(X_prs)
    X_ehr = sc_ehr.fit_transform(X_ehr)
    if X_cov is not None:
        X_cov = sc_cov.fit_transform(X_cov)

    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
    y_true_all, y_prob_all, masks_per_fold, fold_auc = [], [], [], []
    ids_all = []

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"Fold {fold}/{args.kfolds}")
        print(len(tr), "train samples;", len(va), "validation samples.")
        print(f"The number of cases in train: {int(y[tr].sum())}, val: {int(y[va].sum())}")
        print(
            f"The number of controls in train: {len(tr) - int(y[tr].sum())}, "
            f"val: {len(va) - int(y[va].sum())}"
        )

        ds_tr = ArrayDataset(
            X_prs[tr], X_ehr[tr], y[tr], X_cov[tr] if X_cov is not None else None
        )
        ds_va = ArrayDataset(
            X_prs[va], X_ehr[va], y[va], X_cov[va] if X_cov is not None else None
        )
        ld_tr = DataLoader(
            ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False
        )
        ld_va = DataLoader(
            ds_va, batch_size=4096, shuffle=False, num_workers=0, drop_last=False
        )

        model = OrthoFusion(
            d_prs=d_prs,
            d_ehr=d_ehr,
            d_sem=d_sem,
            S_sem=S_sem,
            d_latent=args.d_latent,
            d_shared=args.d_shared,
            d_dist=args.d_dist,
            p_drop=args.dropout,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        alpha_pos, alpha_neg, gamma_focal, pos_frac = infer_focal_params(y[tr])
        args.alpha_pos = alpha_pos
        args.alpha_neg = alpha_neg
        args.gamma_focal = gamma_focal
        best_pr, best_state, patience = -1.0, None, 0

        # -------- Training loop --------
        for ep in range(1, args.epochs + 1):
            model.train()
            total_loss = task_accum = aux_accum = align_accum = orth_accum = sparse_accum = 0.0

            for xprs, xehr, yy, xcov in ld_tr:
                xprs = xprs.to(device)
                xehr = xehr.to(device)
                yy = yy.to(device)
                xcov = xcov.to(device) if xcov is not None else None

                opt.zero_grad(set_to_none=True)
                out = model(xprs, xehr, xcov)

                loss_task = args.lam_pred * focal_loss(
                    out["logit"],
                    yy,
                    alpha_pos=args.alpha_pos,
                    alpha_neg=args.alpha_neg,
                    gamma=args.gamma_focal,
                )
                loss_aux = args.lam_pred * (
                    focal_loss(
                        out["logit_dist_ehr"],
                        yy,
                        alpha_pos=args.alpha_pos,
                        alpha_neg=args.alpha_neg,
                        gamma=args.gamma_focal,
                    )
                    + focal_loss(
                        out["logit_shared"],
                        yy,
                        alpha_pos=args.alpha_pos,
                        alpha_neg=args.alpha_neg,
                        gamma=args.gamma_focal,
                    )
                    + focal_loss(
                        out["logit_covariates"],
                        yy,
                        alpha_pos=args.alpha_pos,
                        alpha_neg=args.alpha_neg,
                        gamma=args.gamma_focal,
                    )
                    + focal_loss(
                        out["logit_dist_prs"],
                        yy,
                        alpha_pos=args.alpha_pos,
                        alpha_neg=args.alpha_neg,
                        gamma=args.gamma_focal,
                )) / 4.0

                loss_align = args.lam_align * cosine_align(out["z_prs_shared"], out["z_ehr_shared"])
                z_fused = torch.cat([out["z_prs_shared"], out["z_ehr_shared"]], dim=-1)
                loss_orth = args.lam_orth * (cross_covariance_penalty(z_fused, out["z_ehr_dist"])+ cross_covariance_penalty(z_fused, out["z_prs_dist"]))/2.0
                loss_stg = stg_regularizer(out["mu"], sigma=model.sigma, lam=args.lam_stg)

                total_loss_batch = (
                    loss_task + loss_aux + loss_align + loss_orth + loss_stg
                )
                total_loss_batch.backward()
                opt.step()

                total_loss += float(total_loss_batch.item())
                task_accum += float(loss_task.item())
                aux_accum += float(loss_aux.item())
                align_accum += float(loss_align.item())
                orth_accum += float(loss_orth.item())
                sparse_accum += float(loss_stg.item())

            sch.step()

            # -------- Validation (for early stopping) --------
            model.eval()
            with torch.no_grad():
                y_true, y_prob = [], []
                for xprs, xehr, yy, xcov in ld_va:
                    xprs = xprs.to(device)
                    xehr = xehr.to(device)
                    yy = yy.to(device)
                    xcov = xcov.to(device) if xcov is not None else None
                    out = model(xprs, xehr, xcov)
                    prob = torch.sigmoid(out["logit"]).cpu().numpy()
                    y_true.append(yy.cpu().numpy())
                    y_prob.append(prob)

                y_true = np.concatenate(y_true)
                y_prob = np.concatenate(y_prob)
                metrics = compute_metrics(y_true, y_prob)
                pr_auc = metrics["pr_auc"]
                auc = metrics["roc_auc"]

            print(
                f"Epoch {ep}: Total Loss={total_loss/len(ld_tr):.5f} "
                f"Task = {task_accum/len(ld_tr):.5f} Aux = {aux_accum/len(ld_tr):.5f} "
                f"Align={align_accum/len(ld_tr):.5f} Orth={orth_accum/len(ld_tr):.5f} "
                f"STG Reg={sparse_accum/len(ld_tr):.5f} | "
                f"Va PR AUC={pr_auc:.4f} ROC AUC={metrics['roc_auc']:.4f}"
            )
            print(
                f"Mask: min: {out['mask'].min().item():.4f}, "
                f"max: {out['mask'].max().item():.4f}, "
                f"mean: {out['mask'].mean().item():.4f}"
            )

            if pr_auc > best_pr:
                best_pr = pr_auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    print(f"Early stopping at epoch {ep}.")
                    break


        # -------- Re-evaluate best model on val & collect OOF --------
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            y_true, y_prob = [], []
            for xprs, xehr, yy, xcov in ld_va:
                xprs = xprs.to(device)
                xehr = xehr.to(device)
                yy = yy.to(device)
                xcov = xcov.to(device) if xcov is not None else None
                out = model(xprs, xehr, xcov)
                prob = torch.sigmoid(out["logit"]).cpu().numpy()
                y_true.append(yy.cpu().numpy())
                y_prob.append(prob)

            y_true = np.concatenate(y_true)
            y_prob = np.concatenate(y_prob)
            auc = float(compute_metrics(y_true, y_prob)["roc_auc"])

            mask_est = out["mask"].mean(dim=0).detach().cpu().numpy()
            np.save(os.path.join(args.out_dir, f"fold{fold}_mask.npy"), mask_est)

        ids_fold = sample_ids[va]

        torch.save(best_state, os.path.join(args.out_dir, f"fold{fold}_model.pt"))

        y_true_all.append(y_true)
        y_prob_all.append(y_prob)
        ids_all.append(ids_fold)
        masks_per_fold.append(mask_est)
        fold_auc.append(auc)

    # -------- Aggregate across folds --------
    w = np.asarray(fold_auc)
    w = w / (w.sum() + 1e-8)
    normalized_masks_per_fold = (np.stack(masks_per_fold, axis=0)- np.stack(masks_per_fold, axis=0).min())/(np.stack(masks_per_fold, axis=0).max()- np.stack(masks_per_fold, axis=0).min()+1e-8)
    weighted_mask = np.average(np.stack(normalized_masks_per_fold, axis=0), axis=0, weights=w)
    np.save(os.path.join(args.out_dir, "weighted_mask.npy"), weighted_mask)

    y_true_concat = np.concatenate(y_true_all)
    y_prob_concat = np.concatenate(y_prob_all)
    ids_concat = np.concatenate(ids_all)

    overall = compute_metrics(y_true_concat, y_prob_concat)
    print("Overall metrics:", overall)

    np.save(os.path.join(args.out_dir, "oof_true.npy"), y_true_concat)
    np.save(os.path.join(args.out_dir, "oof_prob.npy"), y_prob_concat)

    df = pd.DataFrame(
        {
            "EID": ids_concat,
            "y_true": y_true_concat.astype(np.int64),
            "y_prob": y_prob_concat.astype(np.float64),
        }
    )
    df.to_csv(os.path.join(args.out_dir, "oof_pred.csv"), index=False)


if __name__ == "__main__":
    main()
