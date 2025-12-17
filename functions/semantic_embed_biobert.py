
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python /home/yinghaofu2/augmented_prs/py_function_alternative/semantic_embed_biobert.py \
  --in_csv /home/yinghaofu2/augmented_prs/data_prepare/clean_data/T2D/semantic.csv \
  --text_col text \
  --batch_size 64 \
  --model dmis-lab/biobert-base-cased-v1.1 \
  --out_sem /home/yinghaofu2/augmented_prs/data_prepare/clean_data/T2D/sem_emb.npy
"""
import argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summ = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summ / denom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--text_col", default="description")
    ap.add_argument("--model", default="dmis-lab/biobert-v1.1")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_sem", default="sem_emb.npy")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"text column {args.text_col} not found")
    texts = df[args.text_col].fillna("").astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()

    vecs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch = texts[i:i+args.batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            vec = mean_pool(out.last_hidden_state, enc["attention_mask"])  # [B, H]
            vecs.append(vec.cpu())

    sem = torch.cat(vecs, dim=0).numpy().astype("float32")
    np.save(args.out_sem, sem)
    print("Saved:", args.out_sem, sem.shape)

if __name__ == "__main__":
    main()
