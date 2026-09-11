"""Mean-pooled BioBERT embeddings with row-order and model provenance."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions.utils import sha256_file, write_json


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_csv", required=True)
    p.add_argument("--text_col", default="text")
    p.add_argument("--model", default="dmis-lab/biobert-base-cased-v1.1")
    p.add_argument(
        "--revision", default="main", help="Use a model commit hash for reproducibility"
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--out_sem", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local_files_only", action="store_true")
    args = p.parse_args()
    frame = pd.read_csv(args.in_csv, dtype=str, keep_default_na=False)
    path = Path(args.out_sem)
    from transformers import AutoModel, AutoTokenizer

    kwargs = {"revision": args.revision, "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    model = AutoModel.from_pretrained(args.model, **kwargs).to(args.device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            tokens = tokenizer(
                frame[args.text_col].iloc[start : start + args.batch_size].tolist(),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(args.device) for k, v in tokens.items()}
            vectors.append(
                mean_pool(model(**tokens).last_hidden_state, tokens["attention_mask"])
                .cpu()
                .numpy()
            )
    embeddings = np.concatenate(vectors).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    write_json(
        path.with_suffix(".json"),
        {
            "model": args.model,
            "requested_revision": args.revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "pooling": "attention-mask mean, including special tokens",
            "max_length": args.max_length,
            "input_sha256": sha256_file(args.in_csv),
            "embedding_sha256": sha256_file(path),
            "shape": list(embeddings.shape),
            "text_column": args.text_col,
        },
    )
    print(f"Saved {path}: {embeddings.shape}")


if __name__ == "__main__":
    main()
