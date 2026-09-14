# PEAR

PEAR is an interpretable deep learning method for PRS- and EHR-augmented risk prediction.

This repository provides R scripts to prepare the data and Python code to generate semantic vectors, train and evaluate PEAR, and export EHR feature gates.

![PEAR workflow](figure/pipeline.jpg)

## Repository structure

1. `data_prepare/`: R scripts for phenotype-specific data preparation and PRS calculation.
2. `functions/`: Python code for semantic vectors, datasets, models, losses, training, and prediction.
3. `reference/`: Phecode definitions and reference table instructions.

## Installation

### Clone the repository

```bash
git clone https://github.com/EddieFua/PEAR.git
cd PEAR
```

### Install Python packages

Use Python 3.10 or newer. The commands below use Bash or zsh.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 'transformers>=4.40,<5'
```

Choose a PyTorch build that matches your CPU or CUDA setup. The tutorial runs on the CPU. To use an NVIDIA GPU, set `--device cuda`. You only need `transformers` to generate semantic vectors.

### Install R packages

If you plan to use the data preparation scripts, install these packages in R:

```r
install.packages(c(
  "data.table", "bigreadr", "bigsnpr", "bigstatsr", "RcppCNPy", "tidyr"
))
```

## Files prepared for PEAR

Keep all sample-level arrays and IDs in the same order. Save numeric arrays as `.npy` files or as `.npz` files containing an array named `arr_0`. All numeric values must be finite.

### Required files

1. PRS matrix `X_prs`: shape `[N, d_prs]`, before scaling.
2. EHR matrix `X_ehr`: shape `[N, d_ehr]`, with the target disease and its exclusion-range features removed.
3. Label vector `y`: shape `[N]` or `[N, 1]`, with values `0` or `1`.
4. Sample IDs `EID`: `N` unique IDs, as a text file with one ID per line and no header, or a `.npy` file.
5. Semantic vectors `S_sem`: shape `[d_ehr, d_sem]`, with nonzero rows in EHR feature order.

### Optional files

- Covariate matrix `X_cov`: shape `[N, cov_dim]`. The R scripts export age, sex, and 20 genetic principal components.
- `semantic.csv`: columns `feature,text`, with one row per EHR column. This file is used to generate semantic vectors and include feature names in the training outputs. If you already have semantic vectors, it is optional for training.

Use unique feature IDs and nonempty descriptions. Keep the same EHR column order in `X_ehr`, `semantic.csv`, and `S_sem`.

## Tutorial

Run all commands from the repository root with the Python environment activated. This tutorial uses atrial fibrillation and the filenames produced by its R scripts. If your arrays are already prepared, skip Step 1 and use your own paths below.

### Step 1. Prepare UKB data

Open the two scripts in `data_prepare/Atrial_fibrillation/` and set their input paths:

| Setting | Input |
|---|---|
| `fp` | UKB phenotype CSV, in both scripts |
| `kinship_fp` | Relatedness table with `ID1`, `ID2`, and `Kinship` |
| `map_fp` | ICD-10-to-phecode mapping table, described in [reference/README.md](reference/README.md) |
| `sumstats_fp` | The phenotype's GWAS summary statistics in GRCh37 coordinates |
| `mfi_pattern`, `bgen_pattern` | Variant metadata and genotype paths for chromosomes 1–22, with `%d` standing for the chromosome number |
| `sample_fp` | PSAM file with an `IID` column in the same sample order as every BGEN file |
| `out_dir` | The same output directory in both scripts |

The phenotype CSV must contain `eid`, paired `41270-*` diagnosis and `41280-*` date columns, `21000-*` ancestry fields, `21022-0.0` (age), `22001-0.0` (sex), and `22009-0.1` through `22009-0.20` (genetic PCs).

The repository includes phecode definitions. You will need to provide the mapping table and participant data separately. Keep `semantic_fp` pointing to the definitions table, and set `NCORES` to the number of cores to use for PRS computation. If you use a different GWAS source, adjust the column selection and effect-value conversion in the PRS script to match your file.

```bash
Rscript data_prepare/Atrial_fibrillation/1_clear_feature.R
Rscript data_prepare/Atrial_fibrillation/2_compute_prs.R
```

The first script creates `feature.RData`. The second calculates 4,200 PRS and exports the following files to `outputs/Atrial_fibrillation/data/` by default:

- `multi_PRS.npy`
- `ukb_final_features.npy`
- `ukb_final_y.npy`
- `ukb_final_covariates.npy`
- `ukb_final_ids.txt`
- `semantic.csv`

### Step 2. Create semantic vectors

Set `DATA_DIR` to the directory containing your prepared data. Skip this step if you already have semantic vectors in EHR column order.

```bash
DATA_DIR="outputs/Atrial_fibrillation/data"
python functions/semantic_embed_biobert.py \
  --in_csv "$DATA_DIR/semantic.csv" \
  --text_col text \
  --model dmis-lab/biobert-base-cased-v1.1 \
  --batch_size 32 \
  --out_sem "$DATA_DIR/sem_emb.npy" --device cpu
```

The script downloads BioBERT on the first run and averages the token representations to create one vector per EHR feature. It saves the vectors in `sem_emb.npy` with shape `[d_ehr, 768]` and records the model and pooling details in `sem_emb.json`. Change `--text_col` if your description column has a different name. To use a specific model version, pass its commit hash with `--revision`.

### Step 3. Train PEAR

```bash
DATA_DIR="outputs/Atrial_fibrillation/data"
RUN_DIR="outputs/Atrial_fibrillation/pear"
python functions/train.py \
  --prs "$DATA_DIR/multi_PRS.npy" \
  --ehr "$DATA_DIR/ukb_final_features.npy" \
  --y "$DATA_DIR/ukb_final_y.npy" \
  --covariates "$DATA_DIR/ukb_final_covariates.npy" \
  --semantic "$DATA_DIR/semantic.csv" \
  --sem "$DATA_DIR/sem_emb.npy" \
  --EID "$DATA_DIR/ukb_final_ids.txt" \
  --out_dir "$RUN_DIR" \
  --epochs 100 --batch_size 4096 --kfolds 5 --seed 1 --device cpu
```

Leave out `--covariates` if you do not have covariates. You can also leave out `--semantic` if you do not need feature names in the results. The script will use numeric feature IDs instead. Use a separate output directory for each run to avoid overwriting earlier results.

For each outer fold, 20% of the training samples are held out for inner validation. PRS min-max scaling and the 1% EHR prevalence filter are fitted using only the inner training samples. Covariates are used without rescaling. Training saves the checkpoint with the highest inner validation ROC-AUC and stops after five epochs without improvement. That checkpoint is then used to predict the outer test fold. Each training, validation, and test set must contain both cases and controls.

Use `--inner_fraction`, `--prevalence`, and `--patience` to change the validation fraction, prevalence threshold, and early stopping patience. For repeated runs, change `--seed` and `--out_dir`. To reuse outer folds, pass a saved `split_info.csv` with `--split_info`. Its rows must follow the current input sample order, with fold IDs from `1` to `kfolds`. Run `python functions/train.py --help` to see all options.

### Step 4. Read the results

Files are saved in `RUN_DIR`:

| File | Contents |
|---|---|
| `oof_pred.csv` | `EID,y_true,fold,y_prob` in input sample order, with one out-of-fold prediction per sample |
| `metrics.json` | Pooled and per-fold ROC-AUC and PR-AUC (average precision) |
| `fold*_model.pt` | Selected model weights, semantic vectors, preprocessing state, and feature IDs |
| `fold*_features.csv` | `eligible` and `gate` for each EHR feature in its original order. Filtered features have blank gates |
| `feature_summary.csv` | `mean_gate` across eligible folds and `eligible_folds` for each feature |
| `ehr_feature_semantic.csv` | Feature table in the original EHR column order |
| `config.json`, `split_info.csv`, `fold*_indices.npz`, `fold*_history.json` | Settings, outer fold assignments, zero-based train/validation/test row indices, and training history |

Larger gate values send more of an EHR feature to the distinct channel, while smaller values send more to the shared channel. The mean across folds provides a descriptive summary. If you evaluate feature selection on a fold's held-out samples, select features using that fold's own gates.

### Step 5. Predict for new samples

Prepare PRS, EHR, covariates, and IDs for the new samples. Keep the feature columns in the same order as the training inputs, including any EHR columns that were removed by the fold's prevalence filter. Set `NEW_DIR` to the directory containing these files:

```bash
NEW_DIR="/path/to/new_samples"
python functions/predict.py \
  --checkpoint outputs/Atrial_fibrillation/pear/fold1_model.pt \
  --prs "$NEW_DIR/prs.npy" \
  --ehr "$NEW_DIR/ehr.npy" \
  --covariates "$NEW_DIR/covariates.npy" \
  --EID "$NEW_DIR/ids.txt" \
  --out outputs/Atrial_fibrillation/new_predictions.csv --device cpu
```

Include `--covariates` only if the model was trained with covariates. The checkpoint stores the semantic vectors, EHR filter, and PRS scaling, so you do not need labels, semantic vectors, or a feature CSV for prediction. Keep new PRS values on the same input scale as the original training files.

The output contains `EID,y_prob` in the same order as the input samples. Predictions come from the selected fold model. The scores are uncalibrated sigmoid outputs between 0 and 1.

## Contact

For questions, open an [issue](https://github.com/EddieFua/PEAR/issues), send an [email](mailto:yinghao.fu@my.cityu.edu.hk), or visit my [personal homepage](https://eddiefua.github.io/).
