# PEAR

PEAR is an interpretable deep learning method for PRS- and EHR-augmented risk prediction.

This repository contains R scripts for data preparation and Python scripts for semantic vectors, model fit, evaluation, and EHR feature gate export.

![Pipeline](figure/pipeline.jpg)

## Repository structure

1. `data_prepare`  
   R scripts for phenotype specific data preparation and PRS calculation

2. `functions`  
   Python code for semantic vectors, datasets, model, loss, metrics, and model fit

3. `reference`  
   Phecode mapping tables

## Installation

### Clone the repository

```bash
git clone https://github.com/EddieFua/PEAR.git
cd PEAR
```

### Install Python packages

```bash
pip install numpy pandas scikit-learn scipy tqdm transformers pyliftover
```
PyTorch install depends on your CUDA or CPU setup.

### Install R packages

```r
install.packages(c(
  "data.table",
  "stringr",
  "dplyr",
  "tidyr",
  "glue",
  "RcppCNPy"
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
BiocManager::install(c(
  "bigsnpr",
  "bigstatsr",
  "bigreadr"
))
```


## Files prepared for PEAR
All sample level arrays must share the same sample order.
### Required files
1. PRS matrix `X_prs` with shape `[N, d_prs]`
2. EHR matrix `X_ehr` with shape `[N, d_ehr]`
3. Label vector `y` with shape `[N]` with values `0` or `1`
4. Sample id file `EID` with length `N`  
   It can be a text file with one id per line or a `.npy` file
5. Semantic vector matrix `S_sem` with shape `[d_ehr, d_sem]`
6. Semantic feature table `semantic.csv` with `d_ehr` rows  
   Each row maps to one EHR feature in `X_ehr` and one row in `S_sem`
### Optional files
Covariate matrix `X_cov` with shape `[N, cov_dim]`
### Clean UKB data

We provide the complete R scripts and reference files used to clean the UKB data for each target phenotype. The scripts are in the `data_prepare` folder, and the reference files are in the `reference` folder.



## Quick start
###Step 1 Create semantic vectors for EHR features
```bash
cd functions

python semantic_embed_biobert.py \
  --in_csv /path/to/semantic.csv \
  --text_col description \
  --model dmis-lab/biobert-v1.1 \
  --batch_size 32 \
  --out_sem /path/to/sem_emb.npy
```
**Output**
1. `sem_emb.npy` with shape `[d_ehr, hidden_dim]`

###Step 2 Fit the fusion model with K fold split
```bash
cd functions

python train.py \
  --prs /path/to/X_prs.npy \
  --ehr /path/to/X_ehr.npy \
  --y /path/to/y.npy \
  --covariates /path/to/X_cov.npy \
  --semantic /path/to/semantic.csv \
  --sem /path/to/sem_emb.npy \
  --EID /path/to/EID.txt \
  --out_dir ./outputs \
  --epochs 100 \
  --batch_size 1024 \
  --kfolds 5 \
  --d_latent 128 \
  --d_shared 32 \
  --d_dist 32 \
  --dropout 0.1
```
**Output**

1. `ehr_feature_semantic.csv`  
   A filtered copy of your `semantic.csv` that matches the EHR feature filter inside `train.py`
2. `fold1_model.pt` to `foldK_model.pt`  
   Best model parameters for each fold
3. `fold1_mask.npy` to `foldK_mask.npy`  
   EHR feature gate values for each fold
4. `weighted_mask.npy`  
   Weighted average of fold masks with fold ROC AUC weights
5. `oof_true.npy` and `oof_prob.npy`  
   Out of fold labels and probabilities for all samples
6. `oof_pred.csv`  
   A table with `EID`, `y_true`, `y_prob`

## Contact

For questions, open an issue on GitHub or [email](yinghao.fu@my.cityu.edu.hk), or visit my [personal homepage](https://eddiefua.github.io/).
