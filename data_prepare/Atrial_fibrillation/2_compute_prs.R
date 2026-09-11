library(data.table)
library(bigreadr)
library(bigsnpr)
library(bigstatsr)
library(RcppCNPy)

# Edit these paths; out_dir must match 1_clear_feature.R.
fp <- "/path/to/ukb673555.csv"
sumstats_fp <- "/path/to/GCST90475967_GRCh37.txt.gz"
mfi_pattern <- "/path/to/filtered_mfi/ukb_mfi_chr%d_v3_filtered.txt"
bgen_pattern <- "/path/to/bgen/ukb22828_c%d_qc_snp.bgen"
sample_fp <- "/path/to/ukb22828_c12_qc_snp.psam"
semantic_fp <- "reference/phecode_definitions1.2.csv"
out_dir <- "outputs/Atrial_fibrillation/data"
NCORES <- 8L

# 1. Read GRCh37 GWAS statistics and match alleles to imputed SNP metadata.
sumstats <- fread2(sumstats_fp, select = c(1, 23, 3, 4, 5, 8))
names(sumstats) <- c("chr", "pos", "a1", "a0", "beta", "p")
sumstats$beta <- log(sumstats$beta)  # Source column contains odds ratios.
sumstats <- subset(sumstats, p < 0.1)
sumstats$p[sumstats$p == 0] <- min(sumstats$p[sumstats$p > 0])
stopifnot(all(is.finite(sumstats$beta)), all(is.finite(sumstats$p)), all(sumstats$p > 0))
info_snp_UKBB <- rbind_df(lapply(1:22, function(chr) {
  df <- fread2(sprintf(mfi_pattern, chr), select = c(3:5, 8),
               col.names = c("pos", "a0", "a1", "info"))
  cbind.data.frame(chr = chr, df)
}))
info_snp <- snp_match(sumstats, info_snp_UKBB, strand_flip = FALSE)
info_snp <- subset(na.omit(info_snp), info > 0.3)
snp_ids <- with(info_snp, paste(as.integer(chr), pos, a0, a1, sep = "_"))
stopifnot(!anyDuplicated(snp_ids))
list_snp_id <- split(snp_ids, factor(info_snp$chr, levels = 1:22))

# 2. Align labels, EHR and genotype row indices by participant ID.
# The PSAM must have the same sample order as every BGEN file.
load(file.path(out_dir, "feature.RData"))
samples <- fread(sample_fp, colClasses = "character")
stopifnot("IID" %in% names(samples), !anyDuplicated(samples$IID),
          !anyDuplicated(names(y)), !anyDuplicated(X$eid))
sub_eid <- sort(Reduce(intersect, list(names(y), X$eid, samples$IID)))
sub_id <- match(sub_eid, samples$IID)
y <- y[match(sub_eid, names(y))]
X <- X[match(sub_eid, X$eid)]
stopifnot(length(y) > 0, !anyNA(sub_id), identical(as.character(X$eid), names(y)))

# 3. Read genotypes once and align GWAS weights to the returned SNP order.
rds <- snp_readBGEN(
  bgenfiles = sprintf(bgen_pattern, 1:22), list_snp_id = list_snp_id,
  ind_row = sub_id, backingfile = file.path(out_dir, "genotypes"), ncores = NCORES
)
ukbb <- snp_attach(rds)
G <- ukbb$genotypes
CHR <- as.integer(ukbb$map$chromosome)
POS <- ukbb$map$physical.pos
snp_order <- match(paste(CHR, POS, ukbb$map$allele1, ukbb$map$allele2, sep = "_"), snp_ids)
stopifnot(!anyNA(snp_order), !anyDuplicated(snp_order), nrow(G) == length(y))
info_snp <- info_snp[snp_order, ]
beta <- info_snp$beta
lpval <- -log10(info_snp$p)

# 4. Compute the 3 x 7 x 4 x 50 = 4,200 clumping-and-thresholding scores.
options(bigstatsr.check.parallel.blas = FALSE)
all_keep <- snp_grid_clumping(
  G, CHR, POS, lpS = lpval, infos.imp = info_snp$info,
  grid.thr.imp = c(0.8, 0.9, 0.95),
  grid.thr.r2 = c(0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95),
  grid.base.size = c(50, 100, 200, 500), ncores = NCORES
)
multi_PRS <- snp_grid_PRS(
  G, all_keep, betas = beta, lpS = lpval, n_thr_lpS = 50,
  backingfile = file.path(out_dir, "PRS_by_chromosome"), ncores = NCORES
)
# snp_grid_PRS returns one block per chromosome represented in G.
s <- nrow(attr(all_keep, "grid")) * length(attr(multi_PRS, "grid.lpS.thr"))
stopifnot(s == 4200L, ncol(multi_PRS) == s * length(unique(CHR)))
multi_PRS_mat <- big_apply(
  multi_PRS, a.FUN = function(X, ind, s, n_chr) rowSums(X[, ind + s * (seq_len(n_chr) - 1L)]),
  ind = seq_len(s), s = s, n_chr = length(unique(CHR)),
  a.combine = "cbind", block.size = 1, ncores = NCORES
)

# 5. Export aligned NumPy arrays, IDs and phecode descriptions for PEAR.
features <- X[, -1, with = FALSE]
npySave(file.path(out_dir, "multi_PRS.npy"), multi_PRS_mat)
npySave(file.path(out_dir, "ukb_final_features.npy"), as.matrix(features))
npySave(file.path(out_dir, "ukb_final_y.npy"), as.matrix(y))
writeLines(names(y), file.path(out_dir, "ukb_final_ids.txt"))
all_semantic <- fread(semantic_fp, colClasses = "character")
semantic <- data.frame(feature = names(features),
                       text = all_semantic$phenotype[match(names(features), all_semantic$phecode)])
stopifnot(!anyNA(semantic$text), all(nzchar(trimws(semantic$text))))
write.csv(semantic, file.path(out_dir, "semantic.csv"), row.names = FALSE)
covariates <- fread(fp, select = c("eid", "21022-0.0", "22001-0.0", paste0("22009-0.", 1:20)),
                    colClasses = list(character = "eid"))
covariates <- covariates[match(names(y), covariates$eid)]
stopifnot(identical(covariates$eid, names(y)), !anyNA(covariates))
npySave(file.path(out_dir, "ukb_final_covariates.npy"), as.matrix(covariates[, -1, with = FALSE]))
cat("Saved PEAR inputs to", out_dir, "\n")
