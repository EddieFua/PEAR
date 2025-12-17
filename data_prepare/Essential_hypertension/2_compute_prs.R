rm(list = ls())
library(dplyr)
library(bigreadr)
library(data.table)
folder = '/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/'
sumstats <- fread2("/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/GCST90475922_GRCh37.txt.gz", select = c(1,23,3,4,5,8))
colnames(sumstats)
names(sumstats) <- c("chr", "pos", "a1", "a0", "beta", "p")
sumstats$beta <- log(sumstats$beta)
sumstats <- subset(sumstats, p < 0.1)
info_snp_UKBB <- rbind_df(lapply(1:22, function(chr) {
  file <- paste0("/home/group3/yinghaofu2/data/ukb/filtered_mfi/ukb_mfi_chr", chr, "_v3_filtered.txt")
  df <- fread2(file, select = c(3:5, 8), col.names = c("pos", "a0", "a1", "info"))
  cbind.data.frame(chr = chr, df)
}))
info_snp <- bigsnpr::snp_match(sumstats, info_snp_UKBB)
info_snp <- bigsnpr::snp_match(sumstats, info_snp_UKBB, strand_flip = FALSE)
info_snp <- subset(na.omit(info_snp), info > 0.3)
list_snp_id <- with(info_snp, split(
  paste0(as.integer(chr), "_", pos, "_", a0, "_", a1),
  factor(chr, levels = 1:22)
))
beta <- info_snp$beta
lpval <- -log10(info_snp$p)
info <- info_snp$info

library(bigreadr)
load('/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/feature.RData')
table(y)
####     0      1 
####304180  60759

sample <- fread2('/home/group3/yinghaofu2/data/ukb/imppgen_qc_unrelated_final/ukb22828_c18_qc_unrelated_final.sample')[-1,]
intersect_y_gen <- sort(intersect(names(y), sample$ID_2), decreasing = FALSE)
y <- y[match(intersect_y_gen, names(y))]
# set.seed(1)
# id_pos <- names(y)[y == 1]
# id_neg <- names(y)[y == 0]
# k <- min(length(id_pos), length(id_neg))
# id_pos_k <- sample(id_pos, k)
# id_neg_k <- sample(id_neg, k)
# balanced_ids <- c(id_pos_k, id_neg_k)
# y <- y[balanced_ids]
# table(y)
####     0      1 
####15896 15896 
sub_eid <- sort(intersect(intersect(names(y), X$eid), sample$ID_2), decreasing = FALSE)
sub_id <- match(sub_eid, sample$ID_2)
y <- y[match(sub_eid, names(y))]
X <- X[eid %in% sub_eid]
saveRDS(y, file = '/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/ukb_final_y.rds')
saveRDS(X, file = '/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/ukb_final_X.rds')

NCORES <- 100
system.time(
rds <- bigsnpr::snp_readBGEN(
  bgenfiles = glue::glue("/home/group3/yinghaofu2/data/ukb/imppgen_qc_unrelated_final/ukb22828_c{chr}_qc_unrelated_final.bgen", chr = 1:22),
  list_snp_id = list_snp_id,
  ind_row = sub_id,
  backingfile = '/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/imppgen_qc_unrelated_final',
  ncores = NCORES)
)#149s


library(bigsnpr)
ukbb <- snp_attach("/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/imppgen_qc_unrelated_final.rds")
G <- ukbb$genotypes
file.size(G$backingfile) / 1024^3  # 184GB
CHR <- as.integer(ukbb$map$chromosome)
POS <- ukbb$map$physical.pos

options(bigstatsr.check.parallel.blas = FALSE)
system.time(
  all_keep <- snp_grid_clumping(
    G, CHR, POS, lpS = lpval, ind.row = rows_along(G), infos.imp = info,
    grid.thr.imp = c(0.3, 0.6, 0.9, 0.95),
    grid.thr.r2 = c(0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95),
    grid.base.size = c(50, 100, 200, 500),
    ncores = NCORES)
) # 2370s
saveRDS(all_keep, file = '/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/all_keep.rds')
system.time(
  multi_PRS <- snp_grid_PRS(
    G, all_keep, betas = beta, lpS = lpval, ind.row = rows_along(G),
    n_thr_lpS = 50, backingfile = "/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/UKBB_scores", ncores = NCORES)
) # 226s
saveRDS(multi_PRS, file = '/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/multi_PRS.rds')


all_keep <- readRDS('/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/all_keep.rds')
multi_PRS <- readRDS('/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/multi_PRS.rds')

library(tidyr)
grid2 <- attr(all_keep, "grid") %>%
  mutate(thr.lp = list(attr(multi_PRS, "grid.lpS.thr")), num = row_number()) %>%
  unnest()
write.csv(grid2, paste0(folder, 'multi_PRS_grid.csv'), row.names = FALSE)
s <- nrow(grid2)
grid2$auc <- big_apply(multi_PRS, a.FUN = function(X, ind, s, y.train) {
  single_PRS <- rowSums(X[, ind + s * (0:21)])
  bigstatsr::AUC(single_PRS, y.train)
}, ind = 1:s, s = s, y.train = y,
a.combine = 'c', block.size = 1, ncores = NCORES)
max(grid2$auc)  # 0.5916347
min(grid2$auc)  # 0.5431746


library(RcppCNPy)
multi_PRS_mat <- big_apply(
  multi_PRS,
  a.FUN = function(X, ind, s) {
      rowSums(X[, ind + s * (0:21)])
  },
  ind = 1:s,
  s = s,
  a.combine = 'cbind',
  block.size = 1,
  ncores = NCORES
)
dim(multi_PRS_mat)
npySave("/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/multi_PRS.npy", multi_PRS_mat)
y = readRDS('/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/ukb_final_y.rds')
ids <- names(y)
write.table(
  ids,
  file =  "/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/ukb_final_ids.txt",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
npySave("/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/ukb_final_y.npy", as.matrix(y))
features <- readRDS('/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/ukb_final_X.rds')
features <- features[,-1]
npySave('/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/ukb_final_features.npy', as.matrix(features))
all_semantic = read.csv('/home/yinghaofu2/augmented_prs/data_prepare/phecode_definitions1.2.csv')
text = as.matrix(all_semantic$phenotype[match(colnames(features), all_semantic$phecode)])
feature = c(colnames(features))
semantic = data.frame(feature = feature, text = text)
write.csv(semantic, '/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/semantic.csv', row.names = FALSE)


sex_age = fread2('/home/group3/data/ukb/phenotype/ukb673555.csv', select = c('eid','21022-0.0','22001-0.0')) 
col_names = fread2('/home/group3/data/ukb/phenotype/ukb673555.csv', nrows = 0) %>% colnames()
pcs_all <- grep("^22009", col_names, value = TRUE)
cols <- c("eid", paste0("22009-0.", 1:20))
top20_pcs = fread2('/home/group3/data/ukb/phenotype/ukb673555.csv', select = cols)
covariates <- merge(sex_age, top20_pcs, by = 'eid')
covariates <- covariates[match(names(y), covariates$eid),]
dim(covariates)
length(y)
npySave('/home/yinghaofu2/augmented_prs/data_prepare/clean_data/Essential_hypertension/ukb_final_covariates.npy', as.matrix(covariates[, -1]))

