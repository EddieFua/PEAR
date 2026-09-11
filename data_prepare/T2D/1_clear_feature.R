library(data.table)

# Edit these paths; run from the PEAR repository root.
fp <- "/path/to/ukb673555.csv"
map_fp <- "reference/Phecode_map_v1_2_icd10_WHO_beta.csv"
kinship_fp <- "/path/to/ukb_rel_a68136_s488243.dat"
out_dir <- "outputs/T2D/data"
target_phe <- "250.2"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Read paired ICD-10 codes and diagnosis dates.
all_cols <- names(fread(fp, nrows = 0))
icd_cols <- grep("^41270-", all_cols, value = TRUE)
date_cols <- sub("^41270", "41280", icd_cols)
stopifnot(length(icd_cols) > 0, all(date_cols %in% all_cols))
dt <- fread(fp, select = c("eid", icd_cols, date_cols), colClasses = list(character = "eid"))
long <- melt(dt, id.vars = "eid", measure.vars = list(icd_cols, date_cols),
             value.name = c("icd10", "icd_date"))
long[, icd_clean := toupper(gsub("\\.", "", trimws(icd10)))]
long[, icd_date := as.IDate(icd_date)]
long <- long[!is.na(icd_clean) & icd_clean != "" & !is.na(icd_date)]

# 2. Map each ICD-10 code using its longest matching prefix.
map_raw <- fread(map_fp, colClasses = "character")
map <- map_raw[, .(code_clean = toupper(gsub("\\.", "", trimws(ICD10))),
                  phe = trimws(PHECODE))]
map <- map[!is.na(code_clean) & code_clean != "" & !is.na(phe) & phe != ""]
map[, len := nchar(code_clean)]
uniq_icd <- unique(long[, .(icd_clean)])
hits <- list()
for (L in sort(unique(map$len))) {
  prefixes <- uniq_icd[, .(icd_clean, pref = substr(icd_clean, 1, L))]
  hits[[as.character(L)]] <- map[len == L][prefixes, on = .(code_clean = pref),
                                          nomatch = 0L][, .(icd_clean, phe, len)]
}
code2phe <- unique(rbindlist(hits))
code2phe <- code2phe[, .SD[len == max(len)], by = icd_clean][, len := NULL]
long <- code2phe[long, on = "icd_clean", nomatch = 0L]

# 3. Observe 1999-2008; wash out 2009-2010; predict incident disease in 2011-2018.
obs_start <- as.IDate("1999-01-01")
obs_end <- as.IDate("2008-12-31")
wash_end <- as.IDate("2010-12-31")
pred_start <- as.IDate("2011-01-01")
pred_end <- as.IDate("2018-12-31")
first_dx <- long[, .(first_dx = min(icd_date)), by = .(eid, phe)]
prev_ids <- first_dx[phe %in% target_phe & first_dx <= wash_end, eid]
case_ids <- first_dx[phe %in% target_phe & first_dx >= pred_start & first_dx <= pred_end, eid]
cohort <- data.table(eid = setdiff(dt$eid, prev_ids))
cohort[, y := as.integer(eid %in% case_ids)]

# 4. Build binary EHR features and remove the target and its exclusion range.
ehr_obs <- unique(long[icd_date >= obs_start & icd_date <= obs_end, .(eid, phe)])
ehr_obs[, present := 1L]
X <- dcast(ehr_obs, eid ~ phe, value.var = "present", fill = 0L)
X <- merge(cohort[, .(eid)], X, by = "eid", all.x = TRUE)
for (j in setdiff(names(X), "eid")) set(X, which(is.na(X[[j]])), j, 0L)
phe_cols <- setdiff(names(X), "eid")
phe_num <- as.numeric(phe_cols)
cols_to_drop <- intersect(phe_cols, target_phe)
exclusions <- na.omit(map_raw[PHECODE %in% target_phe, get("Exl. Phecodes")])
for (token in unique(unlist(strsplit(exclusions, "[,，;；]")))) {
  bounds <- suppressWarnings(as.numeric(trimws(unlist(strsplit(token, "[-–—]")))))
  if (length(bounds) && all(is.finite(bounds))) {
    cols_to_drop <- union(cols_to_drop,
                          phe_cols[is.finite(phe_num) & phe_num >= min(bounds) & phe_num <= max(bounds)])
  }
}
X[, (cols_to_drop) := NULL]

# 5. Retain self-reported British participants (1001), removing related ID2s.
race_cols <- grep("^21000-", all_cols, value = TRUE)
stopifnot(length(race_cols) > 0)
race <- fread(fp, select = c("eid", race_cols), colClasses = list(character = "eid"))
race[, ancestry := get(race_cols[1])]
for (col in race_cols[-1]) race[is.na(ancestry), ancestry := get(col)]
kinship <- fread(kinship_fp, colClasses = list(character = c("ID1", "ID2")))
keep_ids <- setdiff(race[ancestry == 1001, eid], kinship[Kinship >= 0.0884, ID2])
X <- X[eid %in% keep_ids]
y <- cohort$y[match(X$eid, cohort$eid)]
names(y) <- X$eid
stopifnot(!anyDuplicated(X$eid), !anyNA(y), identical(names(y), X$eid))
save(X, y, file = file.path(out_dir, "feature.RData"))
cat("Saved", nrow(X), "participants and", ncol(X) - 1, "EHR features to", out_dir, "\n")
