rm(list = ls())
library(data.table)
library(stringr)
fp     <- "/home/group3/data/ukb/phenotype/ukb673555.csv"
map_fp <- "/home/yinghaofu2/augmented_prs/data_prepare/Phecode_map_v1_2_icd10_WHO_beta.csv"
all_cols <- names(fread(fp, nrows = 0))
icd_cols  <- grep("^41270-", all_cols, value = TRUE)
date_cols <- sub("^41270", "41280", icd_cols)
use_cols  <- c("eid", icd_cols, date_cols)
dt <- fread(fp, select = use_cols)
long <- melt(
  dt,
  id.vars       = "eid",
  measure.vars  = patterns("^41270-", "^41280-"),
  value.name    = c("icd10", "icd_date"),
  variable.name = "slot"
)
long <- long[!is.na(icd10) & icd10 != ""]
long[, icd_clean := toupper(gsub("\\.", "", trimws(icd10)))]
long[, icd_date  := as.IDate(icd_date)]
long <- long[!is.na(icd_date)]

map_raw <- fread(
  map_fp,
  sep = if (grepl("\\.tsv$", map_fp, ignore.case = TRUE)) "\t" else ",",
  check.names = FALSE
)
col_icd  <- grep("ICD10", names(map_raw), ignore.case = TRUE, value = TRUE)[1]
col_phe  <- grep("PHECODE", names(map_raw), ignore.case = TRUE, value = TRUE)[1]
col_excl_num <- grep("(Excl|Exl).*Phecode", names(map_raw), ignore.case = TRUE, value = TRUE)[1]
col_excl_txt <- grep("(Excl|Exl).*Phenotype", names(map_raw), ignore.case = TRUE, value = TRUE)[1]
map <- data.table(
  code = as.character(map_raw[[col_icd]]),
  phe  = as.character(map_raw[[col_phe]]),
  excl_num = if (!is.na(col_excl_num)) as.character(map_raw[[col_excl_num]]) else NA_character_,
  excl_txt = if (!is.na(col_excl_txt)) as.character(map_raw[[col_excl_txt]]) else NA_character_
)
map[, code_clean := toupper(gsub("\\.", "", trimws(code)))]
map[, phe       := trimws(phe)]
map <- map[nchar(code_clean) > 0 & phe != "" & !is.na(phe)]
map[, len := nchar(code_clean)]
uniq_icd <- unique(long[, .(icd_clean)])
lens     <- sort(unique(map$len))
code2phe_list <- vector("list", length(lens))
for (i in seq_along(lens)) {
  L <- lens[i]
  map_L <- map[len == L, .(code_clean, phe, len)]
  icd_L <- uniq_icd[, .(icd_clean, pref = substr(icd_clean, 1L, L))]
  setkey(map_L, code_clean); setkey(icd_L, pref)
  hit_L <- map_L[icd_L, on = .(code_clean = pref), nomatch = 0L][, .(icd_clean, phe, len)]
  code2phe_list[[i]] <- unique(hit_L)
}
code2phe_all <- unique(rbindlist(code2phe_list, use.names = TRUE))
code2phe_all <- unique(rbindlist(code2phe_list, use.names = TRUE))
code2phe <- code2phe_all[, .SD[len == max(len)], by = icd_clean][, len := NULL]
setkey(code2phe, icd_clean)
setkey(long, icd_clean)
long <- code2phe[long, on = .(icd_clean), nomatch = 0L]

STRICT_TWO_YEAR <- TRUE
if (!STRICT_TWO_YEAR) {
  obs_start  <- as.IDate("1999-01-01"); obs_end  <- as.IDate("2009-12-31")
  wash_start <- as.IDate("2010-01-01"); wash_end <- as.IDate("2010-12-31")
  pred_start <- as.IDate("2011-01-01"); pred_end <- as.IDate("2018-12-31")
} else {
  obs_start  <- as.IDate("1999-01-01"); obs_end  <- as.IDate("2008-12-31")
  wash_start <- as.IDate("2009-01-01"); wash_end <- as.IDate("2010-12-31")
  pred_start <- as.IDate("2011-01-01"); pred_end <- as.IDate("2018-12-31")
}
ehr_obs  <- long[icd_date >= obs_start  & icd_date <= obs_end]
ehr_wash <- long[icd_date >= wash_start & icd_date <= wash_end]
ehr_pred <- long[icd_date >= pred_start & icd_date <= pred_end]
dis_all <- long[, .(first_dx = min(icd_date, na.rm = TRUE)), by = .(eid, phe)]


####T2D: 250.2
####ADHD: 290.11
# ----------------------------- 目标疾病与标签（y） -----------------------------
target_phe <- c("250.2")
all_eids <- unique(dt$eid)
prev_ids <- unique(dis_all[phe %in% target_phe & first_dx <= wash_end, eid])
cohort <- data.table(eid = setdiff(all_eids, prev_ids))
dis_pred <- dis_all[
  eid %in% cohort$eid & phe %in% target_phe & first_dx >= pred_start & first_dx <= pred_end,
  .(eid, first_dx_pred = min(first_dx)), by = eid
][, .(eid, y = 1L)]
y_dt <- merge(cohort, dis_pred, by = "eid", all.x = TRUE)
y_dt[is.na(y), y := 0L]
eid_phe <- unique(ehr_obs[, .(eid, phe)])
eid_phe[, present := 1L]
X <- dcast(eid_phe, eid ~ phe, value.var = "present", fill = 0)
X <- merge(cohort, X, by = "eid", all.x = TRUE)
for (j in setdiff(names(X), "eid")) set(X, i = which(is.na(X[[j]])), j = j, value = 0L)

drop_similar_cols <- function(X_cols, target_phe, map_raw, col_phe, col_excl_num) {
  if (is.null(col_excl_num) || is.na(col_excl_num) || !(col_excl_num %in% names(map_raw))) {
    return(character(0))
  }
  idx <- as.character(map_raw[[col_phe]]) %in% target_phe
  if (!any(idx)) return(character(0))

  excl_str <- unique(na.omit(as.character(map_raw[[col_excl_num]][idx])))
  if (!length(excl_str)) return(character(0))
  phe_cols <- setdiff(X_cols, "eid")
  phe_num  <- suppressWarnings(as.numeric(phe_cols))

  to_drop <- logical(length(phe_cols)); names(to_drop) <- phe_cols
  split_tokens <- function(s) {
    unlist(strsplit(s, "\\s*[,，;；]\\s*"))
  }
  parse_one_range <- function(tok) {
    tok <- trimws(tok)
    if (tok == "") return(NULL)
    if (grepl("[-–—]", tok)) {
      ab <- unlist(strsplit(tok, "[-–—]"))
      a  <- suppressWarnings(as.numeric(trimws(ab[1])))
      b  <- suppressWarnings(as.numeric(trimws(ab[length(ab)])))
      if (!is.na(a) && !is.na(b)) {
        return(list(type = "range", lo = min(a, b), hi = max(a, b)))
      }
      return(NULL)
    } else {
      v <- suppressWarnings(as.numeric(tok))
      if (!is.na(v)) return(list(type = "point", v = v))
      return(NULL)
    }
  }
  tokens <- unlist(lapply(excl_str, split_tokens))
  specs  <- Filter(Negate(is.null), lapply(tokens, parse_one_range))
  if (!length(specs)) return(character(0))

  for (sp in specs) {
    if (sp$type == "range") {
      hit <- !is.na(phe_num) & phe_num >= sp$lo & phe_num <= sp$hi
      to_drop[hit] <- TRUE
    } else { # point
      hit <- !is.na(phe_num) & abs(phe_num - sp$v) < .Machine$double.eps^0.5
      to_drop[hit] <- TRUE
    }
  }
  to_drop[names(to_drop) %in% target_phe] <- TRUE
  names(to_drop)[to_drop]
}

cols_to_drop <- drop_similar_cols(colnames(X), target_phe, map_raw, col_phe, col_excl_num)
if (length(cols_to_drop)) {
  X <- X[, setdiff(names(X), cols_to_drop), with = FALSE]
}


phe_file_path <- "/home/group3/data/ukb/phenotype/ukb673555.csv"
race_cols <- names(fread(phe_file_path, nrows = 0))
race_cols <- race_cols[grepl("^21000-", race_cols)]
use_cols <- c("eid", race_cols)
phe_race <- fread(phe_file_path, select = use_cols)
phe_race[, race := get(race_cols[1])]
if (length(race_cols) > 1) {
  for (col in race_cols[-1]) {
    phe_race[is.na(race), race := get(col)]
  }
}
map_race <- function(code) {
  if (is.na(code)) {
    return(NA_integer_)
  } else if (code %in% c(1001)) {
    return(1)  # White
  } else if (code %in% c(3, 3001, 3002, 3003, 3004)) {
    return(2)  # South Asian
  } else if (code %in% c(4, 4001, 4002, 4003)) {
    return(3)  # Black
  } else if (code == 5) {
    return(4)  # East Asian (Chinese)
  } else {
    return(NA_integer_)
  }
}
phe_race[, race_group := sapply(race, map_race)]
phe_white <- phe_race[race_group == 1, .(eid, race, race_group)]
white_eids <- phe_white$eid
kinship <- data.table::fread('/home/group3/data/ukb/ukb_rel_a68136_s488243.dat')
threshold <- 0.0884
related <- kinship[kinship$Kinship >= threshold, ]
remove_ids <- unique(related$ID2)
final_white_eids <- setdiff(white_eids, remove_ids)
X <- X[eid %in% final_white_eids]
y <- y_dt[eid %in% final_white_eids, y]
names(y) <- y_dt[eid %in% final_white_eids, as.character(eid)]
save(X, y, file = "/home/yinghaofu2/augmented_prs/data_prepare/clean_data/T2D/feature.RData")
