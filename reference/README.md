# Reference tables

- `phecode_definitions1.2.csv` is included and supplies phecode descriptions.
- `Phecode_map_v1_2_icd10_WHO_beta.csv` is required but not included. Place it here before running the R scripts, or update `map_fp`. The scripts require the columns `ICD10`, `PHECODE`, and `Exl. Phecodes`.

Read code IDs as strings to preserve leading zeros. The preparation scripts remove the target phecode and its exclusion ranges, then export descriptions in EHR column order as `semantic.csv`.