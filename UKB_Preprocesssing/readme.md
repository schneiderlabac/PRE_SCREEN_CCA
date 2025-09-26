# Analysis Workflow

This README provides a concise walkthrough of the R Markdown (`*.Rmd`) notebooks that reproduce the data preparation, analyses and tables for the manuscript.

## Prerequisites

- **R ≥ 4.2** (tested with 4.3) and RStudio
- Packages: `data.table`, `tidyverse`, `readxl`, `lubridate`, `knitr`, `gtsummary`, `arrow`, `ggplot2` (install on first run)
- Set base paths once at the top of each notebook:
  ```r
  sharepoint_ukb <- "<path-to-UKB-sharepoint>"
  drive          <- "<path-to-local-drive>"
  ```
- Raw UK Biobank bulk data must be available under `raw/`.
- Master Table providing data on input variables, UKB field IDs, units etc.

## Recommended run order

| Step | Notebook                                                   | Purpose (one‑liner)                                                       | Key output(s)                              |
| ---- | --------------------------------------------------         | ------------------------------------------------------------------------- | ------------------------------------------ |
| 1    | **01 Extract and process covariates Notebook.Rmd**         | Pull covariate fields from raw UKB and clean/recode them.                 | `extracted/covariates.csv`                 |
| 2    | **02 Extract\_multiple\_diagnosis.Rmd**                    | Extract HESIN hospital‑episode diagnoses & filter ICD codes.              | `extracted/diagnoses.csv`                  |
| 3    | **03 Table Y CCA Extract.Rmd**                             | Identify hepatocellular‑carcinoma (HCC) cases and build outcome variable. | `derived/hcc_cases.csv`                    |
| 4    | **04 Table X Preprocessing CCA and Table1 creation.Rmd**   | Pre‑process multi‑omic & lab variables for Table X and modelling.         | `derived/tableX_data.parquet`              |
|      |                                                            | Generate descriptive Table 1 for whole cohort & PAR subset.              | `output/table1.docx`, `output/table1.xlsx` |

> **Tip:** Knit the notebooks sequentially (1→4). Each script writes the datasets required by the next one—no manual edits needed.

## Refreshing with new data

1. Replace raw UKB files in `raw/`.
2. Re‑run notebooks **01**–**07** in order; downstream outputs refresh automatically.

## Contact

Questions or issues? Contact **Felix van Haag** *(*[*felix.van.haag@rwth-aachen.de*](mailto\:felix.van.haag@rwth-aachen.de)*).*

