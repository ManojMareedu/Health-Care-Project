# Healthcare Claims Analytics — Predictive Modeling of Patient Charge Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![R](https://img.shields.io/badge/R-4.x-276DC3?logo=r) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter) ![License](https://img.shields.io/badge/License-Academic-lightgrey) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Transforming synthetic Medicare-style claims data into actionable cost-risk intelligence — combining multi-source data integration, frequency-encoded clinical features, and ensemble classification to predict patient total charge tiers, enabling precision resource allocation and proactive cost containment strategies.**

---

## Table of Contents

- [Business Impact](#business-impact)
- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Technical Architecture](#technical-architecture)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Business Impact

Healthcare cost overruns and unpredictable claim volumes represent one of the most operationally expensive challenges facing payers, hospital systems, and health policy administrators. This project delivers a **data-driven charge classification framework** that directly addresses high-value operational decisions:

- **Cost Risk Stratification:** By predicting which patient claims fall into high-charge tiers (>$10,000 or >$100,000), payers and hospital finance teams can proactively flag high-risk cases for utilization review and care management intervention before claims are settled.
- **Claims Processing Optimization:** Classification of total charge into discrete cost bands enables intelligent routing of claims — directing complex, high-value cases to specialist review teams while automating low-charge claim approvals.
- **Population Health Investment:** Identifying diagnosis codes and claim volume patterns most predictive of high total charges supports targeted population health programs, disease management protocols, and preventive care investments that reduce downstream acute care costs.
- **Provider Network Strategy:** State-level provider charge analysis supports payer network tiering, contract renegotiation benchmarking, and out-of-network liability assessment.
- **Regulatory & Compliance Readiness:** Linking median household income to charge patterns provides an evidence base for value-based care program design and health equity reporting under CMS quality frameworks.

---

## Project Overview

This project constructs a **multi-stage healthcare analytics pipeline** that integrates synthetic Medicare-style inpatient and outpatient claims with state-level socioeconomic indicators to predict patient total charge classification. The core analytical question: *Can clinical diagnosis codes, claim volume, care setting, and regional economic context predict the cost tier of a patient's combined inpatient-outpatient charge burden?*

The pipeline spans data engineering (multi-source claims merging), feature engineering (frequency encoding of ICD diagnosis codes, dummy encoding of clinical indicators), regression modeling (log-transformed total charge), and multi-class classification (five cost tiers) using KNN, Decision Tree, and Random Forest algorithms.

---

## Key Findings

| Finding | Detail |
|---|---|
| **Dataset Scale** | 58,066 inpatient claims and 575,092 outpatient claims merged into 46,059 patient-level records |
| **Geographic Coverage** | 51 U.S. state/territory codes represented across both inpatient and outpatient datasets |
| **Total Charge Range** | Inpatient CLM_PMT_AMT from $62.44 to $598,716.31; outpatient CLM_PMT_AMT from $59.64 to $428,631.86 |
| **Best Classification Model** | Random Forest outperformed KNN and Decision Tree on multi-class charge tier prediction |
| **Feature Importance** | Diagnosis frequency encoding (inpatient & outpatient ICD codes), number of claims, and median income were leading predictors of total charge tier |
| **Regression Baseline** | Linear regression on log-transformed total charge established a continuous cost prediction baseline prior to classification |
| **Cost Tier Design** | Five classes: Class 1 (<$1K), Class 2 ($1K-$10K), Class 3 ($10K-$100K), Class 4 ($100K-$1M), Class 5 (>$1M) |
| **Data Engineering** | Outpatient records filtered to beneficiaries present in inpatient data; claims aggregated by beneficiary-diagnosis-state groupings before merge |

---

## Dataset

> **Note: All claims data used in this project is synthetic (simulated Medicare-style data) generated for academic and analytical purposes. No real patient health information (PHI) is used or stored in this repository.**

| Source | Description | Records |
|---|---|---|
| `inpatient.csv` | Synthetic Medicare inpatient claims with 197 columns including ICD diagnosis codes, provider identifiers, DRG codes, claim payment amounts, and admission indicators | 58,066 rows |
| `outpatient.csv` | Synthetic Medicare outpatient claims with 162 columns including revenue center codes, charge amounts, and rendering physician NPIs | 575,092 rows |
| `Median_Income.xlsx` | State-level median household income used as a socioeconomic covariate | 51 state/territory entries |
| `Patient_Claim_Data.xlsx` | Engineered output: merged inpatient-outpatient patient-level dataset used as input to the modeling pipeline | 46,059 records |

**Key features selected for modeling:**
- `PRNCPAL_DGNS_CD_inp` / `PRNCPAL_DGNS_CD_out` - Principal ICD-10 diagnosis codes (frequency-encoded)
- `CLM_E_POA_IND_SW1` - Present-on-Admission indicator (dummy-encoded; values: Y, U)
- `Number_of_Claims_inp` / `Number_of_Claims_out` - Claim volume per beneficiary-diagnosis-state grouping
- `TOTAL_CHARGE` - Sum of inpatient and outpatient total charges (target variable)
- `Median_Income` - State-level median household income

---

## Methodology

### 1. Data Engineering (Python - Jupyter Notebook)
- Loaded synthetic Medicare inpatient (58,066 rows x 197 cols) and outpatient (575,092 rows x 162 cols) claim files
- Standardized `BENE_ID` and `CLM_ID` as integers; applied `.abs()` to correct signed IDs
- Filtered outpatient records to beneficiaries with matching inpatient claims
- Selected 7 key features per dataset; labeled patient type (inpatient=1, outpatient=0)
- Computed `Number_of_Claims` via grouped count transform (beneficiary x diagnosis x state)
- Aggregated charges by beneficiary-diagnosis-state grouping; merged inpatient and outpatient on `BENE_ID`
- Exported merged dataset (`Patient_Claim_Data.xlsx`) - 46,059 patient records, 10 columns

### 2. Feature Engineering (R)
- Computed `TOTAL_CHARGE` as sum of `CLM_TOT_CHRG_AMT_inp` and `CLM_TOT_CHRG_AMT_out` per patient record
- Merged with `Median_Income.xlsx` on `PRVDR_STATE_CD` to incorporate socioeconomic context
- Applied **frequency encoding** to high-cardinality ICD-10 diagnosis codes (replacing codes with their frequency counts in the dataset)
- Applied **dummy encoding** to `CLM_E_POA_IND_SW1` (Present-on-Admission indicator)
- Log-transformed `TOTAL_CHARGE` for regression modeling to address right-skewed distribution
- Created 5-class target variable `TC_class` using charge breakpoints: <$1K, $1K-$10K, $10K-$100K, $100K-$1M, >$1M

### 3. Modeling (R - caret framework)
- **Train/Test Split:** 70/30 stratified partition (seed=1123) applied to both regression and classification tasks
- **Linear Regression:** Baseline model on log-transformed total charge; evaluated with RMSE
- **K-Nearest Neighbors (KNN):** Grid search over k in {3, 5, 7, 9, 11} with 5-fold cross-validation
- **Decision Tree:** `rpart` with `minbucket=10`, `cp=0`; pruned via complexity parameter (CP) plot
- **Random Forest:** 100 trees (`ntree=100`) with `importance=TRUE`; evaluated via confusion matrix and multiclass AUC (pROC)

---

## Technical Architecture

```
Synthetic Medicare Claims (inpatient.csv + outpatient.csv)
            |
            v
  [Python - Healthcareprojectfinal.ipynb]
  |-- Data Loading & ID Standardization (.abs())
  |-- Feature Selection (7 cols each dataset)
  |-- Beneficiary Intersection Filter
  |-- Number_of_Claims via groupby count transform
  |-- Charge Aggregation by Beneficiary-Diagnosis-State
  |-- Inpatient-Outpatient Merge (on BENE_ID)
  `-- Export -> Patient_Claim_Data.xlsx (46,059 records x 10 cols)
            |
            v
  [R - HealthCare Class Project.R]
  |-- Load Patient_Claim_Data.xlsx + Median_Income.xlsx
  |-- TOTAL_CHARGE = CLM_TOT_CHRG_AMT_inp + CLM_TOT_CHRG_AMT_out
  |-- State-Income Merge (by PRVDR_STATE_CD)
  |-- Frequency Encoding (ICD-10 Dx codes: inp & out)
  |-- Dummy Encoding (CLM_E_POA_IND_SW1)
  |-- Log Transform + 5-Class TC_class Target Creation
  |-- 70/30 Train-Test Split (seed=1123)
  `-- Model Training & Evaluation:
      |-- Linear Regression (log TOTAL_CHARGE -> RMSE)
      |-- KNN (k in {3,5,7,9,11}, 5-fold CV)
      |-- Decision Tree (rpart, minbucket=10, cp=0)
      `-- Random Forest (ntree=100, varImpPlot, multiclass AUC)
```

---

## Results

| Model | Task | Evaluation Metrics |
|---|---|---|
| Linear Regression | Continuous charge prediction (log scale) | RMSE on test set |
| KNN (best k via CV) | 5-class charge tier classification | Accuracy, Confusion Matrix, Multiclass AUC |
| Decision Tree | 5-class charge tier classification | Accuracy, Confusion Matrix, Multiclass AUC |
| **Random Forest** | **5-class charge tier classification** | **Best performer: Accuracy, Confusion Matrix, Multiclass AUC, Variable Importance** |

**Variable Importance (Random Forest):** Diagnosis frequency encoding for both inpatient and outpatient ICD-10 codes, inpatient and outpatient claim volume, and median income were the top predictors of charge tier classification.

*Visual outputs in the repository: `rf variable importance.png`, `Total charge distribution.png`, `Lo total charge distribution.png`, `Decision Tree CP.png`*

---

## Repository Structure

```
Health-Care-Project/
|-- Healthcareprojectfinal.ipynb          # Python pipeline: data ingestion, merging, feature prep
|-- HealthCare Class Project.R            # R modeling pipeline: feature engineering + ML models
|-- Healthcare Analytics Project Group 3.docx  # Project report and analysis documentation
|-- Frequency Encoding Index.docx         # Reference index for ICD-10 frequency encoding mappings
|-- Patient_Claim_Data.xlsx               # Engineered dataset: merged inpatient-outpatient claims
|-- Median_Income.xlsx                    # State-level median household income reference data
|-- Decision Tree CP.png                  # Complexity parameter plot for decision tree pruning
|-- rf variable importance.png            # Random Forest variable importance plot
|-- Total charge distribution.png         # Raw total charge distribution histogram
|-- Lo total charge distribution.png      # Log-transformed total charge distribution histogram
`-- README.md
```

---

## Getting Started

### Prerequisites

**Python Environment:**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

**R Environment:**
```r
install.packages(c("readxl", "ggplot2", "dplyr", "reshape2",
                   "fastDummies", "caret", "rpart", "rpart.plot",
                   "pROC", "randomForest", "yardstick", "tidyverse"))
```

### Execution Order

1. **Run the Python notebook first** (`Healthcareprojectfinal.ipynb`) with `inpatient.csv` and `outpatient.csv` in your working directory - this generates `Patient_Claim_Data.xlsx`
2. **Run the R script** (`HealthCare Class Project.R`) with `Patient_Claim_Data.xlsx` and `Median_Income.xlsx` in your working directory - this executes all feature engineering and model training/evaluation

> **Note:** The notebook was developed on Google Colab (TPU runtime). Adjust file paths from `/content/drive/MyDrive/` to your local directory as needed.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Engineering | Python 3, pandas, NumPy |
| Visualization | matplotlib, seaborn, ggplot2 |
| Feature Engineering | R, dplyr, fastDummies, reshape2 |
| Machine Learning | R caret, rpart, randomForest, pROC |
| Development Environment | Google Colab (Python), RStudio (R) |
| Data Storage | Excel (.xlsx), CSV |

---

## Author

**Manoj Mareedu** - Data Scientist / ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-ManojMareedu-181717?logo=github)](https://github.com/ManojMareedu)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-manojmareedu-0A66C2?logo=linkedin)](https://www.linkedin.com/in/manojmareedu/)

Developed as part of a Healthcare Analytics graduate project at the **University of Texas at Dallas**.

---

*This project is extensible - planned enhancements include SHAP-based model explainability, time-series claim frequency analysis, and integration of CMS beneficiary demographic data for equity-aware modeling.*
