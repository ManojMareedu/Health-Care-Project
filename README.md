# Healthcare Claims Cost Intelligence

An end-to-end ML system that predicts the **cost tier** and **total charge** of a
Medicare-style claim, with SHAP attributions behind every prediction, served
through a containerized FastAPI service and a Streamlit dashboard.

This started as an academic R + Jupyter analysis. Rebuilding it as a production
system surfaced five methodology defects in the original — including one that
had silently deleted an entire state from the dataset. Those fixes are
documented below, because finding them is the more interesting part of the work.

---

## Why this matters

Payers and hospital finance teams settle claims faster than they can review
them. Knowing which claims will land in a high-charge tier *before* settlement
lets an organization:

- **Flag high-cost claims for utilization review** while intervention is still possible
- **Route intelligently** — auto-approve the routine, escalate the complex
- **Target population-health investment** at the diagnoses and states driving cost

State-level median income is carried as a socioeconomic covariate so cost
patterns can be read with health-equity context rather than in isolation.

---

## Architecture

```
data/raw/  (DVC-tracked: Patient_Claim_Data.xlsx, Median_Income.xlsx)
      |
      v
data_validation.py ......... pandera schemas; rejects malformed rows,
      |                      asserts cross-column invariants
      v
feature_engineering.py ..... GroupShuffleSplit on BENE_ID *before* encoding;
      |                      frequency + one-hot encoders fit on train only
      v
train_pipeline.py (ZenML) .. 4 classifiers + 2 regressors
      |                      -> MLflow: params, metrics, confusion matrices,
      |                         feature importance, serialized pipelines
      v
exported_model/ ............ best model by documented rule, committed so the
      |                      demo runs without retraining
      |
      +--> inference.py ........... single scoring path, shared by both surfaces
      +--> app/api/main.py ........ FastAPI: /predict/tier, /predict/charge
      |                             with SHAP contributions in the response
      +--> app/dashboard/app.py ... Streamlit: comparison, scoring, methodology
      |                             (scores in-process; this is the public demo)
      +--> monitoring/ ............ Evidently feature + prediction drift
      |
      v
.github/workflows/ci.yml ... ruff -> pytest -> docker build -> live API smoke test
```

---

## Corrected methodology

The original implementation is preserved in `legacy/`. Each defect below was
verified against the actual data before being fixed.

### 1. Beneficiary leakage across the train/test split

`BENE_ID` is **not** a unique row key — 46,059 rows cover only **5,416
beneficiaries** (~8.5 rows each, one per outpatient diagnosis). Those rows share
a beneficiary's inpatient charge, claim counts, and state.

The original used a random 70/30 **row** split, which put **4,074 of 5,416
beneficiaries (75%) on both sides**. The test set was largely a paraphrase of
the training set.

**Fix:** `GroupShuffleSplit` on `BENE_ID`. Beneficiary overlap is now zero, and
the pipeline raises if it ever isn't.

This correction is why the random forest scores **76.1%** here against the
**79.3%** the original reported. The old number was measuring memorization.

### 2. Encoding fit on the full dataset

Diagnosis-code frequencies and the POA dummy were computed over all rows before
splitting, leaking test-set distribution into every model's features.

**Fix:** both encoders live in a `ColumnTransformer` fit on the training fold
only. Unseen diagnosis codes fall back to the training-set minimum frequency —
not zero, which would collide with "absent" in distance-based models.

### 3. Ambiguous state-code rule

Both `PRVDR_STATE_CD_inp` and `_out` exist; the original kept `_inp` and
silently dropped `_out`, leaving no way to tell whether signal was lost.

**Fix:** the two columns agree on **all 46,059 rows**, so `_inp` drives the
income join and `_out` is redundant. The rule is stated explicitly *and*
enforced as a schema check, so a future extract where they disagree fails loudly
instead of quietly modeling a half-truth.

### 4. Incomplete POA encoding

The original produced only a `CLM_E_POA_IND_SW1_Y` column, treating `U` as an
implicit reference level with no handling for unseen values at inference.

**Fix:** `OneHotEncoder(handle_unknown="ignore")`. Both levels are now explicit,
and a novel value encodes as all-zero rather than raising.

### 5. An entire state silently deleted

`Median_Income.xlsx` stores 50 states as integers and **Wyoming as the string
`"$60,510 "`**. The original ran `as.numeric()` over that column, which turned
the value into `NA`; a later `na.omit()` dropped every row that inherited it.

Wyoming has exactly **47 claims rows**. The original write-up records this as
routine *"missing value handling, 47 NAs"* — but nothing was missing. A currency
formatting artifact removed a whole state from an analysis whose stated purpose
includes health-equity reporting by state income.

**Fix:** parse the currency string at ingestion. All 51 states and all 46,059
rows now reach the model.

### Also addressed

- **No validation existed.** pandera schemas now gate every load.
- **Three outpatient rows carry ICD-10-PCS *procedure* codes** (`0FB43ZZ`,
  `0FC44ZZ`) in a *diagnosis* column. Too few to fail a run, so they are
  reported rather than raised — visible instead of silent.
- **No tracking, registry, API, or tests.** All present now.

---

## Results

Test fold: **13,288 rows / 1,625 beneficiaries**, disjoint by beneficiary from
the 32,771-row training fold.

### Cost tier classification

| Model | Accuracy | Macro-F1 | ROC-AUC (OvR) | Selection score |
|---|---|---|---|---|
| KNN | 0.6273 | 0.5101 | 0.7894 | 0.5570 |
| Decision tree | 0.6930 | 0.6177 | 0.9055 | 0.6478 |
| Random forest | 0.7614 | 0.7103 | 0.9424 | 0.7307 |
| **XGBoost** | **0.7911** | **0.7542** | **0.9549** | **0.7690** |

**Selection rule: `0.6 × macro-F1 + 0.4 × accuracy`.** Accuracy alone is the
wrong criterion here — tier 3 is half the rows, so a model that ignores the rare
catastrophic tier entirely can still post a respectable number while being
useless for the actual question. Macro-F1 weights every tier equally, so a model
has to handle tier 5 (521 rows, 1.1%) to win.

### Charge regression, on `log(TOTAL_CHARGE)`

| Model | RMSE (log) | R² | Median absolute error |
|---|---|---|---|
| Linear regression | 1.7395 | 0.2576 | $14,855 |
| **Ridge regression** | **1.7394** | **0.2577** | **$14,785** |

The regression is honestly weak — R² ≈ 0.26 — and it is reported that way. Six
features cannot explain charges spanning $129 to $32.6M. The classification
framing is the one that answers a usable business question, which is why the
tier model is the primary production surface.

### Cost tier distribution

| Tier | Range | Rows | Share |
|---|---|---|---|
| 1 | under $1K | 4,823 | 10.5% |
| 2 | $1K – $10K | 9,061 | 19.7% |
| 3 | $10K – $100K | 22,976 | 49.9% |
| 4 | $100K – $1M | 8,678 | 18.8% |
| 5 | over $1M | 521 | 1.1% |

---

## Quick start

### What is hosted, and what you run yourself

Being precise about this, because "has a live demo" is easy to overstate:

| Surface | Where it runs | Status |
|---|---|---|
| **Streamlit dashboard** | Streamlit Community Cloud | **Publicly hosted** |
| **FastAPI service** | Your machine, via Docker | **Not publicly hosted** — run it locally |

The FastAPI service is a fully built and verified component, not a sketch: it is
Dockerized, covered by the test suite, and smoke-tested against a live container
on every CI run. It simply has no public URL, because every free tier that would
host an always-on container either requires a paid plan or a payment card, which
this project does not use.

The hosted dashboard therefore scores claims **in-process** from the committed
`exported_model/` rather than calling the API over HTTP. Both surfaces import the
same `src/healthcare_mlops/inference.py`, so they cannot return different answers
for the same claim — there is one scoring path, exposed two ways.

### Deploying the dashboard yourself

Streamlit Community Cloud is free for public repositories and needs no card. The
repo is already configured for it — `requirements.txt` and `packages.txt`
(`libgomp1`, which XGBoost needs) sit at the root, and the app loads
`exported_model/` from the repo, so there is no database, secret, or environment
variable to set.

1. Sign in at <https://share.streamlit.io> with GitHub and authorize it
2. **Create app** -> **Deploy a public app from GitHub**
3. Repository `ManojMareedu/Health-Care-Project`, branch `main`, main file
   `app/dashboard/app.py`
4. Under **Advanced settings**, set Python version to **3.12**
5. Deploy — the first build takes a few minutes while it installs the dependencies

### Docker Compose — the full stack, including the API

```bash
git clone https://github.com/ManojMareedu/Health-Care-Project.git
cd Health-Care-Project
docker compose up --build
```

- API: <http://localhost:8000> — interactive docs at `/docs`
- Dashboard: <http://localhost:8501>

Verify the running stack with the exact checks CI uses:

```bash
python scripts/smoke_test.py http://localhost:8000
```

### Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest                                        # 48 tests
ruff check . && ruff format --check .
```

### Retraining

The raw data is DVC-tracked, so a fresh clone needs `dvc pull` (or the source
workbooks placed in `data/raw/`) before training. The committed
`exported_model/` means the API and dashboard run without this step.

```bash
dvc pull
python -m src.healthcare_mlops.train_pipeline
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Setting `DAGSHUB_MLFLOW_URI` points tracking at a hosted MLflow server instead;
unset, everything stays local and offline.

### Drift monitoring

```bash
python -m monitoring.drift_report
# writes monitoring/reports/{feature,prediction}_drift.html
```

---

## Example request

```bash
curl -X POST http://localhost:8000/predict/tier \
  -H 'Content-Type: application/json' \
  -d '{
    "PRNCPAL_DGNS_CD_inp": "I10",
    "PRNCPAL_DGNS_CD_out": "E119",
    "CLM_E_POA_IND_SW1": "Y",
    "Number_of_Claims_inp": 3,
    "Number_of_Claims_out": 7,
    "Median_Income": 60510
  }'
```

```json
{
  "tier": 3,
  "tier_description": "$10K-$100K - elevated cost, sample for review",
  "confidence": 0.7399,
  "probabilities": {"1": 0.0000, "2": 0.1541, "3": 0.7399, "4": 0.1059, "5": 0.0000},
  "top_contributions": [
    {"feature": "PRNCPAL_DGNS_CD_inp_freq", "contribution": 1.199069, "direction": "increases"},
    {"feature": "Number_of_Claims_out", "contribution": 0.628999, "direction": "increases"},
    {"feature": "Median_Income", "contribution": 0.353163, "direction": "increases"}
  ],
  "model_name": "xgboost"
}
```

---

## Dataset

Synthetic CMS-style claims data — no real patient information, no PHI.

**`Patient_Claim_Data.xlsx`** — 46,059 rows × 10 columns, no nulls. Derived by
the legacy notebook from CMS inpatient (58,066 × 197) and outpatient
(575,092 × 162) extracts, which are too large to redistribute and are not in
this repository. This merged file is the ingestion boundary.

| Column | Type | Notes |
|---|---|---|
| `BENE_ID` | int | Beneficiary key — **repeats**, not a row key |
| `PRNCPAL_DGNS_CD_inp` / `_out` | str | ICD-10, 174 / 211 distinct values |
| `CLM_E_POA_IND_SW1` | str | Present on admission, `Y` / `U` |
| `PRVDR_STATE_CD_inp` / `_out` | int | FIPS-style, identical on every row |
| `Number_of_Claims_inp` / `_out` | int | Claim counts |
| `CLM_TOT_CHRG_AMT_inp` / `_out` | float | Charge amounts |

**`Median_Income.xlsx`** — 51 rows: state code, name, median household income.

**Target:** `TOTAL_CHARGE = CLM_TOT_CHRG_AMT_inp + CLM_TOT_CHRG_AMT_out`,
modeled as `log(TOTAL_CHARGE)` for regression and binned into `TC_class` for
classification.

---

## Tech stack

| Layer | Tool |
|---|---|
| Orchestration | ZenML |
| Experiment tracking | MLflow (local SQLite; DagsHub optional via env var) |
| Data versioning | DVC (local remote) |
| Validation | pandera |
| Modeling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Monitoring | Evidently |
| Serving | FastAPI, Uvicorn, Pydantic v2 |
| Dashboard | Streamlit (hosted on Streamlit Community Cloud) |
| Packaging | Docker, Docker Compose |
| Quality | pytest, ruff, pre-commit |
| CI | GitHub Actions |

Every component is free and open source. No paid service, subscription, or API
key is needed to run or view any part of this project.

---

## Testing

48 tests covering schema validation, the grouped split, encoder behaviour, the
selection rule, SHAP explainer dispatch, API endpoints, in-process inference, and
the dashboard booting and scoring with no API running.

The leakage tests were **mutation-checked** rather than trusted. Reverting to a
naive row split, zeroing the unseen-category fallback, restoring the strict POA
encoder, and switching frequency encoding to raw counts each make a specific
test fail. Two tests were rewritten after that check found them passing against
a deliberately broken implementation — one compared the encoder against itself,
and one had no test-only categories to detect.

A test that cannot fail is not a test.

---

## Project layout

```
src/healthcare_mlops/   config, ingestion, validation, features, models,
                        evaluation, SHAP, ZenML pipeline
app/api/                FastAPI service
app/dashboard/          Streamlit dashboard (scores in-process)
monitoring/             Evidently drift reporting
scripts/                smoke test, service launcher, Space deployment
tests/                  pytest suite
exported_model/         committed production models + SHAP background
legacy/                 original R script, notebook, figures, write-ups
```

---

## Author

**Manoj Mareedu**

- GitHub: <https://github.com/ManojMareedu>
- LinkedIn: <https://www.linkedin.com/in/manojmareedu/>
