# LendingClub Loan Default Risk Analysis

An end-to-end machine learning project predicting loan defaults using 1.35 million LendingClub loans (2007–2018). Built for both technical rigour and business impact.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Dataset | 1,348,092 loans (2007–2018) |
| Best Model | XGBoost |
| ROC-AUC | 0.7170 |
| F2 Score | optimised at threshold = 0.35 |
| Default Catch Rate | 88.8% |
| Default Rate Reduction | 64% among approved loans |
| Total Applications Analysed | 29.9M (including 27.6M rejected) |

---

## Project Structure

```
LendingClub_Loan/
│
├── notebooks/
│   ├── 01_data_loading.ipynb           # Load, filter, map binary target
│   ├── 02_eda.ipynb                    # Exploratory data analysis & hypothesis testing
│   ├── 03_preprocessing.ipynb          # Feature engineering, encoding, train/test split
│   ├── 03b_feature_significance.ipynb  # Statistical tests: Point-Biserial, Mann-Whitney, Chi-Square, VIF
│   ├── 04_modeling.ipynb               # Train LR, Random Forest, XGBoost; threshold optimisation
│   ├── 05_evaluation.ipynb             # Final evaluation, confusion matrix, business impact
│   └── 06_rejected_analysis.ipynb      # Analyse 27.6M rejected applications
│
├── scripts/
│   └── generate_report.py             # Generates full DOCX project report
│
├── docs/
│   └── LendingClub_Loan_Default_Risk_Report_v4.docx   # Full project report
│
└── data/                              # (not committed — see Data Setup below)
    ├── accepted_2007_to_2018Q4.csv
    ├── rejected_2007_to_2018Q4.csv
    └── threshold.txt
```

---

## Notebooks Walkthrough

### 01 — Data Loading
- Loads the full accepted loans CSV (2.26M rows, 151 columns)
- Filters to resolved loans only (drops "Current", "In Grace Period", etc.)
- Maps `loan_status` to binary target: `1 = Default`, `0 = Fully Paid`
- Saves `loans_filtered.parquet` (1,348,092 rows)

### 02 — Exploratory Data Analysis
Tests five domain-driven hypotheses about what drives defaults:

| Hypothesis | Finding |
|---|---|
| H1: Higher DTI → more defaults | CONFIRMED — DTI in top 5 features |
| H2: Shorter credit history → more defaults | WEAK — only 0.83yr difference |
| H3: Loan purpose affects default rate | CONFIRMED — Chi² = 4,182, p ≈ 0 |
| H4: Lower income → more defaults | CONFIRMED — $5K median income gap |
| H5: More derogatory marks → more defaults | WEAK — converted to binary flag |

### 03 — Preprocessing
- Feature engineering: `earliest_cr_line` → `credit_history_years`; `pub_rec` → `has_derogatory` (binary flag)
- Drops features with data leakage risk or post-loan information
- One-hot encoding for categoricals (drop_first=True)
- StandardScaler fit on train only, applied to test
- Stratified 80/20 train/test split (preserves class ratio)

### 03b — Feature Significance Testing
Validates every feature statistically before modelling:
- **Point-Biserial Correlation** — linear relationship with target
- **Mann-Whitney U Test** — distribution difference between default / paid groups
- **Chi-Square Test** — independence of categorical features from target
- **Correlation Matrix + VIF** — multicollinearity check

Key finding: `loan_amnt` ↔ `installment` (r = 0.953), `int_rate` ↔ `grade` (r = 0.952) → dropped `installment` and `total_acc`

### 04 — Modelling

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.6996 |
| Random Forest | 0.7072 |
| **XGBoost** | **0.7170** |

Threshold optimised using **F2-score** (recall weighted 2× over precision) to prioritise catching defaults. Optimal threshold = **0.35**.

### 05 — Evaluation
- Confusion matrix at threshold 0.35
- ROC and Precision-Recall curves
- Feature importance (top 15 features)
- Business impact summary: default rate reduced from 20.0% → 7.2% among approved loans

### 06 — Rejected Applications Analysis
Analyses LendingClub's 27.6M rejected applications (2007–2018) to complete the full picture:

Key findings:
- **92.4%** of all applications were rejected before a single dollar was lent
- **70.2%** of rejected applicants had < 1 year of employment (vs 8.5% of accepted) — the single biggest differentiator
- **62.2%** of rejected applicants had credit scores below 670 (Very Poor or Fair)
- Rejected applicants requested *less* money ($10K median vs $12K) — loan amount was not the rejection driver
- Rejection volume grew from 5,274 in 2007 to **9.5 million** in 2018

---

## Setup & Running

### 1. Download the Data
Download the two CSV files from [Kaggle — LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club):
- `accepted_2007_to_2018Q4.csv`
- `rejected_2007_to_2018Q4.csv`

Place both in the `data/` folder.

### 2. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install pandas numpy matplotlib seaborn scikit-learn xgboost pyarrow \
            scipy statsmodels python-docx jupyter
```

### 3. Run Notebooks in Order

```
01_data_loading → 02_eda → 03_preprocessing → 03b_feature_significance
→ 04_modeling → 05_evaluation → 06_rejected_analysis
```

### 4. Generate the Report (optional)

```bash
python scripts/generate_report.py
```

Output: `docs/LendingClub_Loan_Default_Risk_Report_v4.docx`

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy, pyarrow |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn, XGBoost |
| Statistics | scipy, statsmodels |
| Reporting | python-docx |
| Notebooks | Jupyter |

---

## Business Impact

> For every 100 loan applications the model screens:
> - **64 fewer defaults** reach the approved portfolio (default rate 20% → 7.2%)
> - **88.8%** of actual defaults are caught and rejected
> - The model adds a second layer of risk screening on top of LendingClub's existing rules-based system, catching complex non-linear default patterns that simple rules miss

---

## Key Learnings

- **Class imbalance** handled via `scale_pos_weight` in XGBoost (~4:1 ratio)
- **Threshold optimisation** using F2-score delivered better business outcomes than default 0.5 cutoff
- **Feature leakage** prevention — only pre-loan application features used
- **Multicollinearity** addressed via VIF analysis before modelling
- **Rejected loans** provide critical context: the accepted applicants our model sees are already pre-screened — yet 20% still default, justifying ML over simple rules

---

## Author

Vivek — [LinkedIn](https://www.linkedin.com/in/vivekanandanandam) | [GitHub](https://github.com/vivek0402)
