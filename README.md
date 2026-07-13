# Ensemble Anomaly Detection on Medicare Claims

> **Role in the research series:** This is the **thesis core project** — a complete five-notebook end-to-end pipeline.  
> For the baseline algorithm comparison see [`Anomaly-Detection-IF-LOF-OCSVM`](https://github.com/nyan-dev/Anomaly-Detection-IF-LOF-OCSVM) · For the main case-study pipeline see [`Anomaly-Detection-Medicare-Claims`](https://github.com/nyan-dev/Anomaly-Detection-Medicare-Claims)

---

## Overview

This repository implements a **full unsupervised anomaly detection pipeline** for Medicare outpatient claims, culminating in an **ensemble consensus method** that combines Isolation Forest (IF), Local Outlier Factor (LOF), and One-Class SVM (OCSVM) to produce high-confidence anomaly flags for expert review.

This is the primary research repository for a Master's thesis on machine learning-based anomaly detection in healthcare billing data, using the CMS DE-SynPUF 2010 synthetic dataset.

---

## Why Ensemble?

Single anomaly detection models are inherently unstable — each algorithm uses a different definition of "anomalous":
- **IF** isolates points by random partitioning (global, tree-based)
- **LOF** compares local density to neighbours (local, density-based)
- **OCSVM** learns a decision boundary around normal data (kernel-based)

By taking the **intersection of IF and OCSVM flags** (consensus voting), we produce a conservative set of anomalies that two fundamentally different algorithms agree on — reducing false positives and increasing confidence for domain expert review.

---

## Dataset

| Property | Details |
|---|---|
| Source | [CMS DE-SynPUF 2010 Outpatient Claims](https://www.kaggle.com/datasets/kukulauren/cms-desynpuf-2010-outpatient-claims) (Kaggle) |
| Type | **Fully synthetic** — no real patient data, safe for public research |
| Size | ~175,000 outpatient claims after cleaning |
| Features | 14 engineered features (monetary, temporal, utilization, provider-level) |

---

## Five-Notebook Pipeline

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_setup_and_kaggle_download` | Environment setup, Kaggle API config, raw data download and initial inspection |
| 02 | `02_data_preparation_and_feature_engineering` | Data cleaning, quality checks, EDA, and 14-feature engineering |
| 03 | `03_controlled_anomaly_injection_and_baseline_benchmark` | Semi-synthetic anomaly injection (3% contamination), baseline IF/LOF/OCSVM evaluation with PR-AUC, ROC-AUC, Precision@K |
| 04 | `04_tuning_and_ensemble_analysis` | Hyperparameter grid search, stability analysis, IF ∩ OCSVM ensemble consensus definition |
| 05 | `05_expert_review_pack_and_interpretation` | Top-ranked consensus anomalies packaged as human-readable case files for domain expert review and inter-rater reliability assessment |

> Run notebooks **in order** from 01 to 05. Each notebook saves outputs used by the next.

---

## Ensemble Consensus Method

```
IF flags ∩ OCSVM flags = Consensus anomaly set
```

- **Why IF + OCSVM (not LOF)?** LOF has no native `predict()` for unseen data, making it unsuitable for the consensus step. IF and OCSVM are complementary — tree-based vs. kernel-based — so agreement between them is a strong signal.
- **Contamination rate:** 0.05 (5%) used as the anomaly prior for IF and OCSVM
- **Output:** Ranked list of high-confidence anomaly cases with feature-level explanations for expert review

---

## Key Results

| Model | ROC-AUC | PR-AUC | Precision@100 | Recall@100 |
|---|---|---|---|---|
| Isolation Forest | — | — | — | — |
| Local Outlier Factor | — | — | — | — |
| One-Class SVM | — | — | — | — |
| IF ∩ OCSVM Ensemble | — | — | — | — |

> Results will be populated after thesis evaluation run. See Notebook 03 and 04 for current outputs.

---

## Repository Structure

```
Anomaly-Detection-Ensemble/
├── 01_data_setup_and_kaggle_download.ipynb
├── 02_data_preparation_and_feature_engineering.ipynb
├── 03_controlled_anomaly_injection_and_baseline_benchmark.ipynb
├── 04_tuning_and_ensemble_analysis.ipynb
├── 05_expert_review_pack_and_interpretation.ipynb
├── data/
│   └── README.md          # Dataset provenance and feature notes
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/nyan-dev/Anomaly-Detection-Ensemble.git
cd Anomaly-Detection-Ensemble
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle API
```bash
# Place your kaggle.json in ~/.kaggle/
# Then run Notebook 01 to download the dataset automatically
```

### 4. Run notebooks in order
```
01 → 02 → 03 → 04 → 05
```

---

## Limitations

- Dataset is **fully synthetic** — results may not generalise directly to real Medicare claims
- LOF excluded from ensemble consensus due to lack of native `predict()` method
- OCSVM is computationally expensive — applied on a stratified sample for tuning
- Anomaly injection is semi-synthetic; real fraud patterns are more complex

---

## Related Repositories

| Repo | Role |
|---|---|
| [`Anomaly-Detection-IF-LOF-OCSVM`](https://github.com/nyan-dev/Anomaly-Detection-IF-LOF-OCSVM) | Baseline algorithm comparison (IF vs LOF vs OCSVM) |
| **This repo** | Full end-to-end ensemble pipeline — thesis core |
| [`Anomaly-Detection-Medicare-Claims`](https://github.com/nyan-dev/Anomaly-Detection-Medicare-Claims) | Complementary case-study pipeline |

---

## Citation

```
Nyan Lynn Htet. (2026). Ensemble Anomaly Detection in Medicare Outpatient Claims:
An Unsupervised Machine Learning Pipeline with Consensus Voting.
INTI International University. Master's Thesis.
```

---

## Author

**Nyan Lynn Htet** — Master's Researcher, Data Science & Machine Learning  
INTI International University, Malaysia  
[GitHub](https://github.com/nyan-dev)
