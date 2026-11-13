# ICS 485 Project: Dataset Analysis and Preprocessing

Exploratory analysis and a reproducible preprocessing pipeline for a highly imbalanced binary classification dataset, built for the ICS 485 machine learning course at KFUPM.

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

> **Scope:** this repository contains the data-analysis and preprocessing stage (`Part-A.ipynb`). Model training and evaluation are not included.

## The problem

| | |
|---|---|
| Task | Binary classification of `y` (`square` vs `circle`) |
| Data | 4,480 labelled training rows and 792 unlabelled test rows, 11 features (`f1`–`f11`) |
| Class balance | `square` 4,181 vs `circle` 299, a **13.98 : 1** imbalance (the minority class is 6.7%) |
| Target metric | F1-score of the minority class |
| Features | 10 numeric and 1 categorical (`f11`, four levels `C1`–`C4`) |

## What the notebook does

**1. Dataset analysis**
- Class distribution, missing values, feature distributions, correlation structure, outliers and class separability.
- Findings that drove the pipeline:
  - **7,134 missing values (15.9% of all cells)**, concentrated in `f10` (87.3% missing), `f1` (41.0%) and `f3` (30.9%).
  - **Near-perfect multicollinearity:** `f2` and `f7` correlate at 1.000, and `f2`/`f9`/`f7` and `f5`/`f6` are also highly correlated.
  - **Outliers** in several numeric features, which motivated robust imputation.

**2. Preprocessing pipeline**
- **Missing values:** `IterativeImputer` (MICE) for the numeric features, plus `*_missing_flag` indicators for the heavily missing ones, since missingness may itself be informative. The categorical feature is filled with its mode. Result: 0 missing values remaining.
- **Feature selection:** drop the non-predictive `ID` column and the redundant `f7` (r = 1.000 with `f2`), `f6` and `f5`.
- **Class imbalance:** a hybrid strategy. SMOTE is applied to the training data only, partially balancing the classes from 299 to 1,393 minority samples (25% of the set, +1,094 synthetic rows), and `class_weight='balanced'` is planned for the training stage to avoid over-generating synthetic data.
- **Encoding and scaling:** categorical encoding of `f11`, then `StandardScaler`. The identical transformation is applied to the test set.

## Files

| File | Description |
|---|---|
| `Part-A.ipynb` | Analysis and preprocessing notebook |
| `dataset-train-vf.csv` / `dataset-test-vf.csv` | Course-provided training and test data |
| `Submission.csv` | Submission template with the test IDs |

## Run it

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn jupyter
jupyter notebook Part-A.ipynb
```

## Tech stack

Python, pandas, NumPy, scikit-learn (`IterativeImputer`, `StandardScaler`), imbalanced-learn (SMOTE), Matplotlib, Seaborn.
