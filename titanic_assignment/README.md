# Titanic Survival Prediction – AI Assignment 2

## Overview
This project builds a predictive model pipeline for Titanic survival using data cleaning,
feature engineering, and feature selection on the Kaggle Titanic dataset.

---

## Project Structure

```
titanic_assignment/
├── data/
│   ├── train.csv               ← Original dataset
│   ├── train_cleaned.csv       ← After Part 1 (Data Cleaning)
│   ├── train_engineered.csv    ← After Part 2 (Feature Engineering)
│   └── train_selected.csv      ← After Part 3 (Feature Selection)
│
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb   ← Main notebook (all parts)
│
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
│
├── README.md
└── requirements.txt
```

---

## How to Run

### Option 1: Run scripts directly
```bash
cd titanic_assignment/
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
```

### Option 2: Run the Jupyter Notebook
```bash
pip install jupyter
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

---

## Approach

### Part 1 – Data Cleaning
- **Cabin** dropped (77% missing – not recoverable)
- **Age** imputed with median (robust to skew)
- **Fare** imputed with median (only 1–3 missing)
- **Embarked** imputed with mode (only 2–5 missing)
- Outliers in `Fare` and `Age` capped at the 99th percentile
- Duplicate rows removed

### Part 2 – Feature Engineering
| Feature | Description |
|---------|-------------|
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 if travelling alone |
| `Title` | Extracted from Name (Mr/Mrs/Miss/Master/Other) |
| `AgeGroup` | Child/Teen/Adult/Senior bins |
| `FarePerPerson` | Fare / FamilySize |
| `Fare_log` | log(1 + Fare) — reduces right skew |
| `Age_log` | log(1 + Age) — reduces right skew |
| One-hot columns | Sex, Embarked, Title, AgeGroup |

### Part 3 – Feature Selection
- Correlation analysis to remove redundant features (threshold > 0.95)
- Random Forest feature importance ranking
- Features with importance > 0.01 kept for final model

---

## Key Findings
- **Fare and Age** are the strongest individual predictors
- **Pclass** captures socioeconomic status well
- **FamilySize** matters — small families survived more than solo travelers or large groups
- Log transforms improve distribution normality for distance-based models
- Title feature captures social status and gender simultaneously

---

## Dependencies
See `requirements.txt`
