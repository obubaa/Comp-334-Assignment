# Comp 334 – Artificial Intelligence Assignments

This repository contains assignments completed for **COMP 334 – Artificial Intelligence**. Each folder is a self-contained project with its own dataset, code, and documentation.

---

##  Repository Structure

```
Comp-334-Assignment/
├── football_exercise/      ← AI Exercise 1
├── titanic_assignment/     ← AI Assignment 2
└── README.md               ← You are here
```

---

##  Assignment Summaries

### 1. [Football Analysis Exercise](./football_exercise/)
> **AI Exercise 1** – Exploratory Data Analysis

- **Goal:** Analyze international football results from 1872 to 2024.
- **Techniques:** Data exploration, goals analysis, match result classification, and visualizations.
- **Key Findings:**
  - ~45,000 international matches analyzed
  - Home wins account for ~48–50% of matches, confirming home advantage
  - Brazil, England, and Germany dominate all-time wins
- **Tools:** Python, Pandas, Matplotlib, Seaborn

### 2. [Titanic Survival Prediction](./titanic_assignment/)
> **AI Assignment 2** – Data Cleaning, Feature Engineering & Feature Selection

- **Goal:** Build a predictive model pipeline for Titanic survival prediction.
- **Techniques:** Data cleaning, feature engineering (FamilySize, Title extraction, log transforms), and feature selection (correlation analysis, Random Forest importance).
- **Key Findings:**
  - Fare and Age are the strongest individual predictors
  - Small families survived more than solo travelers or large groups
  - Title feature captures social status and gender simultaneously
- **Tools:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---

##  Getting Started

Each project has its own `requirements.txt`. To run a specific assignment:

```bash
# Navigate to the project folder
cd football_exercise/       # or titanic_assignment/

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| Pandas | Data manipulation & analysis |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning & feature selection |
| Jupyter Notebook | Interactive development environment |

---

##  Author

**TOM OBUBA** – 

---
