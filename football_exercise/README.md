# Football Analysis Exercise – AI Exercise 1

## Overview
Exploratory data analysis on international football results from 1872 to 2024.
Covers basic exploration, goals analysis, match result classification, and visualizations.

## Dataset
- Source: [Kaggle – International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- File: `data/results.csv`
- Columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country`, `neutral`

## Project Structure
```
football_exercise/
├── data/
│   └── results.csv
├── notebooks/
│   ├── Football_Analysis.ipynb
│   ├── hist_goals.png
│   ├── bar_outcomes.png
│   └── bar_top10_wins.png
├── README.md
└── requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Football_Analysis.ipynb
```

## Key Findings
- ~45,000 international matches spanning 1872–2024
- Average goals per match: ~2.7
- Most common match total: 2 goals (right-skewed distribution)
- Home wins account for ~48–50% of all matches → **home advantage confirmed**
- Brazil, England, Germany dominate all-time wins

## Dependencies
See `requirements.txt`
