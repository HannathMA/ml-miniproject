# Machine Learning on Poverty Data

A machine learning project applying supervised learning, PCA, DBSCAN clustering, and fairness/ethics analysis to a poverty dataset.

## Project Overview

This project is split into three parts:

1. **PAC Learning / Overfitting / PCA / Ethics** — train models of different complexity to demonstrate overfitting, apply PCA for dimensionality reduction, and discuss fairness implications.
2. **DBSCAN Hotspot Detection** — unsupervised clustering to identify geographic poverty hotspots.
3. **Model Comparison** — training vs. testing performance, with and without PCA.

## Dataset

- **Source:** [Add your Kaggle dataset name and link here]
- **Description:** [e.g. "Poverty and Equity Database — contains poverty headcount ratios, Gini index, and inequality indicators by country/region/year."]
- **Rows / Columns:** [fill in after running `01_explore.py`]
- **Target variable:** [e.g. `Poverty_Rate`]

## Project Structure

```
machine-learning-poverty-project/
│
├── data/
│   └── poverty_data.csv          # dataset (not committed if private/large)
│
├── src/
│   ├── 01_explore.py             # inspect columns, missing values, shape
│   ├── 02_pac_learning.py        # preprocessing, overfitting demo, PCA
│   ├── 03_dbscan.py              # DBSCAN hotspot clustering
│   └── 04_model_pca.py           # training vs testing comparison + PCA
│
├── outputs/
│   ├── overfitting_plot.png
│   ├── pca_variance_plot.png
│   └── dbscan_plot.png
│
└── README.md
```

## Setup

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   (macOS/Linux: `source .venv/bin/activate`)

2. Install dependencies:
   ```powershell
   pip install pandas numpy scikit-learn matplotlib
   ```

3. Place the dataset CSV inside `data/`.

## How to Run

Run scripts in order:

```powershell
python src/01_explore.py
python src/02_pac_learning.py
python src/03_dbscan.py
python src/04_model_pca.py
```

Each script prints metrics to the terminal and saves any plots to `outputs/`.

## Methodology

- **Preprocessing:** missing values handled via median imputation (numeric) and mode imputation (categorical); features scaled with `StandardScaler` fit only on training data to avoid leakage; 80/20 train-test split.
- **Overfitting demonstration:** compared a simple model (Linear Regression) against a more complex one (Decision Tree / Random Forest) across training vs. testing performance.
- **PCA:** applied to reduce dimensionality while retaining a target percentage of variance; models were re-evaluated on PCA-transformed features and compared to the original.
- **DBSCAN:** density-based clustering on location + poverty features to detect hotspot regions; `eps` chosen using a k-distance elbow plot; `-1` labels treated as noise/outliers.
- **Fairness check:** prediction error compared across subgroups (e.g. region, sex) to check for uneven model performance.

## Results

| Model | Training R² / Accuracy | Testing R² / Accuracy | Notes |
|---|---|---|---|
| Linear Regression | | | Baseline |
| Decision Tree | | | Overfitting check |
| Random Forest | | | Generalization |
| PCA Model | — | | Effect of dimensionality reduction |


## Ethics & Fairness Notes

- **Representation bias:** some regions/groups may be under- or over-represented in the data.
- **Historical bias:** poverty data reflects existing social and economic inequalities the model may reproduce.
- **Fairness evaluation:** prediction error was compared across groups (see `04_model_pca.py` output) to check for unequal accuracy.
- **Deployment risk:** predictions about poverty could influence real decisions (aid, loans, policy) — errors or bias can cause real harm, so results should not be used for automated decisions without human oversight.

