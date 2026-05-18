# Online Payment Fraud Detection

## 1. Project Title & Context

This repository implements a binary fraud-classification pipeline on the
PaySim online payment dataset (~6 million transactions).  Two tree-based
models are trained — `DecisionTreeClassifier` and `RandomForestClassifier` —
using 5-fold stratified cross-validation.

The pipeline corrects three deficiencies present in the original experimental
notebook:

| Deficiency | Original | Production Fix |
|---|---|---|
| Data truncation | First 100,000 rows only (sequential bias) | Chunked stratified streaming — 500,000 rows preserving true class ratio |
| Categorical encoding | `LabelEncoder` (ordinal treatment of `type`) | `pd.get_dummies` One-Hot Encoding |
| Class imbalance | No penalty weighting → Recall = 0.391 | `class_weight='balanced'` on both classifiers |

---

## 2. System Architecture

```
fraud-detection-tree/
├── src/
│   ├── __init__.py
│   ├── pipeline.py       # Chunked streaming, stratified sampling, OHE
│   ├── train.py          # Balanced DT & RF, StratifiedKFold CV
│   └── evaluation.py     # Metrics dict, confusion matrix, PR curve
├── main.py               # Pipeline orchestrator
├── outputs/
│   └── fraud_evaluation.png   # Generated at runtime
└── README.md
```

---

## 3. Engineering Optimisations

### 3.1 Chunked Stratified Data Parsing

The original `nrows=100000` read loaded only the opening rows of the CSV.
Because the dataset is ordered by time step, early rows contain a
disproportionately small number of fraud events, destroying the true class
ratio before any model sees the data.

The production `pipeline.build_stratified_subset` function executes two
streaming passes:

**Pass 1** — Stream the entire file in 100,000-row chunks to compute the
global fraud prevalence ratio `p = count(isFraud==1) / total_rows`.

**Pass 2** — Stream again, collecting all fraud rows and all non-fraud rows
into separate buffers.  Draw a random sample of each class proportional to
`p` such that the combined sample contains 500,000 rows and exactly mirrors
the population ratio.

This guarantees that both classifiers are exposed to a representative
frequency of fraud events regardless of their position in the raw file.

### 3.2 Elimination of 0.391 Recall via `class_weight='balanced'`

With the truncated sequential sample, fraud examples constituted fewer than
0.2 % of training rows.  Both classifiers defaulted to predicting "Legit"
for nearly every input, achieving high precision (1.000) but near-zero recall
(0.391).

Setting `class_weight='balanced'` instructs sklearn to compute per-class
loss weights as:

```
w_class = n_samples / (n_classes * n_samples_in_class)
```

This gives the minority fraud class a weight proportional to its inverse
frequency, penalising the model heavily for missing fraud events.  Combined
with the stratified 500,000-row dataset, this resolves the recall collapse.

### 3.3 One-Hot Encoding Replacing Label Encoding

The original notebook applied `LabelEncoder` to the `type` column, mapping
transaction types (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN) to integers
0–4.  This imposed an ordinal relationship (e.g. CASH_OUT > PAYMENT
numerically) that does not exist semantically, biasing tree splits.

The production pipeline replaces this with `pd.get_dummies`, producing one
binary column per transaction type with no ordinal assumption.

---

## 4. Execution Instructions

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run

```bash
# Assumes onlinefraud.csv is in the working directory
python main.py

# Explicit paths
python main.py --data /path/to/onlinefraud.csv --output outputs/fraud_evaluation.png
```

Cross-validation scores and hold-out metrics are printed to stdout.
The confusion matrix and Precision-Recall curve are saved to `outputs/fraud_evaluation.png`.

---

## 5. Deterministic Metrics Table

Metrics below are populated after execution on the full stratified subset.

| ──────────────────────────────────────────────────────────────────────────────────────────────── |               |               |       |        |            |            |           |
|--------------------------------------------------------------------------------------------------|---------------|---------------|-------|--------|------------|------------|-----------|
| Cross-Validation Summary: Decision Tree (class_weight=balanced, max_depth=10)                    |               |               |       |        |            |            |           |
| ──────────────────────────────────────────────────────────────────────────────────────────────── |               |               |       |        |            |            |           |
| Precision: [0.128                                                                                | 0.121         | 0.129         | 0.123 | 0.127] | mean=0.126 | std=0.003  |           |
| Recall                                                                                           | : [0.913      | 0.874         | 0.864 | 0.845  | 0.837]     | mean=0.866 | std=0.027 |
| F1-Score : [0.225                                                                                | 0.213         | 0.225         | 0.215 | 0.220] | mean=0.219 | std=0.005  |           |
| ROC-AUC                                                                                          | : [0.955      | 0.936         | 0.931 | 0.921  | 0.917]     | mean=0.932 | std=0.013 |
| ──────────────────────────────────────────────────────────────────────────────────────────────── |               |               |       |        |            |            |           |
| ──────────────────────────────────────────────────────────────────────────────────────────────── |               |               |       |        |            |            |           |
| Cross-Validation Summary: Random Forest (class_weight=balanced, n_estimators=10, max_depth=10)   |               |               |       |        |            |            |           |
| ──────────────────────────────────────────────────────────────────────────────────────────────── |               |               |       |        |            |            |           |
| Precision: [0.185                                                                                | 0.164         | 0.160         | 0.159 | 0.160] | mean=0.166 | std=0.010  |           |
| Recall                                                                                           | : [0.874      | 0.903         | 0.874 | 0.893  | 0.827]     | mean=0.874 | std=0.026 |
| F1-Score : [0.306                                                                                | 0.277         | 0.271         | 0.270 | 0.268] | mean=0.278 | std=0.014  |           |
| ROC-AUC                                                                                          | : [0.989      | 0.997         | 0.998 | 0.998  | 0.986]     | mean=0.994 | std=0.005 |
| ──────────────────────────────────────────────────────────────────────────────────────────────── |               |               |       |        |            |            |           |
| ────────────────────────────────────────────                                                     |               |               |       |        |            |            |           |
| Hold-Out Test Performance                                                                        |               |               |       |        |            |            |           |
| ────────────────────────────────────────────                                                     |               |               |       |        |            |            |           |
| Metric                                                                                           | Decision Tree | Random Forest |       |        |            |            |           |
| ────────────────────────────────────────────                                                     |               |               |       |        |            |            |           |
| PRECISION                                                                                        | 0.1409        | 0.1262        |       |        |            |            |           |
| RECALL                                                                                           | 0.8915        | 0.9225        |       |        |            |            |           |
| F1                                                                                               | 0.2434        | 0.2220        |       |        |            |            |           |
| ROC_AUC                                                                                          | 0.9448        | 0.9933        |       |        |            |            |           |
| ────────────────────────────────────────────                                                     |               |               |       |        |            |            |           |
