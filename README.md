# Online Payment Fraud Detection

## 1. Project Title & Context

This repository implements a binary fraud-classification pipeline on the
PaySim online payment dataset (~6 million transactions).  Two tree-based
models are trained — `DecisionTreeClassifier` and `RandomForestClassifier` —
using 5-fold stratified cross-validation.

The pipeline corrects three deficiencies present in the experimental jupyter
notebook:

| Deficiency | Jupyter Notebook | Production Fix |
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
