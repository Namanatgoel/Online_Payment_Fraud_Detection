"""
Entry point for the Online Payment Fraud Detection Pipeline.
Orchestration order:
    1. Chunked stratified data ingestion       (pipeline)
    2. Feature preprocessing & encoding        (pipeline)
    3. Train/test split
    4. 5-fold StratifiedKFold cross-validation (train)
    5. Final model fit on training split       (train)
    6. Hold-out evaluation & metric logging    (evaluation)
    7. Save diagnostic plots                   (evaluation)
Usage:
    python main.py --data onlinefraud.csv
    python main.py --data /path/to/onlinefraud.csv --output outputs/fraud_evaluation.png
"""

import argparse
import sys
import numpy as np

from sklearn.model_selection import train_test_split

from src.pipeline import build_stratified_subset, preprocess_features
from src.train import (
    build_decision_tree,
    build_random_forest,
    cross_validate_model,
    train_final_model,
)
from src.evaluation import compute_metrics, save_evaluation_plots, print_cv_summary


MAIN_CONFIG: dict = {
    "test_size": 0.2,
    "random_state": 42,
    "default_csv": "onlinefraud.csv",
    "default_output": "outputs/fraud_evaluation.png",
}

def parse_args():
    """Parse command-line arguments.
    Returns:
        Namespace with attributes:
            - data (str): Path to the onlinefraud CSV file.
            - output (str): Output path for the evaluation PNG.
    """
    parser = argparse.ArgumentParser(
        description="Online Payment Fraud Detection - Decision Tree & Random Forest"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=MAIN_CONFIG["default_csv"],
        help="Path to onlinefraud.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=MAIN_CONFIG["default_output"],
        help="Output path for evaluation PNG (default: %(default)s)",
    )
    return parser.parse_args()

def main():
    """
    Returns:
        None
    Raises:
        SystemExit: On unrecoverable errors (file not found, zero fraud rows).
    """
    args: argparse.Namespace = parse_args()

    print("[1/7] Streaming CSV and building stratified subset...")
    try:
        df = build_stratified_subset(args.data)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] Data loading failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[2/7] Preprocessing features...")
    X, y, feature_names = preprocess_features(df)
    print(f"      Feature matrix shape: {X.shape}")
    print(f"      Fraud prevalence in subset: {y.mean():.4f}")
    print(f"      Features: {feature_names}")

    print("[3/7] Splitting into train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MAIN_CONFIG["test_size"],
        random_state=MAIN_CONFIG["random_state"],
        stratify=y,
    )
    print(f"      Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    print("[4/7] Running 5-fold Stratified K-Fold cross-validation...")

    dt = build_decision_tree()
    rf = build_random_forest()

    print("      Cross-validating Decision Tree...")
    dt_cv = cross_validate_model(dt, X_train, y_train)
    print_cv_summary("Decision Tree (class_weight=balanced, max_depth=10)", dt_cv)

    print("      Cross-validating Random Forest...")
    rf_cv = cross_validate_model(rf, X_train, y_train)
    print_cv_summary("Random Forest (class_weight=balanced, n_estimators=10, max_depth=10)", rf_cv)

    print("[5/7] Fitting final models on full training split...")
    dt_fitted = train_final_model(build_decision_tree(), X_train, y_train)
    rf_fitted = train_final_model(build_random_forest(), X_train, y_train)

    print("[6/7] Evaluating on held-out test set...")

    y_pred_dt  = dt_fitted.predict(X_test)
    y_prob_dt  = dt_fitted.predict_proba(X_test)[:, 1]
    dt_metrics = compute_metrics(y_test, y_pred_dt, y_prob_dt)

    y_pred_rf  = rf_fitted.predict(X_test)
    y_prob_rf  = rf_fitted.predict_proba(X_test)[:, 1]
    rf_metrics = compute_metrics(y_test, y_pred_rf, y_prob_rf)

    header = f"{'Metric':<12} {'Decision Tree':>15} {'Random Forest':>15}"
    separator = "─" * len(header)
    print(f"\n{separator}")
    print(f"  Hold-Out Test Performance")
    print(separator)
    print(f"  {header}")
    print(f"  {separator}")
    for metric in ["precision", "recall", "f1", "roc_auc"]:
        print(
            f"  {metric.upper():<12}"
            f" {dt_metrics[metric]:>15.4f}"
            f" {rf_metrics[metric]:>15.4f}"
        )
    print(separator)

    print("[7/7] Saving evaluation plots...")
    save_evaluation_plots(
        y_true=y_test,
        y_pred=y_pred_rf,
        y_prob=y_prob_rf,
        model_name="Random Forest",
        output_path=args.output,
    )
    print("\n Pipeline complete !")

if __name__ == "__main__":
    main()