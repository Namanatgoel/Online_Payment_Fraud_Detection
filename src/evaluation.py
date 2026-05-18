"""
Computes Precision, Recall, F1-Score, and ROC-AUC for a fitted binary
classifier, and saves a two-panel diagnostic figure (Confusion Matrix
heatmap + Precision-Recall curve) to the outputs directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


EVAL_CONFIG: dict = {
    "output_path": "outputs/fraud_evaluation.png",
    "figsize": (12, 5),
    "dpi": 150,
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
):
    """Compute binary classification metrics.

    Args:
        y_true: 1-D integer array of ground-truth labels.
        y_pred: 1-D integer array of hard predictions (threshold 0.5).
        y_prob: 1-D float array of predicted positive-class probabilities.

    Returns:
        Dictionary with keys 'precision', 'recall', 'f1', 'roc_auc', each
        mapped to a float rounded to 4 decimal places.

    Raises:
        ValueError: If arrays have mismatched lengths or contain unexpected
            values.
    """
    metrics: dict[str, float] = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }
    return metrics


def save_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    output_path: str = EVAL_CONFIG["output_path"],
):
    """Generate and save a confusion matrix heatmap and Precision-Recall curve.

    The figure contains two panels side by side:
      - Left : Confusion Matrix heatmap with count annotations.
      - Right: Precision-Recall curve with Average Precision annotation.

    Args:
        y_true: 1-D integer array of ground-truth labels.
        y_pred: 1-D integer array of hard class predictions.
        y_prob: 1-D float array of predicted positive-class probabilities.
        model_name: Display name for plot titles (e.g. 'Random Forest').
        output_path: Absolute or relative file path for the output PNG.
            Parent directories are created if absent.

    Returns:
        None

    Raises:
        OSError: If the output directory cannot be created.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cm: np.ndarray = confusion_matrix(y_true, y_pred)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    avg_precision: float = float(average_precision_score(y_true, y_prob))

    fig, axes = plt.subplots(1, 2, figsize=EVAL_CONFIG["figsize"])

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=axes[0],
    )
    axes[0].set_title(f"Confusion Matrix — {model_name}")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    axes[1].plot(recall_vals, precision_vals, lw=2,
                 label=f"AP = {avg_precision:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {model_name}")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=EVAL_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] Plots saved → {os.path.abspath(output_path)}")


def print_cv_summary(
    model_name: str,
    cv_results: dict[str, np.ndarray],
):
    """Print a formatted cross-validation summary table to stdout.

    Args:
        model_name: Display name for the model being summarised.
        cv_results: Dictionary from sklearn cross_validate containing arrays
            keyed by 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc'.

    Returns:
        None
    """
    print(f"\n{'─'*50}")
    print(f"  Cross-Validation Summary: {model_name}")
    print(f"{'─'*50}")
    metric_map: dict[str, str] = {
        "test_precision": "Precision",
        "test_recall":    "Recall   ",
        "test_f1":        "F1-Score ",
        "test_roc_auc":   "ROC-AUC  ",
    }
    for key, label in metric_map.items():
        scores: np.ndarray = cv_results.get(key, np.array([]))
        if len(scores):
            fold_str = "  ".join(f"{s:.3f}" for s in scores)
            print(f"  {label}: [{fold_str}]  mean={scores.mean():.3f}  std={scores.std():.3f}")
    print(f"{'─'*50}")
