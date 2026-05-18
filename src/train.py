import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

TRAIN_CONFIG: dict = {
    "decision_tree": {
        "class_weight": "balanced",
        "max_depth": 10,
        "random_state": 42,
    },
    "random_forest": {
        "class_weight": "balanced",
        "n_estimators": 10,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1,
    },
    "cv_n_splits": 5,
    "cv_random_state": 42,
    "cv_shuffle": True,
}

def build_decision_tree():
    """Instantiate the Decision Tree classifier with production configuration.
    Returns:
        Configured but unfitted DecisionTreeClassifier instance.
    """
    cfg: dict = TRAIN_CONFIG["decision_tree"]
    return DecisionTreeClassifier(
        class_weight=cfg["class_weight"],
        max_depth=cfg["max_depth"],
        random_state=cfg["random_state"],
    )


def build_random_forest():
    """Instantiate the Random Forest classifier with production configuration.
    Returns:
        Configured but unfitted RandomForestClassifier instance.
    """
    cfg: dict = TRAIN_CONFIG["random_forest"]
    return RandomForestClassifier(
        class_weight=cfg["class_weight"],
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
    )


def cross_validate_model(
    model: DecisionTreeClassifier | RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
):
    """
    Run 5-fold stratified cross-validation and return per-fold scores.
    The scoring dictionary includes: precision, recall, f1, and roc_auc.
    Args:
        model: An unfitted sklearn estimator compatible with predict_proba.
        X: 2-D float array of features, shape (n_samples, n_features).
        y: 1-D integer array of binary labels, shape (n_samples,).
    Returns:
        Dictionary mapping metric names to 1-D arrays of per-fold scores.
        Keys: 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc'.
    """
    skf: StratifiedKFold = StratifiedKFold(
        n_splits=TRAIN_CONFIG["cv_n_splits"],
        shuffle=TRAIN_CONFIG["cv_shuffle"],
        random_state=TRAIN_CONFIG["cv_random_state"],
    )
    cv_results: dict = cross_validate(
        model, X, y,
        cv=skf,
        scoring=["precision", "recall", "f1", "roc_auc"],
        return_train_score=False,
        n_jobs=-1,
    )
    return cv_results


def train_final_model(
    model: DecisionTreeClassifier | RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    """Fit the model on the full training split.
    Args:
        model: An unfitted sklearn estimator.
        X_train: 2-D float array of training features.
        y_train: 1-D integer array of training labels.
    Returns:
        The fitted estimator (same object, mutated in-place by sklearn).
    """
    model.fit(X_train, y_train)
    return model