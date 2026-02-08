from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def find_threshold_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_target: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    best_threshold = 0.5
    best_precision = -1.0
    for p, r, t in zip(precision[1:], recall[1:], thresholds):
        if r >= recall_target and p > best_precision:
            best_precision = p
            best_threshold = t
    if best_precision < 0:
        best_threshold = thresholds[np.argmax(recall[1:])]
    return float(best_threshold)


def fpr_at_fixed_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_target: float,
    threshold: float,
) -> float:
    y_pred = (y_score >= threshold).astype(int)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    if fp + tn == 0:
        return 0.0
    return float(fp / (fp + tn))
