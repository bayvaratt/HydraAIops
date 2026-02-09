from __future__ import annotations

import hashlib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROC_AUC_TOL_SMALL = 0.10
ROC_AUC_TOL_MED = 0.05
ROC_AUC_TOL_LARGE = 0.03
PR_AUC_TOL = 0.10


def compute_pr_auc(y_true, scores):
    return float(average_precision_score(y_true, scores))


def pr_auc_sanity_check(y_true, eps: float = 1e-6):
    y = np.asarray(y_true)
    prevalence = float(np.mean(y)) if len(y) else 0.0
    const_scores = np.full(len(y), prevalence, dtype=float)
    ap = float(average_precision_score(y, const_scores))
    if abs(ap - prevalence) > eps:
        raise RuntimeError("PR-AUC calculation inconsistent — check pos_label / score column")


def compute_roc_auc(y_true, scores, logger):
    if len(np.unique(y_true)) < 2:
        logger.warning("ROC-AUC undefined: only one class present")
        return float("nan")
    if np.var(scores) == 0:
        logger.warning("ROC-AUC undefined: constant scores")
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def compute_brier(y_true, scores):
    return float(brier_score_loss(y_true, scores))


def compute_log_loss(y_true, scores, eps: float = 1e-12):
    scores = np.clip(scores, eps, 1.0 - eps)
    return float(log_loss(y_true, scores, labels=[0, 1]))


def compute_ks_statistic(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    all_scores = np.sort(s)
    cdf_pos = np.searchsorted(pos_sorted, all_scores, side="right") / len(pos_sorted)
    cdf_neg = np.searchsorted(neg_sorted, all_scores, side="right") / len(neg_sorted)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def roc_auc_null_tolerance(y_true) -> float:
    y = np.asarray(y_true)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    min_count = min(n_pos, n_neg)
    if min_count < 200:
        return ROC_AUC_TOL_SMALL
    if min_count < 1000:
        return ROC_AUC_TOL_MED
    return ROC_AUC_TOL_LARGE


def warn_on_constant_scores(scores, pr_auc, prevalence, logger):
    if np.var(scores) == 0:
        logger.warning("constant score detected")
        if abs(pr_auc - prevalence) > 0.05:
            logger.warning("PR-AUC deviates from prevalence under constant scores: pr_auc=%.4f prevalence=%.4f", pr_auc, prevalence)


def majority_baseline_sanity(pr_auc, prevalence, logger):
    if abs(pr_auc - prevalence) > 0.05:
        logger.warning(
            "Majority baseline PR-AUC differs from prevalence: pr_auc=%.4f prevalence=%.4f",
            pr_auc,
            prevalence,
        )


def hash_rows_postprocess(X) -> list[str]:
    if hasattr(X, "toarray"):
        X = X.toarray()
    arr = np.asarray(X, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e12, neginf=-1e12)
    arr = np.ascontiguousarray(arr)
    return [hashlib.sha256(arr[i].tobytes()).hexdigest() for i in range(arr.shape[0])]
