from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
