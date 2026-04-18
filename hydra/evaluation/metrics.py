"""Evaluation metrics for binary intrusion detection.

Provides PR-AUC, ROC-AUC, Brier score, log-loss, KS statistic, DeLong
paired AUC tests, and sanity-check utilities for the HYDRA pipeline.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

# Permutation-probe null-tolerance thresholds: how far ROC-AUC may deviate
# from 0.5 under a shuffled-label null before we flag leakage.
# Wider tolerance for smaller test sets (higher sampling variance).
ROC_AUC_TOL_SMALL = 0.10   # minority class < 200 samples
ROC_AUC_TOL_MED = 0.05     # minority class 200–999 samples
ROC_AUC_TOL_LARGE = 0.03   # minority class >= 1000 samples
SMALL_CLASS_THRESHOLD = 200
MEDIUM_CLASS_THRESHOLD = 1000

PR_AUC_TOL = 0.10

# Threshold for flagging PR-AUC / prevalence mismatch in sanity checks
PREVALENCE_DEVIATION_THRESHOLD = 0.05

ArrayLike = Union[np.ndarray, pd.Series, list]


def compute_pr_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Compute area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, scores))


def pr_auc_sanity_check(y_true: ArrayLike, eps: float = 1e-6) -> None:
    """Verify PR-AUC calculation consistency.

    Scores a constant-prevalence vector and checks that AP matches prevalence.
    Raises RuntimeError if the deviation exceeds *eps*.
    """
    y = np.asarray(y_true)
    prevalence = float(np.mean(y)) if len(y) else 0.0
    const_scores = np.full(len(y), prevalence, dtype=float)
    ap = float(average_precision_score(y, const_scores))
    if abs(ap - prevalence) > eps:
        raise RuntimeError("PR-AUC calculation inconsistent — check pos_label / score column")


def compute_roc_auc(
    y_true: ArrayLike, scores: ArrayLike, logger: logging.Logger
) -> float:
    """Compute ROC-AUC, returning NaN for degenerate inputs."""
    if len(np.unique(y_true)) < 2:
        logger.warning("ROC-AUC undefined: only one class present")
        return float("nan")
    if np.var(scores) == 0:
        logger.warning("ROC-AUC undefined: constant scores")
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def compute_brier(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Compute Brier score (mean squared error of probability estimates)."""
    return float(brier_score_loss(y_true, scores))


def compute_log_loss(
    y_true: ArrayLike, scores: ArrayLike, eps: float = 1e-12
) -> float:
    """Compute log-loss with score clipping for numerical stability."""
    scores = np.clip(scores, eps, 1.0 - eps)
    return float(log_loss(y_true, scores, labels=[0, 1]))


def compute_ks_statistic(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Compute Kolmogorov-Smirnov statistic between positive and negative score distributions."""
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


def roc_auc_null_tolerance(y_true: ArrayLike) -> float:
    """Return permutation-probe tolerance based on minority class size."""
    y = np.asarray(y_true)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    min_count = min(n_pos, n_neg)
    if min_count < SMALL_CLASS_THRESHOLD:
        return ROC_AUC_TOL_SMALL
    if min_count < MEDIUM_CLASS_THRESHOLD:
        return ROC_AUC_TOL_MED
    return ROC_AUC_TOL_LARGE


def warn_on_constant_scores(
    scores: ArrayLike, pr_auc: float, prevalence: float, logger: logging.Logger
) -> None:
    """Log a warning if scores are constant and PR-AUC deviates from prevalence."""
    if np.var(scores) == 0:
        logger.warning("constant score detected")
        if abs(pr_auc - prevalence) > PREVALENCE_DEVIATION_THRESHOLD:
            logger.warning("PR-AUC deviates from prevalence under constant scores: pr_auc=%.4f prevalence=%.4f", pr_auc, prevalence)


def majority_baseline_sanity(
    pr_auc: float, prevalence: float, logger: logging.Logger
) -> None:
    """Log a warning if majority-baseline PR-AUC deviates from prevalence."""
    if abs(pr_auc - prevalence) > PREVALENCE_DEVIATION_THRESHOLD:
        logger.warning(
            "Majority baseline PR-AUC differs from prevalence: pr_auc=%.4f prevalence=%.4f",
            pr_auc,
            prevalence,
        )


def _structural_components(
    pos_scores: np.ndarray, neg_scores: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute AUC and DeLong structural components (V01, V10).

    Returns (auc, v_pos, v_neg) where:
      v_pos[i] = P(neg < pos[i]) + 0.5 * P(neg == pos[i])  — shape (n_pos,)
      v_neg[j] = P(pos > neg[j]) + 0.5 * P(pos == neg[j])  — shape (n_neg,)
    """
    neg_sorted = np.sort(neg_scores)
    lo = np.searchsorted(neg_sorted, pos_scores, side="left")
    hi = np.searchsorted(neg_sorted, pos_scores, side="right")
    v_pos = (lo + hi) / 2.0 / len(neg_scores)

    pos_sorted = np.sort(pos_scores)
    lo2 = np.searchsorted(pos_sorted, neg_scores, side="left")
    hi2 = np.searchsorted(pos_sorted, neg_scores, side="right")
    n_pos = len(pos_scores)
    v_neg = 1.0 - (lo2 + hi2) / 2.0 / n_pos

    auc = float(np.mean(v_pos))
    return auc, v_pos, v_neg


def delong_auc_test(
    y_true,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict:
    """DeLong et al. (1988) paired AUC significance test.

    Tests H₀: AUC(A) == AUC(B) on the same test set.

    Parameters
    ----------
    y_true   : binary labels (0/1)
    scores_a : predicted probabilities from model A
    scores_b : predicted probabilities from model B

    Returns
    -------
    dict with keys: auc_a, auc_b, delta_auc, z_stat, p_value, significant_005
    """
    from scipy import stats as _stats

    y = np.asarray(y_true, dtype=int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {
            "auc_a": float("nan"), "auc_b": float("nan"),
            "delta_auc": float("nan"), "z_stat": float("nan"),
            "p_value": float("nan"), "significant_005": False,
            "error": "need both positive and negative samples",
        }

    auc_a, v_pos_a, v_neg_a = _structural_components(sa[pos_idx], sa[neg_idx])
    auc_b, v_pos_b, v_neg_b = _structural_components(sb[pos_idx], sb[neg_idx])

    n_pos, n_neg = len(pos_idx), len(neg_idx)

    # DeLong covariance: S = S_pos / n_pos + S_neg / n_neg
    # where S_pos = cov matrix of [V01_a, V01_b], S_neg = cov of [V10_a, V10_b]
    mat_pos = np.stack([v_pos_a, v_pos_b])  # (2, n_pos)
    mat_neg = np.stack([v_neg_a, v_neg_b])  # (2, n_neg)

    # np.cov with ddof=1; handle single-sample edge case
    cov_pos = np.cov(mat_pos) if n_pos > 1 else np.zeros((2, 2))
    cov_neg = np.cov(mat_neg) if n_neg > 1 else np.zeros((2, 2))

    s11 = cov_pos[0, 0] / n_pos + cov_neg[0, 0] / n_neg
    s22 = cov_pos[1, 1] / n_pos + cov_neg[1, 1] / n_neg
    s12 = cov_pos[0, 1] / n_pos + cov_neg[0, 1] / n_neg

    var_diff = s11 + s22 - 2 * s12
    if var_diff <= 0:
        return {
            "auc_a": float(auc_a), "auc_b": float(auc_b),
            "delta_auc": float(auc_a - auc_b),
            "z_stat": float("nan"), "p_value": float("nan"),
            "significant_005": False,
        }

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * _stats.norm.sf(abs(z)))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "delta_auc": float(auc_a - auc_b),
        "z_stat": float(z),
        "p_value": p,
        "significant_005": p < 0.05,
    }


def delong_pairwise(
    y_true,
    model_scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    """All-pairs DeLong AUC tests.

    Parameters
    ----------
    y_true        : binary labels
    model_scores  : {model_name: score_array} — same test set for all

    Returns
    -------
    DataFrame indexed by (model_a, model_b) with DeLong result columns.
    """
    models = sorted(model_scores)
    rows = []
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            if i >= j:
                continue
            result = delong_auc_test(y_true, model_scores[ma], model_scores[mb])
            result["model_a"] = ma
            result["model_b"] = mb
            rows.append(result)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["model_a", "model_b"])


def hash_rows_postprocess(X) -> list[str]:
    if hasattr(X, "toarray"):
        X = X.toarray()
    arr = np.asarray(X, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e12, neginf=-1e12)
    arr = np.ascontiguousarray(arr)
    return [hashlib.sha256(arr[i].tobytes()).hexdigest() for i in range(arr.shape[0])]
