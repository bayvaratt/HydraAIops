import math
import numpy as np
import pytest

from hydra.evaluation.metrics import compute_pr_auc, compute_roc_auc, compute_ks_statistic
from hydra.evaluation.thresholds import coverage_at_threshold, fpr_at_threshold, select_threshold_at_recall


class _NullLogger:
    def warning(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Existing test
# ---------------------------------------------------------------------------

def test_threshold_selection_and_fpr():
    y = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])

    thr = select_threshold_at_recall(y, scores, 0.9, _NullLogger())
    fpr = fpr_at_threshold(y, scores, thr)
    cov = coverage_at_threshold(scores, thr)

    assert 0.0 <= fpr <= 1.0
    assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
# ROC-AUC edge cases
# ---------------------------------------------------------------------------

def test_roc_auc_single_class_returns_nan():
    y = np.array([1, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    result = compute_roc_auc(y, scores, _NullLogger())
    assert math.isnan(result)


def test_roc_auc_constant_scores_returns_nan():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    result = compute_roc_auc(y, scores, _NullLogger())
    assert math.isnan(result)


def test_roc_auc_perfect_prediction():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    result = compute_roc_auc(y, scores, _NullLogger())
    assert result == pytest.approx(1.0)


def test_roc_auc_random_prediction():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=1000)
    scores = rng.random(size=1000)
    result = compute_roc_auc(y, scores, _NullLogger())
    # Random classifier should be near 0.5
    assert 0.4 < result < 0.6


# ---------------------------------------------------------------------------
# PR-AUC edge cases
# ---------------------------------------------------------------------------

def test_pr_auc_majority_baseline_equals_prevalence():
    # PR-AUC of a constant-score classifier equals prevalence.
    y = np.array([1, 1, 0, 0, 0])
    prevalence = y.mean()
    scores = np.full(len(y), prevalence)
    result = compute_pr_auc(y, scores)
    assert result == pytest.approx(prevalence, abs=1e-5)


def test_pr_auc_perfect_prediction():
    y = np.array([1, 1, 0, 0])
    scores = np.array([1.0, 0.9, 0.1, 0.0])
    result = compute_pr_auc(y, scores)
    assert result == pytest.approx(1.0)


def test_pr_auc_high_prevalence():
    # CIC-IoT-2023-like scenario: 97% attack prevalence.
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.97).astype(int)
    scores = rng.random(n)
    result = compute_pr_auc(y, scores)
    # With random scores, PR-AUC ≈ prevalence
    assert result == pytest.approx(y.mean(), abs=0.05)


# ---------------------------------------------------------------------------
# KS statistic edge cases
# ---------------------------------------------------------------------------

def test_ks_statistic_perfect_separation():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    result = compute_ks_statistic(y, scores)
    assert result == pytest.approx(1.0, abs=1e-3)


def test_ks_statistic_identical_distributions():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    result = compute_ks_statistic(y, scores)
    assert result == pytest.approx(0.0, abs=1e-3)
