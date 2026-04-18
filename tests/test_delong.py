"""Tests for DeLong AUC significance test in hydra.evaluation.metrics."""
import numpy as np
import pandas as pd
import pytest

from hydra.evaluation.metrics import delong_auc_test, delong_pairwise


def test_delong_identical_scores():
    """Identical models should have delta=0 and non-significant p-value."""
    y = np.array([1, 1, 1, 0, 0, 0] * 50)
    rng = np.random.default_rng(42)
    scores = rng.random(len(y))
    result = delong_auc_test(y, scores, scores)
    assert abs(result["delta_auc"]) < 1e-10
    # Identical scores → var_diff=0 → p_value is NaN (edge case)
    assert result["p_value"] > 0.05 or np.isnan(result["p_value"])


def test_delong_different_scores():
    """Perfect vs random model should be significant."""
    rng = np.random.default_rng(42)
    y = np.array([1, 1, 1, 0, 0, 0] * 100)
    perfect_scores = y.astype(float) + rng.normal(0, 0.01, len(y))
    random_scores = rng.random(len(y))
    result = delong_auc_test(y, perfect_scores, random_scores)
    assert result["auc_a"] > result["auc_b"]
    assert result["significant_005"] == True


def test_delong_single_class_returns_nan():
    """All-positive labels should return NaN gracefully."""
    y = np.ones(100, dtype=int)
    scores = np.random.default_rng(42).random(100)
    result = delong_auc_test(y, scores, scores)
    assert "error" in result
    assert np.isnan(result["auc_a"])


def test_delong_pairwise_returns_dataframe():
    rng = np.random.default_rng(42)
    y = np.array([1, 1, 0, 0] * 50)
    model_scores = {
        "model_a": rng.random(len(y)),
        "model_b": rng.random(len(y)),
        "model_c": rng.random(len(y)),
    }
    df = delong_pairwise(y, model_scores)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 3 pairs from 3 models (C(3,2))
    assert "auc_a" in df.columns
    assert "delta_auc" in df.columns


def test_delong_pairwise_single_model():
    """Single model → no pairs → empty DataFrame."""
    y = np.array([1, 0] * 50)
    df = delong_pairwise(y, {"only_model": np.random.default_rng(0).random(100)})
    assert len(df) == 0


def test_delong_result_keys():
    """Check all expected keys are present in result dict."""
    y = np.array([1, 0] * 50)
    rng = np.random.default_rng(42)
    result = delong_auc_test(y, rng.random(100), rng.random(100))
    for key in ("auc_a", "auc_b", "delta_auc", "z_stat", "p_value", "significant_005"):
        assert key in result
