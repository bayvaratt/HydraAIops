import numpy as np
import pytest

from hydra.xai.record import (
    compute_faithfulness,
    compute_simplicity,
    compute_stability,
    compute_plausibility,
    run_xai_eval,
)


def _dummy_predict_fn(X):
    # simple linear decision boundary on first feature
    return np.clip(0.1 * X[:, 0] + 0.5, 0.0, 1.0)


def _dummy_explain_fn(X):
    # attribution strongest for feature 0
    return np.vstack((X[:, 0], np.zeros_like(X[:, 1]))).T


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

def test_compute_faithfulness_baseline_modes():
    X = np.array([
        [1.0, 0.2, 0.3],
        [2.0, 0.1, 0.0],
        [0.5, 0.4, 0.6],
    ], dtype=np.float32)
    attr = np.array([
        [1.0, 0.1, 0.0],
        [1.5, 0.05, 0.0],
        [0.5, 0.2, 0.0],
    ], dtype=np.float32)

    for baseline in ["zero", "mean", "median", "min", "max"]:
        res = compute_faithfulness(_dummy_predict_fn, X, attr, top_k_values=(1, 2), max_samples=10, baseline=baseline)
        assert "comprehensiveness_k1" in res
        assert "sufficiency_k1" in res


def test_sufficiency_unclamped_field_present_when_exceeds_1():
    # Construct a case where sufficient features produce higher confidence than all features.
    # predict_fn that scores higher when only feature 0 is present (i.e., feature 1 suppresses).
    def suppress_predict(X):
        return np.clip(0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.5, 0.0, 1.0)

    X = np.array([[2.0, 0.1], [2.0, 0.1], [2.0, 0.1]], dtype=np.float32)
    attr = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    res = compute_faithfulness(suppress_predict, X, attr, top_k_values=(1,), max_samples=10, baseline="zero")
    # If sufficiency > 1.0, unclamped field is saved; clamped field is always <= 1.0
    assert res["sufficiency_k1"] <= 1.0
    if "sufficiency_unclamped_k1" in res:
        assert res["sufficiency_unclamped_k1"] > 1.0


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def test_stability_all_zero_attributions():
    # Explainer always returns zeros — Spearman undefined, should return NaN gracefully.
    def zero_explain(X):
        return np.zeros((len(X), 3), dtype=np.float32)

    X = np.random.default_rng(0).random((10, 3)).astype(np.float32)
    res = compute_stability(zero_explain, X, n_samples=5, n_perturb=3)
    assert np.isnan(res["mean_spearman_rank_corr"])


def test_stability_perfect_stability():
    # Explainer always returns the same vector regardless of input noise → Spearman = 1.0.
    def constant_explain(X):
        out = np.zeros((len(X), 3), dtype=np.float32)
        out[:, 0] = 1.0
        return out

    X = np.random.default_rng(1).random((10, 3)).astype(np.float32)
    res = compute_stability(constant_explain, X, n_samples=5, n_perturb=3)
    assert res["mean_spearman_rank_corr"] == pytest.approx(1.0, abs=1e-3)


def test_stability_returns_nan_when_no_valid_pairs():
    # Explainer always raises — should return NaN, not crash.
    def broken_explain(X):
        raise RuntimeError("broken")

    X = np.random.default_rng(2).random((5, 3)).astype(np.float32)
    res = compute_stability(broken_explain, X, n_samples=3, n_perturb=2)
    assert np.isnan(res["mean_spearman_rank_corr"])


# ---------------------------------------------------------------------------
# Simplicity
# ---------------------------------------------------------------------------

def test_simplicity_uniform_attributions():
    # Uniform attributions → Gini = 0 (perfectly equal), k90_frac = high.
    attr = np.ones((5, 10), dtype=np.float32)
    res = compute_simplicity(attr)
    assert res["gini_coeff"] == pytest.approx(0.0, abs=1e-3)
    assert res["k90_frac"] > 0.5


def test_simplicity_single_feature_dominates():
    # All weight on feature 0 → Gini close to 1, k90_frac = 1/n_features.
    attr = np.zeros((5, 10), dtype=np.float32)
    attr[:, 0] = 1.0
    res = compute_simplicity(attr)
    assert res["gini_coeff"] > 0.8
    assert res["k90_frac"] == pytest.approx(1 / 10, abs=1e-3)


def test_simplicity_all_zero_returns_nan_k90():
    # All-zero attributions → k90_frac is NaN (no valid rows), gini is 0.0 (clamped via max).
    attr = np.zeros((5, 4), dtype=np.float32)
    res = compute_simplicity(attr)
    assert np.isnan(res["k90_frac"])
    assert res["gini_coeff"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Plausibility
# ---------------------------------------------------------------------------

def test_plausibility_all_features_match():
    attr = np.array([[1.0, 0.5, 0.2]], dtype=np.float32)
    feat_names = ["num__src_bytes", "num__duration", "num__dst_bytes"]
    expert_kws = ["src_bytes", "duration", "dst_bytes"]
    res = compute_plausibility(attr, feat_names, expert_kws, top_k=3)
    assert res["rma_at_k"] == pytest.approx(1.0)


def test_plausibility_no_features_match():
    attr = np.array([[1.0, 0.5, 0.2]], dtype=np.float32)
    feat_names = ["num__foo", "num__bar", "num__baz"]
    expert_kws = ["src_bytes", "duration"]
    res = compute_plausibility(attr, feat_names, expert_kws, top_k=3)
    assert res["rma_at_k"] == pytest.approx(0.0)


def test_plausibility_empty_expert_list():
    attr = np.array([[1.0, 0.5, 0.2]], dtype=np.float32)
    feat_names = ["num__src_bytes", "num__duration", "num__dst_bytes"]
    res = compute_plausibility(attr, feat_names, expert_feature_keywords=[], top_k=3)
    assert res["rma_at_k"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_xai_eval integration
# ---------------------------------------------------------------------------

def test_run_xai_eval_respects_baseline_parameter():
    X_test = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    res = run_xai_eval(
        _dummy_explain_fn,
        _dummy_predict_fn,
        X_test,
        feature_names=["f0", "f1"],
        dataset_name="ton_iot",
        out_path="/tmp/xai_eval_dummy.json",
        faithfulness_baseline="mean",
        timeliness_n=1,
    )
    assert res is not None
    assert "faithfulness" in res
    assert "stability" in res
