"""Extended unit tests for metrics and baselines modules."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from hydra.evaluation.metrics import (
    compute_brier,
    compute_log_loss,
    hash_rows_postprocess,
    majority_baseline_sanity,
    pr_auc_sanity_check,
    roc_auc_null_tolerance,
    warn_on_constant_scores,
)
from hydra.models.baselines import baseline_majority_scores, baseline_threshold_scores

logger = logging.getLogger("test")


# ---------------------------------------------------------------------------
# pr_auc_sanity_check
# ---------------------------------------------------------------------------
class TestPrAucSanityCheck:
    def test_well_formed_balanced(self):
        """Should not raise for well-formed balanced data."""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        pr_auc_sanity_check(y)  # no exception

    def test_well_formed_imbalanced(self):
        """Should not raise for imbalanced but well-formed data."""
        y = np.array([0] * 90 + [1] * 10)
        pr_auc_sanity_check(y)

    def test_all_positive(self):
        """Should not raise when all labels are positive."""
        y = np.ones(50)
        pr_auc_sanity_check(y)

    def test_large_eps_no_raise(self):
        """With a very large eps the check is always lenient."""
        y = np.array([0, 1, 0, 1])
        pr_auc_sanity_check(y, eps=10.0)

    def test_list_input(self):
        """Accepts plain python lists."""
        pr_auc_sanity_check([0, 1, 1, 0, 1])


# ---------------------------------------------------------------------------
# compute_brier
# ---------------------------------------------------------------------------
class TestComputeBrier:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1])
        scores = np.array([0.0, 1.0, 0.0, 1.0])
        assert compute_brier(y, scores) == pytest.approx(0.0)

    def test_worst_predictions(self):
        y = np.array([0, 1, 0, 1])
        scores = np.array([1.0, 0.0, 1.0, 0.0])
        assert compute_brier(y, scores) == pytest.approx(1.0)

    def test_random_predictions(self):
        """Constant 0.5 predictions on balanced data → Brier ≈ 0.25."""
        y = np.array([0, 1] * 500)
        scores = np.full(1000, 0.5)
        assert compute_brier(y, scores) == pytest.approx(0.25)

    def test_single_sample(self):
        assert compute_brier(np.array([1]), np.array([0.7])) == pytest.approx(0.09)

    def test_returns_float(self):
        result = compute_brier(np.array([0, 1]), np.array([0.3, 0.8]))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_log_loss
# ---------------------------------------------------------------------------
class TestComputeLogLoss:
    def test_perfect_near_zero(self):
        y = np.array([0, 1, 0, 1])
        scores = np.array([0.0001, 0.9999, 0.0001, 0.9999])
        result = compute_log_loss(y, scores)
        assert result < 0.01

    def test_random_predictions(self):
        y = np.array([0, 1] * 100)
        scores = np.full(200, 0.5)
        result = compute_log_loss(y, scores)
        assert result == pytest.approx(np.log(2), abs=0.01)

    def test_clips_exact_zero(self):
        """Exact 0/1 scores should not cause -inf log; eps clipping handles it."""
        y = np.array([0, 1])
        scores = np.array([0.0, 1.0])
        result = compute_log_loss(y, scores)
        assert np.isfinite(result)

    def test_returns_float(self):
        result = compute_log_loss(np.array([0, 1]), np.array([0.2, 0.8]))
        assert isinstance(result, float)

    def test_custom_eps(self):
        y = np.array([0, 1])
        scores = np.array([0.0, 1.0])
        result_tight = compute_log_loss(y, scores, eps=1e-15)
        result_loose = compute_log_loss(y, scores, eps=1e-3)
        # Looser eps → larger clipping → larger loss
        assert result_loose > result_tight


# ---------------------------------------------------------------------------
# roc_auc_null_tolerance
# ---------------------------------------------------------------------------
class TestRocAucNullTolerance:
    def test_small_minority(self):
        """<200 min class → 0.10."""
        y = np.array([0] * 900 + [1] * 100)
        assert roc_auc_null_tolerance(y) == 0.10

    def test_medium_minority(self):
        """200 <= min class < 1000 → 0.05."""
        y = np.array([0] * 800 + [1] * 500)
        assert roc_auc_null_tolerance(y) == 0.05

    def test_large_minority(self):
        """>= 1000 min class → 0.03."""
        y = np.array([0] * 5000 + [1] * 2000)
        assert roc_auc_null_tolerance(y) == 0.03

    def test_boundary_200(self):
        """Exactly 200 in minority → medium tolerance."""
        y = np.array([0] * 800 + [1] * 200)
        assert roc_auc_null_tolerance(y) == 0.05

    def test_boundary_1000(self):
        """Exactly 1000 in minority → large tolerance."""
        y = np.array([0] * 2000 + [1] * 1000)
        assert roc_auc_null_tolerance(y) == 0.03

    def test_all_same_class(self):
        """All one class → min_count=0 → small tolerance."""
        y = np.ones(500)
        assert roc_auc_null_tolerance(y) == 0.10

    def test_single_element(self):
        y = np.array([1])
        assert roc_auc_null_tolerance(y) == 0.10


# ---------------------------------------------------------------------------
# warn_on_constant_scores
# ---------------------------------------------------------------------------
class TestWarnOnConstantScores:
    def test_constant_scores_with_deviation_warns(self, caplog):
        scores = np.full(100, 0.5)
        with caplog.at_level(logging.WARNING, logger="test"):
            warn_on_constant_scores(scores, pr_auc=0.8, prevalence=0.5, logger=logger)
        assert "constant score detected" in caplog.text
        assert "PR-AUC deviates" in caplog.text

    def test_constant_scores_no_deviation(self, caplog):
        scores = np.full(100, 0.5)
        with caplog.at_level(logging.WARNING, logger="test"):
            warn_on_constant_scores(scores, pr_auc=0.52, prevalence=0.5, logger=logger)
        assert "constant score detected" in caplog.text
        assert "PR-AUC deviates" not in caplog.text

    def test_non_constant_scores_no_warning(self, caplog):
        scores = np.array([0.1, 0.9, 0.3, 0.7])
        with caplog.at_level(logging.WARNING, logger="test"):
            warn_on_constant_scores(scores, pr_auc=0.8, prevalence=0.5, logger=logger)
        assert "constant score" not in caplog.text

    def test_single_element_constant(self, caplog):
        scores = np.array([0.5])
        with caplog.at_level(logging.WARNING, logger="test"):
            warn_on_constant_scores(scores, pr_auc=0.5, prevalence=0.5, logger=logger)
        assert "constant score detected" in caplog.text


# ---------------------------------------------------------------------------
# majority_baseline_sanity
# ---------------------------------------------------------------------------
class TestMajorityBaselineSanity:
    def test_warns_on_large_deviation(self, caplog):
        with caplog.at_level(logging.WARNING, logger="test"):
            majority_baseline_sanity(pr_auc=0.9, prevalence=0.5, logger=logger)
        assert "differs from prevalence" in caplog.text

    def test_no_warning_within_tolerance(self, caplog):
        with caplog.at_level(logging.WARNING, logger="test"):
            majority_baseline_sanity(pr_auc=0.52, prevalence=0.50, logger=logger)
        assert "differs from prevalence" not in caplog.text

    def test_exact_match(self, caplog):
        with caplog.at_level(logging.WARNING, logger="test"):
            majority_baseline_sanity(pr_auc=0.7, prevalence=0.7, logger=logger)
        assert caplog.text == ""

    def test_boundary_just_over(self, caplog):
        with caplog.at_level(logging.WARNING, logger="test"):
            majority_baseline_sanity(pr_auc=0.56, prevalence=0.5, logger=logger)
        assert "differs from prevalence" in caplog.text


# ---------------------------------------------------------------------------
# hash_rows_postprocess
# ---------------------------------------------------------------------------
class TestHashRowsPostprocess:
    def test_basic_array(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        hashes = hash_rows_postprocess(X)
        assert len(hashes) == 2
        assert all(isinstance(h, str) and len(h) == 64 for h in hashes)

    def test_identical_rows_same_hash(self):
        X = np.array([[1.0, 2.0], [1.0, 2.0]])
        hashes = hash_rows_postprocess(X)
        assert hashes[0] == hashes[1]

    def test_different_rows_different_hash(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        hashes = hash_rows_postprocess(X)
        assert hashes[0] != hashes[1]

    def test_nan_handling(self):
        """NaN values are replaced with 0.0 so the hash is stable."""
        X_nan = np.array([[np.nan, 2.0]])
        X_zero = np.array([[0.0, 2.0]])
        assert hash_rows_postprocess(X_nan) == hash_rows_postprocess(X_zero)

    def test_inf_handling(self):
        """Inf values are clamped."""
        X = np.array([[np.inf, -np.inf]])
        hashes = hash_rows_postprocess(X)
        assert len(hashes) == 1
        assert isinstance(hashes[0], str)

    def test_single_row(self):
        X = np.array([[42.0]])
        hashes = hash_rows_postprocess(X)
        assert len(hashes) == 1

    def test_dataframe_input(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        hashes = hash_rows_postprocess(df.values)
        assert len(hashes) == 2

    def test_sparse_matrix_input(self):
        from scipy.sparse import csr_matrix

        X = csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        hashes = hash_rows_postprocess(X)
        assert len(hashes) == 2


# ---------------------------------------------------------------------------
# baseline_majority_scores
# ---------------------------------------------------------------------------
class TestBaselineMajorityScores:
    def test_basic(self):
        y_train = pd.Series([0, 0, 1, 1, 1])
        scores = baseline_majority_scores(y_train, n_samples=10)
        assert len(scores) == 10
        assert np.all(scores == pytest.approx(0.6))

    def test_all_positive(self):
        y_train = pd.Series([1, 1, 1])
        scores = baseline_majority_scores(y_train, n_samples=5)
        assert np.all(scores == 1.0)

    def test_all_negative(self):
        y_train = pd.Series([0, 0, 0])
        scores = baseline_majority_scores(y_train, n_samples=5)
        assert np.all(scores == 0.0)

    def test_empty_train(self):
        y_train = pd.Series([], dtype=float)
        scores = baseline_majority_scores(y_train, n_samples=3)
        assert len(scores) == 3
        assert np.all(scores == 0.0)

    def test_single_sample(self):
        y_train = pd.Series([1])
        scores = baseline_majority_scores(y_train, n_samples=1)
        assert scores[0] == 1.0

    def test_returns_ndarray(self):
        y_train = pd.Series([0, 1])
        result = baseline_majority_scores(y_train, n_samples=4)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float


# ---------------------------------------------------------------------------
# baseline_threshold_scores
# ---------------------------------------------------------------------------
class TestBaselineThresholdScores:
    def _make_df(self, **kwargs):
        n = kwargs.pop("n", 5)
        data = {}
        for col, vals in kwargs.items():
            data[col] = vals if vals is not None else np.zeros(n)
        return pd.DataFrame(data)

    def test_all_columns_present(self):
        df = pd.DataFrame({
            "duration": [1.0, 2.0, 3.0],
            "src_bytes": [100.0, 200.0, 300.0],
            "dst_bytes": [50.0, 100.0, 150.0],
            "src_pkts": [10, 20, 30],
            "dst_pkts": [5, 10, 15],
        })
        colmap = {}
        scores = baseline_threshold_scores(df, colmap, logger)
        assert len(scores) == 3
        assert scores[2] > scores[0]  # larger values → larger score

    def test_missing_all_columns_returns_zeros(self):
        df = pd.DataFrame({"unrelated": [1, 2, 3]})
        colmap = {}
        scores = baseline_threshold_scores(df, colmap, logger)
        assert np.all(scores == 0.0)

    def test_colmap_override(self):
        df = pd.DataFrame({
            "my_dur": [10.0, 20.0],
            "my_sb": [100.0, 200.0],
            "my_db": [50.0, 100.0],
            "my_sp": [5, 10],
            "my_dp": [2, 4],
        })
        colmap = {
            "duration_col": "my_dur",
            "src_bytes_col": "my_sb",
            "dst_bytes_col": "my_db",
            "src_pkts_col": "my_sp",
            "dst_pkts_col": "my_dp",
        }
        scores = baseline_threshold_scores(df, colmap, logger)
        assert len(scores) == 2
        assert all(np.isfinite(scores))

    def test_nan_values_treated_as_zero(self):
        df = pd.DataFrame({
            "duration": [np.nan, 5.0],
            "src_bytes": [np.nan, 100.0],
            "dst_bytes": [np.nan, 50.0],
            "src_pkts": [np.nan, 10],
            "dst_pkts": [np.nan, 5],
        })
        colmap = {}
        scores = baseline_threshold_scores(df, colmap, logger)
        assert scores[0] == pytest.approx(0.0)  # all NaN → log1p(0)=0
        assert scores[1] > 0.0

    def test_alternative_column_names(self):
        """Should find columns by candidate names like sbytes, dbytes, etc."""
        df = pd.DataFrame({
            "dur": [1.0, 2.0],
            "sbytes": [100.0, 200.0],
            "dbytes": [50.0, 100.0],
            "spkts": [10, 20],
            "dpkts": [5, 10],
        })
        colmap = {}
        scores = baseline_threshold_scores(df, colmap, logger)
        assert all(s > 0 for s in scores)

    def test_single_row(self):
        df = pd.DataFrame({
            "duration": [5.0],
            "src_bytes": [100.0],
            "dst_bytes": [50.0],
            "src_pkts": [10],
            "dst_pkts": [5],
        })
        scores = baseline_threshold_scores(df, {}, logger)
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_zero_values(self):
        df = pd.DataFrame({
            "duration": [0.0, 0.0],
            "src_bytes": [0.0, 0.0],
            "dst_bytes": [0.0, 0.0],
            "src_pkts": [0, 0],
            "dst_pkts": [0, 0],
        })
        scores = np.asarray(baseline_threshold_scores(df, {}, logger))
        assert np.allclose(scores, 0.0)
