"""Unit tests for hydra.data.preprocess module."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from hydra.data.preprocess import (
    FeatureSpec,
    _high_cardinality,
    _infer_types,
    _is_text_col,
    _log1p_nonneg,
    apply_feature_spec,
    bucket_port,
    build_feature_spec,
    fit_preprocessor,
)

logger = logging.getLogger("test")


# ---------------------------------------------------------------------------
# _log1p_nonneg
# ---------------------------------------------------------------------------
class TestLog1pNonneg:
    def test_positive_values(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _log1p_nonneg(X)
        expected = np.log1p(X)
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_values(self):
        X = np.array([[0.0, 0.0]])
        result = _log1p_nonneg(X)
        np.testing.assert_array_equal(result, np.array([[0.0, 0.0]]))

    def test_negative_values_clipped(self):
        X = np.array([[-5.0, -1.0, 0.0, 1.0]])
        result = _log1p_nonneg(X)
        # negatives should be clipped to 0, then log1p(0) = 0
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert result[0, 2] == 0.0
        assert result[0, 3] == pytest.approx(np.log1p(1.0))

    def test_large_values(self):
        X = np.array([[1e10]])
        result = _log1p_nonneg(X)
        assert result[0, 0] == pytest.approx(np.log1p(1e10))

    def test_1d_array(self):
        X = np.array([-1.0, 0.0, 5.0])
        result = _log1p_nonneg(X)
        expected = np.log1p(np.array([0.0, 0.0, 5.0]))
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_array(self):
        X = np.array([]).reshape(0, 2)
        result = _log1p_nonneg(X)
        assert result.shape == (0, 2)


# ---------------------------------------------------------------------------
# _infer_types
# ---------------------------------------------------------------------------
class TestInferTypes:
    def test_explicit_cols_returned_as_is(self):
        df = pd.DataFrame({"a": [1], "b": ["x"], "c": [2.0]})
        cat, num = _infer_types(df, categorical_cols=["b"], numeric_cols=["a", "c"])
        assert cat == ["b"]
        assert num == ["a", "c"]

    def test_explicit_cat_only(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        cat, num = _infer_types(df, categorical_cols=["b"], numeric_cols=None)
        assert cat == ["b"]
        assert num == []

    def test_explicit_num_only(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        cat, num = _infer_types(df, categorical_cols=None, numeric_cols=["a"])
        assert cat == []
        assert num == ["a"]

    def test_auto_inference_object_is_categorical(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
        cat, num = _infer_types(df, None, None)
        assert "b" in cat
        assert "a" in num
        assert "c" in num
        assert "b" not in num

    def test_auto_inference_category_dtype(self):
        df = pd.DataFrame({"a": pd.Categorical(["x", "y"]), "b": [1, 2]})
        cat, num = _infer_types(df, None, None)
        assert "a" in cat
        assert "b" in num

    def test_all_numeric(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        cat, num = _infer_types(df, None, None)
        assert cat == []
        assert set(num) == {"a", "b"}

    def test_all_object(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        cat, num = _infer_types(df, None, None)
        assert set(cat) == {"a", "b"}
        assert num == []

    def test_deduplicates_explicit_cols(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        cat, num = _infer_types(df, ["a", "a"], ["b", "b", "b"])
        assert cat == ["a"]
        assert num == ["b"]

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        cat, num = _infer_types(df, None, None)
        assert cat == []
        assert num == []


# ---------------------------------------------------------------------------
# _is_text_col
# ---------------------------------------------------------------------------
class TestIsTextCol:
    def test_short_strings_not_text(self):
        s = pd.Series(["tcp", "udp", "icmp", "tcp"])
        assert not _is_text_col(s)

    def test_long_avg_is_text(self):
        long_str = "x" * 50
        s = pd.Series([long_str] * 10)
        assert _is_text_col(s)

    def test_one_very_long_string_is_text(self):
        # max_len > 100 triggers text
        s = pd.Series(["short"] * 9 + ["x" * 101])
        assert _is_text_col(s)

    def test_numeric_column_not_text(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert not _is_text_col(s)

    def test_empty_series(self):
        s = pd.Series(dtype="object")
        assert not _is_text_col(s)

    def test_all_nan(self):
        s = pd.Series([None, None, None], dtype="object")
        assert not _is_text_col(s)

    def test_borderline_avg_30(self):
        # avg_len exactly 30 should NOT be text (> 30 required)
        s = pd.Series(["x" * 30] * 10)
        assert not _is_text_col(s)

    def test_borderline_max_100(self):
        # max_len exactly 100 should NOT be text (> 100 required)
        s = pd.Series(["a"] * 9 + ["x" * 100])
        assert not _is_text_col(s)

    def test_avg_31_is_text(self):
        s = pd.Series(["x" * 31] * 10)
        assert _is_text_col(s)


# ---------------------------------------------------------------------------
# _high_cardinality
# ---------------------------------------------------------------------------
class TestHighCardinality:
    def test_low_cardinality(self):
        s = pd.Series(["a", "b", "c"] * 10)
        assert _high_cardinality(s) is False

    def test_exactly_50_not_high(self):
        s = pd.Series(list(range(50)))
        assert _high_cardinality(s) is False

    def test_51_is_high(self):
        s = pd.Series(list(range(51)))
        assert _high_cardinality(s) is True

    def test_empty_series(self):
        s = pd.Series(dtype="object")
        assert _high_cardinality(s) is False

    def test_with_nan(self):
        # 51 unique non-null + NaN; nunique(dropna=True) = 51 → high
        vals = list(range(51)) + [None]
        s = pd.Series(vals)
        assert _high_cardinality(s) is True

    def test_all_same(self):
        s = pd.Series(["same"] * 100)
        assert _high_cardinality(s) is False


# ---------------------------------------------------------------------------
# bucket_port
# ---------------------------------------------------------------------------
class TestBucketPort:
    def test_well_known(self):
        s = pd.Series([0, 22, 80, 443, 1023])
        result = bucket_port(s)
        assert (result == "well_known").all()

    def test_registered(self):
        s = pd.Series([1024, 8080, 49151])
        result = bucket_port(s)
        assert (result == "registered").all()

    def test_dynamic(self):
        s = pd.Series([49152, 55000, 65535])
        result = bucket_port(s)
        assert (result == "dynamic").all()

    def test_unknown_negative(self):
        s = pd.Series([-1])
        result = bucket_port(s)
        assert result.iloc[0] == "unknown"

    def test_unknown_non_numeric(self):
        s = pd.Series(["abc", None])
        result = bucket_port(s)
        assert (result == "unknown").all()

    def test_mixed(self):
        s = pd.Series([22, 8080, 50000, "bad", -1])
        result = bucket_port(s)
        assert result.iloc[0] == "well_known"
        assert result.iloc[1] == "registered"
        assert result.iloc[2] == "dynamic"
        assert result.iloc[3] == "unknown"
        assert result.iloc[4] == "unknown"

    def test_empty_series(self):
        s = pd.Series(dtype="object")
        result = bucket_port(s)
        assert len(result) == 0

    def test_above_65535_is_unknown(self):
        s = pd.Series([70000])
        result = bucket_port(s)
        assert result.iloc[0] == "unknown"


# ---------------------------------------------------------------------------
# build_feature_spec
# ---------------------------------------------------------------------------
class TestBuildFeatureSpec:
    @staticmethod
    def _make_df():
        """A minimal network-flow-like DataFrame."""
        return pd.DataFrame({
            "src_ip": ["1.2.3.4"] * 10,
            "dst_ip": ["5.6.7.8"] * 10,
            "src_port": [22, 80, 443, 8080, 55000, 22, 80, 443, 8080, 55000],
            "dst_port": [12345] * 10,
            "proto": ["tcp", "udp"] * 5,
            "service": ["http", "dns"] * 5,
            "duration": np.random.rand(10),
            "src_bytes": np.random.randint(0, 1000, 10),
            "dst_bytes": np.random.randint(0, 1000, 10),
            "label": [0, 1] * 5,
        })

    def test_behaviour_only_drops_identifiers(self):
        df = self._make_df()
        spec = build_feature_spec(df, "label", "behaviour_only", None, None, logger)
        assert spec.regime == "behaviour_only"
        for col in ["src_ip", "dst_ip", "src_port", "dst_port", "service"]:
            assert col in spec.dropped
        # numeric flow features should be kept
        assert "duration" in spec.keep_numeric
        assert "src_bytes" in spec.keep_numeric

    def test_operational_creates_port_buckets(self):
        df = self._make_df()
        spec = build_feature_spec(df, "label", "operational", None, None, logger)
        # Port cols should be dropped but bucket derivatives created
        assert "src_port" in spec.dropped
        assert "dst_port" in spec.dropped
        assert "src_port_bucket" in spec.derived_categorical
        assert "dst_port_bucket" in spec.derived_categorical
        assert "src_port_bucket" in spec.keep_categorical
        assert "dst_port_bucket" in spec.keep_categorical

    def test_identifier_inclusive_keeps_ips(self):
        df = self._make_df()
        spec = build_feature_spec(df, "label", "identifier_inclusive", None, None, logger)
        # IPs should NOT be dropped
        assert "src_ip" not in spec.dropped
        assert "dst_ip" not in spec.dropped

    def test_core_flow_keeps_only_core_cols(self):
        df = pd.DataFrame({
            "duration": [1.0, 2.0],
            "src_bytes": [100, 200],
            "dst_bytes": [50, 60],
            "src_pkts": [10, 20],
            "dst_pkts": [5, 10],
            "proto": ["tcp", "udp"],
            "conn_state": ["SF", "REJ"],
            "extra_col": [1, 2],
            "label": [0, 1],
        })
        spec = build_feature_spec(df, "label", "core_flow", None, None, logger)
        assert "extra_col" in spec.dropped
        assert "duration" in spec.keep_numeric
        assert "proto" in spec.keep_categorical

    def test_paper_5feat(self):
        df = pd.DataFrame({
            "duration": [1.0, 2.0],
            "src_bytes": [100, 200],
            "dst_bytes": [50, 60],
            "src_ip_bytes": [1000, 2000],
            "dst_ip_bytes": [500, 600],
            "label": [0, 1],
        })
        spec = build_feature_spec(df, "label", "paper_5feat", None, None, logger)
        assert spec.regime == "paper_5feat"
        assert len(spec.keep_numeric) == 5
        assert spec.keep_categorical == []
        assert spec.missing_required == []

    def test_paper_5feat_with_missing_cols(self):
        df = pd.DataFrame({
            "duration": [1.0, 2.0],
            "src_bytes": [100, 200],
            "dst_bytes": [50, 60],
            "label": [0, 1],
        })
        spec = build_feature_spec(df, "label", "paper_5feat", None, None, logger)
        assert spec.keep_numeric == ["duration", "src_bytes", "dst_bytes"]
        assert "src_ip_bytes" in spec.missing_required
        assert "dst_ip_bytes" in spec.missing_required

    def test_paper_5feat_too_few_cols_raises(self):
        df = pd.DataFrame({
            "duration": [1.0, 2.0],
            "src_bytes": [100, 200],
            "label": [0, 1],
        })
        with pytest.raises(RuntimeError, match="at least 3 of 5"):
            build_feature_spec(df, "label", "paper_5feat", None, None, logger)

    def test_unknown_regime_raises(self):
        df = pd.DataFrame({"a": [1], "label": [0]})
        with pytest.raises(ValueError, match="Unknown feature regime"):
            build_feature_spec(df, "label", "nonexistent", None, None, logger)

    def test_label_col_excluded(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]})
        spec = build_feature_spec(df, "label", "behaviour_only", None, None, logger)
        all_features = spec.keep_numeric + spec.keep_categorical
        assert "label" not in all_features
        assert "label" not in spec.dropped

    def test_behaviour_only_drops_text_cols(self):
        long_text = "x" * 50
        df = pd.DataFrame({
            "notes": [long_text] * 10,
            "value": list(range(10)),
            "label": [0, 1] * 5,
        })
        spec = build_feature_spec(df, "label", "behaviour_only", None, None, logger)
        assert "notes" in spec.dropped

    def test_explicit_col_types(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "label": [0, 1, 0],
        })
        spec = build_feature_spec(
            df, "label", "behaviour_only",
            categorical_cols=["b"], numeric_cols=["a"], logger=logger,
        )
        assert "a" in spec.keep_numeric


# ---------------------------------------------------------------------------
# apply_feature_spec
# ---------------------------------------------------------------------------
class TestApplyFeatureSpec:
    def test_basic_apply(self):
        df = pd.DataFrame({
            "duration": [1.0, 2.0],
            "proto": ["tcp", "udp"],
            "src_ip": ["1.2.3.4", "5.6.7.8"],
            "label": [0, 1],
        })
        spec = FeatureSpec(
            regime="behaviour_only",
            keep_categorical=["proto"],
            keep_numeric=["duration"],
            derived_categorical={},
            dropped=["src_ip"],
        )
        out, cat, num = apply_feature_spec(df, spec, "label", logger)
        assert "label" not in out.columns
        assert "src_ip" not in out.columns
        assert list(out.columns) == ["duration", "proto"]
        assert cat == ["proto"]
        assert num == ["duration"]

    def test_derived_port_buckets(self):
        df = pd.DataFrame({
            "src_port": [22, 8080, 55000],
            "duration": [1.0, 2.0, 3.0],
            "label": [0, 1, 0],
        })
        spec = FeatureSpec(
            regime="operational",
            keep_categorical=["src_port_bucket"],
            keep_numeric=["duration"],
            derived_categorical={"src_port_bucket": "src_port"},
            dropped=["src_port"],
        )
        out, cat, num = apply_feature_spec(df, spec, "label", logger)
        assert "src_port_bucket" in out.columns
        assert out["src_port_bucket"].iloc[0] == "well_known"
        assert out["src_port_bucket"].iloc[1] == "registered"
        assert out["src_port_bucket"].iloc[2] == "dynamic"

    def test_missing_source_port_col(self):
        df = pd.DataFrame({
            "duration": [1.0],
            "label": [0],
        })
        spec = FeatureSpec(
            regime="operational",
            keep_categorical=["src_port_bucket"],
            keep_numeric=["duration"],
            derived_categorical={"src_port_bucket": "src_port"},
            dropped=[],
        )
        out, cat, num = apply_feature_spec(df, spec, "label", logger)
        assert "src_port_bucket" in out.columns
        assert out["src_port_bucket"].iloc[0] == "unknown"

    def test_label_removed(self):
        df = pd.DataFrame({"a": [1], "label": [0]})
        spec = FeatureSpec(
            regime="behaviour_only",
            keep_categorical=[],
            keep_numeric=["a"],
            derived_categorical={},
            dropped=[],
        )
        out, _, _ = apply_feature_spec(df, spec, "label", logger)
        assert "label" not in out.columns

    def test_missing_keep_col_gracefully_skipped(self):
        df = pd.DataFrame({"a": [1], "label": [0]})
        spec = FeatureSpec(
            regime="behaviour_only",
            keep_categorical=[],
            keep_numeric=["a", "nonexistent_col"],
            derived_categorical={},
            dropped=[],
        )
        out, cat, num = apply_feature_spec(df, spec, "label", logger)
        assert "nonexistent_col" not in out.columns
        assert "a" in out.columns
        assert num == ["a"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series(dtype="float64"), "label": pd.Series(dtype="int64")})
        spec = FeatureSpec(
            regime="behaviour_only",
            keep_categorical=[],
            keep_numeric=["a"],
            derived_categorical={},
            dropped=[],
        )
        out, cat, num = apply_feature_spec(df, spec, "label", logger)
        assert len(out) == 0
        assert "a" in out.columns


# ---------------------------------------------------------------------------
# fit_preprocessor
# ---------------------------------------------------------------------------
class TestFitPreprocessor:
    def test_numeric_only(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        ct = fit_preprocessor(X, categorical_cols=[], numeric_cols=["a", "b"])
        result = ct.fit_transform(X)
        # log1p + standard scaling: should have 2 output features
        if sparse.issparse(result):
            result = result.toarray()
        assert result.shape == (3, 2)

    def test_categorical_only(self):
        X = pd.DataFrame({"c": ["tcp", "udp", "tcp"]})
        ct = fit_preprocessor(X, categorical_cols=["c"], numeric_cols=[])
        result = ct.fit_transform(X)
        if sparse.issparse(result):
            result = result.toarray()
        # OneHotEncoder with 2 categories → 2 columns
        assert result.shape == (3, 2)

    def test_mixed_types(self):
        X = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "c": ["tcp", "udp", "tcp"],
        })
        ct = fit_preprocessor(X, categorical_cols=["c"], numeric_cols=["a"])
        result = ct.fit_transform(X)
        if sparse.issparse(result):
            result = result.toarray()
        # 1 numeric + 2 one-hot = 3 columns
        assert result.shape == (3, 3)

    def test_no_features_raises(self):
        X = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="No features available"):
            fit_preprocessor(X, categorical_cols=[], numeric_cols=[])

    def test_handles_nan(self):
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "c": ["tcp", None, "udp"]})
        ct = fit_preprocessor(X, categorical_cols=["c"], numeric_cols=["a"])
        result = ct.fit_transform(X)
        if sparse.issparse(result):
            result = result.toarray()
        # Should not contain NaN after imputation
        assert not np.isnan(result).any()

    def test_negative_numeric_clipped_in_log1p(self):
        X = pd.DataFrame({"a": [-10.0, 0.0, 100.0]})
        ct = fit_preprocessor(X, categorical_cols=[], numeric_cols=["a"])
        result = ct.fit_transform(X)
        if sparse.issparse(result):
            result = result.toarray()
        # After impute→log1p(clip)→scale, no NaN or inf
        assert np.isfinite(result).all()

    def test_unknown_categories_handled(self):
        X_train = pd.DataFrame({"c": ["tcp", "udp", "tcp"]})
        ct = fit_preprocessor(X_train, categorical_cols=["c"], numeric_cols=[])
        ct.fit(X_train)
        X_test = pd.DataFrame({"c": ["tcp", "icmp"]})
        result = ct.transform(X_test)
        if sparse.issparse(result):
            result = result.toarray()
        # "icmp" is unknown → all zeros for that row
        assert result.shape == (2, 2)
        assert result[1].sum() == 0.0  # unknown category → zeros

    def test_transform_output_is_finite(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "bytes": np.random.exponential(1000, 50),
            "duration": np.random.exponential(10, 50),
            "proto": np.random.choice(["tcp", "udp", "icmp"], 50),
        })
        ct = fit_preprocessor(X, categorical_cols=["proto"], numeric_cols=["bytes", "duration"])
        result = ct.fit_transform(X)
        if sparse.issparse(result):
            result = result.toarray()
        assert np.isfinite(result).all()
