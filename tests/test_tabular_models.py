"""Unit tests for hydra.models.tabular model builders."""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest
from sklearn.datasets import make_classification

from hydra.models.tabular import (
    ModelSpec,
    build_lightgbm,
    build_logreg,
    build_random_forest,
    build_sklearn_gbdt,
    build_xgboost,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RANDOM_STATE = 42


@pytest.fixture()
def small_data():
    """Small synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=RANDOM_STATE,
    )
    return X, y


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _check_fit_predict(spec: ModelSpec, X: np.ndarray, y: np.ndarray) -> None:
    """Fit on first 60 rows, predict on last 20; verify shapes."""
    spec.model.fit(X[:60], y[:60])
    preds = spec.model.predict(X[60:])
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# build_logreg
# ---------------------------------------------------------------------------


class TestBuildLogreg:
    def test_returns_model_spec(self):
        spec = build_logreg(RANDOM_STATE)
        assert isinstance(spec, ModelSpec)
        assert spec.name == "logreg"
        assert spec.backend == "sklearn"

    def test_random_state_passed(self):
        spec = build_logreg(123)
        assert spec.model.random_state == 123

    def test_fit_predict(self, small_data):
        spec = build_logreg(RANDOM_STATE)
        _check_fit_predict(spec, *small_data)


# ---------------------------------------------------------------------------
# build_random_forest
# ---------------------------------------------------------------------------


class TestBuildRandomForest:
    def test_returns_model_spec(self):
        spec = build_random_forest(RANDOM_STATE)
        assert isinstance(spec, ModelSpec)
        assert spec.name == "random_forest"
        assert spec.backend == "sklearn"

    def test_hyperparams(self):
        spec = build_random_forest(RANDOM_STATE)
        assert spec.model.n_estimators == 100
        assert spec.model.max_depth == 15
        assert spec.model.class_weight == "balanced"

    def test_random_state_passed(self):
        spec = build_random_forest(99)
        assert spec.model.random_state == 99

    def test_fit_predict(self, small_data):
        spec = build_random_forest(RANDOM_STATE)
        _check_fit_predict(spec, *small_data)


# ---------------------------------------------------------------------------
# build_sklearn_gbdt
# ---------------------------------------------------------------------------


class TestBuildSklearnGbdt:
    def test_returns_model_spec(self):
        spec = build_sklearn_gbdt(RANDOM_STATE)
        assert isinstance(spec, ModelSpec)
        assert spec.name == "sklearn_gbdt"
        assert spec.backend == "sklearn"

    def test_random_state_passed(self):
        spec = build_sklearn_gbdt(7)
        assert spec.model.random_state == 7

    def test_fit_predict(self, small_data):
        spec = build_sklearn_gbdt(RANDOM_STATE)
        _check_fit_predict(spec, *small_data)


# ---------------------------------------------------------------------------
# build_xgboost
# ---------------------------------------------------------------------------


class TestBuildXgboost:
    def test_returns_model_spec(self):
        spec = build_xgboost(RANDOM_STATE)
        assert isinstance(spec, ModelSpec)
        assert spec.name == "xgboost"
        assert spec.backend == "xgboost"

    def test_random_state_passed(self):
        spec = build_xgboost(55)
        assert spec.model.random_state == 55

    def test_scale_pos_weight(self):
        spec = build_xgboost(RANDOM_STATE, scale_pos_weight=3.5)
        assert spec.model.scale_pos_weight == 3.5

    def test_fit_predict(self, small_data):
        spec = build_xgboost(RANDOM_STATE)
        _check_fit_predict(spec, *small_data)

    def test_raises_when_xgboost_unavailable(self):
        """Simulate xgboost not being installed."""
        with mock.patch.dict(sys.modules, {"xgboost": None}):
            with pytest.raises(RuntimeError, match="xgboost is not available"):
                build_xgboost(RANDOM_STATE)


# ---------------------------------------------------------------------------
# build_lightgbm
# ---------------------------------------------------------------------------


class TestBuildLightgbm:
    def test_returns_model_spec(self):
        spec = build_lightgbm(RANDOM_STATE)
        assert isinstance(spec, ModelSpec)
        assert spec.name == "lightgbm"
        # backend depends on what is installed; just check it is a string
        assert isinstance(spec.backend, str)

    def test_fit_predict(self, small_data):
        spec = build_lightgbm(RANDOM_STATE)
        _check_fit_predict(spec, *small_data)

    def test_fallback_when_lightgbm_unavailable(self):
        """With lightgbm blocked, should fall back to xgboost or sklearn."""
        with mock.patch.dict(sys.modules, {"lightgbm": None}):
            spec = build_lightgbm(RANDOM_STATE)
            assert spec.name == "lightgbm"
            # Should NOT be lightgbm backend since we blocked it
            assert spec.backend in ("xgboost", "sklearn_gbdt", "sklearn_hist_gbdt")

    def test_fallback_to_sklearn_when_both_unavailable(self):
        """With both lightgbm and xgboost blocked, should fall back to sklearn."""
        with mock.patch.dict(sys.modules, {"lightgbm": None, "xgboost": None}):
            spec = build_lightgbm(RANDOM_STATE)
            assert spec.name == "lightgbm"
            assert spec.backend in ("sklearn_gbdt", "sklearn_hist_gbdt")

    def test_env_var_disables_lightgbm(self):
        """HYDRA_DISABLE_LIGHTGBM=1 should trigger sklearn_gbdt fallback."""
        with mock.patch.dict("os.environ", {"HYDRA_DISABLE_LIGHTGBM": "1"}):
            spec = build_lightgbm(RANDOM_STATE)
            assert spec.name == "lightgbm"
            assert spec.backend == "sklearn_gbdt"
            from sklearn.ensemble import GradientBoostingClassifier

            assert isinstance(spec.model, GradientBoostingClassifier)

    def test_env_var_not_set_uses_normal_path(self):
        """Without the env var, should not use sklearn_gbdt fallback."""
        with mock.patch.dict("os.environ", {}, clear=False):
            # Remove the key if present
            import os

            env_backup = os.environ.pop("HYDRA_DISABLE_LIGHTGBM", None)
            try:
                spec = build_lightgbm(RANDOM_STATE)
                assert spec.name == "lightgbm"
                assert spec.backend != "sklearn_gbdt"
            finally:
                if env_backup is not None:
                    os.environ["HYDRA_DISABLE_LIGHTGBM"] = env_backup
