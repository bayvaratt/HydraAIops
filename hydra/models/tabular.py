from __future__ import annotations

from dataclasses import dataclass
import os

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelSpec:
    name: str
    backend: str
    model: object


def build_logreg(random_state: int) -> ModelSpec:
    model = LogisticRegression(
        max_iter=5000,
        tol=1e-3,
        solver="saga",
        random_state=random_state,
    )
    return ModelSpec(name="logreg", backend="sklearn", model=model)


def build_random_forest(random_state: int) -> ModelSpec:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=random_state,
        n_jobs=2,
        class_weight="balanced",
    )
    return ModelSpec(name="random_forest", backend="sklearn", model=model)


def build_sklearn_gbdt(random_state: int) -> ModelSpec:
    # GradientBoostingClassifier has no class_weight; use sample_weight at fit time instead.
    # The pipeline passes sample_weight via fit_params when class_weight is set on the spec.
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )
    return ModelSpec(name="sklearn_gbdt", backend="sklearn", model=model)


def _build_xgboost_model(random_state: int):
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost(random_state: int) -> ModelSpec:
    try:
        model = _build_xgboost_model(random_state)
        return ModelSpec(name="xgboost", backend="xgboost", model=model)
    except Exception as exc:
        raise RuntimeError("xgboost is not available; install xgboost to use this model.") from exc


def build_lightgbm(random_state: int) -> ModelSpec:
    if os.environ.get("HYDRA_DISABLE_LIGHTGBM") == "1":
        spec = build_sklearn_gbdt(random_state)
        return ModelSpec(name="lightgbm", backend="sklearn_gbdt", model=spec.model)
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=1,
            num_threads=1,
        )
        return ModelSpec(name="lightgbm", backend="lightgbm", model=model)
    except Exception:
        pass

    try:
        model = _build_xgboost_model(random_state)
        return ModelSpec(name="lightgbm", backend="xgboost", model=model)
    except Exception:
        pass

    model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.05,
        random_state=random_state,
    )
    return ModelSpec(name="lightgbm", backend="sklearn_hist_gbdt", model=model)
