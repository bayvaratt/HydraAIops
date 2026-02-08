from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelSpec:
    name: str
    backend: str
    model: object


def build_logreg(random_state: int) -> ModelSpec:
    model = LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        random_state=random_state,
    )
    return ModelSpec(name="logreg", backend="sklearn", model=model)


def build_random_forest(random_state: int) -> ModelSpec:
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight=None,
    )
    return ModelSpec(name="random_forest", backend="sklearn", model=model)


def build_lightgbm(random_state: int) -> ModelSpec:
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
        return ModelSpec(name="lightgbm", backend="lightgbm", model=model)
    except Exception:
        pass

    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
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
        return ModelSpec(name="lightgbm", backend="xgboost", model=model)
    except Exception:
        pass

    model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.05,
        random_state=random_state,
    )
    return ModelSpec(name="lightgbm", backend="sklearn_hist_gbdt", model=model)
