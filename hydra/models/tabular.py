from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from hydra.config import TabularModelConfig

logger = logging.getLogger(__name__)


def _maybe_lightgbm(cfg: TabularModelConfig):
    try:
        import lightgbm as lgb

        params = cfg.hist_gbt.copy()
        params.pop("random_state", None)
        return "lightgbm", lgb.LGBMClassifier(
            objective="binary",
            n_estimators=params.get("max_iter", 300),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", -1),
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        return None, None


def _maybe_xgboost(cfg: TabularModelConfig):
    try:
        import xgboost as xgb

        params = cfg.hist_gbt.copy()
        return "xgboost", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=params.get("max_iter", 300),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=6 if params.get("max_depth") is None else params.get("max_depth"),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        return None, None


def get_tabular_models(cfg: TabularModelConfig) -> Dict[str, object]:
    models: Dict[str, object] = {
        "logreg": LogisticRegression(**cfg.logreg),
        "random_forest": RandomForestClassifier(**cfg.random_forest),
    }

    name, model = _maybe_lightgbm(cfg)
    if model is not None:
        models[name] = model
    else:
        name, model = _maybe_xgboost(cfg)
        if model is not None:
            models[name] = model
        else:
            models["hist_gbt"] = HistGradientBoostingClassifier(**cfg.hist_gbt)

    return models


def predict_proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim > 1:
            scores = scores[:, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores
    preds = model.predict(X)
    return preds.astype(float)
