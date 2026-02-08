from __future__ import annotations

import logging
from typing import List

import numpy as np
import scipy.sparse as sp
from sklearn.inspection import permutation_importance

from hydra.explain.records import ExplanationRecord, default_metadata

logger = logging.getLogger(__name__)


def _row_to_dense(X, idx: int) -> np.ndarray:
    if sp.issparse(X):
        return X[idx].toarray().ravel()
    return np.asarray(X[idx]).ravel()


def _topk_from_attribs(attribs: np.ndarray, feature_names: List[str], top_k: int):
    scores = np.abs(attribs)
    if scores.sum() == 0:
        norm = scores
    else:
        norm = scores / (scores.sum() + 1e-8)
    idx = np.argsort(scores)[::-1][:top_k]
    items = []
    for i in idx:
        direction = "positive" if attribs[i] >= 0 else "negative"
        items.append(
            {
                "feature_name": feature_names[i] if i < len(feature_names) else str(i),
                "attribution": float(attribs[i]),
                "direction": direction,
                "normalized_score": float(norm[i]),
            }
        )
    return items


def explain_logreg(
    model,
    X,
    feature_names: List[str],
    sample_ids: List[int],
    pred_proba: np.ndarray,
    model_name: str,
    feature_regime: str,
    dataset_name: str,
    seed: int,
    top_k: int = 10,
):
    coef = model.coef_.ravel()
    records = []
    for idx in sample_ids:
        row = _row_to_dense(X, idx)
        attribs = row * coef
        top_features = _topk_from_attribs(attribs, feature_names, top_k)
        rec = ExplanationRecord(
            sample_id=int(idx),
            model_name=model_name,
            pred_label=int(pred_proba[idx] >= 0.5),
            pred_proba=float(pred_proba[idx]),
            top_features=top_features,
            metadata=default_metadata(feature_regime, dataset_name, seed),
        )
        records.append(rec)
    return records


def explain_tree_shap(
    model,
    X,
    feature_names: List[str],
    sample_ids: List[int],
    pred_proba: np.ndarray,
    model_name: str,
    feature_regime: str,
    dataset_name: str,
    seed: int,
    top_k: int = 10,
):
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("shap is required for TreeSHAP explanations") from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    records = []
    for idx in sample_ids:
        attribs = np.array(shap_values[idx]).ravel()
        top_features = _topk_from_attribs(attribs, feature_names, top_k)
        rec = ExplanationRecord(
            sample_id=int(idx),
            model_name=model_name,
            pred_label=int(pred_proba[idx] >= 0.5),
            pred_proba=float(pred_proba[idx]),
            top_features=top_features,
            metadata=default_metadata(feature_regime, dataset_name, seed),
        )
        records.append(rec)
    return records


def permutation_importance_global(model, X, y, feature_names: List[str], n_repeats: int = 5):
    if sp.issparse(X):
        X = X.toarray()
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    importances = result.importances_mean
    idx = np.argsort(importances)[::-1]
    top = []
    for i in idx[: min(20, len(idx))]:
        top.append({"feature_name": feature_names[i], "importance": float(importances[i])})
    return top
