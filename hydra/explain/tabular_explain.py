from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _get_model(pipeline):
    return pipeline.named_steps["model"]


def _get_preprocessor(pipeline):
    return pipeline.named_steps["prep"]


def save_global_importance(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    out_path: Path,
    logger,
):
    model = _get_model(pipeline)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        if importances.shape[0] != len(feature_names):
            logger.warning("Feature importance length mismatch; falling back to permutation importance")
        else:
            df = pd.DataFrame({"feature": feature_names, "importance": importances})
            df.sort_values("importance", ascending=False).to_csv(out_path, index=False)
            return

    logger.info("Using permutation importance for global feature importance")
    X_trans = _get_preprocessor(pipeline).transform(X_val)
    result = permutation_importance(
        model,
        X_trans,
        y_val,
        n_repeats=5,
        random_state=0,
        scoring="average_precision",
    )
    df = pd.DataFrame({"feature": feature_names, "importance": result.importances_mean})
    df.sort_values("importance", ascending=False).to_csv(out_path, index=False)


def save_local_explanations(
    pipeline,
    X_val: pd.DataFrame,
    feature_names: List[str],
    out_dir: Path,
    n_samples: int,
    random_state: int,
    logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X_val), size=min(n_samples, len(X_val)), replace=False)
    X_sample = X_val.iloc[sample_idx]

    model = _get_model(pipeline)
    prep = _get_preprocessor(pipeline)
    X_trans = prep.transform(X_sample)

    try:
        import shap  # noqa: F401

        logger.info("Using SHAP for local explanations")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]
            df = pd.DataFrame(shap_values, columns=feature_names)
            df.insert(0, "sample_id", X_sample.index.astype(str))
            df.to_csv(out_dir / "local_explanations.csv", index=False)
            return
        except Exception as e:
            logger.warning("SHAP failed (%s); falling back to occlusion", e)
    except Exception:
        logger.info("SHAP not available; using occlusion fallback")

    # Occlusion fallback: zero-out each feature column and measure delta
    X_dense = X_trans.toarray() if hasattr(X_trans, "toarray") else np.asarray(X_trans)
    base_scores = model.predict_proba(X_dense)[:, 1]

    contributions = np.zeros_like(X_dense)
    for j in range(X_dense.shape[1]):
        X_mut = X_dense.copy()
        X_mut[:, j] = 0.0
        new_scores = model.predict_proba(X_mut)[:, 1]
        contributions[:, j] = base_scores - new_scores

    df = pd.DataFrame(contributions, columns=feature_names)
    df.insert(0, "sample_id", X_sample.index.astype(str))
    df.to_csv(out_dir / "local_explanations.csv", index=False)
