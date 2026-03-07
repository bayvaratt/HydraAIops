from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------------
# Deep (PyTorch) SHAP helper
# ---------------------------------------------------------------------------

def _deep_shap(net, device, X_arr: np.ndarray, n_bg: int = 30, max_explain: Optional[int] = 200):
    """Run shap.GradientExplainer on a PyTorch nn.Module.

    GradientExplainer (expected gradients) handles MaxPool1d and other ops
    that DeepExplainer/DeepLIFT cannot.

    Always returns a list of (n_samples, n_features) float32 arrays, one per
    output neuron.  The raw GradientExplainer result is normalised:
      - 2-D ndarray (n, f)    → [arr]
      - 3-D ndarray (n, f, k) → [arr[:,:,i] for i in range(k)]
      - list of arrays        → each element cast to ndarray
    """
    import torch
    import shap

    X_arr = np.asarray(X_arr, dtype=np.float32)
    if max_explain is not None and len(X_arr) > max_explain:
        rng = np.random.default_rng(0)
        X_arr = X_arr[rng.choice(len(X_arr), size=max_explain, replace=False)]

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_arr), size=min(n_bg, len(X_arr)), replace=False)
    X_bg = torch.tensor(X_arr[bg_idx], dtype=torch.float32).to(device)
    X_exp = torch.tensor(X_arr, dtype=torch.float32).to(device)

    net.eval()
    # GradientExplainer requires 2-D model output; wrap if model returns 1-D (binary sigmoid)
    with torch.no_grad():
        sample_out = net(X_bg[:1])
    if sample_out.dim() == 1:
        import torch.nn as nn
        class _Wrap2D(nn.Module):
            def __init__(self, inner): super().__init__(); self.inner = inner
            def forward(self, x): return self.inner(x).unsqueeze(-1)
        net = _Wrap2D(net)
        net.eval()

    explainer = shap.GradientExplainer(net, X_bg)
    raw = explainer.shap_values(X_exp)

    # Normalise to list of (n_samples, n_features) arrays
    if isinstance(raw, np.ndarray):
        if raw.ndim == 2:        # (n_samples, n_features) — single scalar output
            return [raw]
        elif raw.ndim == 3:      # (n_samples, n_features, n_outputs)
            return [raw[:, :, i] for i in range(raw.shape[2])]
    if isinstance(raw, list):
        return [np.asarray(sv, dtype=np.float32) for sv in raw]
    return [np.asarray(raw, dtype=np.float32)]


def _get_model(pipeline):
    return pipeline.named_steps["model"]


def _get_preprocessor(pipeline):
    return pipeline.named_steps["prep"]


def _get_selector(pipeline):
    return pipeline.named_steps.get("select")


def _transform_for_model(pipeline, X):
    X_trans = _get_preprocessor(pipeline).transform(X)
    selector = _get_selector(pipeline)
    if selector is not None:
        X_trans = selector.transform(X_trans)
    return X_trans


def save_global_importance(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    out_path: Path,
    logger,
):
    model = _get_model(pipeline)

    # CNN-LSTM: use GradientExplainer (mean |SHAP|) as feature importance;
    # fall back to gradient×input attribution if SHAP fails; never use
    # permutation_importance which doesn't recognise CNNLSTMClassifier.
    if hasattr(model, "net_") and model.net_ is not None:
        import torch
        X_trans = _transform_for_model(pipeline, X_val)
        X_dense = X_trans.toarray() if hasattr(X_trans, "toarray") else np.asarray(X_trans, dtype=np.float32)
        try:
            shap_vals = _deep_shap(model.net_, model.device_, X_dense, max_explain=200)
            # _deep_shap always returns a list (one array per output neuron)
            importances = np.mean([np.abs(np.asarray(sv)).mean(axis=0) for sv in shap_vals], axis=0)
            df = pd.DataFrame({"feature": feature_names, "importance": importances})
            df.sort_values("importance", ascending=False).to_csv(out_path, index=False)
            logger.info("Saved CNN-LSTM GradientExplainer global importance to %s", out_path)
            return
        except Exception as exc:
            logger.warning("CNN-LSTM GradientExplainer global importance failed (%s); using gradient×input", exc)
        # Gradient × input fallback (use torch.autograd.grad to avoid leaf issues)
        try:
            import torch
            net = model.net_
            net.eval()
            X_sub = X_dense[:200]
            X_t = torch.tensor(X_sub, dtype=torch.float32, device=model.device_, requires_grad=True)
            out = net(X_t)
            if out.dim() > 1:
                out = out.sum(dim=-1)
            grads = torch.autograd.grad(out.sum(), X_t)[0]
            grad = grads.detach().cpu().numpy()
            importances = np.abs(grad * X_sub).mean(axis=0)
            df = pd.DataFrame({"feature": feature_names, "importance": importances})
            df.sort_values("importance", ascending=False).to_csv(out_path, index=False)
            logger.info("Saved CNN-LSTM gradient×input global importance to %s", out_path)
        except Exception as exc2:
            logger.warning("CNN-LSTM gradient×input importance also failed (%s); skipping", exc2)
        return

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        if importances.shape[0] != len(feature_names):
            logger.warning("Feature importance length mismatch; falling back to permutation importance")
        else:
            df = pd.DataFrame({"feature": feature_names, "importance": importances})
            df.sort_values("importance", ascending=False).to_csv(out_path, index=False)
            return

    logger.info("Using permutation importance for global feature importance")
    X_trans = _transform_for_model(pipeline, X_val)
    X_trans = X_trans.toarray() if hasattr(X_trans, "toarray") else np.asarray(X_trans)
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


def save_type_shap(
    model,
    X_proc,
    y_true,
    label_encoder,
    feature_names: List[str],
    out_dir: Path,
    logger,
):
    """Save per-class SHAP values for the stage-2 multiclass type classifier."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "feature_importances_") and len(model.feature_importances_) == len(feature_names):
        df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        df.sort_values("importance", ascending=False).to_csv(out_dir / "type_global_importance.csv", index=False)

    X_arr = X_proc.toarray() if hasattr(X_proc, "toarray") else np.asarray(X_proc, dtype=np.float32)
    classes = label_encoder.classes_

    # CNN-LSTM: use DeepExplainer (limit samples to keep it tractable)
    if hasattr(model, "net_") and model.net_ is not None:
        try:
            import shap  # noqa: F401
            # _deep_shap always returns a list (one array per output neuron)
            shap_values = _deep_shap(model.net_, model.device_, X_arr)
            n_explain = len(shap_values[0])  # actual number of samples explained
            y_arr = np.asarray(y_true)[:n_explain]
            for i, cls in enumerate(classes):
                if i >= len(shap_values):
                    break
                safe = cls.replace(" ", "_").replace("/", "_")
                df = pd.DataFrame(np.asarray(shap_values[i]), columns=feature_names)
                df["true_label"] = y_arr
                df.to_csv(out_dir / f"type_shap_{safe}.csv", index=False)
            logger.info("Saved CNN-LSTM GradientExplainer type SHAP to %s", out_dir)
        except Exception as exc:
            logger.warning("CNN-LSTM DeepExplainer type SHAP failed (%s); skipping", exc)
        return

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_arr)

        if isinstance(shap_values, list):
            for i, cls in enumerate(classes):
                safe = cls.replace(" ", "_").replace("/", "_")
                df = pd.DataFrame(shap_values[i], columns=feature_names)
                df["true_label"] = np.asarray(y_true)
                df.to_csv(out_dir / f"type_shap_{safe}.csv", index=False)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            for i, cls in enumerate(classes):
                safe = cls.replace(" ", "_").replace("/", "_")
                df = pd.DataFrame(shap_values[:, :, i], columns=feature_names)
                df["true_label"] = np.asarray(y_true)
                df.to_csv(out_dir / f"type_shap_{safe}.csv", index=False)
        else:
            logger.warning("Unexpected SHAP shape %s for type classifier; skipping per-class CSVs", np.shape(shap_values))

        logger.info("Saved type classifier SHAP to %s", out_dir)
    except Exception as e:
        logger.warning("Type SHAP failed (%s); skipping", e)


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
    X_trans = _transform_for_model(pipeline, X_sample)

    # CNN-LSTM: use DeepExplainer for local explanations
    if hasattr(model, "net_") and model.net_ is not None:
        try:
            import shap  # noqa: F401
            X_dense = X_trans.toarray() if hasattr(X_trans, "toarray") else np.asarray(X_trans, dtype=np.float32)
            logger.info("Using CNN-LSTM GradientExplainer for local explanations")
            shap_vals = _deep_shap(model.net_, model.device_, X_dense)
            # _deep_shap always returns a list; take last output (attack prob / last class)
            shap_arr = np.asarray(shap_vals[-1])
            df = pd.DataFrame(shap_arr, columns=feature_names)
            df.insert(0, "sample_id", X_sample.index.astype(str))
            df.to_csv(out_dir / "local_explanations.csv", index=False)
            return
        except Exception as exc:
            logger.warning("CNN-LSTM DeepExplainer local explanations failed (%s); falling back to occlusion", exc)

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
