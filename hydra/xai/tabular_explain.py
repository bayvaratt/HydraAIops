from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------------
# SOTA attribution: canonical per-sample attribution array
# ---------------------------------------------------------------------------

def compute_binary_shap(
    model,
    X_dense: np.ndarray,
    model_type: str,
    device=None,
    n_steps: int = 50,
    max_explain: Optional[int] = 300,
) -> Optional[np.ndarray]:
    """Compute per-sample binary classification attributions.

    Returns a (n_samples, n_features) float32 array of signed attribution
    values: positive = pushes toward P(attack=1), negative = toward normal.

    Dispatch
    --------
    tree   → SHAP TreeExplainer (Lundberg & Lee 2020 — SOTA for tree ensembles)
    linear → SHAP LinearExplainer (exact for linear models)
    deep   → Integrated Gradients (Sundararajan et al. 2017 — satisfies
             completeness + sensitivity axioms; preferred over GradientExplainer
             for CNN-LSTM because MaxPool1d subgradients are averaged across
             the full interpolation path)

    Parameters
    ----------
    model      : fitted sklearn estimator (Pipeline.named_steps["model"])
    X_dense    : (n, f) dense float32, already preprocessed
    model_type : "tree" | "linear" | "deep"
    device     : torch.device (deep models only)
    n_steps    : IG integration steps (deep models only)
    max_explain: subsample cap

    Returns None on failure (caller logs and skips eval).
    """
    import shap

    X_dense = np.asarray(X_dense, dtype=np.float32)

    if max_explain is not None and len(X_dense) > max_explain:
        rng = np.random.default_rng(42)
        X_dense = X_dense[rng.choice(len(X_dense), size=max_explain, replace=False)]

    try:
        if model_type == "deep":
            from hydra.xai.integrated_gradients import integrated_gradients_batch
            import torch
            dev = device if device is not None else torch.device("cpu")
            return integrated_gradients_batch(
                model.net_, dev, X_dense,
                n_steps=n_steps, max_explain=None,  # already subsampled above
            )

        if model_type == "linear":
            masker = shap.maskers.Independent(X_dense, max_samples=min(100, len(X_dense)))
            explainer = shap.LinearExplainer(model, masker)
            sv = explainer.shap_values(X_dense)
            # LinearExplainer binary: returns (n, f) directly
            if isinstance(sv, list):
                sv = sv[-1]
            return np.asarray(sv, dtype=np.float32)

        # model_type == "tree" (RF, XGBoost, LightGBM, sklearn_gbdt)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_dense, check_additivity=False)
        if isinstance(sv, list):
            # list of arrays: one per class — take positive class (last)
            return np.asarray(sv[-1], dtype=np.float32)
        if hasattr(sv, "values"):
            sv = sv.values
        sv = np.asarray(sv)
        if sv.ndim == 3:
            # (n, f, n_classes) — take attack class
            return sv[:, :, -1].astype(np.float32)
        return sv.astype(np.float32)

    except Exception:
        return None


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

    # Logistic regression: use |coef_| as feature importance (instant, interpretable)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importances = np.abs(coef).mean(axis=0)  # mean over classes for multiclass
        if importances.shape[0] == len(feature_names):
            df = pd.DataFrame({"feature": feature_names, "importance": importances})
            df.sort_values("importance", ascending=False).to_csv(out_path, index=False)
            logger.info("Saved logreg |coef_| global importance to %s", out_path)
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
    occlusion_baseline: str = "zero",
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

        # Always convert to dense float32 — sparse/object arrays cause isnan cast errors
        X_dense_shap = X_trans.toarray() if hasattr(X_trans, "toarray") else np.asarray(X_trans)
        X_dense_shap = np.asarray(X_dense_shap, dtype=np.float32)

        logger.info("Using SHAP for local explanations")
        try:
            from sklearn.linear_model import LogisticRegression
            _is_linear = isinstance(model, LogisticRegression)
            if _is_linear:
                masker = shap.maskers.Independent(X_dense_shap, max_samples=min(100, len(X_dense_shap)))
                explainer = shap.LinearExplainer(model, masker)
                shap_values = explainer.shap_values(X_dense_shap)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_dense_shap, check_additivity=False)
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

    # Occlusion fallback: ablate each feature and measure delta.
    # Baseline supports zero/mean/median for numeric stability and categorical plug-in.
    X_dense = X_trans.toarray() if hasattr(X_trans, "toarray") else np.asarray(X_trans)
    X_dense = np.asarray(X_dense, dtype=np.float32)
    base_scores = model.predict_proba(X_dense)[:, 1]

    if occlusion_baseline == "zero":
        baseline_value = 0.0
    elif occlusion_baseline == "mean":
        baseline_value = np.mean(X_dense, axis=0)
    elif occlusion_baseline == "median":
        baseline_value = np.median(X_dense, axis=0)
    else:
        raise ValueError("occlusion_baseline must be one of 'zero', 'mean', 'median'.")

    contributions = np.zeros_like(X_dense)
    for j in range(X_dense.shape[1]):
        X_mut = X_dense.copy()
        X_mut[:, j] = baseline_value if np.ndim(baseline_value) == 0 else baseline_value[j]
        new_scores = model.predict_proba(X_mut)[:, 1]
        contributions[:, j] = base_scores - new_scores

    df = pd.DataFrame(contributions, columns=feature_names)
    df.insert(0, "sample_id", X_sample.index.astype(str))
    df.to_csv(out_dir / "local_explanations.csv", index=False)


# ---------------------------------------------------------------------------
# LIME: local interpretable model-agnostic explanations (Ribeiro et al. 2016)
# ---------------------------------------------------------------------------

def save_lime_explanations(
    pipeline,
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    feature_names: List[str],
    out_dir: Path,
    n_samples: int = 50,
    n_lime_samples: int = 2000,
    random_state: int = 42,
    logger=None,
):
    """LIME local linear approximations for a sample of val instances.

    LIME perturbs each instance, queries the model, and fits a sparse local
    linear model.  The resulting coefficients represent each feature's
    contribution to the prediction for that specific instance.

    Provides a model-agnostic comparison baseline alongside SHAP/IG —
    useful for verifying that SHAP attributions are consistent with local
    approximations from an entirely independent method.

    Saves
    -----
    out_dir/lime_explanations.csv — (n_samples × n_features) coefficient matrix
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        if logger:
            logger.warning("LIME not installed (pip install lime); skipping.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _get_model(pipeline)

    # Build preprocessed arrays for LIME background + prediction callable
    X_train_proc = _transform_for_model(pipeline, X_train_raw)
    X_train_dense = (
        X_train_proc.toarray() if hasattr(X_train_proc, "toarray")
        else np.asarray(X_train_proc, dtype=np.float32)
    )

    X_val_proc = _transform_for_model(pipeline, X_val_raw)
    X_val_dense = (
        X_val_proc.toarray() if hasattr(X_val_proc, "toarray")
        else np.asarray(X_val_proc, dtype=np.float32)
    )

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X_val_dense), size=min(n_samples, len(X_val_dense)),
                            replace=False)
    X_sample = X_val_dense[sample_idx]

    # Predict function: operates in preprocessed space
    def _predict_proba(X_arr: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(X_arr)
        if proba.ndim == 1:
            return np.column_stack([1 - proba, proba])
        return proba

    explainer = LimeTabularExplainer(
        training_data=X_train_dense,
        feature_names=list(feature_names),
        mode="classification",
        random_state=random_state,
    )

    rows = []
    for i, x in enumerate(X_sample):
        try:
            exp = explainer.explain_instance(
                x, _predict_proba,
                num_features=len(feature_names),
                num_samples=n_lime_samples,
                labels=(1,),
            )
            coefs = dict(exp.as_list(label=1))
            # Map LIME feature strings back to column names
            row = {fn: 0.0 for fn in feature_names}
            for lime_feat, coef in coefs.items():
                # LIME stringifies features like "feat_name <= 0.5"
                # Extract the feature name part
                for fn in feature_names:
                    if fn in lime_feat:
                        row[fn] = coef
                        break
            rows.append(row)
        except Exception:
            rows.append({fn: float("nan") for fn in feature_names})

    df = pd.DataFrame(rows, columns=list(feature_names))
    df.insert(0, "sample_id", sample_idx.tolist())
    df.to_csv(out_dir / "lime_explanations.csv", index=False)
    if logger:
        logger.info("Saved LIME explanations to %s", out_dir / "lime_explanations.csv")


# ---------------------------------------------------------------------------
# Anchors: rule-based explanations (Ribeiro et al. 2018)
# ---------------------------------------------------------------------------

def save_anchors_explanations(
    pipeline,
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    feature_names: List[str],
    out_dir: Path,
    n_samples: int = 20,
    random_state: int = 42,
    logger=None,
):
    """Anchors — high-precision IF-THEN rules (Ribeiro et al. AAAI 2018).

    An anchor is a set of feature conditions that 'anchors' the prediction:
        IF proto=tcp AND src_bytes > 1024 THEN attack (precision ≥ 0.95)

    Anchors are directly actionable in IDS: an analyst can translate them
    directly into firewall rules or detection signatures.

    Only runs for tree/linear models (fast predict_fn required).
    Falls back gracefully if `alibi` is not installed.

    Saves
    -----
    out_dir/anchors.json — list of {sample_id, anchor, precision, coverage}
    """
    try:
        from alibi.explainers import AnchorTabular
    except ImportError:
        if logger:
            logger.warning("alibi not installed (pip install alibi); skipping Anchors.")
        return

    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _get_model(pipeline)

    X_train_proc = _transform_for_model(pipeline, X_train_raw)
    X_train_dense = (
        X_train_proc.toarray() if hasattr(X_train_proc, "toarray")
        else np.asarray(X_train_proc, dtype=np.float32)
    )
    X_val_proc = _transform_for_model(pipeline, X_val_raw)
    X_val_dense = (
        X_val_proc.toarray() if hasattr(X_val_proc, "toarray")
        else np.asarray(X_val_proc, dtype=np.float32)
    )

    def _predict(X_arr: np.ndarray) -> np.ndarray:
        return model.predict(X_arr)

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X_val_dense), size=min(n_samples, len(X_val_dense)),
                            replace=False)

    try:
        explainer = AnchorTabular(
            predictor=_predict,
            feature_names=list(feature_names),
        )
        explainer.fit(X_train_dense, disc_perc=(25, 50, 75))
    except Exception as exc:
        if logger:
            logger.warning("Anchors fit failed (%s); skipping.", exc)
        return

    # Use signal alarm on UNIX; otherwise, run without a hard global timeout.
    import signal

    use_alarm = hasattr(signal, "SIGALRM")

    def _timeout_handler(signum, frame):
        raise TimeoutError("Anchors explain timed out")

    results = []
    for idx in sample_idx:
        x = X_val_dense[idx]
        try:
            if use_alarm:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(30)  # 30-second hard limit per sample
            try:
                exp = explainer.explain(x, threshold=0.95)
            finally:
                if use_alarm:
                    signal.alarm(0)  # cancel alarm
            anchor_str = " AND ".join(exp.anchor) if exp.anchor else "(empty)"
            results.append({
                "sample_id": int(idx),
                "anchor":    anchor_str,
                "precision": float(exp.precision),
                "coverage":  float(exp.coverage),
                "prediction": int(_predict(x[None])[0]),
            })
        except Exception:
            results.append({
                "sample_id": int(idx),
                "anchor": None,
                "precision": None,
                "coverage": None,
                "prediction": None,
            })

    with open(out_dir / "anchors.json", "w") as f:
        json.dump(results, f, indent=2)
    if logger:
        logger.info("Saved Anchors explanations to %s", out_dir / "anchors.json")


# ---------------------------------------------------------------------------
# Captum LayerConductance: LSTM hidden-unit attribution for CNN-LSTM
# ---------------------------------------------------------------------------

def save_captum_layer_conductance(
    pipeline,
    X_val_raw: pd.DataFrame,
    feature_names: List[str],
    out_dir: Path,
    n_samples: int = 100,
    random_state: int = 42,
    logger=None,
):
    """Captum LayerConductance on the LSTM layer of the CNN-LSTM model.

    LayerConductance (Dhamdhere et al. 2018) is a layer-level extension of
    Integrated Gradients.  It attributes the model output to each HIDDEN UNIT
    of a given layer, revealing which LSTM memory cells are most responsible
    for the attack/normal decision.

    This is COMPLEMENTARY to input-level IG (which attributes to features).
    Together they answer:
      - IG: "which input features matter?"
      - LayerConductance: "which LSTM units encode those features?"

    Saves
    -----
    out_dir/layer_conductance_lstm.csv — (n_samples × hidden_dim) attribution
    out_dir/layer_conductance_lstm_summary.csv — mean |conductance| per unit
    """
    model = _get_model(pipeline)
    if not (hasattr(model, "net_") and model.net_ is not None):
        return  # Not a CNN-LSTM model

    try:
        from captum.attr import LayerConductance
    except ImportError:
        if logger:
            logger.warning("captum not installed (pip install captum); skipping LayerConductance.")
        return

    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = model.net_
    device = model.device_
    net.eval()
    net.to(device)

    X_val_proc = _transform_for_model(pipeline, X_val_raw)
    X_dense = (
        X_val_proc.toarray() if hasattr(X_val_proc, "toarray")
        else np.asarray(X_val_proc, dtype=np.float32)
    )

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_dense), size=min(n_samples, len(X_dense)), replace=False)
    X_sample = X_dense[idx]

    X_t = torch.tensor(X_sample, dtype=torch.float32, device=device)
    baseline_t = torch.zeros_like(X_t)

    # LayerConductance on the LSTM layer
    lc = LayerConductance(net, net.lstm)

    try:
        # binary model returns (batch,) — wrap to (batch, 1) for captum
        with torch.no_grad():
            probe = net(X_t[:1])
        if probe.dim() == 1:
            import torch.nn as nn
            class _Wrap(nn.Module):
                def __init__(self, inner): super().__init__(); self.inner = inner
                def forward(self, x): return self.inner(x).unsqueeze(-1)
            lc = LayerConductance(_Wrap(net).to(device), net.lstm)

        # LSTM outputs (output, (h_n, c_n)); LayerConductance attributes to output
        # target=0 = attack class (positive logit direction)
        conductance = lc.attribute(
            X_t, baselines=baseline_t, target=0,
            n_steps=20,
        )
        # conductance shape: (n_samples, seq_len, hidden_dim) or (n_samples, hidden_dim)
        cond_np = conductance.detach().cpu().numpy()
        if cond_np.ndim == 3:
            cond_np = cond_np.mean(axis=1)  # (n_samples, hidden_dim)

        hidden_dim = cond_np.shape[1]
        col_names = [f"lstm_h{i}" for i in range(hidden_dim)]

        df_full = pd.DataFrame(cond_np, columns=col_names)
        df_full.to_csv(out_dir / "layer_conductance_lstm.csv", index=False)

        summary = pd.DataFrame({
            "lstm_unit": col_names,
            "mean_abs_conductance": np.abs(cond_np).mean(axis=0),
        }).sort_values("mean_abs_conductance", ascending=False)
        summary.to_csv(out_dir / "layer_conductance_lstm_summary.csv", index=False)

        if logger:
            logger.info("Saved Captum LayerConductance to %s", out_dir)
    except Exception as exc:
        if logger:
            logger.warning("Captum LayerConductance failed (%s); skipping.", exc)
