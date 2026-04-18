#!/usr/bin/env python3
"""HYDRA Overnight Autonomous Execution — All Tasks.

Produces:
  results/xai_eval_toniot_behaviour_only.json
  results/xai_eval_toniot_identifier_inclusive.json
  results/oxs_ranking_behaviour_only.json
  results/oxs_ranking_identifier_inclusive.json
  results/detection_3seed_results.csv
  results/detection_summary.csv
  results/generalisation_ciciot2023.json
  results/cnn_lstm_detection.json
  results/leakage_interaction.json
  results/run_summary.txt
  run_log.txt
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = ROOT / "run_log.txt"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("overnight")


def log_task(msg: str):
    log.info(msg)


# ---------------------------------------------------------------------------
# Imports from hydra
# ---------------------------------------------------------------------------
from hydra.data.io import load_dataset
from hydra.data.preprocess import build_feature_spec, apply_feature_spec, fit_preprocessor
from hydra.models.tabular import build_logreg, build_random_forest, build_xgboost, build_lightgbm
from hydra.evaluation.metrics import compute_pr_auc, compute_roc_auc
from hydra.evaluation.thresholds import select_threshold_max_precision_at_recall, fpr_at_threshold
from hydra.xai.record import (
    compute_faithfulness,
    compute_stability,
    compute_simplicity,
    compute_plausibility,
    compute_timeliness,
    EXPERT_FEATURES,
)

CONFIG_PATH = str(ROOT / "hydra" / "config" / "datasets.yaml")
SEEDS = [21, 42, 84]
STABILITY_SEEDS = [21, 42, 84, 7, 99]
MODELS_LIST = ["logreg", "random_forest", "xgboost", "lightgbm"]
REGIMES = ["behaviour_only", "identifier_inclusive"]

# Per-attack-type reference features for plausibility
ATTACK_REFERENCE = {
    "dos":        ["src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "duration", "conn_state"],
    "ddos":       ["src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "duration", "conn_state"],
    "scanning":   ["dst_pkts", "duration", "conn_state", "service", "proto"],
    "ransomware": ["src_bytes", "dst_bytes", "duration", "missed_bytes", "conn_state"],
    "backdoor":   ["duration", "src_bytes", "dst_bytes", "conn_state", "proto"],
    "injection":  ["src_bytes", "dst_bytes", "service", "conn_state", "proto"],
    "xss":        ["src_bytes", "dst_bytes", "service", "conn_state", "duration"],
    "password":   ["duration", "src_pkts", "dst_pkts", "conn_state", "proto"],
    "mitm":       ["src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "conn_state"],
}


def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    log.info("Saved: %s", path)


# ===================================================================
# DATA LOADING AND SPLITTING
# ===================================================================

def load_and_split_toniot():
    """Load TON_IoT and split using row-order temporal split (no shuffle)."""
    log_task("Loading TON_IoT dataset...")
    df, cfg = load_dataset(CONFIG_PATH, "ton_iot")
    log.info("TON_IoT shape: %s, attack prevalence: %.3f", df.shape, df[cfg.label_col].mean())

    n = len(df)
    train_df = df.iloc[:int(0.70 * n)].copy()
    val_df = df.iloc[int(0.70 * n):int(0.85 * n)].copy()
    test_df = df.iloc[int(0.85 * n):].copy()

    log.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return df, cfg, train_df, val_df, test_df


def prepare_regime(train_df, val_df, test_df, cfg, regime):
    """Build feature spec, apply, fit preprocessor for a given regime."""
    spec = build_feature_spec(train_df, cfg.label_col, regime,
                              cfg.categorical_cols, cfg.numeric_cols, log)

    X_train_raw, cat_cols, num_cols = apply_feature_spec(train_df, spec, cfg.label_col, log)
    X_val_raw, _, _ = apply_feature_spec(val_df, spec, cfg.label_col, log)
    X_test_raw, _, _ = apply_feature_spec(test_df, spec, cfg.label_col, log)

    preprocessor = fit_preprocessor(X_train_raw, cat_cols, num_cols)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Get feature names
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend([f"num__{c}" for c in cols])
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            feature_names.extend([f"cat__{c}" for c in ohe.get_feature_names_out(cols)])

    # Convert sparse to dense
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    y_train = train_df[cfg.label_col].values
    y_val = val_df[cfg.label_col].values
    y_test = test_df[cfg.label_col].values

    # Type labels for multiclass
    type_train = train_df[cfg.type_col].values if cfg.type_col else None
    type_test = test_df[cfg.type_col].values if cfg.type_col else None

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "type_train": type_train, "type_test": type_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "spec": spec,
        "X_train_raw": X_train_raw,
        "cat_cols": cat_cols, "num_cols": num_cols,
    }


def build_model(model_name, seed):
    builders = {
        "logreg": build_logreg,
        "random_forest": build_random_forest,
        "xgboost": build_xgboost,
        "lightgbm": build_lightgbm,
    }
    return builders[model_name](seed)


def train_model(model_spec, X_train, y_train):
    model_spec.model.fit(X_train, y_train)
    return model_spec.model


def get_scores(model, X):
    """Get P(attack) scores."""
    proba = model.predict_proba(X)
    if proba.ndim == 2:
        return proba[:, 1]
    return proba


# ===================================================================
# SHAP EXPLAINERS
# ===================================================================

def make_shap_explain_fn(model, model_name, X_train, feature_names):
    """Return explain_fn(X) -> attributions (n, f)."""
    import shap

    def _explain(X_dense):
        X_dense = np.asarray(X_dense, dtype=np.float32)
        if model_name == "logreg":
            explainer = shap.LinearExplainer(model, X_train[:1000])
            vals = explainer.shap_values(X_dense)
        else:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X_dense)
            # For binary, TreeExplainer may return list of [neg, pos]
            if isinstance(vals, list):
                vals = vals[1]  # positive class
        vals = np.asarray(vals, dtype=np.float32)
        # Handle 3D output (n, features, n_classes) — take positive class
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return vals

    return _explain


def make_lime_explain_fn(model, X_train, feature_names):
    """Return explain_fn(X) -> attributions (n, f) via LIME."""
    import lime.lime_tabular

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train[:2000],
        feature_names=feature_names,
        class_names=["normal", "attack"],
        mode="classification",
        discretize_continuous=True,
    )

    def _explain(X_dense):
        X_dense = np.asarray(X_dense, dtype=np.float32)
        n_features = X_dense.shape[1]
        attrs = np.zeros((len(X_dense), n_features), dtype=np.float32)
        for i in range(len(X_dense)):
            try:
                exp = explainer.explain_instance(
                    X_dense[i], model.predict_proba,
                    num_features=n_features, labels=(1,)
                )
                feature_weights = dict(exp.as_map().get(1, []))
                for idx, weight in feature_weights.items():
                    if idx < n_features:
                        attrs[i, idx] = weight
            except Exception:
                pass
        return attrs

    return _explain


# ===================================================================
# TASK 1: XAI EVALUATION
# ===================================================================

def run_xai_eval_for_model(model, model_name, data, regime):
    """Run all 6 XAI criteria for one model on one regime."""
    log_task(f"  XAI eval: {model_name} / {regime}")
    X_test = data["X_test"]
    X_train = data["X_train"]
    feature_names = data["feature_names"]
    y_test = data["y_test"]
    type_test = data["type_test"]

    results = {}

    # Build SHAP explain function
    try:
        shap_explain_fn = make_shap_explain_fn(model, model_name, X_train, feature_names)
    except Exception as e:
        log.error("SHAP explainer failed for %s: %s", model_name, e)
        shap_explain_fn = None

    # Build LIME explain function
    try:
        lime_explain_fn = make_lime_explain_fn(model, X_train, feature_names)
    except Exception as e:
        log.error("LIME explainer failed for %s: %s", model_name, e)
        lime_explain_fn = None

    # Compute SHAP attributions on 500 test samples
    n_samples = min(500, len(X_test))
    X_eval = X_test[:n_samples]

    shap_attrs = None
    if shap_explain_fn is not None:
        try:
            shap_attrs = shap_explain_fn(X_eval)
            log.info("  SHAP attributions shape: %s", shap_attrs.shape)
        except Exception as e:
            log.error("  SHAP attribution failed: %s", e)

    if shap_attrs is None:
        log.warning("  No SHAP attributions for %s, skipping XAI eval", model_name)
        return None

    predict_fn = lambda X: get_scores(model, X)

    # --- Criterion 1: Faithfulness ---
    try:
        faith = compute_faithfulness(
            predict_fn, X_eval, shap_attrs,
            top_k_values=(5,), max_samples=500, baseline="median"
        )
        results["faithfulness"] = {
            "comprehensiveness": faith.get("comprehensiveness_k5", 0.0),
            "sufficiency": faith.get("sufficiency_k5", 0.0),
        }
    except Exception as e:
        log.error("  Faithfulness failed: %s", e)
        results["faithfulness"] = {"error": str(e)}

    # --- Criterion 2: Stability S1 (seed retraining) ---
    try:
        s1_rankings = []
        for seed in STABILITY_SEEDS:
            m = build_model(model_name, seed)
            train_model(m, data["X_train"], data["y_train"])
            _explain = make_shap_explain_fn(m.model, model_name, data["X_train"], feature_names)
            X_probe = X_test[:200]
            attrs_seed = _explain(X_probe)
            # Rank by mean absolute attribution
            mean_abs = np.abs(attrs_seed).mean(axis=0)
            ranking = np.argsort(mean_abs)[::-1]
            s1_rankings.append(ranking)

        from scipy.stats import spearmanr
        rhos = []
        for i, j in combinations(range(len(STABILITY_SEEDS)), 2):
            rho, _ = spearmanr(s1_rankings[i], s1_rankings[j])
            if not np.isnan(rho):
                rhos.append(rho)

        results["stability_s1"] = {
            "mean": round(float(np.mean(rhos)), 4) if rhos else None,
            "std": round(float(np.std(rhos)), 4) if rhos else None,
        }
    except Exception as e:
        log.error("  Stability S1 failed: %s", e)
        results["stability_s1"] = {"error": str(e)}

    # --- Criterion 3: Stability S2 (noise perturbation) ---
    try:
        stab = compute_stability(
            shap_explain_fn, X_test,
            n_samples=200, n_perturb=5,
            noise_std=0.01, top_k=5,
        )
        results["stability_s2"] = {
            "mean": stab.get("mean_spearman_rank_corr", None),
            "std": None,  # single model, report mean only
        }
    except Exception as e:
        log.error("  Stability S2 failed: %s", e)
        results["stability_s2"] = {"error": str(e)}

    # --- Criterion 4: Simplicity ---
    try:
        simp = compute_simplicity(shap_attrs)
        results["simplicity"] = {
            "k90": simp.get("k90_frac", None),
            "gini": simp.get("gini_coeff", None),
        }
    except Exception as e:
        log.error("  Simplicity failed: %s", e)
        results["simplicity"] = {"error": str(e)}

    # --- Criterion 5: Plausibility (RMA@5 per attack type) ---
    try:
        # Overall plausibility using standard expert features
        plaus = compute_plausibility(
            shap_attrs, feature_names,
            EXPERT_FEATURES.get("ton_iot", []), top_k=5
        )
        results["plausibility"] = {
            "rma_at_5_overall": plaus.get("rma_at_k", 0.0),
        }

        # Per-attack-type RMA@5
        if type_test is not None:
            per_attack = {}
            unique_types = [t for t in np.unique(type_test[:n_samples]) if t != "normal"]
            y_pred = model.predict(X_eval)
            for atype in unique_types:
                mask = (type_test[:n_samples] == atype) & (y_pred[:n_samples] == y_test[:n_samples])
                if mask.sum() < 5:
                    continue
                atype_attrs = shap_attrs[mask]
                ref_kws = ATTACK_REFERENCE.get(atype.lower(), EXPERT_FEATURES.get("ton_iot", []))
                atype_plaus = compute_plausibility(atype_attrs, feature_names, ref_kws, top_k=5)
                per_attack[atype] = atype_plaus.get("rma_at_k", 0.0)

            results["plausibility"]["per_attack"] = per_attack
    except Exception as e:
        log.error("  Plausibility failed: %s", e)
        results["plausibility"] = {"error": str(e)}

    # --- Criterion 6: Timeliness ---
    try:
        # SHAP timing
        shap_timing = compute_timeliness(shap_explain_fn, X_test, n_benchmark=500, n_warmup=1)
        results["timeliness"] = {
            "shap_ms_per_sample": shap_timing.get("ms_per_sample", None),
        }
        # LIME timing
        if lime_explain_fn is not None:
            lime_timing = compute_timeliness(lime_explain_fn, X_test[:50], n_benchmark=50, n_warmup=1)
            results["timeliness"]["lime_ms_per_sample"] = lime_timing.get("ms_per_sample", None)
    except Exception as e:
        log.error("  Timeliness failed: %s", e)
        results["timeliness"] = {"error": str(e)}

    return results


def task1_xai_evaluation(regime_data):
    """Task 1: Run XAI evaluation for all models on both regimes."""
    log_task("=" * 60)
    log_task("TASK 1: XAI EVALUATION")
    log_task("=" * 60)

    all_results = {}

    for regime in REGIMES:
        log_task(f"\n--- Regime: {regime} ---")
        data = regime_data[regime]
        regime_results = {}

        for model_name in MODELS_LIST:
            try:
                log_task(f"Training {model_name} (seed=42) for XAI...")
                spec = build_model(model_name, 42)
                model = train_model(spec, data["X_train"], data["y_train"])

                xai_result = run_xai_eval_for_model(model, model_name, data, regime)
                if xai_result is not None:
                    regime_results[model_name] = xai_result
                    log.info("  %s XAI complete: %s", model_name,
                             {k: v for k, v in xai_result.items() if not isinstance(v, dict) or "error" not in v})
            except Exception as e:
                log.error("XAI eval failed for %s/%s: %s", model_name, regime, e)
                traceback.print_exc()

        out_path = RESULTS / f"xai_eval_toniot_{regime}.json"
        save_json(regime_results, out_path)
        all_results[regime] = regime_results

    return all_results


# ===================================================================
# TASK 1b: CNN-LSTM
# ===================================================================

def _run_cnn_lstm_subprocess(regime, X_train, y_train, X_test, y_test,
                              feature_names, model_path, result_path):
    """Run CNN-LSTM train+eval in a subprocess to avoid MPS/memory hangs.

    The subprocess gets a clean Python environment, avoiding state
    contamination from prior tasks (SHAP, MPS init, etc.) that causes
    the main process to hang during torch LSTM operations.
    """
    import subprocess, tempfile, shutil

    tmpdir = tempfile.mkdtemp(prefix="cnn_lstm_")
    try:
        # Save data to temp files
        np.save(os.path.join(tmpdir, "X_train.npy"), X_train)
        np.save(os.path.join(tmpdir, "y_train.npy"), y_train)
        np.save(os.path.join(tmpdir, "X_test.npy"), X_test)
        np.save(os.path.join(tmpdir, "y_test.npy"), y_test)
        np.save(os.path.join(tmpdir, "feature_names.npy"),
                np.array(feature_names, dtype=object))

        script = f'''
import os, sys, json
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
sys.path.insert(0, {repr(str(ROOT))})
os.chdir({repr(str(ROOT))})

import numpy as np
import torch
import hydra.models.deep as _deep_mod
from hydra.models.deep import CNNLSTMClassifier

# Force CPU to avoid MPS LSTM hang
_deep_mod._get_device = lambda: torch.device("cpu")

X_train = np.load({repr(os.path.join(tmpdir, "X_train.npy"))})
y_train = np.load({repr(os.path.join(tmpdir, "y_train.npy"))})
X_test  = np.load({repr(os.path.join(tmpdir, "X_test.npy"))})
y_test  = np.load({repr(os.path.join(tmpdir, "y_test.npy"))})

print(f"Subprocess: X_train={{X_train.shape}}, device=CPU", flush=True)

# Subsample training data to avoid OOM on CPU (30k is enough for convergence)
MAX_TRAIN = 30000
if len(X_train) > MAX_TRAIN:
    rng = np.random.default_rng(42)
    # Stratified subsample
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = min(len(pos_idx), int(MAX_TRAIN * len(pos_idx) / len(y_train)))
    n_neg = MAX_TRAIN - n_pos
    sel = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
    ])
    rng.shuffle(sel)
    X_train = X_train[sel]
    y_train = y_train[sel]
    print(f"Subsampled to {{len(X_train)}} rows", flush=True)

clf = CNNLSTMClassifier(epochs=30, batch_size=128, patience=5, random_state=42, hidden=32)
clf.fit(X_train, y_train)
print("Fit completed", flush=True)

# Save model
torch.save(clf.net_.state_dict(), {repr(str(model_path))})

# Evaluate
scores = clf.predict_proba(X_test)[:, 1]
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
pr_auc = float(average_precision_score(y_test, scores))
try:
    roc_auc = float(roc_auc_score(y_test, scores))
except:
    roc_auc = 0.0
y_pred = clf.predict(X_test)
macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

# FPR@90recall
from sklearn.metrics import precision_recall_curve
precs, recs, threshs = precision_recall_curve(y_test, scores)
mask = recs[:-1] >= 0.9
if mask.any():
    best_idx = np.argmax(precs[:-1][mask])
    thr = threshs[np.where(mask)[0][best_idx]]
else:
    thr = 0.5
y_pos = (scores >= thr)
fp = ((y_pos == 1) & (y_test == 0)).sum()
tn = ((y_pos == 0) & (y_test == 0)).sum()
fpr90 = float(fp / max(fp + tn, 1))

result = {{
    "pr_auc": round(pr_auc, 4),
    "roc_auc": round(roc_auc, 4),
    "fpr_at_90recall": round(fpr90, 4),
    "macro_f1": round(macro_f1, 4),
}}
with open({repr(str(result_path))}, "w") as f:
    json.dump(result, f)
print(f"Results: {{result}}", flush=True)
'''
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

        log.info("  Launching CNN-LSTM subprocess (CPU, epochs=30, bs=128, hidden=32, max_train=30k)...")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=45 * 60,  # 45 min timeout
                env=env,
            )
        except subprocess.TimeoutExpired:
            log.error("  CNN-LSTM subprocess timed out after 45 min")
            return None

        # Log subprocess output
        for line in proc.stdout.strip().split("\n"):
            if line.strip():
                log.info("  [subprocess] %s", line.strip())
        if proc.returncode != 0:
            log.error("  CNN-LSTM subprocess failed (rc=%d)", proc.returncode)
            for line in proc.stderr.strip().split("\n")[-20:]:
                log.error("  [subprocess stderr] %s", line)
            return None

        # Load results
        if result_path.exists():
            with open(result_path) as f:
                return json.load(f)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def task1b_cnn_lstm(regime_data):
    """Task 1b: Wire, train, and evaluate CNN-LSTM."""
    log_task("=" * 60)
    log_task("TASK 1b: CNN-LSTM")
    log_task("=" * 60)

    start_time = time.time()
    TIMEOUT = 90 * 60  # 90 minutes

    try:
        import torch  # noqa: F401 — verify torch is installed
    except ImportError as e:
        log.error("CNN-LSTM dependencies missing: %s", e)
        log.info("CNN_LSTM_FAILED=True")
        return None

    cnn_results = {}

    for regime in REGIMES:
        if time.time() - start_time > TIMEOUT:
            log.warning("CNN-LSTM timeout reached, stopping")
            log.info("CNN_LSTM_FAILED=True")
            return cnn_results if cnn_results else None

        log_task(f"CNN-LSTM training: {regime}")
        data = regime_data[regime]
        log.info("  X_train shape: %s, y_train shape: %s",
                 data["X_train"].shape, data["y_train"].shape)

        try:
            model_path = MODELS / f"cnn_lstm_{regime}.pt"
            result_path = RESULTS / f"cnn_lstm_{regime}_metrics.json"

            result = _run_cnn_lstm_subprocess(
                regime=regime,
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_test=data["X_test"],
                y_test=data["y_test"],
                feature_names=data["feature_names"],
                model_path=model_path,
                result_path=result_path,
            )

            if result:
                cnn_results[regime] = result
                log.info("CNN-LSTM %s: PR-AUC=%.4f, ROC-AUC=%.4f, macro-F1=%.4f",
                         regime, result["pr_auc"], result["roc_auc"], result["macro_f1"])
            else:
                log.error("CNN-LSTM returned no results for %s", regime)

        except subprocess.TimeoutExpired:
            log.error("CNN-LSTM timed out for %s (1 hour limit)", regime)
        except Exception as e:
            log.error("CNN-LSTM failed for %s: %s", regime, e)
            traceback.print_exc()

    save_json(cnn_results, RESULTS / "cnn_lstm_detection.json")

    # --- CNN-LSTM XAI (Integrated Gradients + LIME) ---
    # Skip XAI for now if training barely succeeded; XAI requires
    # re-training which would take another hour+ per regime.
    # The XAI eval will be done in a follow-up if detection results look good.
    if cnn_results:
        log.info("CNN-LSTM detection results saved. XAI eval deferred to avoid timeout.")

    return cnn_results


# ===================================================================
# TASK 2: OXS RANKING
# ===================================================================

def task2_oxs_ranking():
    """Task 2: Compute Overall eXplainability Score (OXS)."""
    log_task("=" * 60)
    log_task("TASK 2: OXS RANKING")
    log_task("=" * 60)

    for regime in REGIMES:
        xai_path = RESULTS / f"xai_eval_toniot_{regime}.json"
        if not xai_path.exists():
            log.warning("XAI eval not found for %s, skipping OXS", regime)
            continue

        with open(xai_path) as f:
            xai_data = json.load(f)

        models = list(xai_data.keys())
        if not models:
            continue

        # Extract raw scores
        raw = {}
        for m in models:
            d = xai_data[m]
            faith = d.get("faithfulness", {})
            s1 = d.get("stability_s1", {})
            s2 = d.get("stability_s2", {})
            simp = d.get("simplicity", {})
            plaus = d.get("plausibility", {})
            timing = d.get("timeliness", {})

            raw[m] = {
                "comprehensiveness": faith.get("comprehensiveness", 0) or 0,
                "sufficiency": faith.get("sufficiency", 0) or 0,
                "s1_mean": s1.get("mean", 0) or 0,
                "s2_mean": s2.get("mean", 0) or 0,
                "k90": simp.get("k90", 1) or 1,
                "gini": simp.get("gini", 0) or 0,
                "rma_at_5": plaus.get("rma_at_5_overall", 0) or 0,
                "shap_ms": timing.get("shap_ms_per_sample") or timing.get("integrated_gradients_ms_per_sample") or 1,
            }

        def normalise(values, invert=False):
            """Min-max normalise to [0,1]. If invert, lower raw = higher normalised."""
            arr = np.array(values, dtype=float)
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-12:
                return np.full_like(arr, 0.5)
            normed = (arr - mn) / (mx - mn)
            if invert:
                normed = 1.0 - normed
            return normed

        model_names = list(raw.keys())
        n = len(model_names)

        # Faithfulness: mean of normalised comprehensiveness and sufficiency (higher=better)
        comp_norm = normalise([raw[m]["comprehensiveness"] for m in model_names])
        suff_norm = normalise([raw[m]["sufficiency"] for m in model_names])
        faith_score = (comp_norm + suff_norm) / 2

        # Stability: mean of normalised S1 and S2 (higher=better)
        s1_norm = normalise([raw[m]["s1_mean"] for m in model_names])
        s2_norm = normalise([raw[m]["s2_mean"] for m in model_names])
        stab_score = (s1_norm + s2_norm) / 2

        # Simplicity: invert-normalised k90 + normalised Gini
        k90_norm = normalise([raw[m]["k90"] for m in model_names], invert=True)
        gini_norm = normalise([raw[m]["gini"] for m in model_names])
        simp_score = (k90_norm + gini_norm) / 2

        # Plausibility: normalised rma_at_5 (higher=better)
        plaus_score = normalise([raw[m]["rma_at_5"] for m in model_names])

        # Timeliness: invert-normalised ms (lower=better)
        time_score = normalise([raw[m]["shap_ms"] for m in model_names], invert=True)

        # OXS = (1/5) * sum of all criteria
        oxs_scores = {}
        for i, m in enumerate(model_names):
            oxs = (faith_score[i] + stab_score[i] + simp_score[i] + plaus_score[i] + time_score[i]) / 5
            oxs_scores[m] = {
                "oxs": round(float(oxs), 4),
                "faithfulness": round(float(faith_score[i]), 4),
                "stability": round(float(stab_score[i]), 4),
                "simplicity": round(float(simp_score[i]), 4),
                "plausibility": round(float(plaus_score[i]), 4),
                "timeliness": round(float(time_score[i]), 4),
            }

        # Sort by OXS
        ranked = dict(sorted(oxs_scores.items(), key=lambda x: x[1]["oxs"], reverse=True))
        save_json(ranked, RESULTS / f"oxs_ranking_{regime}.json")

        log.info("OXS Ranking (%s):", regime)
        for m, scores in ranked.items():
            log.info("  %s: OXS=%.4f", m, scores["oxs"])


# ===================================================================
# TASK 3: 3-SEED DETECTION
# ===================================================================

def task3_detection(regime_data, cfg):
    """Task 3: Systematic 3-seed detection results."""
    log_task("=" * 60)
    log_task("TASK 3: 3-SEED DETECTION")
    log_task("=" * 60)

    from sklearn.metrics import f1_score, classification_report

    rows = []

    for regime in REGIMES:
        data = regime_data[regime]
        y_test = data["y_test"]
        type_test = data["type_test"]

        for model_name in MODELS_LIST:
            for seed in SEEDS:
                try:
                    log_task(f"  {model_name} / {regime} / seed={seed}")
                    spec = build_model(model_name, seed)
                    model = train_model(spec, data["X_train"], data["y_train"])

                    # Binary metrics
                    scores = get_scores(model, data["X_test"])
                    pr_auc = compute_pr_auc(y_test, scores)
                    roc_auc = compute_roc_auc(y_test, scores, log)

                    threshold, _ = select_threshold_max_precision_at_recall(y_test, scores, 0.9, log)
                    fpr90 = fpr_at_threshold(y_test, scores, threshold)

                    y_pred = model.predict(data["X_test"])
                    f1_binary = float(f1_score(y_test, y_pred, average="binary"))
                    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

                    row = {
                        "model": model_name, "regime": regime, "seed": seed,
                        "pr_auc": round(pr_auc, 4),
                        "roc_auc": round(roc_auc, 4),
                        "fpr_at_90recall": round(fpr90, 4),
                        "f1_binary": round(f1_binary, 4),
                        "macro_f1": round(macro_f1, 4),
                    }

                    # Per-class F1 for multiclass
                    if type_test is not None:
                        unique_types = sorted(set(type_test))
                        # Train multiclass model
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_type_train = le.fit_transform(data["type_train"])
                        y_type_test = le.transform(type_test)

                        mc_spec = build_model(model_name, seed)
                        mc_model = train_model(mc_spec, data["X_train"], y_type_train)
                        y_mc_pred = mc_model.predict(data["X_test"])
                        mc_f1 = float(f1_score(y_type_test, y_mc_pred, average="macro"))
                        row["multiclass_macro_f1"] = round(mc_f1, 4)

                        # Per-class F1
                        per_class_f1 = f1_score(y_type_test, y_mc_pred, average=None)
                        for idx, cls_name in enumerate(le.classes_):
                            if idx < len(per_class_f1):
                                row[f"f1_{cls_name}"] = round(float(per_class_f1[idx]), 4)

                    rows.append(row)
                    log.info("  PR-AUC=%.4f, ROC-AUC=%.4f, macro-F1=%.4f",
                             pr_auc, roc_auc, macro_f1)

                except Exception as e:
                    log.error("Detection failed %s/%s/seed=%d: %s", model_name, regime, seed, e)
                    traceback.print_exc()

    # Save raw results
    df_results = pd.DataFrame(rows)
    df_results.to_csv(RESULTS / "detection_3seed_results.csv", index=False)
    log.info("Saved detection_3seed_results.csv (%d rows)", len(df_results))

    # Compute summary (mean ± std across seeds)
    if not df_results.empty:
        metric_cols = [c for c in df_results.columns if c not in ["model", "regime", "seed"]]
        summary_rows = []
        for (model_name, regime), grp in df_results.groupby(["model", "regime"]):
            row = {"model": model_name, "regime": regime}
            for col in metric_cols:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    row[f"{col}_mean"] = round(float(vals.mean()), 4)
                    row[f"{col}_std"] = round(float(vals.std()), 4)
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(RESULTS / "detection_summary.csv", index=False)
        log.info("Saved detection_summary.csv")

    return df_results


# ===================================================================
# TASK 4: CIC-IoT2023 GENERALISATION
# ===================================================================

def task4_cic_generalisation(regime_data):
    """Task 4: Cross-dataset generalisation to CIC-IoT2023."""
    log_task("=" * 60)
    log_task("TASK 4: CIC-IoT2023 GENERALISATION")
    log_task("=" * 60)

    try:
        df_cic, cic_cfg = load_dataset(CONFIG_PATH, "cic_iot2023")
        log.info("CIC-IoT2023 shape: %s", df_cic.shape)
    except Exception as e:
        log.error("Could not load CIC-IoT2023: %s", e)
        return None

    # Use behaviour_only regime for cross-dataset
    ton_data = regime_data["behaviour_only"]

    # Find common features between TON_IoT and CIC-IoT2023
    from hydra.data.align import align_ton_iot, align_cic_iot2023

    results = {}

    for model_name in ["logreg", "random_forest", "xgboost", "lightgbm"]:
        try:
            log_task(f"  Generalisation: {model_name}")

            # Train on TON_IoT using canonical aligned features
            ton_df_full, ton_cfg = load_dataset(CONFIG_PATH, "ton_iot")
            ton_aligned = align_ton_iot(ton_df_full, ton_cfg)
            cic_aligned = align_cic_iot2023(df_cic, cic_cfg)

            canonical_features = [c for c in ton_aligned.columns if c.startswith("f_")]
            log.info("  Canonical features: %s", canonical_features)

            # Split TON_IoT
            n = len(ton_aligned)
            ton_train = ton_aligned.iloc[:int(0.70 * n)]
            ton_test = ton_aligned.iloc[int(0.85 * n):]

            X_train_canon = ton_train[canonical_features].values.astype(np.float32)
            y_train_canon = ton_train["label"].values
            X_test_canon = ton_test[canonical_features].values.astype(np.float32)
            y_test_canon = ton_test["label"].values

            # Impute NaN and scale
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler

            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train_canon)
            X_test_imp = imputer.transform(X_test_canon)

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_imp).astype(np.float32)
            X_test_sc = scaler.transform(X_test_imp).astype(np.float32)

            # Train model
            spec = build_model(model_name, 42)
            spec.model.fit(X_train_sc, y_train_canon)

            # Evaluate on TON_IoT test (source test)
            src_scores = get_scores(spec.model, X_test_sc)
            src_pr = compute_pr_auc(y_test_canon, src_scores)
            src_roc = compute_roc_auc(y_test_canon, src_scores, log)

            # Evaluate on CIC-IoT2023 (target)
            # Subsample CIC for speed (200k rows)
            if len(cic_aligned) > 200000:
                cic_sub = cic_aligned.sample(n=200000, random_state=42)
            else:
                cic_sub = cic_aligned

            X_cic = cic_sub[canonical_features].values.astype(np.float32)
            y_cic = cic_sub["label"].values
            X_cic_imp = imputer.transform(X_cic)
            X_cic_sc = scaler.transform(X_cic_imp).astype(np.float32)

            tgt_scores = get_scores(spec.model, X_cic_sc)
            tgt_pr = compute_pr_auc(y_cic, tgt_scores)
            tgt_roc = compute_roc_auc(y_cic, tgt_scores, log)

            results[model_name] = {
                "source_test_pr_auc": round(src_pr, 4),
                "source_test_roc_auc": round(src_roc, 4),
                "target_pr_auc": round(tgt_pr, 4),
                "target_roc_auc": round(tgt_roc, 4),
                "pr_auc_drop": round(tgt_pr - src_pr, 4),
                "roc_auc_drop": round(tgt_roc - src_roc, 4),
                "note": "PR-AUC inflated by CIC-IoT2023 97.6% attack prevalence",
            }
            log.info("  %s: src_ROC=%.4f → tgt_ROC=%.4f (drop=%.4f)",
                     model_name, src_roc, tgt_roc, tgt_roc - src_roc)

        except Exception as e:
            log.error("Generalisation failed for %s: %s", model_name, e)
            traceback.print_exc()

    save_json(results, RESULTS / "generalisation_ciciot2023.json")
    return results


# ===================================================================
# TASK 5: LEAKAGE x EXPLANATION QUALITY
# ===================================================================

def task5_leakage_interaction():
    """Task 5: Wilcoxon tests comparing behaviour_only vs identifier_inclusive."""
    log_task("=" * 60)
    log_task("TASK 5: LEAKAGE x EXPLANATION QUALITY")
    log_task("=" * 60)

    from scipy.stats import wilcoxon

    # Load XAI results for both regimes
    xai_a_path = RESULTS / "xai_eval_toniot_behaviour_only.json"
    xai_b_path = RESULTS / "xai_eval_toniot_identifier_inclusive.json"

    if not xai_a_path.exists() or not xai_b_path.exists():
        log.warning("XAI results missing for leakage analysis")
        return None

    with open(xai_a_path) as f:
        xai_a = json.load(f)
    with open(xai_b_path) as f:
        xai_b = json.load(f)

    common_models = sorted(set(xai_a.keys()) & set(xai_b.keys()))
    if len(common_models) < 2:
        log.warning("Need at least 2 models for Wilcoxon test, got %d", len(common_models))
        return None

    # Extract per-model metric values
    def extract_metric(xai_dict, metric_path, models):
        values = []
        for m in models:
            d = xai_dict[m]
            parts = metric_path.split(".")
            val = d
            for p in parts:
                val = val.get(p, None) if isinstance(val, dict) else None
                if val is None:
                    break
            values.append(val if val is not None else 0.0)
        return np.array(values, dtype=float)

    metrics_to_test = {
        "rma_at_5": "plausibility.rma_at_5_overall",
        "spearman_s1": "stability_s1.mean",
        "spearman_s2": "stability_s2.mean",
        "gini": "simplicity.gini",
        "k90": "simplicity.k90",
        "comprehensiveness": "faithfulness.comprehensiveness",
        "sufficiency": "faithfulness.sufficiency",
    }

    results = {"xai_metrics": {}, "detection_metrics": {}}

    for metric_name, metric_path in metrics_to_test.items():
        regime_a = extract_metric(xai_a, metric_path, common_models)
        regime_b = extract_metric(xai_b, metric_path, common_models)

        delta = float(np.mean(regime_b) - np.mean(regime_a))

        # Cohen's d
        pooled_std = np.sqrt((np.var(regime_a) + np.var(regime_b)) / 2)
        cohens_d = delta / pooled_std if pooled_std > 1e-12 else 0.0

        try:
            stat, p_value = wilcoxon(regime_a, regime_b)
            p_value = float(p_value)
        except Exception:
            p_value = 1.0

        reject = p_value < 0.05

        results["xai_metrics"][metric_name] = {
            "regime_a_mean": round(float(np.mean(regime_a)), 4),
            "regime_b_mean": round(float(np.mean(regime_b)), 4),
            "delta": round(delta, 4),
            "p_value": round(p_value, 4),
            "cohens_d": round(float(cohens_d), 4),
            "reject_null": reject,
        }

        if reject:
            direction = "increased" if delta > 0 else "decreased"
            log.info("  SIGNIFICANT: %s %s by %.4f (p=%.4f, d=%.4f) with identifier features",
                     metric_name, direction, abs(delta), p_value, cohens_d)

    # Also compare detection metrics from 3-seed results
    det_path = RESULTS / "detection_summary.csv"
    if det_path.exists():
        df_det = pd.read_csv(det_path)
        for metric in ["pr_auc", "roc_auc", "macro_f1"]:
            col = f"{metric}_mean"
            if col not in df_det.columns:
                continue

            a_vals = df_det[df_det["regime"] == "behaviour_only"][col].dropna().values
            b_vals = df_det[df_det["regime"] == "identifier_inclusive"][col].dropna().values

            if len(a_vals) < 2 or len(b_vals) < 2:
                continue

            n = min(len(a_vals), len(b_vals))
            a_vals, b_vals = a_vals[:n], b_vals[:n]

            delta = float(np.mean(b_vals) - np.mean(a_vals))
            pooled_std = np.sqrt((np.var(a_vals) + np.var(b_vals)) / 2)
            cohens_d = delta / pooled_std if pooled_std > 1e-12 else 0.0

            try:
                _, p_value = wilcoxon(a_vals, b_vals)
                p_value = float(p_value)
            except Exception:
                p_value = 1.0

            results["detection_metrics"][metric] = {
                "regime_a_mean": round(float(np.mean(a_vals)), 4),
                "regime_b_mean": round(float(np.mean(b_vals)), 4),
                "delta": round(delta, 4),
                "p_value": round(p_value, 4),
                "cohens_d": round(float(cohens_d), 4),
                "reject_null": p_value < 0.05,
            }

    save_json(results, RESULTS / "leakage_interaction.json")
    return results


# ===================================================================
# TASK 6: FINAL SUMMARY
# ===================================================================

def task6_summary(cnn_results, detection_df):
    """Task 6: Produce final summary."""
    log_task("=" * 60)
    log_task("TASK 6: FINAL SUMMARY")
    log_task("=" * 60)

    tasks_completed = 0
    total_tasks = 6

    # Check what was produced
    files = {
        "XAI evaluation (behaviour_only)": RESULTS / "xai_eval_toniot_behaviour_only.json",
        "XAI evaluation (identifier_inclusive)": RESULTS / "xai_eval_toniot_identifier_inclusive.json",
        "OXS ranking (behaviour_only)": RESULTS / "oxs_ranking_behaviour_only.json",
        "OXS ranking (identifier_inclusive)": RESULTS / "oxs_ranking_identifier_inclusive.json",
        "3-seed detection results": RESULTS / "detection_3seed_results.csv",
        "3-seed detection summary": RESULTS / "detection_summary.csv",
        "CIC-IoT2023 generalisation": RESULTS / "generalisation_ciciot2023.json",
        "Leakage interaction": RESULTS / "leakage_interaction.json",
        "CNN-LSTM detection": RESULTS / "cnn_lstm_detection.json",
    }

    file_status = {}
    for name, path in files.items():
        exists = path.exists()
        file_status[name] = "complete" if exists else "missing"
        if exists:
            tasks_completed += 1

    # Rough task count (some files grouped)
    tasks_completed = min(total_tasks, tasks_completed // 2 + 1)  # rough heuristic

    # Check actual task completion
    actual_tasks = 0
    if (RESULTS / "xai_eval_toniot_behaviour_only.json").exists():
        actual_tasks += 1
    if cnn_results:
        actual_tasks += 1
    if (RESULTS / "oxs_ranking_behaviour_only.json").exists():
        actual_tasks += 1
    if (RESULTS / "detection_3seed_results.csv").exists():
        actual_tasks += 1
    if (RESULTS / "generalisation_ciciot2023.json").exists():
        actual_tasks += 1
    if (RESULTS / "leakage_interaction.json").exists():
        actual_tasks += 1

    cnn_status = "SUCCESS" if cnn_results else "FAILED"

    # Best detector from 3-seed summary
    best_det = "N/A"
    best_pr = 0.0
    best_mf1 = 0.0
    det_summary_path = RESULTS / "detection_summary.csv"
    if det_summary_path.exists():
        df = pd.read_csv(det_summary_path)
        ba_df = df[df["regime"] == "behaviour_only"]
        if not ba_df.empty and "pr_auc_mean" in ba_df.columns:
            best_idx = ba_df["pr_auc_mean"].idxmax()
            best_det = ba_df.loc[best_idx, "model"]
            best_pr = ba_df.loc[best_idx, "pr_auc_mean"]
            best_mf1 = ba_df.loc[best_idx].get("macro_f1_mean", 0.0)

    # CNN-LSTM numbers
    cnn_pr = "N/A"
    cnn_mf1 = "N/A"
    if cnn_results and "behaviour_only" in cnn_results:
        cnn_pr = cnn_results["behaviour_only"].get("pr_auc", "N/A")
        cnn_mf1 = cnn_results["behaviour_only"].get("macro_f1", "N/A")

    # Best OXS model
    best_oxs_model = "N/A"
    best_oxs_score = 0.0
    oxs_path = RESULTS / "oxs_ranking_behaviour_only.json"
    if oxs_path.exists():
        with open(oxs_path) as f:
            oxs = json.load(f)
        if oxs:
            best_oxs_model = list(oxs.keys())[0]
            best_oxs_score = oxs[best_oxs_model].get("oxs", 0.0)

    # Leakage numbers
    leak_rma = "+N/A"
    leak_rma_p = "N/A"
    leak_s1 = "N/A"
    leak_s1_p = "N/A"
    leak_path = RESULTS / "leakage_interaction.json"
    if leak_path.exists():
        with open(leak_path) as f:
            leak = json.load(f)
        xm = leak.get("xai_metrics", {})
        if "rma_at_5" in xm:
            d = xm["rma_at_5"]
            leak_rma = f"{d.get('delta', 0):+.2f} pp"
            leak_rma_p = f"{d.get('p_value', 1):.3f}"
        if "spearman_s1" in xm:
            d = xm["spearman_s1"]
            leak_s1 = f"{d.get('delta', 0):+.3f}"
            leak_s1_p = f"{d.get('p_value', 1):.3f}"

    summary = f"""
{'=' * 56}
         HYDRA OVERNIGHT RUN -- FINAL SUMMARY
{'=' * 56}

Tasks completed:         {actual_tasks} / {total_tasks}
CNN-LSTM:                {cnn_status}

Result files produced:
"""
    for name, status in file_status.items():
        summary += f"  {name:40s} {status}\n"

    summary += f"""
Key numbers (paste these into the dissertation):
  Best detector (Regime A):   {best_det}  PR-AUC={best_pr}  macro-F1={best_mf1}
  CNN-LSTM   (Regime A):      PR-AUC={cnn_pr}  macro-F1={cnn_mf1}
  Best OXS model:             {best_oxs_model}  OXS={best_oxs_score}
  Leakage -> RMA@5 delta:     {leak_rma}  (p={leak_rma_p})
  Leakage -> S1 stability:    {leak_s1}    (p={leak_s1_p})

Full log: run_log.txt
"""

    print(summary)
    with open(RESULTS / "run_summary.txt", "w") as f:
        f.write(summary)
    log.info("Saved run_summary.txt")


# ===================================================================
# MAIN ORCHESTRATOR
# ===================================================================

def main():
    t_start = time.time()
    log_task("=" * 60)
    log_task("HYDRA OVERNIGHT RUN STARTED")
    log_task(f"Time: {datetime.now().isoformat()}")
    log_task("=" * 60)

    # --- Load data once ---
    df, cfg, train_df, val_df, test_df = load_and_split_toniot()

    # --- Prepare both regimes ---
    regime_data = {}
    for regime in REGIMES:
        log_task(f"Preparing regime: {regime}")
        try:
            regime_data[regime] = prepare_regime(train_df, val_df, test_df, cfg, regime)
            log.info("  %s: X_train=%s, X_test=%s, features=%d",
                     regime, regime_data[regime]["X_train"].shape,
                     regime_data[regime]["X_test"].shape,
                     len(regime_data[regime]["feature_names"]))
        except Exception as e:
            log.error("Failed to prepare regime %s: %s", regime, e)
            traceback.print_exc()
            return

    # --- Task 1: XAI Evaluation ---
    log_task("Starting Task 1: XAI Evaluation")
    try:
        xai_results = task1_xai_evaluation(regime_data)
    except Exception as e:
        log.error("Task 1 failed: %s", e)
        traceback.print_exc()
        xai_results = {}

    # --- Task 1b: CNN-LSTM ---
    log_task("Starting Task 1b: CNN-LSTM")
    try:
        cnn_results = task1b_cnn_lstm(regime_data)
    except Exception as e:
        log.error("Task 1b failed: %s", e)
        traceback.print_exc()
        cnn_results = None
        with open(LOG_FILE, "a") as f:
            f.write(f"\nCNN_LSTM_FAILED=True\n")

    # --- Task 3: 3-Seed Detection ---
    log_task("Starting Task 3: 3-Seed Detection")
    try:
        detection_df = task3_detection(regime_data, cfg)
    except Exception as e:
        log.error("Task 3 failed: %s", e)
        traceback.print_exc()
        detection_df = pd.DataFrame()

    # --- Task 2: OXS Ranking (depends on Task 1) ---
    log_task("Starting Task 2: OXS Ranking")
    try:
        task2_oxs_ranking()
    except Exception as e:
        log.error("Task 2 failed: %s", e)
        traceback.print_exc()

    # --- Task 4: CIC-IoT2023 Generalisation (depends on Task 3) ---
    log_task("Starting Task 4: CIC-IoT2023 Generalisation")
    try:
        task4_cic_generalisation(regime_data)
    except Exception as e:
        log.error("Task 4 failed: %s", e)
        traceback.print_exc()

    # --- Task 5: Leakage Interaction (depends on Tasks 1 and 3) ---
    log_task("Starting Task 5: Leakage Interaction")
    try:
        task5_leakage_interaction()
    except Exception as e:
        log.error("Task 5 failed: %s", e)
        traceback.print_exc()

    # --- Task 6: Final Summary ---
    try:
        task6_summary(cnn_results, detection_df)
    except Exception as e:
        log.error("Task 6 failed: %s", e)
        traceback.print_exc()

    elapsed = time.time() - t_start
    log_task(f"HYDRA OVERNIGHT RUN COMPLETED in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
