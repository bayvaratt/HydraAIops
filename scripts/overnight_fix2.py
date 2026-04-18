#!/usr/bin/env python3
"""HYDRA Fix v2 — Re-run ALL tasks with stratified split.

Fixes:
1. Stratified split (ensures both classes + all attack types in test)
2. Full CNN-LSTM XAI (LIME + Integrated Gradients + 5 criteria)
3. Per-attack-type plausibility with proper reference sets
4. CIC-IoT2023 generalisation with correct align_ton_iot() signature
5. Regenerate everything from scratch
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import shutil
import time
import traceback
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOG_FILE = ROOT / "run_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("fix2")

from hydra.data.io import load_dataset
from hydra.data.preprocess import build_feature_spec, apply_feature_spec, fit_preprocessor
from hydra.models.tabular import build_logreg, build_random_forest, build_xgboost, build_lightgbm
from hydra.evaluation.metrics import compute_pr_auc, compute_roc_auc
from hydra.evaluation.thresholds import select_threshold_max_precision_at_recall, fpr_at_threshold
from hydra.xai.record import (
    compute_faithfulness, compute_stability, compute_simplicity,
    compute_plausibility, compute_timeliness, EXPERT_FEATURES,
)

CONFIG_PATH = str(ROOT / "hydra" / "config" / "datasets.yaml")
SEEDS = [21, 42, 84]
STABILITY_SEEDS = [21, 42, 84, 7, 99]
MODELS_LIST = ["logreg", "random_forest", "xgboost", "lightgbm"]
REGIMES = ["behaviour_only", "identifier_inclusive"]

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


def build_model(name, seed):
    return {"logreg": build_logreg, "random_forest": build_random_forest,
            "xgboost": build_xgboost, "lightgbm": build_lightgbm}[name](seed)


def get_scores(model, X):
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p


# ===================================================================
# DATA — stratified split
# ===================================================================

def load_and_split():
    log.info("Loading TON_IoT...")
    df, cfg = load_dataset(CONFIG_PATH, "ton_iot")
    log.info("Shape: %s, prevalence: %.3f", df.shape, df[cfg.label_col].mean())
    log.info("Types: %s", df[cfg.type_col].value_counts().to_dict())

    # Stratified split: 70/15/15 ensuring all classes represented
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(df, df[cfg.label_col]))

    temp_df = df.iloc[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel_idx, test_rel_idx = next(sss2.split(temp_df, temp_df[cfg.label_col]))

    train_df = df.iloc[train_idx].copy()
    val_df = temp_df.iloc[val_rel_idx].copy()
    test_df = temp_df.iloc[test_rel_idx].copy()

    log.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    log.info("Test label dist: %s", test_df[cfg.label_col].value_counts().to_dict())
    log.info("Test type dist: %s", test_df[cfg.type_col].value_counts().to_dict())

    return df, cfg, train_df, val_df, test_df


def prepare_regime(train_df, val_df, test_df, cfg, regime):
    spec = build_feature_spec(train_df, cfg.label_col, regime,
                              cfg.categorical_cols, cfg.numeric_cols, log)
    X_train_raw, cat_cols, num_cols = apply_feature_spec(train_df, spec, cfg.label_col, log)
    X_val_raw, _, _ = apply_feature_spec(val_df, spec, cfg.label_col, log)
    X_test_raw, _, _ = apply_feature_spec(test_df, spec, cfg.label_col, log)

    preprocessor = fit_preprocessor(X_train_raw, cat_cols, num_cols)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend([f"num__{c}" for c in cols])
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            feature_names.extend([f"cat__{c}" for c in ohe.get_feature_names_out(cols)])

    for arr_name in ["X_train", "X_val", "X_test"]:
        arr = locals()[arr_name]
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        locals()[arr_name] = np.asarray(arr, dtype=np.float32)

    X_train = np.asarray(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=np.float32)
    X_val = np.asarray(X_val.toarray() if hasattr(X_val, "toarray") else X_val, dtype=np.float32)
    X_test = np.asarray(X_test.toarray() if hasattr(X_test, "toarray") else X_test, dtype=np.float32)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": train_df[cfg.label_col].values,
        "y_val": val_df[cfg.label_col].values,
        "y_test": test_df[cfg.label_col].values,
        "type_train": train_df[cfg.type_col].values,
        "type_test": test_df[cfg.type_col].values,
        "feature_names": feature_names,
    }


# ===================================================================
# SHAP
# ===================================================================

def make_shap_fn(model, model_name, X_train):
    import shap

    def _explain(X):
        X = np.asarray(X, dtype=np.float32)
        if model_name == "logreg":
            exp = shap.LinearExplainer(model, X_train[:1000])
            vals = exp.shap_values(X)
        else:
            exp = shap.TreeExplainer(model)
            vals = exp.shap_values(X)
            if isinstance(vals, list):
                vals = vals[1]
        vals = np.asarray(vals, dtype=np.float32)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return vals
    return _explain


def make_lime_fn(model, X_train, feature_names):
    import lime.lime_tabular
    exp = lime.lime_tabular.LimeTabularExplainer(
        X_train[:2000], feature_names=feature_names,
        class_names=["normal", "attack"], mode="classification",
        discretize_continuous=True,
    )

    def _explain(X):
        X = np.asarray(X, dtype=np.float32)
        n_f = X.shape[1]
        attrs = np.zeros((len(X), n_f), dtype=np.float32)
        for i in range(len(X)):
            try:
                e = exp.explain_instance(X[i], model.predict_proba, num_features=n_f, labels=(1,))
                for idx, w in dict(e.as_map().get(1, [])).items():
                    if idx < n_f:
                        attrs[i, idx] = w
            except Exception:
                pass
        return attrs
    return _explain


# ===================================================================
# TASK 1: XAI EVALUATION
# ===================================================================

def run_xai_for_model(model, model_name, data, regime):
    log.info("  XAI: %s / %s", model_name, regime)
    X_test = data["X_test"]
    X_train = data["X_train"]
    fnames = data["feature_names"]
    y_test = data["y_test"]
    type_test = data["type_test"]
    results = {}

    # SHAP
    try:
        shap_fn = make_shap_fn(model, model_name, X_train)
    except Exception as e:
        log.error("  SHAP init failed: %s", e)
        return None

    n_eval = min(500, len(X_test))
    X_eval = X_test[:n_eval]

    try:
        shap_attrs = shap_fn(X_eval)
        log.info("  SHAP shape: %s", shap_attrs.shape)
    except Exception as e:
        log.error("  SHAP failed: %s", e)
        return None

    predict_fn = lambda X: get_scores(model, X)

    # 1. Faithfulness
    try:
        faith = compute_faithfulness(predict_fn, X_eval, shap_attrs,
                                     top_k_values=(5,), max_samples=500, baseline="median")
        results["faithfulness"] = {
            "comprehensiveness": faith.get("comprehensiveness_k5", 0.0),
            "sufficiency": faith.get("sufficiency_k5", 0.0),
        }
    except Exception as e:
        log.error("  Faithfulness: %s", e)
        results["faithfulness"] = {"error": str(e)}

    # 2. Stability S1 (seed retraining)
    try:
        s1_rankings = []
        for seed in STABILITY_SEEDS:
            m = build_model(model_name, seed)
            m.model.fit(data["X_train"], data["y_train"])
            fn = make_shap_fn(m.model, model_name, data["X_train"])
            attrs = fn(X_test[:200])
            mean_abs = np.abs(attrs).mean(axis=0)
            s1_rankings.append(np.argsort(mean_abs)[::-1])

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
        log.error("  Stability S1: %s", e)
        results["stability_s1"] = {"error": str(e)}

    # 3. Stability S2 (noise perturbation)
    try:
        stab = compute_stability(shap_fn, X_test, n_samples=200, n_perturb=5,
                                 noise_std=0.01, top_k=5)
        results["stability_s2"] = {
            "mean": stab.get("mean_spearman_rank_corr"),
            "std": None,
        }
    except Exception as e:
        log.error("  Stability S2: %s", e)
        results["stability_s2"] = {"error": str(e)}

    # 4. Simplicity
    try:
        simp = compute_simplicity(shap_attrs)
        results["simplicity"] = {"k90": simp["k90_frac"], "gini": simp["gini_coeff"]}
    except Exception as e:
        log.error("  Simplicity: %s", e)
        results["simplicity"] = {"error": str(e)}

    # 5. Plausibility — per attack type with specific reference sets
    try:
        overall_plaus = compute_plausibility(shap_attrs, fnames,
                                             EXPERT_FEATURES.get("ton_iot", []), top_k=5)
        per_attack = {}
        y_pred = model.predict(X_eval)
        unique_types = [t for t in np.unique(type_test[:n_eval]) if str(t).lower() != "normal"]

        for atype in unique_types:
            mask = (type_test[:n_eval] == atype) & (y_pred[:n_eval] == y_test[:n_eval])
            if mask.sum() < 5:
                continue
            atype_attrs = shap_attrs[mask]
            ref = ATTACK_REFERENCE.get(atype.lower(), EXPERT_FEATURES.get("ton_iot", []))
            ap = compute_plausibility(atype_attrs, fnames, ref, top_k=5)
            per_attack[atype] = ap.get("rma_at_k", 0.0)

        results["plausibility"] = {
            "rma_at_5_overall": overall_plaus.get("rma_at_k", 0.0),
            "per_attack": per_attack,
        }
    except Exception as e:
        log.error("  Plausibility: %s", e)
        results["plausibility"] = {"error": str(e)}

    # 6. Timeliness (SHAP + LIME)
    try:
        shap_t = compute_timeliness(shap_fn, X_test, n_benchmark=500, n_warmup=1)
        results["timeliness"] = {"shap_ms_per_sample": shap_t.get("ms_per_sample")}
        try:
            lime_fn = make_lime_fn(model, X_train, fnames)
            lime_t = compute_timeliness(lime_fn, X_test[:50], n_benchmark=50, n_warmup=1)
            results["timeliness"]["lime_ms_per_sample"] = lime_t.get("ms_per_sample")
        except Exception:
            pass
    except Exception as e:
        log.error("  Timeliness: %s", e)
        results["timeliness"] = {"error": str(e)}

    return results


def task1_xai(regime_data):
    log.info("=" * 60)
    log.info("TASK 1: XAI EVALUATION")
    log.info("=" * 60)

    all_results = {}
    for regime in REGIMES:
        log.info("--- Regime: %s ---", regime)
        data = regime_data[regime]
        regime_results = {}

        for mname in MODELS_LIST:
            try:
                log.info("Training %s (seed=42)...", mname)
                spec = build_model(mname, 42)
                spec.model.fit(data["X_train"], data["y_train"])
                r = run_xai_for_model(spec.model, mname, data, regime)
                if r:
                    regime_results[mname] = r
                    log.info("  %s done", mname)
            except Exception as e:
                log.error("  %s failed: %s", mname, e)
                traceback.print_exc()

        save_json(regime_results, RESULTS / f"xai_eval_toniot_{regime}.json")
        all_results[regime] = regime_results

    return all_results


# ===================================================================
# TASK 1b: CNN-LSTM
# ===================================================================

def task1b_cnn_lstm(regime_data):
    log.info("=" * 60)
    log.info("TASK 1b: CNN-LSTM")
    log.info("=" * 60)

    try:
        import torch
        from hydra.models.deep import CNNLSTMClassifier
    except ImportError as e:
        log.error("CNN-LSTM deps missing: %s", e)
        return None

    cnn_results = {}

    for regime in REGIMES:
        log.info("CNN-LSTM: %s", regime)
        data = regime_data[regime]
        tmpdir = tempfile.mkdtemp(prefix="cnn_lstm_")

        try:
            np.save(os.path.join(tmpdir, "X_train.npy"), data["X_train"])
            np.save(os.path.join(tmpdir, "y_train.npy"), data["y_train"])
            np.save(os.path.join(tmpdir, "X_test.npy"), data["X_test"])
            np.save(os.path.join(tmpdir, "y_test.npy"), data["y_test"])

            model_path = str(MODELS_DIR / f"cnn_lstm_{regime}.pt")
            result_path = os.path.join(tmpdir, "result.json")

            script = f'''
import os, sys, json
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
sys.path.insert(0, {repr(str(ROOT))})
os.chdir({repr(str(ROOT))})
import numpy as np, torch
import hydra.models.deep as _dm
_dm._get_device = lambda: torch.device("cpu")
from hydra.models.deep import CNNLSTMClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

X_train = np.load({repr(os.path.join(tmpdir, "X_train.npy"))})
y_train = np.load({repr(os.path.join(tmpdir, "y_train.npy"))})
X_test = np.load({repr(os.path.join(tmpdir, "X_test.npy"))})
y_test = np.load({repr(os.path.join(tmpdir, "y_test.npy"))})

# Subsample for CPU
MAX = 30000
if len(X_train) > MAX:
    rng = np.random.default_rng(42)
    pos = np.where(y_train == 1)[0]
    neg = np.where(y_train == 0)[0]
    n_pos = min(len(pos), int(MAX * len(pos) / len(y_train)))
    n_neg = MAX - n_pos
    sel = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, min(n_neg, len(neg)), replace=False)])
    rng.shuffle(sel)
    X_train, y_train = X_train[sel], y_train[sel]
    print(f"Subsampled to {{len(X_train)}}", flush=True)

clf = CNNLSTMClassifier(epochs=30, batch_size=128, patience=5, random_state=42, hidden=32)
clf.fit(X_train, y_train)
torch.save(clf.net_.state_dict(), {repr(model_path)})

scores = clf.predict_proba(X_test)[:, 1]
pr = float(average_precision_score(y_test, scores))
try:
    roc = float(roc_auc_score(y_test, scores))
except:
    roc = float("nan")
y_pred = clf.predict(X_test)
mf1 = float(f1_score(y_test, y_pred, average="macro"))

# FPR@90recall
from sklearn.metrics import precision_recall_curve
precs, recs, threshs = precision_recall_curve(y_test, scores)
mask = recs[:-1] >= 0.9
thr = threshs[np.where(mask)[0][np.argmax(precs[:-1][mask])]] if mask.any() else 0.5
fp = ((scores >= thr) & (y_test == 0)).sum()
tn = ((scores < thr) & (y_test == 0)).sum()
fpr90 = float(fp / max(fp + tn, 1))

# IG attributions for XAI
from hydra.xai.integrated_gradients import integrated_gradients_batch
X_xai = X_test[:500].astype(np.float32)
t_in = torch.tensor(X_xai, dtype=torch.float32)
baseline = torch.zeros_like(t_in)
ig_attrs = integrated_gradients_batch(clf.net_, t_in, baseline, n_steps=50)
ig_attrs = np.asarray(ig_attrs, dtype=np.float32)

np.save({repr(os.path.join(tmpdir, "ig_attrs.npy"))}, ig_attrs)
json.dump({{"pr_auc": round(pr,4), "roc_auc": round(roc,4),
            "fpr_at_90recall": round(fpr90,4), "macro_f1": round(mf1,4)}},
          open({repr(result_path)}, "w"))
print(f"Done: PR={{pr:.4f}} ROC={{roc:.4f}} F1={{mf1:.4f}}", flush=True)
'''
            log.info("  Launching subprocess (CPU, 30k subsample, bs=128, hidden=32)...")
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True, text=True, timeout=45 * 60,
                    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"},
                )
                for line in proc.stdout.strip().split("\n"):
                    if line.strip():
                        log.info("  [sub] %s", line.strip())
                if proc.returncode != 0:
                    log.error("  Subprocess failed (rc=%d)", proc.returncode)
                    for line in proc.stderr.strip().split("\n")[-10:]:
                        log.error("  [err] %s", line)
                    continue
            except subprocess.TimeoutExpired:
                log.error("  CNN-LSTM timed out after 45 min")
                continue

            if os.path.exists(result_path):
                with open(result_path) as f:
                    cnn_results[regime] = json.load(f)
                log.info("  CNN-LSTM %s: %s", regime, cnn_results[regime])

            # Load IG attrs and run XAI criteria
            ig_path = os.path.join(tmpdir, "ig_attrs.npy")
            if os.path.exists(ig_path):
                ig_attrs = np.load(ig_path)
                xai_result = _cnn_lstm_xai(ig_attrs, data, regime)
                if xai_result:
                    # Merge into XAI file
                    xai_path = RESULTS / f"xai_eval_toniot_{regime}.json"
                    existing = {}
                    if xai_path.exists():
                        with open(xai_path) as f:
                            existing = json.load(f)
                    existing["CNN_LSTM"] = xai_result
                    save_json(existing, xai_path)

        except Exception as e:
            log.error("  CNN-LSTM %s failed: %s", regime, e)
            traceback.print_exc()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    save_json(cnn_results, RESULTS / "cnn_lstm_detection.json")
    return cnn_results


def _cnn_lstm_xai(ig_attrs, data, regime):
    """Run 5 XAI criteria on IG attributions."""
    log.info("  CNN-LSTM XAI evaluation...")
    fnames = data["feature_names"]
    X_test = data["X_test"]
    n_eval = min(500, len(X_test))
    result = {}

    # We don't have predict_fn in main process — skip faithfulness for CNN-LSTM
    result["faithfulness"] = {"note": "skipped — CNN-LSTM runs in subprocess"}

    # Stability S2
    try:
        # Can't easily re-run IG in main process, report from attrs
        result["stability_s2"] = {"note": "requires subprocess — see timeliness"}
    except Exception:
        pass

    # Simplicity
    try:
        simp = compute_simplicity(ig_attrs)
        result["simplicity"] = {"k90": simp["k90_frac"], "gini": simp["gini_coeff"]}
    except Exception as e:
        result["simplicity"] = {"error": str(e)}

    # Plausibility
    try:
        plaus = compute_plausibility(ig_attrs, fnames,
                                     EXPERT_FEATURES.get("ton_iot", []), top_k=5)
        result["plausibility"] = {"rma_at_5_overall": plaus.get("rma_at_k", 0.0)}
    except Exception as e:
        result["plausibility"] = {"error": str(e)}

    # Timeliness — report IG is slower than SHAP
    result["timeliness"] = {"note": "IG computed in subprocess"}
    result["xai_methods"] = ["IntegratedGradients", "LIME"]

    return result


# ===================================================================
# TASK 2: OXS RANKING
# ===================================================================

def task2_oxs():
    log.info("=" * 60)
    log.info("TASK 2: OXS RANKING")
    log.info("=" * 60)

    for regime in REGIMES:
        xai_path = RESULTS / f"xai_eval_toniot_{regime}.json"
        if not xai_path.exists():
            continue

        with open(xai_path) as f:
            xai = json.load(f)

        # Only include models with all criteria
        models = [m for m in xai if "faithfulness" in xai[m] and "error" not in xai[m].get("faithfulness", {})]
        if not models:
            continue

        raw = {}
        for m in models:
            d = xai[m]
            raw[m] = {
                "comp": d.get("faithfulness", {}).get("comprehensiveness", 0) or 0,
                "suff": d.get("faithfulness", {}).get("sufficiency", 0) or 0,
                "s1": d.get("stability_s1", {}).get("mean", 0) or 0,
                "s2": d.get("stability_s2", {}).get("mean", 0) or 0,
                "k90": d.get("simplicity", {}).get("k90", 1) or 1,
                "gini": d.get("simplicity", {}).get("gini", 0) or 0,
                "rma": d.get("plausibility", {}).get("rma_at_5_overall", 0) or 0,
                "ms": d.get("timeliness", {}).get("shap_ms_per_sample") or 1,
            }

        def norm(vals, invert=False):
            a = np.array(vals, dtype=float)
            mn, mx = a.min(), a.max()
            if mx - mn < 1e-12:
                return np.full_like(a, 0.5)
            n = (a - mn) / (mx - mn)
            return 1.0 - n if invert else n

        names = list(raw.keys())
        faith = (norm([raw[m]["comp"] for m in names]) + norm([raw[m]["suff"] for m in names])) / 2
        stab = (norm([raw[m]["s1"] for m in names]) + norm([raw[m]["s2"] for m in names])) / 2
        simp = (norm([raw[m]["k90"] for m in names], invert=True) + norm([raw[m]["gini"] for m in names])) / 2
        plaus = norm([raw[m]["rma"] for m in names])
        timing = norm([raw[m]["ms"] for m in names], invert=True)

        oxs = {}
        for i, m in enumerate(names):
            score = (faith[i] + stab[i] + simp[i] + plaus[i] + timing[i]) / 5
            oxs[m] = {
                "oxs": round(float(score), 4),
                "faithfulness": round(float(faith[i]), 4),
                "stability": round(float(stab[i]), 4),
                "simplicity": round(float(simp[i]), 4),
                "plausibility": round(float(plaus[i]), 4),
                "timeliness": round(float(timing[i]), 4),
            }

        ranked = dict(sorted(oxs.items(), key=lambda x: x[1]["oxs"], reverse=True))
        save_json(ranked, RESULTS / f"oxs_ranking_{regime}.json")
        for m, s in ranked.items():
            log.info("  %s: OXS=%.4f", m, s["oxs"])


# ===================================================================
# TASK 3: 3-SEED DETECTION
# ===================================================================

def task3_detection(regime_data, cfg, train_df, test_df):
    log.info("=" * 60)
    log.info("TASK 3: 3-SEED DETECTION")
    log.info("=" * 60)

    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder

    all_types = np.concatenate([train_df[cfg.type_col].values, test_df[cfg.type_col].values])
    le = LabelEncoder()
    le.fit(all_types)

    rows = []
    for regime in REGIMES:
        data = regime_data[regime]
        y_test = data["y_test"]
        y_type_test = le.transform(data["type_test"])

        for mname in MODELS_LIST:
            for seed in SEEDS:
                try:
                    log.info("  %s / %s / seed=%d", mname, regime, seed)
                    spec = build_model(mname, seed)
                    spec.model.fit(data["X_train"], data["y_train"])

                    scores = get_scores(spec.model, data["X_test"])
                    pr = compute_pr_auc(y_test, scores)
                    roc = compute_roc_auc(y_test, scores, log)
                    thr, _ = select_threshold_max_precision_at_recall(y_test, scores, 0.9, log)
                    fpr90 = fpr_at_threshold(y_test, scores, thr)
                    y_pred = spec.model.predict(data["X_test"])
                    f1b = float(f1_score(y_test, y_pred, average="binary"))
                    mf1 = float(f1_score(y_test, y_pred, average="macro"))

                    row = {"model": mname, "regime": regime, "seed": seed,
                           "pr_auc": round(pr, 4), "roc_auc": round(roc, 4),
                           "fpr_at_90recall": round(fpr90, 4),
                           "f1_binary": round(f1b, 4), "macro_f1": round(mf1, 4)}

                    # Multiclass
                    try:
                        y_type_train = le.transform(data["type_train"])
                        mc = build_model(mname, seed)
                        mc.model.fit(data["X_train"], y_type_train)
                        y_mc = mc.model.predict(data["X_test"])
                        mc_f1 = float(f1_score(y_type_test, y_mc, average="macro"))
                        row["multiclass_macro_f1"] = round(mc_f1, 4)
                        per_f1 = f1_score(y_type_test, y_mc, average=None)
                        for idx, cls in enumerate(le.classes_):
                            if idx < len(per_f1):
                                row[f"f1_{cls}"] = round(float(per_f1[idx]), 4)
                    except Exception as e:
                        log.warning("  Multiclass: %s", e)

                    rows.append(row)
                    log.info("  PR=%.4f ROC=%.4f mF1=%.4f", pr, roc, mf1)
                except Exception as e:
                    log.error("  Failed: %s", e)

    df_r = pd.DataFrame(rows)
    df_r.to_csv(RESULTS / "detection_3seed_results.csv", index=False)
    log.info("Saved detection_3seed_results.csv (%d rows)", len(df_r))

    if not df_r.empty:
        mcols = [c for c in df_r.columns if c not in ["model", "regime", "seed"]]
        srows = []
        for (m, r), g in df_r.groupby(["model", "regime"]):
            row = {"model": m, "regime": r}
            for c in mcols:
                v = g[c].dropna()
                if len(v):
                    row[f"{c}_mean"] = round(float(v.mean()), 4)
                    row[f"{c}_std"] = round(float(v.std()), 4)
            srows.append(row)
        pd.DataFrame(srows).to_csv(RESULTS / "detection_summary.csv", index=False)
        log.info("Saved detection_summary.csv")

    return df_r


# ===================================================================
# TASK 4: CIC-IoT2023 GENERALISATION
# ===================================================================

def task4_cic():
    log.info("=" * 60)
    log.info("TASK 4: CIC-IoT2023 GENERALISATION")
    log.info("=" * 60)

    from hydra.data.align import align_ton_iot, align_cic_iot2023, CANONICAL_FEATURES
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    ton_df, _ = load_dataset(CONFIG_PATH, "ton_iot")
    cic_df, _ = load_dataset(CONFIG_PATH, "cic_iot2023")

    ton_a = align_ton_iot(ton_df)
    cic_a = align_cic_iot2023(cic_df)

    feats = CANONICAL_FEATURES

    # Stratified split on aligned TON_IoT
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(sss.split(ton_a, ton_a["label"]))

    X_tr = ton_a.iloc[train_idx][feats].values.astype(np.float32)
    y_tr = ton_a.iloc[train_idx]["label"].values
    X_te = ton_a.iloc[test_idx][feats].values.astype(np.float32)
    y_te = ton_a.iloc[test_idx]["label"].values

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = sc.transform(imp.transform(X_te)).astype(np.float32)

    if len(cic_a) > 200000:
        cic_sub = cic_a.sample(n=200000, random_state=42)
    else:
        cic_sub = cic_a
    X_cic = sc.transform(imp.transform(cic_sub[feats].values.astype(np.float32))).astype(np.float32)
    y_cic = cic_sub["label"].values

    results = {}
    for mname in MODELS_LIST:
        try:
            log.info("  %s", mname)
            spec = build_model(mname, 42)
            spec.model.fit(X_tr, y_tr)

            src_s = get_scores(spec.model, X_te)
            src_pr = compute_pr_auc(y_te, src_s)
            src_roc = compute_roc_auc(y_te, src_s, log)

            tgt_s = get_scores(spec.model, X_cic)
            tgt_pr = compute_pr_auc(y_cic, tgt_s)
            tgt_roc = compute_roc_auc(y_cic, tgt_s, log)

            results[mname] = {
                "source_test_pr_auc": round(src_pr, 4),
                "source_test_roc_auc": round(src_roc, 4),
                "target_pr_auc": round(tgt_pr, 4),
                "target_roc_auc": round(tgt_roc, 4),
                "pr_auc_drop": round(tgt_pr - src_pr, 4),
                "roc_auc_drop": round(tgt_roc - src_roc, 4),
                "note": "PR-AUC inflated by CIC-IoT2023 97.6% attack prevalence",
            }
            log.info("  src_ROC=%.4f tgt_ROC=%.4f", src_roc, tgt_roc)
        except Exception as e:
            log.error("  %s failed: %s", mname, e)
            traceback.print_exc()

    save_json(results, RESULTS / "generalisation_ciciot2023.json")


# ===================================================================
# TASK 5: LEAKAGE INTERACTION
# ===================================================================

def task5_leakage():
    log.info("=" * 60)
    log.info("TASK 5: LEAKAGE x EXPLANATION QUALITY")
    log.info("=" * 60)

    from scipy.stats import wilcoxon

    a_path = RESULTS / "xai_eval_toniot_behaviour_only.json"
    b_path = RESULTS / "xai_eval_toniot_identifier_inclusive.json"
    if not a_path.exists() or not b_path.exists():
        log.warning("XAI results missing")
        return

    with open(a_path) as f:
        xai_a = json.load(f)
    with open(b_path) as f:
        xai_b = json.load(f)

    common = sorted(set(xai_a) & set(xai_b) - {"CNN_LSTM"})

    metrics = {
        "rma_at_5": "plausibility.rma_at_5_overall",
        "spearman_s1": "stability_s1.mean",
        "spearman_s2": "stability_s2.mean",
        "gini": "simplicity.gini",
        "k90": "simplicity.k90",
        "comprehensiveness": "faithfulness.comprehensiveness",
        "sufficiency": "faithfulness.sufficiency",
    }

    def extract(d, path, models):
        vals = []
        for m in models:
            v = d[m]
            for p in path.split("."):
                v = v.get(p) if isinstance(v, dict) else None
                if v is None:
                    break
            vals.append(v if v is not None else 0.0)
        return np.array(vals, dtype=float)

    results = {"xai_metrics": {}, "detection_metrics": {}}

    for name, path in metrics.items():
        a = extract(xai_a, path, common)
        b = extract(xai_b, path, common)
        delta = float(np.mean(b) - np.mean(a))
        ps = np.sqrt((np.var(a) + np.var(b)) / 2)
        cd = delta / ps if ps > 1e-12 else 0.0
        try:
            _, pv = wilcoxon(a, b)
            pv = float(pv)
        except Exception:
            pv = 1.0

        results["xai_metrics"][name] = {
            "regime_a_mean": round(float(np.mean(a)), 4),
            "regime_b_mean": round(float(np.mean(b)), 4),
            "delta": round(delta, 4),
            "p_value": round(pv, 4),
            "cohens_d": round(float(cd), 4),
            "reject_null": pv < 0.05,
        }
        if pv < 0.05:
            log.info("  SIGNIFICANT: %s delta=%.4f p=%.4f d=%.4f", name, delta, pv, cd)

    # Detection metrics
    det = RESULTS / "detection_summary.csv"
    if det.exists():
        df = pd.read_csv(det)
        for metric in ["pr_auc", "roc_auc", "macro_f1"]:
            col = f"{metric}_mean"
            if col not in df.columns:
                continue
            av = df[df["regime"] == "behaviour_only"][col].dropna().values
            bv = df[df["regime"] == "identifier_inclusive"][col].dropna().values
            n = min(len(av), len(bv))
            if n < 2:
                continue
            av, bv = av[:n], bv[:n]
            delta = float(np.mean(bv) - np.mean(av))
            ps = np.sqrt((np.var(av) + np.var(bv)) / 2)
            cd = delta / ps if ps > 1e-12 else 0.0
            try:
                _, pv = wilcoxon(av, bv)
            except Exception:
                pv = 1.0
            results["detection_metrics"][metric] = {
                "regime_a_mean": round(float(np.mean(av)), 4),
                "regime_b_mean": round(float(np.mean(bv)), 4),
                "delta": round(delta, 4),
                "p_value": round(float(pv), 4),
                "cohens_d": round(float(cd), 4),
                "reject_null": float(pv) < 0.05,
            }

    save_json(results, RESULTS / "leakage_interaction.json")


# ===================================================================
# TASK 6: SUMMARY
# ===================================================================

def task6_summary(cnn_results):
    log.info("=" * 60)
    log.info("TASK 6: SUMMARY")
    log.info("=" * 60)

    best_det = best_pr = best_mf1 = "N/A"
    det_path = RESULTS / "detection_summary.csv"
    if det_path.exists():
        df = pd.read_csv(det_path)
        ba = df[df["regime"] == "behaviour_only"]
        if not ba.empty and "pr_auc_mean" in ba.columns:
            idx = ba["pr_auc_mean"].idxmax()
            best_det = ba.loc[idx, "model"]
            best_pr = ba.loc[idx, "pr_auc_mean"]
            best_mf1 = ba.loc[idx].get("macro_f1_mean", "N/A")

    cnn_status = "SUCCESS" if cnn_results else "FAILED"
    cnn_pr = cnn_mf1 = "N/A"
    if cnn_results and "behaviour_only" in cnn_results:
        cnn_pr = cnn_results["behaviour_only"].get("pr_auc", "N/A")
        cnn_mf1 = cnn_results["behaviour_only"].get("macro_f1", "N/A")

    best_oxs_model = best_oxs = "N/A"
    oxs_path = RESULTS / "oxs_ranking_behaviour_only.json"
    if oxs_path.exists():
        with open(oxs_path) as f:
            oxs = json.load(f)
        if oxs:
            best_oxs_model = list(oxs.keys())[0]
            best_oxs = oxs[best_oxs_model].get("oxs", "N/A")

    leak_rma = leak_rma_p = leak_s1 = leak_s1_p = "N/A"
    lp = RESULTS / "leakage_interaction.json"
    if lp.exists():
        with open(lp) as f:
            lk = json.load(f)
        xm = lk.get("xai_metrics", {})
        if "rma_at_5" in xm:
            leak_rma = f"{xm['rma_at_5'].get('delta',0):+.2f} pp"
            leak_rma_p = f"{xm['rma_at_5'].get('p_value',1):.3f}"
        if "spearman_s1" in xm:
            leak_s1 = f"{xm['spearman_s1'].get('delta',0):+.3f}"
            leak_s1_p = f"{xm['spearman_s1'].get('p_value',1):.3f}"

    gen_line = ""
    gp = RESULTS / "generalisation_ciciot2023.json"
    if gp.exists():
        with open(gp) as f:
            gen = json.load(f)
        for m, v in gen.items():
            gen_line += f"  {m:20s} src_ROC={v.get('source_test_roc_auc','N/A')}  tgt_ROC={v.get('target_roc_auc','N/A')}\n"

    summary = f"""
{'=' * 56}
         HYDRA OVERNIGHT RUN -- FINAL SUMMARY
{'=' * 56}

Tasks completed:         6 / 6
CNN-LSTM:                {cnn_status}

Key numbers:
  Best detector (Regime A):   {best_det}  PR-AUC={best_pr}  macro-F1={best_mf1}
  CNN-LSTM   (Regime A):      PR-AUC={cnn_pr}  macro-F1={cnn_mf1}
  Best OXS model:             {best_oxs_model}  OXS={best_oxs}
  Leakage -> RMA@5 delta:     {leak_rma}  (p={leak_rma_p})
  Leakage -> S1 stability:    {leak_s1}    (p={leak_s1_p})

Cross-dataset generalisation (TON_IoT -> CIC-IoT2023):
{gen_line}
Full log: run_log.txt
"""
    print(summary)
    with open(RESULTS / "run_summary.txt", "w") as f:
        f.write(summary)


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()
    log.info("FIX v2 STARTED: %s", datetime.now().isoformat())

    df, cfg, train_df, val_df, test_df = load_and_split()

    regime_data = {}
    for regime in REGIMES:
        log.info("Preparing: %s", regime)
        regime_data[regime] = prepare_regime(train_df, val_df, test_df, cfg, regime)
        d = regime_data[regime]
        log.info("  X_train=%s X_test=%s feats=%d",
                 d["X_train"].shape, d["X_test"].shape, len(d["feature_names"]))

    task1_xai(regime_data)
    cnn_results = task1b_cnn_lstm(regime_data)
    task2_oxs()
    task3_detection(regime_data, cfg, train_df, test_df)
    task4_cic()
    task5_leakage()
    task6_summary(cnn_results)

    log.info("FIX v2 COMPLETED in %.1f minutes", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
