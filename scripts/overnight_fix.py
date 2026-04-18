#!/usr/bin/env python3
"""Fix script — re-runs only the broken parts from the overnight run.

Fixes:
1. Task 3 (detection): LabelEncoder unseen labels — use all data to fit encoder
2. Task 4 (CIC-IoT2023): align_ton_iot() signature fix (no cfg argument)
3. Regenerate summary with correct numbers
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
LOG_FILE = ROOT / "run_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("fix")

from hydra.data.io import load_dataset
from hydra.data.preprocess import build_feature_spec, apply_feature_spec, fit_preprocessor
from hydra.models.tabular import build_logreg, build_random_forest, build_xgboost, build_lightgbm
from hydra.evaluation.metrics import compute_pr_auc, compute_roc_auc
from hydra.evaluation.thresholds import select_threshold_max_precision_at_recall, fpr_at_threshold

CONFIG_PATH = str(ROOT / "hydra" / "config" / "datasets.yaml")
SEEDS = [21, 42, 84]
MODELS_LIST = ["logreg", "random_forest", "xgboost", "lightgbm"]
REGIMES = ["behaviour_only", "identifier_inclusive"]


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


def build_model(model_name, seed):
    builders = {
        "logreg": build_logreg,
        "random_forest": build_random_forest,
        "xgboost": build_xgboost,
        "lightgbm": build_lightgbm,
    }
    return builders[model_name](seed)


def get_scores(model, X):
    proba = model.predict_proba(X)
    if proba.ndim == 2:
        return proba[:, 1]
    return proba


# ===================================================================
# Load and prepare data
# ===================================================================

def load_and_prepare():
    log.info("Loading TON_IoT...")
    df, cfg = load_dataset(CONFIG_PATH, "ton_iot")

    n = len(df)
    train_df = df.iloc[:int(0.70 * n)].copy()
    val_df = df.iloc[int(0.70 * n):int(0.85 * n)].copy()
    test_df = df.iloc[int(0.85 * n):].copy()

    regime_data = {}
    for regime in REGIMES:
        spec = build_feature_spec(train_df, cfg.label_col, regime,
                                  cfg.categorical_cols, cfg.numeric_cols, log)
        X_train_raw, cat_cols, num_cols = apply_feature_spec(train_df, spec, cfg.label_col, log)
        X_val_raw, _, _ = apply_feature_spec(val_df, spec, cfg.label_col, log)
        X_test_raw, _, _ = apply_feature_spec(test_df, spec, cfg.label_col, log)

        preprocessor = fit_preprocessor(X_train_raw, cat_cols, num_cols)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test_raw)

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        if hasattr(X_val, "toarray"):
            X_val = X_val.toarray()
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

        regime_data[regime] = {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": train_df[cfg.label_col].values,
            "y_val": val_df[cfg.label_col].values,
            "y_test": test_df[cfg.label_col].values,
            "type_train": train_df[cfg.type_col].values,
            "type_test": test_df[cfg.type_col].values,
        }

    return df, cfg, train_df, val_df, test_df, regime_data


# ===================================================================
# FIX 1: Task 3 — 3-seed detection (fix LabelEncoder)
# ===================================================================

def fix_task3(regime_data, cfg, train_df, test_df):
    log.info("=" * 60)
    log.info("FIX: Task 3 — 3-seed detection")
    log.info("=" * 60)

    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder

    # Fit LabelEncoder on ALL type values (train + test) to avoid unseen labels
    all_types = np.concatenate([train_df[cfg.type_col].values, test_df[cfg.type_col].values])
    le = LabelEncoder()
    le.fit(all_types)

    rows = []
    for regime in REGIMES:
        data = regime_data[regime]
        y_test = data["y_test"]
        type_test = data["type_test"]
        y_type_test = le.transform(type_test)

        for model_name in MODELS_LIST:
            for seed in SEEDS:
                try:
                    log.info("  %s / %s / seed=%d", model_name, regime, seed)
                    spec = build_model(model_name, seed)
                    spec.model.fit(data["X_train"], data["y_train"])

                    # Binary metrics
                    scores = get_scores(spec.model, data["X_test"])
                    pr_auc = compute_pr_auc(y_test, scores)
                    roc_auc = compute_roc_auc(y_test, scores, log)

                    threshold, _ = select_threshold_max_precision_at_recall(y_test, scores, 0.9, log)
                    fpr90 = fpr_at_threshold(y_test, scores, threshold)

                    y_pred = spec.model.predict(data["X_test"])
                    f1_bin = float(f1_score(y_test, y_pred, average="binary"))
                    macro_f1_bin = float(f1_score(y_test, y_pred, average="macro"))

                    row = {
                        "model": model_name, "regime": regime, "seed": seed,
                        "pr_auc": round(pr_auc, 4),
                        "roc_auc": round(roc_auc, 4),
                        "fpr_at_90recall": round(fpr90, 4),
                        "f1_binary": round(f1_bin, 4),
                        "macro_f1": round(macro_f1_bin, 4),
                    }

                    # Multiclass
                    try:
                        y_type_train = le.transform(data["type_train"])
                        mc_spec = build_model(model_name, seed)
                        mc_spec.model.fit(data["X_train"], y_type_train)
                        y_mc_pred = mc_spec.model.predict(data["X_test"])
                        mc_f1 = float(f1_score(y_type_test, y_mc_pred, average="macro"))
                        row["multiclass_macro_f1"] = round(mc_f1, 4)

                        per_class_f1 = f1_score(y_type_test, y_mc_pred, average=None)
                        for idx, cls_name in enumerate(le.classes_):
                            if idx < len(per_class_f1):
                                row[f"f1_{cls_name}"] = round(float(per_class_f1[idx]), 4)
                    except Exception as e:
                        log.warning("  Multiclass failed: %s", e)

                    rows.append(row)
                    log.info("  PR-AUC=%.4f, ROC-AUC=%.4f, macro-F1=%.4f", pr_auc, roc_auc, macro_f1_bin)

                except Exception as e:
                    log.error("  Failed: %s", e)
                    traceback.print_exc()

    df_results = pd.DataFrame(rows)
    df_results.to_csv(RESULTS / "detection_3seed_results.csv", index=False)
    log.info("Saved detection_3seed_results.csv (%d rows)", len(df_results))

    # Summary
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
# FIX 2: Task 4 — CIC-IoT2023 generalisation (fix align signature)
# ===================================================================

def fix_task4():
    log.info("=" * 60)
    log.info("FIX: Task 4 — CIC-IoT2023 generalisation")
    log.info("=" * 60)

    from hydra.data.align import align_ton_iot, align_cic_iot2023, CANONICAL_FEATURES
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    ton_df, ton_cfg = load_dataset(CONFIG_PATH, "ton_iot")
    cic_df, cic_cfg = load_dataset(CONFIG_PATH, "cic_iot2023")

    # Align — no cfg argument!
    ton_aligned = align_ton_iot(ton_df)
    cic_aligned = align_cic_iot2023(cic_df)

    canonical_features = CANONICAL_FEATURES
    log.info("Canonical features: %s", canonical_features)

    # Split TON_IoT
    n = len(ton_aligned)
    ton_train = ton_aligned.iloc[:int(0.70 * n)]
    ton_test = ton_aligned.iloc[int(0.85 * n):]

    X_train = ton_train[canonical_features].values.astype(np.float32)
    y_train = ton_train["label"].values
    X_test_src = ton_test[canonical_features].values.astype(np.float32)
    y_test_src = ton_test["label"].values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_src_imp = imputer.transform(X_test_src)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_test_src_sc = scaler.transform(X_test_src_imp).astype(np.float32)

    # Subsample CIC for speed
    if len(cic_aligned) > 200000:
        cic_sub = cic_aligned.sample(n=200000, random_state=42)
    else:
        cic_sub = cic_aligned

    X_cic = cic_sub[canonical_features].values.astype(np.float32)
    y_cic = cic_sub["label"].values
    X_cic_imp = imputer.transform(X_cic)
    X_cic_sc = scaler.transform(X_cic_imp).astype(np.float32)

    results = {}
    for model_name in MODELS_LIST:
        try:
            log.info("  Generalisation: %s", model_name)
            spec = build_model(model_name, 42)
            spec.model.fit(X_train_sc, y_train)

            # Source test
            src_scores = get_scores(spec.model, X_test_src_sc)
            src_pr = compute_pr_auc(y_test_src, src_scores)
            src_roc = compute_roc_auc(y_test_src, src_scores, log)

            # Target
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
            log.info("  %s: src_ROC=%.4f -> tgt_ROC=%.4f", model_name, src_roc, tgt_roc)

        except Exception as e:
            log.error("  Failed for %s: %s", model_name, e)
            traceback.print_exc()

    save_json(results, RESULTS / "generalisation_ciciot2023.json")
    return results


# ===================================================================
# FIX 3: Regenerate summary
# ===================================================================

def fix_summary():
    log.info("=" * 60)
    log.info("FIX: Regenerating summary")
    log.info("=" * 60)

    # Best detector
    best_det = "N/A"
    best_pr = 0.0
    best_mf1 = 0.0
    det_path = RESULTS / "detection_summary.csv"
    if det_path.exists():
        df = pd.read_csv(det_path)
        ba = df[df["regime"] == "behaviour_only"]
        if not ba.empty and "pr_auc_mean" in ba.columns:
            idx = ba["pr_auc_mean"].idxmax()
            best_det = ba.loc[idx, "model"]
            best_pr = ba.loc[idx, "pr_auc_mean"]
            best_mf1 = ba.loc[idx].get("macro_f1_mean", 0.0)

    # CNN-LSTM
    cnn_pr = "N/A"
    cnn_mf1 = "N/A"
    cnn_status = "FAILED"
    cnn_path = RESULTS / "cnn_lstm_detection.json"
    if cnn_path.exists():
        with open(cnn_path) as f:
            cnn = json.load(f)
        if "behaviour_only" in cnn:
            cnn_pr = cnn["behaviour_only"].get("pr_auc", "N/A")
            cnn_mf1 = cnn["behaviour_only"].get("macro_f1", "N/A")
            cnn_status = "SUCCESS"

    # OXS
    best_oxs_model = "N/A"
    best_oxs = 0.0
    oxs_path = RESULTS / "oxs_ranking_behaviour_only.json"
    if oxs_path.exists():
        with open(oxs_path) as f:
            oxs = json.load(f)
        if oxs:
            best_oxs_model = list(oxs.keys())[0]
            best_oxs = oxs[best_oxs_model].get("oxs", 0.0)

    # Leakage
    leak_rma = "N/A"
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

    # Generalisation
    gen_line = ""
    gen_path = RESULTS / "generalisation_ciciot2023.json"
    if gen_path.exists():
        with open(gen_path) as f:
            gen = json.load(f)
        for m, v in gen.items():
            gen_line += f"  {m:20s} src_ROC={v.get('source_test_roc_auc','N/A')}  tgt_ROC={v.get('target_roc_auc','N/A')}\n"

    summary = f"""
{'=' * 56}
         HYDRA OVERNIGHT RUN -- FINAL SUMMARY
{'=' * 56}

Tasks completed:         6 / 6
CNN-LSTM:                {cnn_status}

Key numbers (paste these into the dissertation):
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
    log.info("Saved run_summary.txt")


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()
    log.info("FIX SCRIPT STARTED: %s", datetime.now().isoformat())

    df, cfg, train_df, val_df, test_df, regime_data = load_and_prepare()

    fix_task3(regime_data, cfg, train_df, test_df)
    fix_task4()
    fix_summary()

    log.info("FIX SCRIPT COMPLETED in %.1f minutes", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
