from __future__ import annotations

import os
import sys

if "LOKY_MAX_CPU_COUNT" not in os.environ:
    # Avoid joblib physical core detection warning on macOS by setting a cap.
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max((os.cpu_count() or 1) - 1, 1))
if "KMP_SHM_DISABLE" not in os.environ:
    # Avoid OpenMP shared memory errors on macOS.
    os.environ["KMP_SHM_DISABLE"] = "1"
if sys.platform == "darwin" and "HYDRA_DISABLE_LIGHTGBM" not in os.environ:
    # LightGBM OpenMP can crash on macOS without /dev/shm.
    os.environ["HYDRA_DISABLE_LIGHTGBM"] = "1"

import argparse
import hashlib
import json
import logging
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from hydra.data.io import config_to_dict, load_dataset
from hydra.data.preprocess import apply_feature_spec, build_feature_spec, fit_preprocessor
from hydra.data.split import (
    split_group_stratified_by_label,
    split_host,
    split_stratified,
    split_stratified_by_column,
    split_temporal,
)
from hydra.eval.metrics import (
    compute_brier,
    compute_ks_statistic,
    compute_log_loss,
    compute_pr_auc,
    compute_roc_auc,
    hash_rows_postprocess,
    majority_baseline_sanity,
    pr_auc_sanity_check,
    roc_auc_null_tolerance,
    PR_AUC_TOL,
    warn_on_constant_scores,
)
from hydra.eval.thresholds import (
    coverage_at_threshold,
    fpr_at_threshold,
    precision_recall_at_threshold,
    select_threshold_max_precision_at_recall,
)
from hydra.explain.tabular_explain import save_global_importance, save_local_explanations, save_type_shap
from hydra.models.baselines import baseline_majority_scores, baseline_threshold_scores
from hydra.models.tabular import build_lightgbm, build_logreg, build_random_forest, build_sklearn_gbdt, build_xgboost
from hydra.models.deep import build_cnn_lstm


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("hydra")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(run_dir / "logs.txt")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _hash_labels(y: pd.Series) -> str:
    arr = np.asarray(y, dtype=np.int8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _format_model_comparison_table(metrics_df: pd.DataFrame) -> str:
    display_map = {
        "model": "model",
        "backend": "backend",
        "pr_auc": "pr_auc",
        "pr_lift": "pr_lift",
        "roc_auc": "roc_auc",
        "precision_test_at_recall_0_90": "precision@0.90",
        "recall_test_at_recall_0_90": "recall@0.90",
        "f1_test_at_recall_0_90": "f1@0.90",
        "fpr_at_recall_0_90": "fpr@0.90",
        "coverage": "coverage",
    }
    available = [col for col in display_map if col in metrics_df.columns]
    if not available:
        return ""
    table_df = metrics_df[available].copy()
    if "pr_lift" in table_df.columns:
        table_df = table_df.sort_values("pr_lift", ascending=False)
    table_df = table_df.rename(columns={k: display_map[k] for k in available})

    def _fmt_cell(val):
        if isinstance(val, (int, float, np.floating)):
            if np.isnan(val):
                return "nan"
            return f"{float(val):.4f}"
        return "nan" if val is None else str(val)

    for col in table_df.columns:
        table_df[col] = table_df[col].map(_fmt_cell)
    return table_df.to_string(index=False)


def _format_type_comparison_table(metrics_df: pd.DataFrame) -> str:
    display_map = {
        "model": "model",
        "type_accuracy_overall": "type_acc",
        "attack_type_accuracy_e2e": "attack_acc",
        "attack_type_accuracy_detected": "attack_acc_det",
        "attack_type_f1_macro_detected": "attack_f1_macro",
        "attack_type_f1_weighted_detected": "attack_f1_weighted",
        "unknown_fraction_attack_detected": "unk_frac",
    }
    available = [col for col in display_map if col in metrics_df.columns]
    if len(available) <= 1:
        return ""
    table_df = metrics_df[available].copy()
    if "attack_type_accuracy_e2e" in table_df.columns:
        table_df = table_df.sort_values("attack_type_accuracy_e2e", ascending=False)
    table_df = table_df.rename(columns={k: display_map[k] for k in available})

    def _fmt_cell(val):
        if isinstance(val, (int, float, np.floating)):
            if np.isnan(val):
                return "nan"
            return f"{float(val):.4f}"
        return "nan" if val is None else str(val)

    for col in table_df.columns:
        table_df[col] = table_df[col].map(_fmt_cell)
    return table_df.to_string(index=False)


class FixedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=int)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.indices]


def _select_feature_indices(
    method: str,
    k: int | None,
    X_train,
    y_train,
    seed: int,
    logger,
) -> np.ndarray | None:
    if method == "none":
        return None
    if not k or k <= 0:
        logger.warning("feature_selection_k must be >0 when feature_selection is enabled; skipping.")
        return None
    n_features = X_train.shape[1]
    if k >= n_features:
        logger.warning("feature_selection_k >= n_features (%d); skipping.", n_features)
        return None

    if method == "mutual_info":
        scores = mutual_info_classif(X_train, y_train, random_state=seed)
        if np.all(np.isnan(scores)):
            logger.warning("Mutual information returned all NaN scores; skipping feature selection.")
            return None
        return np.argsort(scores)[::-1][:k]
    if method == "rfe":
        logger.warning("RFE can be slow on high-dimensional data; consider mutual_info for speed.")
        estimator = RandomForestClassifier(
            n_estimators=150,
            random_state=seed,
            n_jobs=-1,
        )
        rfe = RFE(estimator=estimator, n_features_to_select=k, step=0.1)
        rfe.fit(X_train, y_train)
        return np.flatnonzero(rfe.support_)
    if method == "model_importance":
        estimator = RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        )
        estimator.fit(X_train, y_train)
        scores = estimator.feature_importances_
        return np.argsort(scores)[::-1][:k]

    logger.warning("Unknown feature selection method '%s'; skipping.", method)
    return None


def _assert_binary_labels(name: str, y: pd.Series) -> None:
    values = set(pd.Series(y).dropna().unique().tolist())
    if not values.issubset({0, 1}):
        raise RuntimeError(f"{name} labels are not binary: {sorted(values)}")


def _label_stats(y: pd.Series) -> Dict[str, float]:
    n = int(len(y))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    prevalence = float(y.mean()) if n else 0.0
    return {"n": n, "n_pos": n_pos, "n_neg": n_neg, "prevalence": prevalence}

def _write_evaluation_meta(run_dir: Path, meta: Dict) -> None:
    with open(run_dir / "evaluation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """Returns True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False


def _dataset_fingerprint(path: str) -> str:
    """Stable fingerprint of the raw data file: SHA-256 of resolved path + size + mtime."""
    try:
        p = Path(path)
        if not p.exists():
            return "file_not_found"
        stat = p.stat()
        content = f"{p.resolve()}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(content.encode()).hexdigest()
    except Exception:
        return "unknown"


def _package_versions() -> Dict[str, str]:
    import sklearn
    import pandas
    import numpy

    versions = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import lightgbm  # noqa: F401

        versions["lightgbm"] = lightgbm.__version__
    except Exception:
        versions["lightgbm"] = "not_installed"
    try:
        import xgboost  # noqa: F401

        versions["xgboost"] = xgboost.__version__
    except Exception:
        versions["xgboost"] = "not_installed"
    try:
        import shap  # noqa: F401

        versions["shap"] = shap.__version__
    except Exception:
        versions["shap"] = "not_installed"
    return versions


def _load_defaults(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(args) -> Dict[str, object]:
    defaults = _load_defaults(args.defaults)
    seed = args.seed if args.seed is not None else defaults["seed"]
    split_assertions = bool(defaults.get("split_assertions", True))
    random.seed(seed)
    np.random.seed(seed)

    # Build run directory early for logging
    split_strategy = args.split_strategy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{timestamp}_{split_strategy}_{args.feature_regime}"
    run_dir = Path("runs") / args.dataset / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_dir)
    if os.environ.get("HYDRA_DISABLE_LIGHTGBM") == "1":
        logger.warning("HYDRA_DISABLE_LIGHTGBM=1; LightGBM will fall back to xgboost/sklearn.")

    df, cfg = load_dataset(args.datasets, args.dataset)
    n_rows_loaded = len(df)  # record before any subsampling

    if args.max_rows:
        df = df.sample(n=min(len(df), args.max_rows), random_state=seed)

    label_col = cfg.label_col
    y = df[label_col]
    group_col_used = None
    timestamp_col_used = None
    temporal_missing = False

    if split_strategy == "host":
        group_col = args.group_col or cfg.group_col
        if not group_col:
            raise ValueError("group_col must be provided for host split")
        group_col_used = group_col
        train_idx, val_idx, test_idx = split_host(
            df,
            y,
            group_col=group_col,
            test_size=defaults["split"]["test_size"],
            val_size=defaults["split"]["val_size"],
            seed=seed,
            logger=logger,
            split_assertions=split_assertions,
        )
    elif split_strategy == "temporal":
        timestamp_col = args.timestamp_col or cfg.timestamp_col
        if not timestamp_col:
            raise ValueError("timestamp_col must be provided for temporal split")
        if timestamp_col not in df.columns:
            temporal_missing = True
        timestamp_col_used = timestamp_col
        train_idx, val_idx, test_idx = split_temporal(
            df,
            timestamp_col,
            defaults["split"]["temporal"]["train_frac"],
            defaults["split"]["temporal"]["val_frac"],
            defaults["split"]["temporal"]["test_frac"],
            logger=logger,
            split_assertions=split_assertions,
        )
    elif split_strategy == "stratified":
        train_idx, val_idx, test_idx = split_stratified(
            df,
            y,
            test_size=defaults["split"]["test_size"],
            val_size=defaults["split"]["val_size"],
            seed=seed,
            logger=logger,
            split_assertions=split_assertions,
        )
    elif split_strategy == "type_stratified":
        type_col = args.type_col or cfg.type_col
        if not type_col:
            raise ValueError("type_col must be provided for type_stratified split")
        train_idx, val_idx, test_idx = split_stratified_by_column(
            df,
            type_col,
            test_size=defaults["split"]["test_size"],
            val_size=defaults["split"]["val_size"],
            seed=seed,
            logger=logger,
            split_assertions=split_assertions,
        )
    elif split_strategy == "group_type_stratified":
        group_col = args.group_col or cfg.group_col
        if not group_col:
            raise ValueError("group_col must be provided for group_type_stratified split")
        group_col_used = group_col
        type_col = args.type_col or cfg.type_col
        if not type_col:
            raise ValueError("type_col must be provided for group_type_stratified split")
        normal_type_value = (
            args.normal_type_value
            if args.normal_type_value is not None
            else (cfg.normal_type_value if cfg.normal_type_value is not None else "normal")
        )
        type_series = df[type_col]
        required_labels = set(type_series.dropna().unique().tolist())
        if normal_type_value in required_labels:
            required_labels.remove(normal_type_value)
        if not required_labels:
            raise RuntimeError("No attack types found for group_type_stratified split.")
        train_idx, val_idx, test_idx = split_group_stratified_by_label(
            df,
            y,
            group_col=group_col,
            label_col=type_col,
            test_size=defaults["split"]["test_size"],
            val_size=defaults["split"]["val_size"],
            seed=seed,
            logger=logger,
            required_labels=required_labels,
            split_assertions=split_assertions,
        )
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    logger.info("Run ID: %s", run_id)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Split strategy: %s", split_strategy)
    logger.info("Feature regime: %s", args.feature_regime)

    # Build feature spec from train only
    spec = build_feature_spec(
        df.iloc[train_idx],
        label_col=label_col,
        regime=args.feature_regime,
        categorical_cols=cfg.categorical_cols,
        numeric_cols=cfg.numeric_cols,
        logger=logger,
    )

    X_train_raw, cat_cols, num_cols = apply_feature_spec(df.iloc[train_idx], spec, label_col, logger)
    X_val_raw, _, _ = apply_feature_spec(df.iloc[val_idx], spec, label_col, logger)
    X_test_raw, _, _ = apply_feature_spec(df.iloc[test_idx], spec, label_col, logger)

    # Drop type_col from features — it is the stage-2 label and must not be an input.
    _type_col_to_drop = args.type_col or (cfg.type_col if hasattr(cfg, "type_col") else None)
    if _type_col_to_drop and _type_col_to_drop in X_train_raw.columns:
        logger.info("Dropping type_col '%s' from features to prevent label leakage.", _type_col_to_drop)
        X_train_raw = X_train_raw.drop(columns=[_type_col_to_drop])
        X_val_raw   = X_val_raw.drop(columns=[_type_col_to_drop], errors="ignore")
        X_test_raw  = X_test_raw.drop(columns=[_type_col_to_drop], errors="ignore")
        cat_cols = [c for c in cat_cols if c != _type_col_to_drop]
        num_cols = [c for c in num_cols if c != _type_col_to_drop]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]

    def _assert_two_classes(split_name: str, y_split: pd.Series) -> None:
        if y_split.nunique(dropna=True) < 2:
            raise RuntimeError(f"{split_name} split has <2 classes; aborting")

    _assert_two_classes("train", y_train)
    _assert_two_classes("val", y_val)
    _assert_two_classes("test", y_test)
    _assert_binary_labels("y_train", y_train)
    _assert_binary_labels("y_val", y_val)
    _assert_binary_labels("y_test", y_test)

    pr_auc_sanity_check(y_test)

    label_stats = {
        "train": _label_stats(y_train),
        "val": _label_stats(y_val),
        "test": _label_stats(y_test),
    }
    logger.info(
        "Prevalence: train=%.4f val=%.4f test=%.4f",
        label_stats["train"]["prevalence"],
        label_stats["val"]["prevalence"],
        label_stats["test"]["prevalence"],
    )
    logger.info(
        "Label counts: train=%d (pos=%d neg=%d) val=%d (pos=%d neg=%d) test=%d (pos=%d neg=%d)",
        label_stats["train"]["n"], label_stats["train"]["n_pos"], label_stats["train"]["n_neg"],
        label_stats["val"]["n"], label_stats["val"]["n_pos"], label_stats["val"]["n_neg"],
        label_stats["test"]["n"], label_stats["test"]["n_pos"], label_stats["test"]["n_neg"],
    )

    type_col = args.type_col or cfg.type_col
    normal_type_value = (
        args.normal_type_value
        if args.normal_type_value is not None
        else (cfg.normal_type_value if cfg.normal_type_value is not None else "normal")
    )
    two_stage_enabled = bool(type_col and type_col in df.columns)
    type_series = None
    if type_col and type_col not in df.columns:
        logger.warning("type_col '%s' not found in dataset; two-stage disabled.", type_col)
        two_stage_enabled = False
    if two_stage_enabled:
        type_series = df[type_col]

    # Identifier diagnostics: which categorical columns are included in features.
    categorical_cols = cfg.categorical_cols or []
    categorical_in_features = [c for c in categorical_cols if c in X_train_raw.columns]
    if categorical_in_features:
        logger.warning("Categorical columns included in features: %s", categorical_in_features)
    if group_col_used and group_col_used in X_train_raw.columns:
        logger.warning("group_col '%s' is present in features; host split may leak identifiers.", group_col_used)

    evaluation_meta = {
        "reproducibility": {
            "git_commit_hash": _git_commit_hash(),
            "git_dirty": _git_dirty(),
            "config_snapshot": config_to_dict(cfg),
            "dataset_fingerprint": _dataset_fingerprint(cfg.path),
            "n_rows_loaded": n_rows_loaded,
            "n_rows_used": len(df),
            "split_params": {
                "strategy": split_strategy,
                "group_col": group_col_used,
                "timestamp_col": timestamp_col_used,
                "test_size": defaults["split"]["test_size"],
                "val_size": defaults["split"]["val_size"],
                "feature_selection": args.feature_selection,
                "feature_selection_k": args.feature_selection_k,
                "type_unknown_threshold": float(args.type_unknown_threshold),
            },
            "seed": seed,
        },
        "split_label_counts": label_stats,
        "duplicate_leakage_audit": {
            "train_val_overlap_rate_definition": "by_val",
            "train_test_overlap_rate_definition": "by_test",
            "val_test_overlap_rate_definition": "by_test",
            "train_val_overlap_rate": None,
            "train_test_overlap_rate": None,
            "val_test_overlap_rate": None,
            "train_val_overlap_count": None,
            "train_test_overlap_count": None,
            "val_test_overlap_count": None,
            "n_train_rows": None,
            "n_val_rows": None,
            "n_test_rows": None,
            "duplicate_leakage_flag": False,
            "duplicate_leakage_threshold": float(args.duplicate_leakage_threshold),
            "fail_on_duplicate_leakage": bool(args.fail_on_duplicate_leakage),
        },
        "feature_audit": {
            "categorical_in_features": categorical_in_features,
            "group_col_in_features": bool(group_col_used and group_col_used in X_train_raw.columns),
            "identifier_like_features": [],
        },
        "temporal_proxy_note": "row_order_proxy" if (split_strategy == "temporal" and temporal_missing) else None,
    }
    if two_stage_enabled:
        evaluation_meta["attack_type_audit"] = {
            "type_col": type_col,
            "normal_type_value": normal_type_value,
            "train_type_counts": type_series.iloc[train_idx].value_counts(dropna=False).to_dict(),
            "val_type_counts": type_series.iloc[val_idx].value_counts(dropna=False).to_dict(),
            "test_type_counts": type_series.iloc[test_idx].value_counts(dropna=False).to_dict(),
        }
    _write_evaluation_meta(run_dir, evaluation_meta)

    preprocessor = fit_preprocessor(X_train_raw, cat_cols, num_cols)
    preprocessor.fit(X_train_raw)

    # Duplicate leakage check via row-hash overlap (post-preprocessing features).
    X_train_proc = preprocessor.transform(X_train_raw)
    X_val_proc = preprocessor.transform(X_val_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    train_hash = hash_rows_postprocess(X_train_proc)
    val_hash = hash_rows_postprocess(X_val_proc)
    test_hash = hash_rows_postprocess(X_test_proc)
    train_set = set(train_hash)
    val_set = set(val_hash)
    test_set = set(test_hash)
    overlap_train_val = len(train_set & val_set)
    overlap_train_test = len(train_set & test_set)
    overlap_val_test = len(val_set & test_set)
    train_val_rate_by_val = overlap_train_val / max(len(val_hash), 1)
    train_val_rate_by_train = overlap_train_val / max(len(train_hash), 1)
    train_test_rate_by_test = overlap_train_test / max(len(test_hash), 1)
    train_test_rate_by_train = overlap_train_test / max(len(train_hash), 1)
    val_test_rate_by_test = overlap_val_test / max(len(test_hash), 1)
    val_test_rate_by_val = overlap_val_test / max(len(val_hash), 1)
    duplicate_flag = train_test_rate_by_test > float(args.duplicate_leakage_threshold)
    evaluation_meta["duplicate_leakage_audit"].update(
        {
            # Legacy overlap rates follow the *_overlap_rate_definition fields above.
            "train_val_overlap_rate": train_val_rate_by_val,
            "train_test_overlap_rate": train_test_rate_by_test,
            "val_test_overlap_rate": val_test_rate_by_test,
            "train_val_overlap_rate_by_val": train_val_rate_by_val,
            "train_val_overlap_rate_by_train": train_val_rate_by_train,
            "train_test_overlap_rate_by_test": train_test_rate_by_test,
            "train_test_overlap_rate_by_train": train_test_rate_by_train,
            "val_test_overlap_rate_by_test": val_test_rate_by_test,
            "val_test_overlap_rate_by_val": val_test_rate_by_val,
            "train_val_overlap_count": overlap_train_val,
            "train_test_overlap_count": overlap_train_test,
            "val_test_overlap_count": overlap_val_test,
            "n_train_rows": len(train_hash),
            "n_val_rows": len(val_hash),
            "n_test_rows": len(test_hash),
            "duplicate_leakage_flag": duplicate_flag,
        }
    )
    _write_evaluation_meta(run_dir, evaluation_meta)
    logger.info(
        "Row-hash overlap counts: train_val=%d train_test=%d val_test=%d",
        overlap_train_val,
        overlap_train_test,
        overlap_val_test,
    )
    logger.info(
        "Row-hash train->test overlap: count=%d rate_by_test=%.6f rate_by_train=%.6f",
        overlap_train_test,
        train_test_rate_by_test,
        train_test_rate_by_train,
    )
    if duplicate_flag:
        logger.warning(
            "Duplicate leakage detected: train_test_overlap_rate_by_test=%.6f train_test_overlap_rate_by_train=%.6f (threshold=%.6f)",
            train_test_rate_by_test,
            train_test_rate_by_train,
            float(args.duplicate_leakage_threshold),
        )
        if args.fail_on_duplicate_leakage:
            raise RuntimeError(
                f"Duplicate leakage detected: rate={train_test_rate_by_test:.6f} threshold={float(args.duplicate_leakage_threshold):.6f}"
            )

    feature_selector = None
    selected_feature_names = None
    if args.feature_selection != "none":
        selected_idx = _select_feature_indices(
            args.feature_selection,
            args.feature_selection_k,
            X_train_proc,
            y_train,
            seed,
            logger,
        )
        if selected_idx is not None and selected_idx.size > 0:
            feature_selector = FixedFeatureSelector(selected_idx)
            X_train_proc = feature_selector.transform(X_train_proc)
            X_val_proc = feature_selector.transform(X_val_proc)
            X_test_proc = feature_selector.transform(X_test_proc)
            try:
                feature_names = preprocessor.get_feature_names_out()
                selected_feature_names = [str(feature_names[i]) for i in selected_idx]
            except Exception:
                selected_feature_names = None
            evaluation_meta["feature_selection"] = {
                "method": args.feature_selection,
                "k": int(args.feature_selection_k),
                "n_selected": int(len(selected_idx)),
                "selected_features": selected_feature_names,
            }
            _write_evaluation_meta(run_dir, evaluation_meta)

    type_train = type_test = None
    attack_train_mask = attack_test_mask = None
    attack_type_classes = None
    majority_attack_type = None
    if two_stage_enabled and type_series is not None:
        type_train = type_series.iloc[train_idx].to_numpy()
        type_test = type_series.iloc[test_idx].to_numpy()
        attack_train_mask = type_train != normal_type_value
        attack_test_mask = type_test != normal_type_value

        attack_types_train = pd.Series(type_train[attack_train_mask]).dropna().unique().tolist()
        attack_types_test = pd.Series(type_test[attack_test_mask]).dropna().unique().tolist()
        overlap_types = sorted(set(attack_types_train) & set(attack_types_test))

        if not overlap_types:
            logger.warning("No overlapping attack types between train/test; two-stage metrics disabled.")
            two_stage_enabled = False
        else:
            overlap_train_mask = attack_train_mask & np.isin(type_train, overlap_types)
            overlap_test_mask = attack_test_mask & np.isin(type_test, overlap_types)
            if not overlap_train_mask.any() or not overlap_test_mask.any():
                logger.warning("Attack type overlap empty after filtering; two-stage metrics disabled.")
                two_stage_enabled = False
            else:
                attack_train_mask = overlap_train_mask
                attack_test_mask = overlap_test_mask
                attack_type_classes = pd.Series(type_train[attack_train_mask]).dropna().unique().tolist()
                majority_attack_type = (
                    pd.Series(type_train[attack_train_mask]).value_counts(dropna=False).idxmax()
                )
                if "attack_type_audit" in evaluation_meta:
                    evaluation_meta["attack_type_audit"].update(
                        {
                            "train_attack_types": sorted(set(attack_types_train)),
                            "test_attack_types": sorted(set(attack_types_test)),
                            "overlap_attack_types": overlap_types,
                            "attack_type_eval_note": "two_stage_metrics_use_overlap_types",
                        }
                    )
                _write_evaluation_meta(run_dir, evaluation_meta)

    models = args.models or defaults["models"]
    if os.environ.get("HYDRA_DISABLE_LIGHTGBM") == "1" and "lightgbm" in models:
        logger.warning("HYDRA_DISABLE_LIGHTGBM=1; lightgbm slot will use sklearn_gbdt fallback")
    metrics_rows: List[Dict] = []

    def _two_stage_metrics(model_name: str, scores_test, threshold: float, type_explain_dir=None, feat_names=None) -> Dict[str, float | int | None]:
        if not two_stage_enabled:
            return {
                "type_accuracy_overall": None,
                "attack_type_accuracy_e2e": None,
                "attack_type_accuracy_detected": None,
                "attack_type_f1_macro_detected": None,
                "attack_type_f1_weighted_detected": None,
                "attack_detected_fraction": None,
                "attack_type_support": None,
                "unknown_fraction_attack_detected": None,
            }
        if attack_train_mask is None or type_test is None or attack_test_mask is None:
            return {
                "type_accuracy_overall": None,
                "attack_type_accuracy_e2e": None,
                "attack_type_accuracy_detected": None,
                "attack_type_f1_macro_detected": None,
                "attack_type_f1_weighted_detected": None,
                "attack_detected_fraction": None,
                "attack_type_support": None,
                "unknown_fraction_attack_detected": None,
            }

        pred_attack = np.asarray(scores_test) >= threshold
        type_pred_all = np.full(len(type_test), normal_type_value, dtype=object)

        pred_attack_idx = np.flatnonzero(pred_attack)
        if pred_attack_idx.size > 0:
            if not attack_type_classes:
                pred_types_attack = np.full(pred_attack_idx.size, normal_type_value, dtype=object)
            elif len(attack_type_classes) < 2 or model_name in {"baseline_majority", "baseline_threshold"}:
                pred_types_attack = np.full(pred_attack_idx.size, majority_attack_type, dtype=object)
            else:
                if model_name == "logreg":
                    spec_model = build_logreg(seed)
                elif model_name == "random_forest":
                    spec_model = build_random_forest(seed)
                elif model_name == "sklearn_gbdt":
                    spec_model = build_sklearn_gbdt(seed)
                elif model_name == "lightgbm":
                    spec_model = build_lightgbm(seed)
                elif model_name == "xgboost":
                    spec_model = build_xgboost(seed)
                elif model_name == "cnn_lstm":
                    spec_model = build_cnn_lstm(seed)
                else:
                    spec_model = None

                if spec_model is None:
                    pred_types_attack = np.full(pred_attack_idx.size, majority_attack_type, dtype=object)
                else:
                    y_train_attack = type_train[attack_train_mask]
                    label_encoder = LabelEncoder()
                    label_encoder.fit(y_train_attack)
                    if len(label_encoder.classes_) < 2:
                        pred_types_attack = np.full(pred_attack_idx.size, majority_attack_type, dtype=object)
                    else:
                        X_train_attack = X_train_proc[attack_train_mask]
                        X_pred_attack = X_test_proc[pred_attack_idx]
                        model = clone(spec_model.model)
                        model.fit(X_train_attack, label_encoder.transform(y_train_attack))
                        if args.type_unknown_threshold and hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X_pred_attack)
                            max_proba = np.max(proba, axis=1)
                            pred_encoded = np.argmax(proba, axis=1)
                            pred_types_attack = label_encoder.inverse_transform(pred_encoded)
                            pred_types_attack = np.asarray(pred_types_attack, dtype=object)
                            pred_types_attack[max_proba < args.type_unknown_threshold] = "unknown"
                        else:
                            pred_encoded = model.predict(X_pred_attack)
                            pred_types_attack = label_encoder.inverse_transform(pred_encoded)

                        if type_explain_dir is not None and feat_names is not None:
                            save_type_shap(
                                model,
                                X_pred_attack,
                                type_test[pred_attack_idx],
                                label_encoder,
                                list(feat_names),
                                type_explain_dir,
                                logger,
                            )
            type_pred_all[pred_attack_idx] = pred_types_attack

        type_accuracy_overall = accuracy_score(type_test, type_pred_all)

        attack_support = int(attack_test_mask.sum())
        if attack_support == 0:
            return {
                "type_accuracy_overall": type_accuracy_overall,
                "attack_type_accuracy_e2e": None,
                "attack_type_accuracy_detected": None,
                "attack_type_f1_macro_detected": None,
                "attack_type_f1_weighted_detected": None,
                "attack_detected_fraction": None,
                "attack_type_support": 0,
            }

        attack_true = type_test[attack_test_mask]
        attack_pred_e2e = type_pred_all[attack_test_mask]
        attack_type_accuracy_e2e = accuracy_score(attack_true, attack_pred_e2e)

        detected_mask = pred_attack & attack_test_mask
        attack_detected_fraction = float(detected_mask.sum() / max(attack_support, 1))
        if detected_mask.any():
            attack_true_detected = type_test[detected_mask]
            attack_pred_detected = type_pred_all[detected_mask]
            attack_type_accuracy_detected = accuracy_score(attack_true_detected, attack_pred_detected)
            attack_type_f1_macro_detected = f1_score(
                attack_true_detected,
                attack_pred_detected,
                average="macro",
                zero_division=0,
            )
            attack_type_f1_weighted_detected = f1_score(
                attack_true_detected,
                attack_pred_detected,
                average="weighted",
                zero_division=0,
            )
            unknown_fraction_attack_detected = float(
                np.mean(np.asarray(attack_pred_detected) == "unknown")
            )
            report = classification_report(
                attack_true_detected,
                attack_pred_detected,
                output_dict=True,
                zero_division=0,
            )
            per_class_metrics: dict = {}
            for cls, cls_m in report.items():
                if cls in ("accuracy", "macro avg", "weighted avg"):
                    continue
                safe = cls.replace(" ", "_").replace("/", "_")
                per_class_metrics[f"f1_{safe}_detected"]        = float(cls_m["f1-score"])
                per_class_metrics[f"precision_{safe}_detected"] = float(cls_m["precision"])
                per_class_metrics[f"recall_{safe}_detected"]    = float(cls_m["recall"])
                per_class_metrics[f"support_{safe}_detected"]   = int(cls_m["support"])
        else:
            attack_type_accuracy_detected = None
            attack_type_f1_macro_detected = None
            attack_type_f1_weighted_detected = None
            unknown_fraction_attack_detected = None
            per_class_metrics = {}

        return {
            "type_accuracy_overall": type_accuracy_overall,
            "attack_type_accuracy_e2e": attack_type_accuracy_e2e,
            "attack_type_accuracy_detected": attack_type_accuracy_detected,
            "attack_type_f1_macro_detected": attack_type_f1_macro_detected,
            "attack_type_f1_weighted_detected": attack_type_f1_weighted_detected,
            "attack_detected_fraction": attack_detected_fraction,
            "attack_type_support": attack_support,
            "unknown_fraction_attack_detected": unknown_fraction_attack_detected,
            **per_class_metrics,
        }

    def evaluate_model(
        model_name: str,
        scores_val,
        scores_test,
        backend: str = "",
        y_val_override: pd.Series | None = None,
        y_test_override: pd.Series | None = None,
        y_test_hash: str | None = None,
    ):
        y_val_used = y_val_override if y_val_override is not None else y_val
        y_test_used = y_test_override if y_test_override is not None else y_test
        if y_test_hash is not None and _hash_labels(y_test_used) != y_test_hash:
            raise RuntimeError("Permutation probe y_true mismatch: hash check failed")
        prevalence = float(y_test_used.mean()) if len(y_test_used) else 0.0
        pr_auc = compute_pr_auc(y_test_used, scores_test)
        roc_auc = compute_roc_auc(y_test_used, scores_test, logger)
        warn_on_constant_scores(scores_test, pr_auc, prevalence, logger)

        threshold, recall_target_met = select_threshold_max_precision_at_recall(
            y_val_used, scores_val, 0.90, logger
        )
        precision_val, recall_val = precision_recall_at_threshold(y_val_used, scores_val, threshold)
        precision_test, recall_test = precision_recall_at_threshold(y_test_used, scores_test, threshold)
        f1_test = (2 * precision_test * recall_test) / max(precision_test + recall_test, 1e-12)
        fpr = fpr_at_threshold(y_test_used, scores_test, threshold)
        coverage = coverage_at_threshold(scores_test, threshold)

        if model_name == "baseline_majority":
            majority_baseline_sanity(pr_auc, prevalence, logger)

        return {
            "model": model_name,
            "backend": backend,
            "model_backend_used": backend,
            "pr_auc": pr_auc,
            "pr_lift": pr_auc - prevalence,
            "roc_auc": roc_auc,
            "fpr_at_recall_0_90": fpr,
            "threshold_at_recall_0_90": threshold,
            "recall_target_met": recall_target_met,
            "precision_val_at_recall_0_90": precision_val,
            "recall_val_at_recall_0_90": recall_val,
            "precision_test_at_recall_0_90": precision_test,
            "recall_test_at_recall_0_90": recall_test,
            "f1_test_at_recall_0_90": f1_test,
            "coverage": coverage,
        }

    for model_name in models:
        logger.info("Training model: %s", model_name)

        if model_name == "baseline_majority":
            scores_val = baseline_majority_scores(y_train, len(y_val))
            scores_test = baseline_majority_scores(y_train, len(y_test))
            row = evaluate_model(model_name, scores_val, scores_test)
            row.update(_two_stage_metrics(model_name, scores_test, row["threshold_at_recall_0_90"]))
            metrics_rows.append(row)
            continue

        if model_name == "baseline_threshold":
            colmap = {
                "duration_col": cfg.duration_col,
                "src_bytes_col": cfg.src_bytes_col,
                "dst_bytes_col": cfg.dst_bytes_col,
                "src_pkts_col": cfg.src_pkts_col,
                "dst_pkts_col": cfg.dst_pkts_col,
            }
            scores_val = baseline_threshold_scores(df.iloc[val_idx], colmap, logger)
            scores_test = baseline_threshold_scores(df.iloc[test_idx], colmap, logger)
            row = evaluate_model(model_name, scores_val, scores_test)
            row.update(_two_stage_metrics(model_name, scores_test, row["threshold_at_recall_0_90"]))
            metrics_rows.append(row)
            continue

        if model_name == "logreg":
            spec_model = build_logreg(seed)
        elif model_name == "random_forest":
            spec_model = build_random_forest(seed)
        elif model_name == "sklearn_gbdt":
            spec_model = build_sklearn_gbdt(seed)
        elif model_name == "lightgbm":
            spec_model = build_lightgbm(seed)
        elif model_name == "xgboost":
            spec_model = build_xgboost(seed)
        elif model_name == "cnn_lstm":
            spec_model = build_cnn_lstm(seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        steps = [("prep", clone(preprocessor))]
        if feature_selector is not None:
            steps.append(("select", feature_selector))
        steps.append(("model", spec_model.model))
        pipeline = Pipeline(steps=steps)
        pipeline.fit(X_train_raw, y_train)

        scores_val = pipeline.predict_proba(X_val_raw)[:, 1]
        scores_test = pipeline.predict_proba(X_test_raw)[:, 1]

        row = evaluate_model(spec_model.name, scores_val, scores_test, backend=spec_model.backend)

        _feat_names = None
        _type_explain_dir = None
        if spec_model.name in {"random_forest", "lightgbm", "xgboost", "sklearn_gbdt", "cnn_lstm"}:
            if selected_feature_names is not None:
                _feat_names = list(selected_feature_names)
            else:
                _feat_names = list(pipeline.named_steps["prep"].get_feature_names_out())
            _type_explain_dir = run_dir / "explain" / spec_model.name / "type_classifier"

        row.update(_two_stage_metrics(model_name, scores_test, row["threshold_at_recall_0_90"], type_explain_dir=_type_explain_dir, feat_names=_feat_names))
        metrics_rows.append(row)

        if spec_model.name in {"random_forest", "lightgbm", "xgboost", "sklearn_gbdt", "cnn_lstm"}:
            explain_dir = run_dir / "explain" / spec_model.name
            explain_dir.mkdir(parents=True, exist_ok=True)

            if _feat_names is not None:
                feature_names = _feat_names
            elif selected_feature_names is not None:
                feature_names = selected_feature_names
            else:
                feature_names = pipeline.named_steps["prep"].get_feature_names_out()
            save_global_importance(
                pipeline,
                X_val_raw,
                y_val,
                feature_names,
                explain_dir / "global_importance.csv",
                logger,
            )
            save_local_explanations(
                pipeline,
                X_val_raw,
                feature_names,
                explain_dir / "local_explanations",
                defaults["explain"]["local_samples"],
                defaults["explain"]["random_state"],
                logger,
            )

    # Label permutation leakage probe
    if args.label_permutation_probe:
        logger.info("Running label permutation probe")
        perm_spec = build_lightgbm(seed)
        if os.environ.get("HYDRA_DISABLE_LIGHTGBM") == "1":
            perm_spec = build_logreg(seed)
        probe = {
            "model": "lightgbm_label_shuffled",
            "backend": perm_spec.backend,
            "seed": seed,
            "permutation_repeats": int(args.permutation_repeats),
            "positive_label": cfg.positive_label,
            "status": "running",
        }
        try:
            _assert_binary_labels("y_train", y_train)
            _assert_binary_labels("y_val", y_val)
            _assert_binary_labels("y_test", y_test)

            hash_y_train = _hash_labels(y_train)
            hash_y_val = _hash_labels(y_val)
            hash_y_test = _hash_labels(y_test)
            probe.update(
                {
                    "hash_y_train": hash_y_train,
                    "hash_y_val": hash_y_val,
                    "hash_y_test": hash_y_test,
                }
            )

            n_test = len(y_test)
            n_pos = int((y_test == 1).sum())
            n_neg = int((y_test == 0).sum())
            if min(n_pos, n_neg) == 0:
                raise RuntimeError(
                    f"Permutation ROC-AUC undefined: single class in test set "
                    f"(n_test={n_test} n_pos={n_pos} n_neg={n_neg})"
                )
            roc_tol = roc_auc_null_tolerance(y_test)
            pr_tol = PR_AUC_TOL
            ks_tol = 0.25 if min(n_pos, n_neg) < 1000 else 0.20
            n_perm = int(args.permutation_repeats)

            perm_rows = []
            pr_aucs = []
            roc_aucs = []
            briers = []
            log_losses = []
            ks_stats = []
            hash_y_train_perm = []
            hash_y_test_perm = []
            pr_auc_constant = []
            pr_auc_noise = []
            roc_auc_noise = []
            prevalence_train_perm_list = []
            prevalence_test_perm_list = []

            for i in range(n_perm):
                rng = np.random.default_rng(seed + i)
                y_train_perm = pd.Series(rng.permutation(y_train.values), index=y_train.index)
                y_val_perm = pd.Series(rng.permutation(y_val.values), index=y_val.index)
                y_test_perm = pd.Series(rng.permutation(y_test.values), index=y_test.index)

                if np.array_equal(y_train_perm.values, y_train.values):
                    raise RuntimeError(
                        f"Permutation probe produced identical y_train (seed={seed + i}, n_train={len(y_train)})"
                    )
                if np.array_equal(y_test_perm.values, y_test.values):
                    raise RuntimeError(
                        f"Permutation probe produced identical y_test (seed={seed + i}, n_test={len(y_test)})"
                    )

                _assert_binary_labels("y_train_perm", y_train_perm)
                _assert_binary_labels("y_val_perm", y_val_perm)
                _assert_binary_labels("y_test_perm", y_test_perm)

                hash_y_train_perm.append(_hash_labels(y_train_perm))
                hash_y_test_perm.append(_hash_labels(y_test_perm))

                prevalence_train_perm = float(y_train_perm.mean()) if len(y_train_perm) else 0.0
                prevalence_test_perm = float(y_test_perm.mean()) if len(y_test_perm) else 0.0
                prevalence_train_perm_list.append(prevalence_train_perm)
                prevalence_test_perm_list.append(prevalence_test_perm)

                perm_pipeline = Pipeline(steps=[("prep", clone(preprocessor)), ("model", perm_spec.model)])
                perm_pipeline.fit(X_train_raw, y_train_perm)
                perm_scores_val = perm_pipeline.predict_proba(X_val_raw)[:, 1]
                perm_scores_test = perm_pipeline.predict_proba(X_test_raw)[:, 1]

                pr_aucs.append(compute_pr_auc(y_test_perm, perm_scores_test))
                roc_auc = compute_roc_auc(y_test_perm, perm_scores_test, logger)
                if np.isnan(roc_auc):
                    logger.warning("Permutation ROC-AUC undefined due to constant scores; treating as 0.5")
                    roc_auc = 0.5
                roc_aucs.append(roc_auc)
                briers.append(compute_brier(y_test_perm, perm_scores_test))
                log_losses.append(compute_log_loss(y_test_perm, perm_scores_test))
                ks_stats.append(compute_ks_statistic(y_test_perm, perm_scores_test))

            perm_rows.append(
                evaluate_model(
                    "lightgbm_label_shuffled",
                    perm_scores_val,
                    perm_scores_test,
                    backend=perm_spec.backend,
                    y_val_override=y_val_perm,
                    y_test_override=y_test_perm,
                    y_test_hash=hash_y_test_perm[-1],
                )
            )

            # Constant-score control
            const_scores = np.full(len(y_test_perm), prevalence_train_perm, dtype=float)
            pr_const = compute_pr_auc(y_test_perm, const_scores)
            pr_auc_constant.append(pr_const)
            pr_const_abs_dev = abs(pr_const - prevalence_test_perm)
            if pr_const_abs_dev > pr_tol:
                raise RuntimeError(
                    "Permutation probe constant-score control failed: "
                    f"pr_auc={pr_const:.4f} prevalence_test={prevalence_test_perm:.4f}"
                )

            # Noise-feature control
            rng_noise = np.random.default_rng(seed + 1000 + i)
            X_train_noise = rng_noise.normal(size=(len(X_train_raw), X_train_raw.shape[1]))
            X_test_noise = rng_noise.normal(size=(len(X_test_raw), X_test_raw.shape[1]))
            noise_model = clone(perm_spec.model)
            noise_model.fit(X_train_noise, y_train_perm)
            noise_scores_test = noise_model.predict_proba(X_test_noise)[:, 1]
            pr_noise = compute_pr_auc(y_test_perm, noise_scores_test)
            roc_noise = compute_roc_auc(y_test_perm, noise_scores_test, logger)
            pr_auc_noise.append(pr_noise)
            roc_auc_noise.append(roc_noise)
            if abs(pr_noise - prevalence_test_perm) > pr_tol:
                raise RuntimeError(
                    "Permutation probe noise-feature control failed: "
                    f"pr_auc={pr_noise:.4f} prevalence_test={prevalence_test_perm:.4f}"
                )

            pr_auc_mean = float(np.mean(pr_aucs))
            pr_auc_std = float(np.std(pr_aucs, ddof=0))
            roc_auc_mean = float(np.mean(roc_aucs))
            roc_auc_std = float(np.std(roc_aucs, ddof=0))
            ks_mean = float(np.nanmean(ks_stats))

            prevalence = float(y_test.mean()) if len(y_test) else 0.0
            pr_abs_dev = abs(pr_auc_mean - prevalence)
            roc_abs_dev = abs(roc_auc_mean - 0.5)

            if pr_abs_dev > pr_tol:
                raise RuntimeError(
                    "Leakage suspected / eval bug: permuted PR-AUC "
                    f"{pr_auc_mean:.4f} vs prevalence {prevalence:.4f} (abs dev {pr_abs_dev:.4f})"
                )
            if roc_abs_dev > roc_tol:
                raise RuntimeError(
                    "Leakage suspected / eval bug: permuted ROC-AUC "
                    f"{roc_auc_mean:.4f} (abs dev {roc_abs_dev:.4f} > tol {roc_tol:.4f})"
                )
            if ks_mean > ks_tol:
                raise RuntimeError(
                    "Leakage suspected / eval bug: permuted KS statistic "
                    f"{ks_mean:.4f} (expected ~0.0)"
                )

            # Average permutation metrics for summary.
            row = {
                k: float(np.nanmean([r[k] for r in perm_rows])) if isinstance(perm_rows[0][k], (int, float)) else perm_rows[0][k]
                for k in perm_rows[0]
            }
            metrics_rows.append(row)

            probe.update(
                {
                    "status": "ok",
                    "pr_auc_mean": pr_auc_mean,
                    "pr_auc_std": pr_auc_std,
                    "pr_auc_permutation": pr_aucs,
                    "roc_auc_mean": roc_auc_mean,
                    "roc_auc_std": roc_auc_std,
                    "roc_auc_permutation": roc_aucs,
                    "brier": float(np.mean(briers)),
                    "brier_permutation": briers,
                    "log_loss": float(np.mean(log_losses)),
                    "log_loss_permutation": log_losses,
                    "ks": ks_mean,
                    "ks_permutation": ks_stats,
                    "n_test": n_test,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "roc_auc_abs_dev_mean": roc_abs_dev,
                    "pr_auc_abs_dev_mean": pr_abs_dev,
                    "tol_used": roc_tol,
                    "pr_tol_used": pr_tol,
                    "prevalence_train_perm": float(np.mean(prevalence_train_perm_list)) if prevalence_train_perm_list else None,
                    "prevalence_test_perm": float(np.mean(prevalence_test_perm_list)) if prevalence_test_perm_list else None,
                    "pr_auc_constant": float(np.mean(pr_auc_constant)) if pr_auc_constant else None,
                    "pr_auc_noise": float(np.mean(pr_auc_noise)) if pr_auc_noise else None,
                    "roc_auc_noise": float(np.mean(roc_auc_noise)) if roc_auc_noise else None,
                    "hash_y_train_perm": hash_y_train_perm,
                    "hash_y_test_perm": hash_y_test_perm,
                }
            )
        except Exception as exc:
            probe["status"] = "failed"
            probe["error_message"] = str(exc)
            with open(run_dir / "permutation_probe.json", "w", encoding="utf-8") as f:
                json.dump(probe, f, indent=2)
            raise
        with open(run_dir / "permutation_probe.json", "w", encoding="utf-8") as f:
            json.dump(probe, f, indent=2)

    # Save metrics summary
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "metrics_summary.csv", index=False)
    table = _format_model_comparison_table(metrics_df)
    if table:
        logger.info("Model comparison:\n%s", table)
    type_table = _format_type_comparison_table(metrics_df)
    if type_table:
        logger.info("Two-stage type comparison:\n%s", type_table)

    paper_missing_cols = spec.missing_required or []

    notes = []
    if split_strategy == "stratified":
        notes.append("stratified split is naive/historical and not deployment-realistic")
    if args.feature_regime == "identifier_inclusive":
        notes.append("identifier_inclusive is an upper bound and not deployable")
    if split_strategy == "temporal" and temporal_missing:
        notes.append("temporal split used row order proxy (timestamp_col missing)")

    # Save run config
    run_config = {
        "dataset": config_to_dict(cfg),
        "feature_regime": args.feature_regime,
        "split_strategy": split_strategy,
        "group_col": group_col_used,
        "timestamp_col": timestamp_col_used,
        "type_col": type_col if two_stage_enabled else None,
        "normal_type_value": normal_type_value if two_stage_enabled else None,
        "seed": seed,
        "run_id": run_id,
        "timestamp": timestamp,
        "commit_hash": _git_commit_hash(),
        "package_versions": _package_versions(),
        "models": models,
        "feature_selection": args.feature_selection,
        "feature_selection_k": args.feature_selection_k,
        "type_unknown_threshold": float(args.type_unknown_threshold),
        "label_permutation_probe": bool(args.label_permutation_probe),
        "permutation_repeats": int(args.permutation_repeats),
        "notes": " | ".join(notes),
        "paper_missing_cols": paper_missing_cols,
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dataset": cfg.name,
        "feature_regime": args.feature_regime,
        "split_strategy": split_strategy,
        "group_col": group_col_used,
        "timestamp_col": timestamp_col_used,
        "metrics_df": metrics_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Run HYDRA tabular IDS experiment")
    parser.add_argument("--dataset", required=True, help="Dataset name in datasets.yaml")
    parser.add_argument(
        "--feature_regime",
        required=True,
        choices=["behaviour_only", "operational", "identifier_inclusive", "paper_5feat", "core_flow"],
    )
    parser.add_argument(
        "--split_strategy",
        required=True,
        choices=["host", "temporal", "stratified", "type_stratified", "group_type_stratified"],
    )
    parser.add_argument("--group_col", default=None)
    parser.add_argument("--timestamp_col", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--label_permutation_probe", action="store_true")
    parser.add_argument("--permutation_repeats", type=int, default=3)
    parser.add_argument("--duplicate_leakage_threshold", type=float, default=0.001)
    parser.add_argument("--fail_on_duplicate_leakage", action="store_true")
    parser.add_argument("--type_col", default=None, help="Attack type column for two-stage classification")
    parser.add_argument("--normal_type_value", default=None, help="Value representing benign in type_col")
    parser.add_argument("--datasets", default="hydra/config/datasets.yaml")
    parser.add_argument("--defaults", default="hydra/config/defaults.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--feature_selection",
        default="none",
        choices=["none", "mutual_info", "rfe", "model_importance"],
        help="Optional feature selection method applied on training data only.",
    )
    parser.add_argument("--feature_selection_k", type=int, default=None)
    parser.add_argument(
        "--type_unknown_threshold",
        type=float,
        default=0.0,
        help="If >0, stage-2 predicts 'unknown' when max attack-type prob < threshold.",
    )
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
