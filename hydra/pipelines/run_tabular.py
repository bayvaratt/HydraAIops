from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from hydra.data.io import config_to_dict, load_dataset
from hydra.data.preprocess import apply_feature_spec, build_feature_spec, fit_preprocessor
from hydra.data.split import split_host, split_stratified, split_temporal
from hydra.eval.metrics import compute_pr_auc, compute_roc_auc, majority_baseline_sanity, warn_on_constant_scores
from hydra.eval.thresholds import coverage_at_threshold, fpr_at_threshold, select_threshold_at_recall
from hydra.explain.tabular_explain import save_global_importance, save_local_explanations
from hydra.models.baselines import baseline_majority_scores, baseline_threshold_scores
from hydra.models.tabular import build_lightgbm, build_logreg, build_random_forest


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


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
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


def run(args) -> None:
    defaults = _load_defaults(args.defaults)
    seed = args.seed if args.seed is not None else defaults["seed"]
    random.seed(seed)
    np.random.seed(seed)

    # Build run directory early for logging
    split_strategy = args.split_strategy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{timestamp}_{split_strategy}_{args.feature_regime}"
    run_dir = Path("runs") / args.dataset / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_dir)

    df, cfg = load_dataset(args.datasets, args.dataset)

    if args.max_rows:
        df = df.sample(n=min(len(df), args.max_rows), random_state=seed)

    label_col = cfg.label_col
    y = df[label_col]
    if split_strategy == "host":
        group_col = args.group_col or cfg.group_col
        if not group_col:
            raise ValueError("group_col must be provided for host split")
        train_idx, val_idx, test_idx = split_host(
            df,
            y,
            group_col=group_col,
            test_size=defaults["split"]["test_size"],
            val_size=defaults["split"]["val_size"],
            seed=seed,
            logger=logger,
        )
    elif split_strategy == "temporal":
        timestamp_col = args.timestamp_col or cfg.timestamp_col
        if not timestamp_col:
            raise ValueError("timestamp_col must be provided for temporal split")
        train_idx, val_idx, test_idx = split_temporal(
            df,
            timestamp_col,
            defaults["split"]["temporal"]["train_frac"],
            defaults["split"]["temporal"]["val_frac"],
            defaults["split"]["temporal"]["test_frac"],
            logger=logger,
        )
    elif split_strategy == "stratified":
        train_idx, val_idx, test_idx = split_stratified(
            df,
            y,
            test_size=defaults["split"]["test_size"],
            val_size=defaults["split"]["val_size"],
            seed=seed,
            logger=logger,
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

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]

    preprocessor = fit_preprocessor(X_train_raw, cat_cols, num_cols)

    models = args.models or defaults["models"]
    metrics_rows: List[Dict] = []

    def evaluate_model(model_name: str, scores_val, scores_test, backend: str = ""):
        prevalence = float(y_train.mean()) if len(y_train) else 0.0
        pr_auc = compute_pr_auc(y_test, scores_test)
        roc_auc = compute_roc_auc(y_test, scores_test, logger)
        warn_on_constant_scores(scores_test, pr_auc, prevalence, logger)

        threshold = select_threshold_at_recall(y_val, scores_val, 0.90, logger)
        fpr = fpr_at_threshold(y_test, scores_test, threshold)
        coverage = coverage_at_threshold(scores_test, threshold)

        if model_name == "baseline_majority":
            majority_baseline_sanity(pr_auc, prevalence, logger)

        return {
            "model": model_name,
            "backend": backend,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "fpr_at_recall_0_90": fpr,
            "threshold": threshold,
            "coverage": coverage,
        }

    for model_name in models:
        logger.info("Training model: %s", model_name)

        if model_name == "baseline_majority":
            scores_val = baseline_majority_scores(y_train, len(y_val))
            scores_test = baseline_majority_scores(y_train, len(y_test))
            row = evaluate_model(model_name, scores_val, scores_test)
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
            metrics_rows.append(row)
            continue

        if model_name == "logreg":
            spec_model = build_logreg(seed)
        elif model_name == "random_forest":
            spec_model = build_random_forest(seed)
        elif model_name == "lightgbm":
            spec_model = build_lightgbm(seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipeline = Pipeline(steps=[("prep", clone(preprocessor)), ("model", spec_model.model)])
        pipeline.fit(X_train_raw, y_train)

        scores_val = pipeline.predict_proba(X_val_raw)[:, 1]
        scores_test = pipeline.predict_proba(X_test_raw)[:, 1]

        row = evaluate_model(spec_model.name, scores_val, scores_test, backend=spec_model.backend)
        metrics_rows.append(row)

        if spec_model.name in {"random_forest", "lightgbm"}:
            explain_dir = run_dir / "explain" / spec_model.name
            explain_dir.mkdir(parents=True, exist_ok=True)

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
        rng = np.random.default_rng(seed)
        y_perm = pd.Series(rng.permutation(y_train.values), index=y_train.index)
        perm_pipeline = Pipeline(steps=[("prep", clone(preprocessor)), ("model", perm_spec.model)])
        perm_pipeline.fit(X_train_raw, y_perm)
        perm_scores_val = perm_pipeline.predict_proba(X_val_raw)[:, 1]
        perm_scores_test = perm_pipeline.predict_proba(X_test_raw)[:, 1]

        prevalence = float(y_train.mean()) if len(y_train) else 0.0
        pr_auc = compute_pr_auc(y_test, perm_scores_test)

        if pr_auc > prevalence + 0.10:
            raise RuntimeError(
                f"Leakage suspected / eval bug: permuted PR-AUC {pr_auc:.4f} > prevalence {prevalence:.4f} + 0.10"
            )

        row = evaluate_model("lightgbm_label_shuffled", perm_scores_val, perm_scores_test, backend=perm_spec.backend)
        metrics_rows.append(row)

        probe = {
            "model": "lightgbm_label_shuffled",
            "backend": perm_spec.backend,
            "pr_auc": pr_auc,
            "prevalence": prevalence,
            "seed": seed,
        }
        with open(run_dir / "permutation_probe.json", "w", encoding="utf-8") as f:
            json.dump(probe, f, indent=2)

    # Save metrics summary
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "metrics_summary.csv", index=False)

    # Save run config
    run_config = {
        "dataset": config_to_dict(cfg),
        "feature_regime": args.feature_regime,
        "split_strategy": split_strategy,
        "group_col": args.group_col or cfg.group_col,
        "timestamp_col": args.timestamp_col or cfg.timestamp_col,
        "seed": seed,
        "run_id": run_id,
        "timestamp": timestamp,
        "commit_hash": _git_commit_hash(),
        "package_versions": _package_versions(),
        "models": models,
        "label_permutation_probe": bool(args.label_permutation_probe),
        "notes": "stratified split is not deployment-realistic" if split_strategy == "stratified" else "",
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run HYDRA tabular IDS experiment")
    parser.add_argument("--dataset", required=True, help="Dataset name in datasets.yaml")
    parser.add_argument("--feature_regime", required=True, choices=["behaviour_only", "operational", "identifier_inclusive"])
    parser.add_argument("--split_strategy", required=True, choices=["host", "temporal", "stratified"])
    parser.add_argument("--group_col", default=None)
    parser.add_argument("--timestamp_col", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--label_permutation_probe", action="store_true")
    parser.add_argument("--datasets", default="hydra/config/datasets.yaml")
    parser.add_argument("--defaults", default="hydra/config/defaults.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
