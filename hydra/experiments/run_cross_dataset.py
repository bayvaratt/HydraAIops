"""Cross-dataset generalisation pipeline.

Trains binary attack/normal classifiers on a source dataset using 6 canonical
flow features, then evaluates on both the source held-out test set and the full
target dataset.  The PR-AUC drop from source_test → target quantifies how well
learned patterns generalise across network environments.

Usage:
    python -m hydra.experiments.run_cross_dataset \\
        --source ton_iot --target cic_iot2023 \\
        --models logreg random_forest xgboost \\
        --seed 42 --max_rows_source 50000 --max_rows_target 50000

    python -m hydra.experiments.run_cross_dataset \\
        --source cic_iot2023 --target ton_iot \\
        --models logreg random_forest xgboost --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hydra.data.align import ALIGNERS, CANONICAL_FEATURES
from hydra.data.io import load_dataset
from hydra.evaluation.metrics import compute_pr_auc, compute_roc_auc
from hydra.models.tabular import (
    build_lightgbm,
    build_logreg,
    build_random_forest,
    build_sklearn_gbdt,
    build_xgboost,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_BUILDERS = {
    "logreg": build_logreg,
    "random_forest": build_random_forest,
    "sklearn_gbdt": build_sklearn_gbdt,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
}

DEFAULT_DATASETS_CFG = "hydra/config/datasets.yaml"


def _fpr_at_recall(y_true, scores, recall_target: float = 0.9) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = tpr >= recall_target
    if not mask.any():
        return float("nan")
    return float(fpr[mask][0])


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.source == args.target:
        raise ValueError("--source and --target must be different datasets")

    # ------------------------------------------------------------------ #
    # 1. Load + align source dataset                                       #
    # ------------------------------------------------------------------ #
    logger.info("Loading source dataset: %s", args.source)
    src_df, _ = load_dataset(args.datasets, args.source)
    src_aligned = ALIGNERS[args.source](src_df)
    del src_df

    if args.max_rows_source:
        src_aligned = src_aligned.sample(
            n=min(len(src_aligned), args.max_rows_source), random_state=args.seed
        ).reset_index(drop=True)
    logger.info("Source rows: %d  (prevalence=%.3f)", len(src_aligned), src_aligned["label"].mean())

    # ------------------------------------------------------------------ #
    # 2. Split source into train / test (stratified by label)             #
    # ------------------------------------------------------------------ #
    X_src = src_aligned[CANONICAL_FEATURES]
    y_src = src_aligned["label"]

    X_train, X_src_test, y_train, y_src_test = train_test_split(
        X_src, y_src,
        test_size=args.test_size,
        stratify=y_src,
        random_state=args.seed,
    )
    logger.info(
        "Train: %d  Source-test: %d  (train prevalence=%.3f)",
        len(X_train), len(X_src_test), float(y_train.mean()),
    )

    # ------------------------------------------------------------------ #
    # 3. Fit shared preprocessor on source training data                  #
    # ------------------------------------------------------------------ #
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_proc = preprocessor.fit_transform(X_train)
    X_src_test_proc = preprocessor.transform(X_src_test)

    # ------------------------------------------------------------------ #
    # 4. Load + align target dataset                                       #
    # ------------------------------------------------------------------ #
    logger.info("Loading target dataset: %s", args.target)
    tgt_df, _ = load_dataset(args.datasets, args.target)
    tgt_aligned = ALIGNERS[args.target](tgt_df)
    del tgt_df

    if args.max_rows_target:
        tgt_aligned = tgt_aligned.sample(
            n=min(len(tgt_aligned), args.max_rows_target), random_state=args.seed
        ).reset_index(drop=True)
    logger.info("Target rows: %d  (prevalence=%.3f)", len(tgt_aligned), tgt_aligned["label"].mean())

    X_tgt = tgt_aligned[CANONICAL_FEATURES]
    y_tgt = tgt_aligned["label"]
    X_tgt_proc = preprocessor.transform(X_tgt)

    # ------------------------------------------------------------------ #
    # 5. Train models and evaluate on both eval sets                       #
    # ------------------------------------------------------------------ #
    rows: list[dict] = []

    for model_name in args.models:
        if model_name not in MODEL_BUILDERS:
            logger.warning("Unknown model '%s'; skipping.", model_name)
            continue

        logger.info("Training %s ...", model_name)
        spec = MODEL_BUILDERS[model_name](args.seed)
        spec.model.fit(X_train_proc, y_train)

        for eval_name, X_eval, y_eval in [
            ("source_test", X_src_test_proc, y_src_test),
            ("target",      X_tgt_proc,      y_tgt),
        ]:
            scores = spec.model.predict_proba(X_eval)[:, 1]
            preds  = (scores >= 0.5).astype(int)
            row = {
                "source":     args.source,
                "target":     args.target,
                "model":      model_name,
                "eval_set":   eval_name,
                "n_train":    len(X_train),
                "n_eval":     len(y_eval),
                "prevalence": float(y_eval.mean()),
                "pr_auc":     compute_pr_auc(y_eval, scores),
                "roc_auc":    compute_roc_auc(y_eval, scores, logger),
                "f1":         float(f1_score(y_eval, preds, zero_division=0)),
                "fpr_at_recall_0_90": _fpr_at_recall(y_eval, scores, 0.9),
            }
            rows.append(row)
            logger.info(
                "  %-12s  eval=%-12s  pr_auc=%.4f  roc_auc=%.4f  f1=%.4f",
                model_name, eval_name, row["pr_auc"], row["roc_auc"], row["f1"],
            )

    metrics_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 6. Save results                                                      #
    # ------------------------------------------------------------------ #
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(f"runs/cross_dataset/{args.source}_to_{args.target}/{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    cfg_out = {
        "source": args.source,
        "target": args.target,
        "models": args.models,
        "seed": args.seed,
        "test_size": args.test_size,
        "max_rows_source": args.max_rows_source,
        "max_rows_target": args.max_rows_target,
        "canonical_features": CANONICAL_FEATURES,
        "run_ts": ts,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)

    logger.info("Results saved → %s", out_dir)
    return metrics_df


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset generalisation: train on source, evaluate on source_test + target"
    )
    parser.add_argument("--source", required=True,
                        choices=list(ALIGNERS.keys()),
                        help="Dataset to train on")
    parser.add_argument("--target", required=True,
                        choices=list(ALIGNERS.keys()),
                        help="Dataset to evaluate generalisation on")
    parser.add_argument("--models", nargs="+",
                        default=["logreg", "random_forest", "xgboost"],
                        help="Models to train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of source data held out as source_test")
    parser.add_argument("--max_rows_source", type=int, default=None,
                        help="Cap source dataset rows before split")
    parser.add_argument("--max_rows_target", type=int, default=None,
                        help="Cap target dataset rows")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS_CFG,
                        help="Path to datasets.yaml")
    args = parser.parse_args()

    metrics_df = run(args)

    print("\n=== Cross-dataset results ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
