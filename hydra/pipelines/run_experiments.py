from __future__ import annotations

import argparse
import logging
import numbers
from types import SimpleNamespace

import pandas as pd
import yaml

from hydra.pipelines.run_tabular import run


def _has_timestamp_col(dataset_path: str, timestamp_col: str | None) -> bool:
    if not timestamp_col:
        return False
    if dataset_path.endswith((".csv", ".gz")):
        df = pd.read_csv(dataset_path, nrows=0)
        return timestamp_col in df.columns
    if dataset_path.endswith((".parquet", ".pq")):
        try:
            pd.read_parquet(dataset_path, columns=[timestamp_col])
            return True
        except Exception:
            return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Run HYDRA experiment matrix")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--label_permutation_probe", action="store_true")
    parser.add_argument("--permutation_repeats", type=int, default=3)
    parser.add_argument("--duplicate_leakage_threshold", type=float, default=0.001)
    parser.add_argument("--fail_on_duplicate_leakage", action="store_true")
    parser.add_argument("--type_col", default=None)
    parser.add_argument("--normal_type_value", default=None)
    parser.add_argument("--feature_selection", default="none")
    parser.add_argument("--feature_selection_k", type=int, default=None)
    parser.add_argument("--type_unknown_threshold", type=float, default=0.0)
    parser.add_argument("--include_paper_comparison", action="store_true")
    parser.add_argument("--datasets", default="hydra/config/datasets.yaml")
    parser.add_argument("--defaults", default="hydra/config/defaults.yaml")
    args = parser.parse_args()

    with open(args.datasets, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    if args.dataset not in ds_cfg:
        raise KeyError(f"Dataset '{args.dataset}' not found in datasets config")
    dataset_path = str(ds_cfg[args.dataset]["path"])
    timestamp_col_cfg = ds_cfg[args.dataset].get("timestamp_col")
    has_timestamp = _has_timestamp_col(dataset_path, timestamp_col_cfg)
    if args.include_paper_comparison:
        feature_regimes = ["paper_5feat", "behaviour_only", "operational"]
        split_specs = [
            {"split_strategy": "host", "group_col": "src_ip", "timestamp_col": None},
            {"split_strategy": "host", "group_col": "dst_ip", "timestamp_col": None},
            {"split_strategy": "stratified", "group_col": None, "timestamp_col": None},
            {"split_strategy": "temporal", "group_col": None, "timestamp_col": timestamp_col_cfg},
        ]
    else:
        feature_regimes = ["behaviour_only", "operational"]
        split_specs = [
            {"split_strategy": "host", "group_col": ds_cfg[args.dataset].get("group_col"), "timestamp_col": None},
            {"split_strategy": "temporal", "group_col": None, "timestamp_col": timestamp_col_cfg},
            {"split_strategy": "stratified", "group_col": None, "timestamp_col": None},
        ]

    summary_rows = []
    for regime in feature_regimes:
        for spec in split_specs:
            split = spec["split_strategy"]
            group_col = spec.get("group_col")
            timestamp_col = spec.get("timestamp_col")

            # Skip temporal if no timestamp_col
            if split == "temporal" and not timestamp_col:
                print(f"Skipping temporal split for {args.dataset} (no timestamp_col configured)")
                continue
            if split == "temporal" and not has_timestamp:
                print(f"Skipping temporal split for {args.dataset} (timestamp_col missing in data)")
                continue
            if split == "host" and not group_col:
                print(f"Skipping host split for {args.dataset} (no group_col)")
                continue

            run_args = SimpleNamespace(
                dataset=args.dataset,
                feature_regime=regime,
                split_strategy=split,
                group_col=group_col,
                timestamp_col=timestamp_col,
                seed=args.seed,
                max_rows=args.max_rows,
                label_permutation_probe=args.label_permutation_probe,
                permutation_repeats=args.permutation_repeats,
                duplicate_leakage_threshold=args.duplicate_leakage_threshold,
                fail_on_duplicate_leakage=args.fail_on_duplicate_leakage,
                type_col=args.type_col,
                normal_type_value=args.normal_type_value,
                feature_selection=args.feature_selection,
                feature_selection_k=args.feature_selection_k,
                type_unknown_threshold=args.type_unknown_threshold,
                datasets=args.datasets,
                defaults=args.defaults,
                models=None,
            )
            result = run(run_args)
            metrics_df = result.get("metrics_df") if result else None
            best_row = None
            if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
                if "pr_lift" in metrics_df.columns:
                    series = pd.to_numeric(metrics_df["pr_lift"], errors="coerce")
                    if series.notna().any():
                        best_row = metrics_df.loc[series.idxmax()]
                if best_row is None and "pr_auc" in metrics_df.columns:
                    series = pd.to_numeric(metrics_df["pr_auc"], errors="coerce")
                    if series.notna().any():
                        best_row = metrics_df.loc[series.idxmax()]
            summary_rows.append(
                {
                    "run_id": result.get("run_id") if result else None,
                    "feature_regime": regime,
                    "split_strategy": split,
                    "group_col": group_col,
                    "best_model": best_row.get("model") if best_row is not None else None,
                    "pr_auc": best_row.get("pr_auc") if best_row is not None else None,
                    "pr_lift": best_row.get("pr_lift") if best_row is not None else None,
                    "roc_auc": best_row.get("roc_auc") if best_row is not None else None,
                    "precision@0.90": best_row.get("precision_test_at_recall_0_90") if best_row is not None else None,
                    "fpr@0.90": best_row.get("fpr_at_recall_0_90") if best_row is not None else None,
                    "coverage": best_row.get("coverage") if best_row is not None else None,
                }
            )

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        columns = [
            "run_id",
            "feature_regime",
            "split_strategy",
            "group_col",
            "best_model",
            "pr_auc",
            "pr_lift",
            "roc_auc",
            "precision@0.90",
            "fpr@0.90",
            "coverage",
        ]
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]

        def _fmt_cell(val):
            if isinstance(val, numbers.Real):
                if pd.isna(val):
                    return "nan"
                return f"{float(val):.4f}"
            return "nan" if val is None else str(val)

        for col in df.columns:
            df[col] = df[col].map(_fmt_cell)
        print("Experiment summary (best model per setting):")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
