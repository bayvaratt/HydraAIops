from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace

import yaml

from hydra.pipelines.run_tabular import run


def main():
    parser = argparse.ArgumentParser(description="Run HYDRA experiment matrix")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--label_permutation_probe", action="store_true")
    parser.add_argument("--permutation_repeats", type=int, default=3)
    parser.add_argument("--duplicate_leakage_threshold", type=float, default=0.001)
    parser.add_argument("--fail_on_duplicate_leakage", action="store_true")
    parser.add_argument("--include_paper_comparison", action="store_true")
    parser.add_argument("--datasets", default="hydra/config/datasets.yaml")
    parser.add_argument("--defaults", default="hydra/config/defaults.yaml")
    args = parser.parse_args()

    with open(args.datasets, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    if args.dataset not in ds_cfg:
        raise KeyError(f"Dataset '{args.dataset}' not found in datasets config")
    if args.include_paper_comparison:
        feature_regimes = ["paper_5feat", "behaviour_only", "operational", "identifier_inclusive"]
        split_specs = [
            {"split_strategy": "host", "group_col": "src_ip", "timestamp_col": None},
            {"split_strategy": "host", "group_col": "dst_ip", "timestamp_col": None},
            {"split_strategy": "stratified", "group_col": None, "timestamp_col": None},
            {"split_strategy": "temporal", "group_col": None, "timestamp_col": ds_cfg[args.dataset].get("timestamp_col")},
        ]
    else:
        feature_regimes = ["behaviour_only", "operational", "identifier_inclusive"]
        split_specs = [
            {"split_strategy": "host", "group_col": ds_cfg[args.dataset].get("group_col"), "timestamp_col": None},
            {"split_strategy": "temporal", "group_col": None, "timestamp_col": ds_cfg[args.dataset].get("timestamp_col")},
            {"split_strategy": "stratified", "group_col": None, "timestamp_col": None},
        ]

    for regime in feature_regimes:
        for spec in split_specs:
            split = spec["split_strategy"]
            group_col = spec.get("group_col")
            timestamp_col = spec.get("timestamp_col")

            # Skip temporal if no timestamp_col
            if split == "temporal" and not timestamp_col:
                print(f"Skipping temporal split for {args.dataset} (no timestamp_col)")
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
                datasets=args.datasets,
                defaults=args.defaults,
                models=None,
            )
            run(run_args)


if __name__ == "__main__":
    main()
