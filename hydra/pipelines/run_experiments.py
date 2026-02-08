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
    parser.add_argument("--datasets", default="hydra/config/datasets.yaml")
    parser.add_argument("--defaults", default="hydra/config/defaults.yaml")
    args = parser.parse_args()

    with open(args.datasets, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    if args.dataset not in ds_cfg:
        raise KeyError(f"Dataset '{args.dataset}' not found in datasets config")

    feature_regimes = ["behaviour_only", "operational", "identifier_inclusive"]
    split_strategies = ["host", "temporal", "stratified"]

    for regime in feature_regimes:
        for split in split_strategies:
            # Skip temporal if no timestamp_col
            if split == "temporal" and not ds_cfg[args.dataset].get("timestamp_col"):
                print(f"Skipping temporal split for {args.dataset} (no timestamp_col)")
                continue
            if split == "host" and not ds_cfg[args.dataset].get("group_col"):
                print(f"Skipping host split for {args.dataset} (no group_col)")
                continue

            run_args = SimpleNamespace(
                dataset=args.dataset,
                feature_regime=regime,
                split_strategy=split,
                group_col=ds_cfg[args.dataset].get("group_col"),
                timestamp_col=ds_cfg[args.dataset].get("timestamp_col"),
                seed=args.seed,
                max_rows=args.max_rows,
                label_permutation_probe=args.label_permutation_probe,
                datasets=args.datasets,
                defaults=args.defaults,
                models=None,
            )
            run(run_args)


if __name__ == "__main__":
    main()
