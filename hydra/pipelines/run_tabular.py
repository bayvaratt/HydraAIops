from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hydra.config import dataset_config, get_config, resolve_path
from hydra.data.load import load_dataset, sample_dataframe
from hydra.eval.protocols import run_experiment_tabular, timestamped_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name: ton_iot | cic_iot2023")
    ap.add_argument(
        "--feature_regime",
        default="behaviour_only",
        choices=["behaviour_only", "operational", "identifier_inclusive"],
    )
    ap.add_argument("--data_path", default=None, help="Override dataset path")
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--type_col", default=None)
    ap.add_argument("--timestamp_col", default=None)
    ap.add_argument("--split_strategy", default=None, choices=["stratified", "temporal", "host"])
    ap.add_argument("--group_col", default=None, help="Grouping column for host-based split")
    ap.add_argument("--label_permutation_probe", action="store_true", help="Enable label permutation leakage probe")
    ap.add_argument("--out", default=None, help="Base output directory")
    ap.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    dcfg = dataset_config(args.dataset, override_path=args.data_path)

    label_col = args.label_col or dcfg.label_col
    type_col = args.type_col if args.type_col is not None else dcfg.type_col
    timestamp_col = args.timestamp_col if args.timestamp_col is not None else dcfg.timestamp_col
    split_strategy = args.split_strategy or cfg.split_strategy
    group_col = args.group_col or dcfg.src_col

    df = load_dataset(dcfg)
    if args.max_rows:
        df = sample_dataframe(df, args.max_rows, label_col=label_col, seed=args.seed)

    base_out = args.out or str(Path("runs") / dcfg.name)
    base_out = resolve_path(base_out)
    out_dir = timestamped_dir(base_out)

    commit = None
    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None

    config_payload = {
        "dataset": dcfg.name,
        "data_path": dcfg.path,
        "feature_regime": args.feature_regime,
        "label_col": label_col,
        "type_col": type_col,
        "timestamp_col": timestamp_col,
        "split_strategy": split_strategy,
        "group_col": group_col,
        "seed": args.seed,
        "label_permutation_probe": bool(args.label_permutation_probe),
        "commit": commit,
    }
    with open(Path(out_dir) / "run_config.json", "w") as f:
        json.dump(config_payload, f, indent=2)

    run_experiment_tabular(
        df,
        dataset_name=dcfg.name,
        cfg=cfg,
        out_dir=out_dir,
        label_col=label_col,
        type_col=type_col,
        feature_regime=args.feature_regime,
        timestamp_col=timestamp_col,
        split_strategy=split_strategy,
        group_col=group_col,
        seed=args.seed,
        label_permutation_probe=args.label_permutation_probe,
    )


if __name__ == "__main__":
    main()
