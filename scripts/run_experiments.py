from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hydra.config import dataset_config, get_config, resolve_path
from hydra.data.load import load_dataset, sample_dataframe
from hydra.eval.protocols import run_experiment_tabular, timestamped_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_rows", type=int, default=5000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_permutation_probe", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()

    ton_cfg = dataset_config("ton_iot")
    ton_df = load_dataset(ton_cfg)
    if args.max_rows:
        ton_df = sample_dataframe(ton_df, args.max_rows, label_col=ton_cfg.label_col, seed=args.seed)

    base_out = resolve_path(args.out or str(Path("runs") / "ton"))

    commit = None
    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None

    split_matrix = [
        {"split_strategy": "host", "group_col": ton_cfg.src_col, "tag": "host_src_ip"},
        {"split_strategy": "host", "group_col": ton_cfg.dst_col, "tag": "host_dst_ip"},
    ]
    if ton_cfg.timestamp_col and ton_cfg.timestamp_col in ton_df.columns:
        split_matrix.append(
            {
                "split_strategy": "temporal",
                "group_col": None,
                "timestamp_col": ton_cfg.timestamp_col,
                "tag": f"temporal_{ton_cfg.timestamp_col}",
            }
        )
    else:
        logger.warning("Skipping temporal split: no timestamp_col configured or present.")

    regimes = ["behaviour_only", "operational", "identifier_inclusive"]
    model_names = ["logreg", "random_forest", "lightgbm"]

    for split_cfg in split_matrix:
        for regime in regimes:
            out_dir = timestamped_dir(resolve_path(str(Path(base_out) / split_cfg["tag"] / regime)))
            run_config = {
                "dataset": ton_cfg.name,
                "data_path": ton_cfg.path,
                "feature_regime": regime,
                "label_col": ton_cfg.label_col,
                "type_col": ton_cfg.type_col,
                "timestamp_col": split_cfg.get("timestamp_col", ton_cfg.timestamp_col),
                "split_strategy": split_cfg["split_strategy"],
                "group_col": split_cfg.get("group_col"),
                "seed": args.seed,
                "label_permutation_probe": bool(args.label_permutation_probe),
                "commit": commit,
            }
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(out_dir) / "run_config.json", "w") as f:
                import json

                json.dump(run_config, f, indent=2)

            run_experiment_tabular(
                ton_df,
                dataset_name=ton_cfg.name,
                cfg=cfg,
                out_dir=out_dir,
                label_col=ton_cfg.label_col,
                type_col=ton_cfg.type_col,
                feature_regime=regime,
                timestamp_col=split_cfg.get("timestamp_col", ton_cfg.timestamp_col),
                split_strategy=split_cfg["split_strategy"],
                group_col=split_cfg.get("group_col"),
                model_names=model_names,
                seed=args.seed,
                label_permutation_probe=args.label_permutation_probe,
            )


if __name__ == "__main__":
    main()
