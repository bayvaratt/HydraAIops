from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hydra.config import dataset_config, get_config, resolve_path
from hydra.data.load import load_dataset, sample_dataframe
from hydra.eval.protocols import run_experiment_gnn, timestamped_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name: cic_iot2023")
    ap.add_argument(
        "--feature_regime",
        default="behaviour_only",
        choices=["behaviour_only", "operational", "identifier_inclusive"],
    )
    ap.add_argument("--data_path", default=None, help="Override dataset path")
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--timestamp_col", default=None)
    ap.add_argument("--src_col", default=None)
    ap.add_argument("--dst_col", default=None)
    ap.add_argument("--out", default=None, help="Base output directory")
    ap.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    dcfg = dataset_config(args.dataset, override_path=args.data_path)

    label_col = args.label_col or dcfg.label_col
    timestamp_col = args.timestamp_col if args.timestamp_col is not None else dcfg.timestamp_col
    src_col = args.src_col or dcfg.src_col
    dst_col = args.dst_col or dcfg.dst_col

    df = load_dataset(dcfg)
    if args.max_rows:
        df = sample_dataframe(df, args.max_rows, label_col=label_col, seed=args.seed)

    base_out = args.out or str(Path("runs") / f"{dcfg.name}_gnn")
    base_out = resolve_path(base_out)
    out_dir = timestamped_dir(base_out)

    config_payload = {
        "dataset": dcfg.name,
        "data_path": dcfg.path,
        "feature_regime": args.feature_regime,
        "label_col": label_col,
        "timestamp_col": timestamp_col,
        "src_col": src_col,
        "dst_col": dst_col,
        "seed": args.seed,
    }
    with open(Path(out_dir) / "run_config.json", "w") as f:
        json.dump(config_payload, f, indent=2)

    run_experiment_gnn(
        df,
        dataset_name=dcfg.name,
        cfg=cfg,
        out_dir=out_dir,
        label_col=label_col,
        feature_regime=args.feature_regime,
        src_col=src_col,
        dst_col=dst_col,
        timestamp_col=timestamp_col,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
