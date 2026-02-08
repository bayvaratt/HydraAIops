from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hydra.config import dataset_config, get_config, resolve_path
from hydra.data.load import load_dataset
from hydra.data.schema import run_smoke_checks
from hydra.eval.protocols import timestamped_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name: ton_iot | cic_iot2023")
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--out", default=None, help="Base output directory")
    ap.add_argument("--max_rows", type=int, default=5000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    dcfg = dataset_config(args.dataset, override_path=args.data_path)

    df = load_dataset(dcfg, nrows=args.max_rows)
    base_out = args.out or str(Path("runs") / f"{dcfg.name}_smoke")
    out_dir = timestamped_dir(resolve_path(base_out))

    results = run_smoke_checks(
        df,
        dataset_name=dcfg.name,
        label_col=dcfg.label_col,
        type_col=dcfg.type_col,
        out_dir=out_dir,
        high_cardinality_threshold=cfg.high_cardinality_threshold,
        high_cardinality_ratio=cfg.high_cardinality_ratio,
    )

    with open(Path(out_dir) / "smoke_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Smoke test artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
