from __future__ import annotations

import logging
from pathlib import Path

from hydra.config import dataset_config, get_config, resolve_path
from hydra.data.load import load_dataset
from hydra.eval.protocols import run_experiment_tabular, run_experiment_gnn, timestamped_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = get_config()

    ton_cfg = dataset_config("ton_iot")
    ton_df = load_dataset(ton_cfg)
    for regime in ["behaviour_only", "operational", "identifier_inclusive"]:
        out_dir = timestamped_dir(resolve_path(str(Path("runs") / f"ton_iot_{regime}")))
        run_experiment_tabular(
            ton_df,
            dataset_name=ton_cfg.name,
            cfg=cfg,
            out_dir=out_dir,
            label_col=ton_cfg.label_col,
            type_col=ton_cfg.type_col,
            feature_regime=regime,
            timestamp_col=ton_cfg.timestamp_col,
            split_strategy=cfg.split_strategy,
            group_col=ton_cfg.src_col,
            seed=cfg.seeds[0],
        )

    cic_cfg = dataset_config("cic_iot2023")
    cic_df = load_dataset(cic_cfg)
    out_dir = timestamped_dir(resolve_path(str(Path("runs") / "cic_iot2023_gnn")))
    run_experiment_gnn(
        cic_df,
        dataset_name=cic_cfg.name,
        cfg=cfg,
        out_dir=out_dir,
        label_col=cic_cfg.label_col,
        feature_regime="behaviour_only",
        src_col=cic_cfg.src_col,
        dst_col=cic_cfg.dst_col,
        timestamp_col=cic_cfg.timestamp_col,
        seed=cfg.seeds[0],
    )


if __name__ == "__main__":
    main()
