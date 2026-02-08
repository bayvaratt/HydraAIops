from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: str
    label_col: str = "label"
    type_col: Optional[str] = "type"
    timestamp_col: Optional[str] = None
    src_col: str = "src_ip"
    dst_col: str = "dst_ip"


@dataclass(frozen=True)
class TabularModelConfig:
    logreg: Dict = field(default_factory=lambda: {
        "max_iter": 2000,
        "solver": "saga",
        "n_jobs": -1,
    })
    random_forest: Dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_jobs": -1,
        "random_state": 42,
    })
    hist_gbt: Dict = field(default_factory=lambda: {
        "max_depth": None,
        "max_iter": 300,
        "learning_rate": 0.1,
        "l2_regularization": 0.0,
        "random_state": 42,
    })


@dataclass(frozen=True)
class GraphConfig:
    window_seconds: int = 60
    split_strategy: str = "temporal"  # temporal | entity
    edge_agg: str = "mean"  # mean | sum
    min_edges_per_window: int = 1
    max_edges: Optional[int] = 20000
    node_feature_mode: str = "degree"  # degree | zeros


@dataclass(frozen=True)
class GNNModelConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5


@dataclass
class HydraConfig:
    datasets: Dict[str, DatasetConfig]
    seeds: List[int] = field(default_factory=lambda: [41, 42, 43, 44, 45])
    recall_target: float = 0.90
    feature_regime: str = "behaviour_only"  # behaviour_only | operational | identifier_inclusive
    test_size: float = 0.2
    val_size: float = 0.2
    split_strategy: str = "host"  # stratified | temporal | host
    high_cardinality_threshold: int = 500
    high_cardinality_ratio: float = 0.2
    categorical_top_k: int = 200
    enable_port_bucketing: bool = True
    port_top_n: int = 20
    max_explain_samples: int = 200
    top_k_explanations: int = 10
    stability_top_k: int = 10
    stability_sample_size: int = 50
    tabular_models: TabularModelConfig = field(default_factory=TabularModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    gnn: GNNModelConfig = field(default_factory=GNNModelConfig)


DEFAULT_DATASETS: Dict[str, DatasetConfig] = {
    "ton_iot": DatasetConfig(
        name="ton_iot",
        path=str(REPO_ROOT / "data/ton_iot/raw/train_test_network.csv"),
        label_col="label",
        type_col="type",
        timestamp_col=None,
        src_col="src_ip",
        dst_col="dst_ip",
    ),
    "cic_iot2023": DatasetConfig(
        name="cic_iot2023",
        path="/Users/varattsaengsiripongpun/.cache/kagglehub/datasets/himadri07/ciciot2023/versions/1/CICIOT23/train/train.csv",
        label_col="label_binary",
        type_col="label",
        timestamp_col=None,
        src_col="Protocol Type",
        dst_col="app_proto",
    ),
}


def get_config() -> HydraConfig:
    return HydraConfig(datasets=DEFAULT_DATASETS)


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def dataset_config(name: str, override_path: Optional[str] = None) -> DatasetConfig:
    cfg = DEFAULT_DATASETS.get(name)
    if cfg is None:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DEFAULT_DATASETS.keys())}")
    if override_path:
        return DatasetConfig(
            name=cfg.name,
            path=resolve_path(override_path),
            label_col=cfg.label_col,
            type_col=cfg.type_col,
            timestamp_col=cfg.timestamp_col,
            src_col=cfg.src_col,
            dst_col=cfg.dst_col,
        )
    return cfg
