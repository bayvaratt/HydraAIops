from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml


@dataclass
class DatasetConfig:
    name: str
    path: str
    label_col: str
    positive_label: Any
    type_col: Optional[str] = None
    normal_type_value: Optional[Any] = None
    timestamp_col: Optional[str] = None
    group_col: Optional[str] = None
    duration_col: Optional[str] = None
    src_bytes_col: Optional[str] = None
    dst_bytes_col: Optional[str] = None
    src_pkts_col: Optional[str] = None
    dst_pkts_col: Optional[str] = None
    dst_ip_col: Optional[str] = None
    categorical_cols: Optional[list[str]] = None
    numeric_cols: Optional[list[str]] = None


def load_dataset_config(config_path: str, dataset_name: str) -> DatasetConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if dataset_name not in cfg:
        raise KeyError(f"Dataset '{dataset_name}' not found in {config_path}")
    d = cfg[dataset_name]
    return DatasetConfig(
        name=dataset_name,
        path=str(d["path"]),
        label_col=str(d["label_col"]),
        positive_label=d.get("positive_label", 1),
        type_col=d.get("type_col"),
        normal_type_value=d.get("normal_type_value"),
        timestamp_col=d.get("timestamp_col"),
        group_col=d.get("group_col"),
        duration_col=d.get("duration_col"),
        src_bytes_col=d.get("src_bytes_col"),
        dst_bytes_col=d.get("dst_bytes_col"),
        src_pkts_col=d.get("src_pkts_col"),
        dst_pkts_col=d.get("dst_pkts_col"),
        dst_ip_col=d.get("dst_ip_col"),
        categorical_cols=d.get("categorical_cols"),
        numeric_cols=d.get("numeric_cols"),
    )


def resolve_dataset_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    parts = candidate.parts
    # Support both historical data/<dataset>/... and current data/raw/<dataset>/...
    if len(parts) >= 3 and parts[0] == "data":
        if parts[1] == "raw" and len(parts) >= 4:
            alt = Path("data", parts[2], *parts[3:])
            if alt.exists():
                return alt
        else:
            alt = Path("data", "raw", *parts[1:])
            if alt.exists():
                return alt

    return candidate


def _read_table(path: str) -> pd.DataFrame:
    p = resolve_dataset_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    if p.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file extension: {p.suffix}")


def _normalize_labels(y: pd.Series, positive_label: Any) -> pd.Series:
    if y.dropna().nunique() <= 2 and set(y.dropna().unique()).issubset({0, 1}):
        return y.astype(int)
    return (y == positive_label).astype(int)


def load_dataset(
    config_path: str,
    dataset_name: str,
) -> Tuple[pd.DataFrame, DatasetConfig]:
    cfg = load_dataset_config(config_path, dataset_name)
    df = _read_table(cfg.path)
    if cfg.label_col not in df.columns:
        raise KeyError(f"Label column '{cfg.label_col}' not found in dataset")
    df[cfg.label_col] = _normalize_labels(df[cfg.label_col], cfg.positive_label)
    return df, cfg


def config_to_dict(cfg: DatasetConfig) -> Dict[str, Any]:
    return json.loads(json.dumps(cfg.__dict__))
