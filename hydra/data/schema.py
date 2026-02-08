from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def required_columns(label_col: str, type_col: Optional[str] = None) -> List[str]:
    cols = [label_col]
    if type_col:
        cols.append(type_col)
    return cols


def validate_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return missing


def validate_label_values(df: pd.DataFrame, label_col: str) -> None:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    values = pd.Series(df[label_col]).dropna().unique()
    allowed = {0, 1}
    try:
        values_set = {int(v) for v in values}
    except Exception:
        values_set = set(values)
    if not values_set.issubset(allowed):
        raise ValueError(f"Label values outside {allowed}: {sorted(values_set)}")


def missingness_report(df: pd.DataFrame) -> Dict[str, float]:
    missing = df.isna().mean().to_dict()
    return {k: float(v) for k, v in missing.items()}


def detect_high_cardinality(
    df: pd.DataFrame,
    threshold_count: int = 500,
    threshold_ratio: float = 0.2,
) -> List[str]:
    high_card = []
    n_rows = max(len(df), 1)
    for col in df.columns:
        if df[col].dtype == "object":
            nunique = df[col].nunique(dropna=True)
            ratio = nunique / n_rows
            if nunique >= threshold_count or ratio >= threshold_ratio:
                high_card.append(col)
    return sorted(high_card)


def dataset_summary(df: pd.DataFrame, label_col: str) -> Dict:
    summary = {
        "shape": list(df.shape),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missingness_pct": {k: float(v) for k, v in df.isna().mean().items()},
        "label_distribution": {},
        "top_unique_counts": {},
    }
    if label_col in df.columns:
        summary["label_distribution"] = df[label_col].value_counts(dropna=False).to_dict()
    summary["top_unique_counts"] = df.nunique(dropna=True).sort_values(ascending=False).head(20).to_dict()
    return summary


def save_json(payload: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_smoke_checks(
    df: pd.DataFrame,
    dataset_name: str,
    label_col: str,
    type_col: Optional[str],
    out_dir: str,
    high_cardinality_threshold: int = 500,
    high_cardinality_ratio: float = 0.2,
) -> Dict:
    logger.info("Running schema smoke checks for %s", dataset_name)
    validate_required_columns(df, required_columns(label_col, type_col))
    validate_label_values(df, label_col)

    missing = missingness_report(df)
    high_card = detect_high_cardinality(
        df,
        threshold_count=high_cardinality_threshold,
        threshold_ratio=high_cardinality_ratio,
    )

    summary = dataset_summary(df, label_col)

    save_json(missing, str(Path(out_dir) / "missingness.json"))
    save_json(high_card, str(Path(out_dir) / "high_cardinality_columns.json"))
    save_json(summary, str(Path(out_dir) / "dataset_summary.json"))

    return {
        "missingness": missing,
        "high_cardinality": high_card,
        "summary": summary,
    }
