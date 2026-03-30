from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _load_importances(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "feature" not in df.columns or "importance" not in df.columns:
        raise ValueError(f"Invalid importance file (missing columns): {path}")
    return df[["feature", "importance"]]


def compute_rank_similarity(
    a_path: Path,
    b_path: Path,
    method: str = "spearman",
) -> Tuple[float | None, List[str]]:
    df_a = _load_importances(a_path)
    df_b = _load_importances(b_path)

    merged = df_a.merge(df_b, on="feature", suffixes=("_a", "_b"))
    if len(merged) < 2:
        return None, merged["feature"].tolist()

    merged["rank_a"] = merged["importance_a"].rank(ascending=False, method="average")
    merged["rank_b"] = merged["importance_b"].rank(ascending=False, method="average")

    corr = float(merged["rank_a"].corr(merged["rank_b"], method=method))
    return corr, merged["feature"].tolist()


def write_similarity_json(
    out_path: Path,
    correlation: float | None,
    compared_regimes: List[str],
    method: str,
    overlap_features: List[str],
) -> None:
    payload: Dict[str, object] = {
        "method": method,
        "correlation": correlation,
        "compared_regimes": compared_regimes,
        "n_overlap": len(overlap_features),
        "overlap_features": overlap_features,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
