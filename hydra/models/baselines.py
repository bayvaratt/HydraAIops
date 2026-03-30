from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def baseline_majority_scores(y_train: pd.Series, n_samples: int) -> np.ndarray:
    prevalence = float(y_train.mean()) if len(y_train) else 0.0
    return np.full(n_samples, prevalence, dtype=float)


def _find_col(df: pd.DataFrame, preferred: Optional[str], candidates: list[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def baseline_threshold_scores(
    df: pd.DataFrame,
    colmap: Dict[str, Optional[str]],
    logger,
) -> np.ndarray:
    duration_col = _find_col(df, colmap.get("duration_col"), ["duration", "dur", "flow_duration"])
    src_bytes_col = _find_col(df, colmap.get("src_bytes_col"), ["src_bytes", "sbytes", "bytes_src", "srcByte"])
    dst_bytes_col = _find_col(df, colmap.get("dst_bytes_col"), ["dst_bytes", "dbytes", "bytes_dst", "dstByte"])
    src_pkts_col = _find_col(df, colmap.get("src_pkts_col"), ["src_pkts", "spkts", "src_packets"])
    dst_pkts_col = _find_col(df, colmap.get("dst_pkts_col"), ["dst_pkts", "dpkts", "dst_packets"])

    score = np.zeros(len(df), dtype=float)

    def add_term(col, label):
        nonlocal score
        if col is None:
            logger.warning("baseline_threshold: missing %s column; skipping term", label)
            return
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
        score += np.log1p(vals)

    if duration_col is not None:
        add_term(duration_col, "duration")
    else:
        logger.warning("baseline_threshold: missing duration column; skipping")

    if src_bytes_col or dst_bytes_col:
        src = pd.to_numeric(df[src_bytes_col], errors="coerce").fillna(0) if src_bytes_col else 0
        dst = pd.to_numeric(df[dst_bytes_col], errors="coerce").fillna(0) if dst_bytes_col else 0
        score += np.log1p(src + dst)
    else:
        logger.warning("baseline_threshold: missing bytes columns; skipping")

    if src_pkts_col or dst_pkts_col:
        srcp = pd.to_numeric(df[src_pkts_col], errors="coerce").fillna(0) if src_pkts_col else 0
        dstp = pd.to_numeric(df[dst_pkts_col], errors="coerce").fillna(0) if dst_pkts_col else 0
        score += np.log1p(srcp + dstp)
    else:
        logger.warning("baseline_threshold: missing packet columns; skipping")

    return score
