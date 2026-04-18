"""Data preprocessing: feature specification, type inference, and sklearn pipelines.

Handles column selection by feature regime, port bucketing, type inference,
and fitting ColumnTransformer pipelines (impute → log1p → scale for numerics,
impute → one-hot for categoricals).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Text detection thresholds: columns with mean string length > this or
# max string length > MAX_TEXT_LENGTH are classified as free-text.
AVG_TEXT_LENGTH_THRESHOLD = 30
MAX_TEXT_LENGTH_THRESHOLD = 100

# Categorical columns with more unique values than this are considered
# high-cardinality and dropped in behaviour_only/operational regimes.
HIGH_CARDINALITY_THRESHOLD = 50

# IANA port ranges for bucket_port()
PORT_WELL_KNOWN_MAX = 1023
PORT_REGISTERED_MAX = 49151
PORT_DYNAMIC_MAX = 65535


def _log1p_nonneg(X: np.ndarray) -> np.ndarray:
    """log1p after clipping negatives to 0 — safe for all network flow numerics."""
    return np.log1p(np.clip(X, 0, None))


@dataclass
class FeatureSpec:
    regime: str
    keep_categorical: List[str]
    keep_numeric: List[str]
    derived_categorical: Dict[str, str]
    dropped: List[str]
    missing_required: List[str] | None = None


def _infer_types(
    df: pd.DataFrame,
    categorical_cols: List[str] | None,
    numeric_cols: List[str] | None,
) -> Tuple[List[str], List[str]]:
    if categorical_cols is not None or numeric_cols is not None:
        cat = categorical_cols or []
        num = numeric_cols or []
        return list(dict.fromkeys(cat)), list(dict.fromkeys(num))

    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, num_cols


def _is_text_col(series: pd.Series) -> bool:
    if series.dtype != "object":
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    avg_len = sample.map(len).mean()
    max_len = sample.map(len).max()
    return avg_len > AVG_TEXT_LENGTH_THRESHOLD or max_len > MAX_TEXT_LENGTH_THRESHOLD


def _high_cardinality(series: pd.Series) -> bool:
    n = len(series)
    if n == 0:
        return False
    nunique = series.nunique(dropna=True)
    return nunique > HIGH_CARDINALITY_THRESHOLD


def _match_cols(cols: List[str], needles: List[str]) -> List[str]:
    out = []
    for c in cols:
        lc = c.lower()
        if any(n in lc for n in needles):
            out.append(c)
    return out


def bucket_port(port_series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(port_series, errors="coerce")
    buckets = pd.Series(index=port_series.index, dtype="object")
    buckets[:] = "unknown"
    buckets[(vals >= 0) & (vals <= PORT_WELL_KNOWN_MAX)] = "well_known"
    buckets[(vals >= PORT_WELL_KNOWN_MAX + 1) & (vals <= PORT_REGISTERED_MAX)] = "registered"
    buckets[(vals >= PORT_REGISTERED_MAX + 1) & (vals <= PORT_DYNAMIC_MAX)] = "dynamic"
    return buckets


def build_feature_spec(
    train_df: pd.DataFrame,
    label_col: str,
    regime: str,
    categorical_cols: List[str] | None,
    numeric_cols: List[str] | None,
    logger,
) -> FeatureSpec:
    if regime == "paper_5feat":
        required = ["duration", "src_bytes", "dst_bytes", "src_ip_bytes", "dst_ip_bytes"]
        missing = [c for c in required if c not in train_df.columns]
        present = [c for c in required if c in train_df.columns]
        if missing:
            logger.warning("paper_5feat missing required columns: %s", missing)
        if len(present) < 3:
            raise RuntimeError(
                f"paper_5feat requires at least 3 of 5 columns; present={present} missing={missing}"
            )
        logger.info("paper_5feat selected columns: %s", present)
        return FeatureSpec(
            regime=regime,
            keep_categorical=[],
            keep_numeric=present,
            derived_categorical={},
            dropped=[],
            missing_required=missing,
        )

    cat_cols, num_cols = _infer_types(train_df.drop(columns=[label_col]), categorical_cols, numeric_cols)
    cat_cols = [c for c in cat_cols if c != label_col]
    num_cols = [c for c in num_cols if c != label_col]

    all_cols = list(dict.fromkeys(cat_cols + num_cols))
    # Exclude traffic-metric columns whose names happen to contain "ip"
    # (e.g. src_ip_bytes, dst_ip_bytes are volume features, not identifiers)
    _metric_suffixes = ("_bytes", "_pkts", "_packets", "_count", "_len")
    ip_cols = [
        c for c in _match_cols(all_cols, ["ip", "addr", "address", "mac"])
        if not c.endswith(_metric_suffixes)
    ]
    port_cols = _match_cols(all_cols, ["port"])
    service_cols = _match_cols(all_cols, ["service", "svc"])
    text_cols = [c for c in cat_cols if _is_text_col(train_df[c])]
    high_card = [c for c in cat_cols if _high_cardinality(train_df[c])]

    drop = set()
    derived = {}

    if regime == "behaviour_only":
        drop.update(ip_cols)
        drop.update(port_cols)
        drop.update(service_cols)
        drop.update(text_cols)
        drop.update(high_card)
    elif regime == "operational":
        drop.update(ip_cols)
        drop.update(port_cols)
        drop.update(text_cols)
        drop.update(high_card)
        for c in port_cols:
            derived[f"{c}_bucket"] = c
    elif regime == "identifier_inclusive":
        drop.update(text_cols)
        # keep IPs/ports even if high-card
        drop.update([c for c in high_card if c not in ip_cols and c not in port_cols])
        for c in port_cols:
            derived[f"{c}_bucket"] = c
    elif regime == "core_flow":
        # Top-14 MI features: 5 flow-volume numerics + 2 connection categoricals
        # + 7 DNS categoricals. Selected by mutual-information analysis on TON_IoT.
        _CORE_NUMERIC = [
            "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
        ]
        _CORE_CATEGORICAL = [
            "proto", "conn_state",
            "dns_rejected", "dns_AA", "dns_RD", "dns_RA", "dns_query",
            "dns_qclass", "dns_qtype",
        ]
        core = set(_CORE_NUMERIC + _CORE_CATEGORICAL)
        drop.update(c for c in all_cols if c not in core)
        missing = [c for c in core if c not in all_cols]
        if missing:
            logger.warning("core_flow: requested columns absent from data: %s", missing)
    else:
        raise ValueError(f"Unknown feature regime: {regime}")

    keep_cat = [c for c in cat_cols if c not in drop]
    keep_num = [c for c in num_cols if c not in drop]

    logger.info("Feature regime: %s", regime)
    logger.info("Dropping columns (%d): %s", len(drop), sorted(drop))
    if derived:
        logger.info("Derived port buckets: %s", sorted(derived.keys()))

    return FeatureSpec(
        regime=regime,
        keep_categorical=keep_cat + list(derived.keys()),
        keep_numeric=keep_num,
        derived_categorical=derived,
        dropped=sorted(drop),
    )


def apply_feature_spec(
    df: pd.DataFrame,
    spec: FeatureSpec,
    label_col: str,
    logger,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    out = df.copy()

    for new_col, src_col in spec.derived_categorical.items():
        if src_col not in out.columns:
            logger.warning("Port column missing for bucket feature: %s", src_col)
            out[new_col] = "unknown"
        else:
            out[new_col] = bucket_port(out[src_col])

    out = out.drop(columns=spec.dropped, errors="ignore")
    if label_col in out.columns:
        out = out.drop(columns=[label_col])

    cols = [c for c in (spec.keep_numeric + spec.keep_categorical) if c in out.columns]
    out = out[cols]

    cat_cols = [c for c in spec.keep_categorical if c in out.columns]
    num_cols = [c for c in spec.keep_numeric if c in out.columns]

    return out, cat_cols, num_cols


def fit_preprocessor(
    X_train: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> ColumnTransformer:
    # Numeric: impute → log1p (compresses right-skewed flow volumes) → z-score scale.
    # log1p clips negatives to 0 first — safe for bytes/packets/duration.
    # Tree models are scale-invariant so StandardScaler doesn't hurt them;
    # logreg and CNN-LSTM benefit significantly from both steps.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p",   FunctionTransformer(_log1p_nonneg, validate=True,
                                       feature_names_out="one-to-one")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    if not transformers:
        raise ValueError("No features available after preprocessing")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor
