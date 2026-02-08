from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_object_dtype, is_string_dtype

from hydra.data.schema import detect_high_cardinality

logger = logging.getLogger(__name__)

IDENTIFIER_COLUMNS = {
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "flow_id",
    "flow_id_hash",
}

IP_IDENTIFIER_COLUMNS = {
    "src_ip",
    "dst_ip",
    "flow_id",
    "flow_id_hash",
}

SERVICE_COLUMNS = {"service"}

RAW_TEXT_COLUMNS = {
    "http_uri",
    "http_user_agent",
    "dns_query",
    "ssl_subject",
    "ssl_issuer",
    "http_orig_mime_types",
    "http_resp_mime_types",
}

PORT_COLUMNS = {"src_port", "dst_port"}

PRESENCE_PATTERNS = {
    "has_dns": ["dns"],
    "has_http": ["http"],
    "has_ssl": ["ssl"],
    "has_weird": ["weird"],
}


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _is_present(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str) and value.strip() in {"", "-", "nan", "NaN"}:
        return False
    return True


def _is_categorical(series: pd.Series) -> bool:
    return (
        is_object_dtype(series.dtype)
        or is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


@dataclass
class TabularPreprocessor:
    feature_regime: str
    label_col: str
    type_col: Optional[str]
    high_cardinality_threshold: int = 500
    high_cardinality_ratio: float = 0.2
    categorical_top_k: int = 200
    enable_port_bucketing: bool = True
    port_top_n: int = 20

    dropped_columns: List[str] = field(default_factory=list, init=False)
    high_cardinality_columns: List[str] = field(default_factory=list, init=False)
    feature_columns: List[str] = field(default_factory=list, init=False)
    category_maps: Dict[str, List[str]] = field(default_factory=dict, init=False)
    port_top_values: Dict[str, List[int]] = field(default_factory=dict, init=False)
    transformer: Optional[ColumnTransformer] = field(default=None, init=False)
    feature_names: List[str] = field(default_factory=list, init=False)

    def fit_transform(self, df: pd.DataFrame):
        X = self._prepare_features(df, fit=True)
        self._fit_transformer(X)
        return self.transformer.transform(X)

    def transform(self, df: pd.DataFrame):
        if self.transformer is None:
            raise RuntimeError("Preprocessor not fitted")
        X = self._prepare_features(df, fit=False)
        return self.transformer.transform(X)

    def _prepare_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        df = self._add_presence_indicators(df)
        if self.enable_port_bucketing and self.feature_regime in {"operational", "identifier_inclusive"}:
            df = self._apply_port_bucketing(df, fit=fit)

        drop_cols = self._compute_drop_columns(df, fit=fit)
        if fit:
            self.dropped_columns = sorted(set(drop_cols))

        drop_cols = set(drop_cols)
        if self.label_col in df.columns:
            drop_cols.add(self.label_col)
        if self.type_col and self.type_col in df.columns:
            drop_cols.add(self.type_col)

        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        for col in X.columns:
            if _is_categorical(X[col]):
                X[col] = X[col].replace({"-": np.nan})

        if fit:
            for col in X.columns:
                if _is_categorical(X[col]):
                    vc = X[col].value_counts(dropna=True)
                    if len(vc) > self.categorical_top_k:
                        keep = set(vc.head(self.categorical_top_k).index)
                        self.category_maps[col] = sorted(keep)
                        X[col] = X[col].where(X[col].isin(keep), "__OTHER__")
        else:
            for col, keep in self.category_maps.items():
                if col in X.columns:
                    X[col] = X[col].where(X[col].isin(keep), "__OTHER__")

        if fit:
            self.feature_columns = list(X.columns)
        else:
            X = self._align_columns(X)

        return X

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        extra = [c for c in X.columns if c not in self.feature_columns]
        if extra:
            X = X.drop(columns=extra)
        X = X[self.feature_columns]
        return X

    def _fit_transformer(self, X: pd.DataFrame) -> None:
        cat_cols = [c for c in X.columns if _is_categorical(X[c])]
        num_cols = [c for c in X.columns if not _is_categorical(X[c])]
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),
                ("cat", _make_ohe(), cat_cols),
            ]
        )
        self.transformer.fit(X)

        try:
            self.feature_names = list(self.transformer.get_feature_names_out())
        except Exception:
            feature_names = []
            for name, trans, cols in self.transformer.transformers_:
                if name == "num":
                    feature_names.extend(cols)
                elif name == "cat":
                    try:
                        feature_names.extend(trans.get_feature_names(cols))
                    except Exception:
                        feature_names.extend(cols)
            self.feature_names = feature_names

    def _compute_drop_columns(self, df: pd.DataFrame, fit: bool) -> List[str]:
        drop_cols: List[str] = []
        high_card = detect_high_cardinality(
            df,
            threshold_count=self.high_cardinality_threshold,
            threshold_ratio=self.high_cardinality_ratio,
        )
        if fit:
            self.high_cardinality_columns = high_card

        if self.feature_regime == "behaviour_only":
            drop_cols.extend(list(IDENTIFIER_COLUMNS))
            drop_cols.extend(list(RAW_TEXT_COLUMNS))
            drop_cols.extend(list(SERVICE_COLUMNS))
            drop_cols.extend(high_card)
        elif self.feature_regime == "operational":
            drop_cols.extend(list(IP_IDENTIFIER_COLUMNS))
            drop_cols.extend(list(RAW_TEXT_COLUMNS))
            drop_cols.extend(list(PORT_COLUMNS))
            drop_cols.extend(high_card)
        return drop_cols

    def _add_presence_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = df.columns
        for indicator, patterns in PRESENCE_PATTERNS.items():
            group_cols = []
            for col in columns:
                lower = col.lower()
                if any(p in lower for p in patterns):
                    group_cols.append(col)
            if group_cols:
                # Avoid deprecated applymap by mapping per column
                presence_df = df[group_cols].apply(lambda s: s.map(_is_present))
                present = presence_df.any(axis=1)
                df[indicator] = present.astype(int)
        return df

    def _apply_port_bucketing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if not self.enable_port_bucketing:
            return df

        for col in PORT_COLUMNS:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if fit:
                vc = series.dropna().astype(int).value_counts()
                top_vals = list(vc.head(self.port_top_n).index)
                self.port_top_values[col] = top_vals
            top_vals = self.port_top_values.get(col, [])
            bucket = []
            for val in series:
                if pd.isna(val):
                    bucket.append("unknown")
                else:
                    ival = int(val)
                    if ival in top_vals:
                        bucket.append(f"port_{ival}")
                    elif ival < 1024:
                        bucket.append("well_known")
                    elif ival >= 49152:
                        bucket.append("ephemeral")
                    else:
                        bucket.append("other")
            df[f"{col}_bucket"] = bucket
            df[f"{col}_is_well_known"] = (series < 1024).fillna(False).astype(int)
            df[f"{col}_is_ephemeral"] = (series >= 49152).fillna(False).astype(int)
        return df
