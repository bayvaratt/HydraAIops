from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from hydra.config import DatasetConfig, resolve_path

logger = logging.getLogger(__name__)


def load_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    path = resolve_path(path)
    logger.info("Loading CSV: %s", path)
    df = pd.read_csv(path, nrows=nrows)
    return df


def load_dataset(cfg: DatasetConfig, nrows: Optional[int] = None) -> pd.DataFrame:
    if not Path(cfg.path).exists():
        logger.warning("Dataset path does not exist: %s", cfg.path)
    df = load_csv(cfg.path, nrows=nrows)
    if cfg.name == "cic_iot2023":
        if "label" in df.columns and "label_binary" not in df.columns:
            if is_string_dtype(df["label"]) or is_object_dtype(df["label"]):
                df["label_binary"] = (df["label"] != "BenignTraffic").astype(int)
            else:
                df["label_binary"] = df["label"].astype(int)
    return df


def sample_dataframe(
    df: pd.DataFrame,
    nrows: Optional[int],
    label_col: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    if nrows is None or len(df) <= nrows:
        return df
    if label_col and label_col in df.columns and df[label_col].nunique() > 1:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=nrows,
            random_state=seed,
        )
        idx, _ = next(splitter.split(df, df[label_col]))
        return df.iloc[idx].reset_index(drop=True)
    return df.sample(n=nrows, random_state=seed).reset_index(drop=True)
