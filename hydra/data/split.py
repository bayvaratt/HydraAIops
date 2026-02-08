from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def _top_groups(groups: pd.Series, idx: np.ndarray, top_n: int = 10):
    counts = groups.iloc[idx].value_counts().head(top_n)
    return list(zip(counts.index.astype(str), counts.values.tolist()))


def check_group_disjointness(
    groups: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    logger,
):
    train_groups = set(groups.iloc[train_idx].unique())
    val_groups = set(groups.iloc[val_idx].unique())
    test_groups = set(groups.iloc[test_idx].unique())

    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    logger.info("Group counts: train=%d val=%d test=%d", len(train_groups), len(val_groups), len(test_groups))
    logger.info("Top train groups: %s", _top_groups(groups, train_idx))
    logger.info("Top val groups: %s", _top_groups(groups, val_idx))
    logger.info("Top test groups: %s", _top_groups(groups, test_idx))

    if overlap:
        sample = list(sorted(overlap))[:20]
        raise RuntimeError(f"Group overlap detected in host split: {sample}")


def split_host(
    df: pd.DataFrame,
    y: pd.Series,
    group_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in dataset")
    groups = df[group_col]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss.split(df, y, groups))

    val_fraction_of_train = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_train, random_state=seed)
    train_idx, val_idx = next(gss2.split(df.iloc[train_val_idx], y.iloc[train_val_idx], groups.iloc[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    check_group_disjointness(groups, train_idx, val_idx, test_idx, logger)
    return train_idx, val_idx, test_idx


def split_temporal(
    df: pd.DataFrame,
    timestamp_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in dataset")
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    order = np.argsort(ts.values)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    logger.info("Temporal split sizes: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))
    return train_idx, val_idx, test_idx


def split_stratified(
    df: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    seed: int,
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss.split(df, y))

    val_fraction_of_train = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction_of_train, random_state=seed)
    train_idx, val_idx = next(sss2.split(df.iloc[train_val_idx], y.iloc[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    logger.warning("Stratified split is NOT deployment-realistic and is provided only as a naive baseline.")
    return train_idx, val_idx, test_idx
