from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logger = logging.getLogger(__name__)


def validate_group_disjointness(
    split_indices: Dict[str, np.ndarray],
    group_series: pd.Series,
    name: str,
) -> None:
    def _groups(indices: np.ndarray) -> pd.Series:
        return group_series.loc[indices].fillna("__MISSING__")

    train_groups = set(_groups(split_indices["train"]).unique())
    val_groups = set(_groups(split_indices["val"]).unique())
    test_groups = set(_groups(split_indices["test"]).unique())

    inter_train_val = train_groups & val_groups
    inter_train_test = train_groups & test_groups
    inter_val_test = val_groups & test_groups

    logger.info(
        "[%s] group counts: train=%d val=%d test=%d",
        name,
        len(train_groups),
        len(val_groups),
        len(test_groups),
    )
    logger.info(
        "[%s] group overlaps: train∩val=%d train∩test=%d val∩test=%d",
        name,
        len(inter_train_val),
        len(inter_train_test),
        len(inter_val_test),
    )

    def _top_groups(indices: np.ndarray) -> Dict[str, int]:
        counts = _groups(indices).value_counts().head(10)
        return {str(k): int(v) for k, v in counts.items()}

    logger.info("[%s] top-10 train groups: %s", name, _top_groups(split_indices["train"]))
    logger.info("[%s] top-10 val groups: %s", name, _top_groups(split_indices["val"]))
    logger.info("[%s] top-10 test groups: %s", name, _top_groups(split_indices["test"]))

    overlap = sorted(
        {
            *inter_train_val,
            *inter_train_test,
            *inter_val_test,
        }
    )
    if overlap:
        sample = overlap[:20]
        counts_train = _groups(split_indices["train"]).value_counts()
        counts_val = _groups(split_indices["val"]).value_counts()
        counts_test = _groups(split_indices["test"]).value_counts()
        sample_counts = {
            str(g): {
                "train": int(counts_train.get(g, 0)),
                "val": int(counts_val.get(g, 0)),
                "test": int(counts_test.get(g, 0)),
            }
            for g in sample
        }
        raise RuntimeError(
            f"[{name}] Group overlap detected across splits. "
            f"train∩val={len(inter_train_val)} "
            f"train∩test={len(inter_train_test)} "
            f"val∩test={len(inter_val_test)}. "
            f"Example groups (counts): {sample_counts}"
        )


def split_tabular(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
    strategy: str = "stratified",
    timestamp_col: Optional[str] = None,
    group_col: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    return_indices: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if strategy == "temporal" and timestamp_col and timestamp_col in df.columns:
        logger.info("Using temporal split on %s", timestamp_col)
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df_sorted)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))
        test = df_sorted.iloc[-n_test:]
        val = df_sorted.iloc[-(n_test + n_val):-n_test]
        train = df_sorted.iloc[: -(n_test + n_val)]
        train_idx = train.index.to_numpy()
        val_idx = val.index.to_numpy()
        test_idx = test.index.to_numpy()
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)
        if return_indices:
            return train, val, test, {"train": train_idx, "val": val_idx, "test": test_idx}
        return train, val, test

    if strategy == "host":
        if not group_col or group_col not in df.columns:
            raise ValueError("Host-based split requires a valid group_col present in the dataframe")
        logger.info("Using host-based split on %s", group_col)
        groups = df[group_col].fillna("__MISSING__")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(df, groups=groups))
        train_val = df.iloc[train_val_idx]
        test = df.iloc[test_idx]

        groups_tv = groups.iloc[train_val_idx]
        val_relative = val_size / (1 - test_size)
        splitter_tv = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
        train_idx_rel, val_idx_rel = next(splitter_tv.split(train_val, groups=groups_tv))
        train = train_val.iloc[train_idx_rel]
        val = train_val.iloc[val_idx_rel]
        train_idx = train.index.to_numpy()
        val_idx = val.index.to_numpy()
        test_idx = test.index.to_numpy()
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)
        if return_indices:
            return train, val, test, {"train": train_idx, "val": val_idx, "test": test_idx}
        return train, val, test

    logger.info("Using stratified split")
    stratify = df[label_col] if label_col in df.columns and df[label_col].nunique() > 1 else None
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    stratify_tv = train_val[label_col] if stratify is not None else None
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=seed,
        stratify=stratify_tv,
    )
    train_idx = train.index.to_numpy()
    val_idx = val.index.to_numpy()
    test_idx = test.index.to_numpy()
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if return_indices:
        return train, val, test, {"train": train_idx, "val": val_idx, "test": test_idx}
    return train, val, test


def split_edges_temporal(
    edges: pd.DataFrame,
    timestamp_col: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if timestamp_col not in edges.columns:
        raise ValueError(f"Missing timestamp column {timestamp_col} for temporal split")
    edges_sorted = edges.sort_values(timestamp_col)
    n = len(edges_sorted)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))
    test_idx = edges_sorted.index[-n_test:].to_numpy()
    val_idx = edges_sorted.index[-(n_test + n_val):-n_test].to_numpy()
    train_idx = edges_sorted.index[: -(n_test + n_val)].to_numpy()
    return train_idx, val_idx, test_idx


def split_edges_entity(
    edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    nodes = pd.unique(edges[[src_col, dst_col]].values.ravel("K"))
    rng.shuffle(nodes)
    n = len(nodes)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))
    test_nodes = set(nodes[:n_test])
    val_nodes = set(nodes[n_test : n_test + n_val])

    test_mask = edges[src_col].isin(test_nodes) | edges[dst_col].isin(test_nodes)
    val_mask = edges[src_col].isin(val_nodes) | edges[dst_col].isin(val_nodes)
    train_mask = ~(test_mask | val_mask)

    train_idx = edges.index[train_mask].to_numpy()
    val_idx = edges.index[val_mask].to_numpy()
    test_idx = edges.index[test_mask].to_numpy()

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        logger.warning("Entity split produced empty split; falling back to temporal split")
        if "window_id" in edges.columns:
            return split_edges_temporal(edges, "window_id", seed, test_size, val_size)
        if "timestamp" in edges.columns:
            return split_edges_temporal(edges, "timestamp", seed, test_size, val_size)
    return train_idx, val_idx, test_idx
