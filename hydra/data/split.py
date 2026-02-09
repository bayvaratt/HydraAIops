from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


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


def _has_both_classes(y: pd.Series, idx: np.ndarray) -> bool:
    return y.iloc[idx].nunique(dropna=True) >= 2


def _group_label_bins(groups: pd.Series, y: pd.Series) -> pd.DataFrame:
    group_means = y.groupby(groups).mean()
    bins = []
    for m in group_means.values:
        if m == 0:
            bins.append("all_neg")
        elif m == 1:
            bins.append("all_pos")
        else:
            bins.append("mixed")
    return pd.DataFrame({"group": group_means.index, "bin": bins})


def _split_groups_stratified(
    group_df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss.split(group_df["group"], group_df["bin"]))
    train_val = group_df.iloc[train_val_idx]

    val_fraction_of_train = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction_of_train, random_state=seed)
    train_idx, val_idx = next(sss2.split(train_val["group"], train_val["bin"]))

    train_groups = train_val.iloc[train_idx]["group"].to_numpy()
    val_groups = train_val.iloc[val_idx]["group"].to_numpy()
    test_groups = group_df.iloc[test_idx]["group"].to_numpy()
    return train_groups, val_groups, test_groups


def _split_groups_random(
    group_df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = group_df["group"].to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_total = len(groups)
    n_test = int(round(n_total * test_size))
    n_val = int(round(n_total * val_size))
    n_train = max(n_total - n_test - n_val, 1)
    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:]
    return train_groups, val_groups, test_groups


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

    group_df = _group_label_bins(groups, y)
    bin_counts = group_df["bin"].value_counts().to_dict()
    logger.info("Group label bins: %s", bin_counts)

    can_stratify = all(v >= 2 for v in bin_counts.values())
    if not can_stratify:
        logger.warning("Insufficient groups per label-bin for stratified group split; using random group split.")

    train_idx = val_idx = test_idx = None
    for attempt in range(50):
        attempt_seed = seed + attempt
        try:
            if can_stratify:
                train_groups, val_groups, test_groups = _split_groups_stratified(
                    group_df, test_size, val_size, attempt_seed
                )
            else:
                train_groups, val_groups, test_groups = _split_groups_random(
                    group_df, test_size, val_size, attempt_seed
                )
        except ValueError:
            continue

        train_idx = np.flatnonzero(groups.isin(train_groups))
        val_idx = np.flatnonzero(groups.isin(val_groups))
        test_idx = np.flatnonzero(groups.isin(test_groups))

        if (
            len(train_idx) > 0
            and len(val_idx) > 0
            and len(test_idx) > 0
            and _has_both_classes(y, train_idx)
            and _has_both_classes(y, val_idx)
            and _has_both_classes(y, test_idx)
        ):
            break
        train_idx = val_idx = test_idx = None

    if train_idx is None or val_idx is None or test_idx is None:
        raise RuntimeError("Host split failed to produce class coverage in all splits after 50 attempts.")

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
