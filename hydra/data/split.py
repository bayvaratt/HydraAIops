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


def _check_split_invariants(
    y: "pd.Series | None",
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    logger,
) -> None:
    """Fail-fast post-split checks: non-empty splits, no NaN labels (if y given), binary class warning."""
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        assert len(idx) > 0, f"Split '{name}' is empty after splitting"
        if y is not None:
            nan_count = int(y.iloc[idx].isna().sum())
            assert nan_count == 0, f"Split '{name}' has {nan_count} NaN labels"
            if not _has_both_classes(y, idx):
                logger.warning(
                    "Split '%s' does not contain both classes; downstream evaluation may be degenerate.", name
                )


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
    split_assertions: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in dataset")
    groups = df[group_col]

    group_df = _group_label_bins(groups, y)
    bin_counts = group_df["bin"].value_counts().to_dict()
    logger.info("Group label bins: %s", bin_counts)
    overall_prev = float(y.mean()) if len(y) else 0.0

    can_stratify = all(v >= 2 for v in bin_counts.values())
    if not can_stratify:
        logger.warning("Insufficient groups per label-bin for stratified group split; using random group split.")

    train_idx = val_idx = test_idx = None
    min_prev = 0.05
    max_prev = 0.95
    max_gap = 0.20
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

        if len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0:
            train_prev = float(y.iloc[train_idx].mean())
            val_prev = float(y.iloc[val_idx].mean())
            test_prev = float(y.iloc[test_idx].mean())
            prevalence_ok = all(
                [
                    min_prev <= train_prev <= max_prev,
                    min_prev <= val_prev <= max_prev,
                    min_prev <= test_prev <= max_prev,
                    abs(train_prev - overall_prev) <= max_gap,
                    abs(val_prev - overall_prev) <= max_gap,
                    abs(test_prev - overall_prev) <= max_gap,
                ]
            )
            if (
                prevalence_ok
                and _has_both_classes(y, train_idx)
                and _has_both_classes(y, val_idx)
                and _has_both_classes(y, test_idx)
            ):
                break
        train_idx = val_idx = test_idx = None

    if train_idx is None or val_idx is None or test_idx is None:
        raise RuntimeError(
            "Host split failed to meet class coverage/prevalence constraints after 50 attempts."
        )

    check_group_disjointness(groups, train_idx, val_idx, test_idx, logger)
    if split_assertions:
        _check_split_invariants(y, train_idx, val_idx, test_idx, logger)
    return train_idx, val_idx, test_idx


def split_temporal(
    df: pd.DataFrame,
    timestamp_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    logger,
    split_assertions: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not timestamp_col or timestamp_col not in df.columns:
        logger.warning("Timestamp column missing; using row order as temporal proxy.")
        order = np.arange(len(df))
    else:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        order = np.argsort(ts.values)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    logger.info("Temporal split sizes: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))
    if split_assertions:
        _check_split_invariants(None, train_idx, val_idx, test_idx, logger)
    return train_idx, val_idx, test_idx


def split_stratified(
    df: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    seed: int,
    logger,
    split_assertions: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss.split(df, y))

    val_fraction_of_train = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction_of_train, random_state=seed)
    train_idx, val_idx = next(sss2.split(df.iloc[train_val_idx], y.iloc[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    logger.warning("Stratified split is NOT deployment-realistic and is provided only as a naive baseline.")
    if split_assertions:
        _check_split_invariants(y, train_idx, val_idx, test_idx, logger)
    return train_idx, val_idx, test_idx


def split_stratified_by_column(
    df: pd.DataFrame,
    stratify_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    logger,
    min_count: int = 1,
    other_label: str = "__other__",
    split_assertions: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stratify_col not in df.columns:
        raise KeyError(f"Stratify column '{stratify_col}' not found in dataset")
    labels = df[stratify_col].astype("object").where(df[stratify_col].notna(), "__missing__")
    value_counts = labels.value_counts(dropna=False)
    rare = value_counts[value_counts < min_count].index.tolist()
    if rare:
        logger.warning(
            "Stratify column '%s' has rare classes (<%d); grouping into '%s': %s",
            stratify_col,
            min_count,
            other_label,
            rare,
        )
        labels = labels.where(~labels.isin(rare), other_label)
        value_counts = labels.value_counts(dropna=False)
    if value_counts.min() < 2:
        raise RuntimeError(
            f"Stratify column '{stratify_col}' has classes with <2 samples after grouping; "
            "cannot stratify reliably."
        )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss.split(df, labels))

    val_fraction_of_train = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction_of_train, random_state=seed)
    train_idx, val_idx = next(sss2.split(df.iloc[train_val_idx], labels.iloc[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    logger.warning(
        "Type-stratified split uses '%s' to balance classes across splits; this is not deployment-realistic.",
        stratify_col,
    )
    if split_assertions:
        _check_split_invariants(None, train_idx, val_idx, test_idx, logger)
    return train_idx, val_idx, test_idx


def split_group_stratified_by_label(
    df: pd.DataFrame,
    y: pd.Series,
    group_col: str,
    label_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    logger,
    required_labels: set | None = None,
    max_attempts: int = 200,
    split_assertions: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in dataset")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataset")

    groups = df[group_col]
    labels = df[label_col].astype("object").where(df[label_col].notna(), "__missing__")
    all_labels = set(labels.dropna().unique().tolist())
    required = set(required_labels) if required_labels is not None else set(all_labels)
    if not required:
        raise RuntimeError("No required labels provided for group-stratified split.")

    unique_groups = groups.unique()
    n_total = len(unique_groups)
    n_test = int(round(n_total * test_size))
    n_val = int(round(n_total * val_size))
    n_test = max(n_test, 1)
    n_val = max(n_val, 1)
    n_train = max(n_total - n_test - n_val, 1)

    # Minimum absolute counts per split — flat count, not fraction.
    # TON_IoT has skewed per-host distributions: "attacker" hosts are ~100% attack,
    # "normal" hosts are ~100% benign. A fraction-based bound (e.g. 0.5%) requires
    # >1000 positives in val, which is impossible when all remaining hosts are normal.
    # 50 positives in val is enough for meaningful threshold calibration.
    MIN_POS = 50
    MIN_NEG = 50

    def _labels_in(idx: np.ndarray) -> set:
        return set(labels.iloc[idx].dropna().unique().tolist())

    def _counts_ok(idx: np.ndarray) -> bool:
        yy = y.iloc[idx]
        return int((yy == 1).sum()) >= MIN_POS and int((yy == 0).sum()) >= MIN_NEG

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        shuffled = rng.permutation(unique_groups)
        train_groups = shuffled[:n_train]
        val_groups = shuffled[n_train:n_train + n_val]
        test_groups = shuffled[n_train + n_val:n_train + n_val + n_test]

        train_idx = np.flatnonzero(groups.isin(train_groups))
        val_idx = np.flatnonzero(groups.isin(val_groups))
        test_idx = np.flatnonzero(groups.isin(test_groups))

        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            continue

        if not (_counts_ok(train_idx) and _counts_ok(val_idx) and _counts_ok(test_idx)):
            continue

        labels_train = _labels_in(train_idx)
        labels_test = _labels_in(test_idx)
        if required.issubset(labels_train) and required.issubset(labels_test):
            logger.info(
                "Group-stratified split: groups by '%s', type-stratified by '%s'. "
                "Group-disjoint — deployment-realistic.",
                group_col,
                label_col,
            )
            check_group_disjointness(groups, train_idx, val_idx, test_idx, logger)
            if split_assertions:
                _check_split_invariants(y, train_idx, val_idx, test_idx, logger)
            return train_idx, val_idx, test_idx

    raise RuntimeError(
        "Group-stratified split failed to find a valid split (prevalence 5–95% in all splits + "
        "all required labels in train and test) "
        f"after {max_attempts} attempts. Consider relaxing constraints."
    )


def split_host_type_aware(
    df: pd.DataFrame,
    y: pd.Series,
    group_col: str,
    type_col: str,
    normal_type_value: str,
    test_size: float,
    val_size: float,
    seed: int,
    logger,
    max_attempts: int = 50,
    split_assertions: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host-disjoint split that guarantees all attack types appear in train.

    Algorithm
    ---------
    1. For each attack type, find the host with the most samples of that type.
    2. Greedily pin one representative host per type to train (rarest type
       first so covering a rare type also covers common co-located types).
    3. Fill remaining train slots randomly from the unpinned hosts.
    4. Assign leftover hosts to val/test randomly, retrying until both splits
       have ≥ MIN_POS positive and ≥ MIN_NEG negative samples.

    This is strictly stronger than `split_host` (all types in train guaranteed)
    while avoiding the infeasibility of `split_group_stratified_by_label`
    (which also requires all types in test — impossible for TON_IoT).
    """
    if group_col not in df.columns:
        raise KeyError(f"group_col '{group_col}' not found")
    if type_col not in df.columns:
        raise KeyError(f"type_col '{type_col}' not found")

    groups = df[group_col]
    types = df[type_col]
    unique_groups = list(groups.unique())
    n_total = len(unique_groups)
    n_test = max(int(round(n_total * test_size)), 1)
    n_val = max(int(round(n_total * val_size)), 1)
    n_train = max(n_total - n_test - n_val, 1)

    MIN_POS = 50
    MIN_NEG = 50

    # Build map: attack_type → [(host, count), ...] sorted by count desc
    attack_mask = types != normal_type_value
    type_host_counts: dict[str, list[tuple]] = {}
    for host in unique_groups:
        host_mask = groups == host
        for t, c in types[host_mask & attack_mask].value_counts().items():
            type_host_counts.setdefault(str(t), []).append((host, int(c)))
    for t in type_host_counts:
        type_host_counts[t].sort(key=lambda x: -x[1])

    # Greedy pin — rarest type (fewest hosting hosts) first
    pinned: set = set()
    covered: set = set()
    for t in sorted(type_host_counts, key=lambda t: len(type_host_counts[t])):
        if t in covered:
            continue
        # Prefer already-pinned hosts (free coverage); else pin the best host
        chosen = next(
            (h for h, _ in type_host_counts[t] if h in pinned),
            type_host_counts[t][0][0],  # best host by sample count
        )
        pinned.add(chosen)
        # Mark all types this host covers as covered
        for t2, hlist in type_host_counts.items():
            if any(h == chosen for h, _ in hlist):
                covered.add(t2)

    logger.info(
        "Type-aware host split: %d hosts pinned to train for attack-type coverage: %s",
        len(pinned),
        sorted(str(h) for h in pinned),
    )
    if len(pinned) > n_train:
        raise RuntimeError(
            f"Need {len(pinned)} pinned hosts but only {n_train} train slots available."
        )

    remaining = [h for h in unique_groups if h not in pinned]

    def _counts_ok(idx: np.ndarray) -> bool:
        yy = y.iloc[idx]
        return int((yy == 1).sum()) >= MIN_POS and int((yy == 0).sum()) >= MIN_NEG

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        shuffled = list(rng.permutation(remaining))

        n_extra_train = n_train - len(pinned)
        train_groups = pinned | set(shuffled[:n_extra_train])
        leftover = shuffled[n_extra_train:]
        val_groups = set(leftover[:n_val])
        test_groups = set(leftover[n_val: n_val + n_test])

        train_idx = np.flatnonzero(groups.isin(train_groups))
        val_idx = np.flatnonzero(groups.isin(val_groups))
        test_idx = np.flatnonzero(groups.isin(test_groups))

        if not (len(train_idx) and len(val_idx) and len(test_idx)):
            continue
        if _counts_ok(train_idx) and _counts_ok(val_idx) and _counts_ok(test_idx):
            check_group_disjointness(groups, train_idx, val_idx, test_idx, logger)
            if split_assertions:
                _check_split_invariants(y, train_idx, val_idx, test_idx, logger)
            return train_idx, val_idx, test_idx

    raise RuntimeError(
        f"Type-aware host split failed to find valid val/test prevalence after {max_attempts} attempts."
    )
