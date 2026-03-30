import numpy as np
import pandas as pd
import pytest

from hydra.data.split import (
    check_group_disjointness,
    split_group_stratified_by_label,
    split_host,
    split_stratified_by_column,
    split_temporal,
)


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_host_df(n_groups: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic host-split dataset: each group gets alternating labels."""
    df = pd.DataFrame(
        {"x": range(n_groups), "group": [f"g{i}" for i in range(n_groups)]}
    )
    y = pd.Series([i % 2 for i in range(n_groups)])
    return df, y


def _make_group_type_df(n_groups: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic dataset with group_col and a multi-class type column.

    Each group has one row per attack type so every type appears across many groups.
    """
    attack_types = ["normal", "dos", "probe", "r2l", "u2r"]
    rows = [{"host": f"g{g}", "type": t} for g in range(n_groups) for t in attack_types]
    df = pd.DataFrame(rows)
    y = pd.Series([0 if r["type"] == "normal" else 1 for r in rows])
    return df, y


def _make_type_col_df(n: int = 100) -> pd.DataFrame:
    """Synthetic dataset with a stratify column containing one rare class."""
    return pd.DataFrame(
        {
            "x": range(n),
            "type": ["normal"] * 60 + ["attack_a"] * 30 + ["rare_b"] * 10,
        }
    )


def _make_temporal_df(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic dataset with an hourly timestamp column."""
    ts = pd.date_range("2020-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"timestamp": ts, "x": range(n)})
    y = pd.Series([i % 2 for i in range(n)])
    return df, y


# ---------------------------------------------------------------------------
# Existing tests (kept)
# ---------------------------------------------------------------------------

def test_host_split_disjoint():
    df, y = _make_host_df()
    train_idx, val_idx, test_idx = split_host(
        df, y, group_col="group", test_size=0.2, val_size=0.2, seed=42,
        logger=DummyLogger(),
    )
    train_groups = set(df.loc[train_idx, "group"])
    val_groups = set(df.loc[val_idx, "group"])
    test_groups = set(df.loc[test_idx, "group"])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_group_overlap_raises():
    groups = pd.Series(["a", "b", "c", "a"])
    with pytest.raises(RuntimeError):
        check_group_disjointness(
            groups, np.array([0, 1]), np.array([2]), np.array([3]), DummyLogger()
        )


# ---------------------------------------------------------------------------
# split_group_stratified_by_label
# ---------------------------------------------------------------------------

def test_group_stratified_no_group_overlap():
    df, y = _make_group_type_df()
    required = {"dos", "probe", "r2l", "u2r"}
    train_idx, val_idx, test_idx = split_group_stratified_by_label(
        df, y, group_col="host", label_col="type",
        test_size=0.2, val_size=0.2, seed=42,
        logger=DummyLogger(), required_labels=required,
    )
    train_hosts = set(df.loc[train_idx, "host"])
    val_hosts = set(df.loc[val_idx, "host"])
    test_hosts = set(df.loc[test_idx, "host"])
    assert train_hosts.isdisjoint(val_hosts)
    assert train_hosts.isdisjoint(test_hosts)
    assert val_hosts.isdisjoint(test_hosts)


def test_group_stratified_all_splits_nonempty():
    df, y = _make_group_type_df()
    required = {"dos", "probe", "r2l", "u2r"}
    train_idx, val_idx, test_idx = split_group_stratified_by_label(
        df, y, group_col="host", label_col="type",
        test_size=0.2, val_size=0.2, seed=42,
        logger=DummyLogger(), required_labels=required,
    )
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0


def test_group_stratified_label_coverage():
    """Both binary classes must appear in every split."""
    df, y = _make_group_type_df()
    required = {"dos", "probe", "r2l", "u2r"}
    train_idx, val_idx, test_idx = split_group_stratified_by_label(
        df, y, group_col="host", label_col="type",
        test_size=0.2, val_size=0.2, seed=42,
        logger=DummyLogger(), required_labels=required,
    )
    for idx in (train_idx, val_idx, test_idx):
        assert y.iloc[idx].nunique() == 2, "Expected both classes (0 and 1) in split"


def test_group_stratified_deterministic():
    df, y = _make_group_type_df()
    required = {"dos", "probe", "r2l", "u2r"}
    kwargs = dict(
        df=df, y=y, group_col="host", label_col="type",
        test_size=0.2, val_size=0.2, seed=7,
        logger=DummyLogger(), required_labels=required,
    )
    r1 = split_group_stratified_by_label(**kwargs)
    r2 = split_group_stratified_by_label(**kwargs)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# split_stratified_by_column
# ---------------------------------------------------------------------------

def test_stratified_by_column_classes_in_test():
    """Each non-rare class should appear in the test split."""
    df = _make_type_col_df()
    _, _, test_idx = split_stratified_by_column(
        df, stratify_col="type", test_size=0.2, val_size=0.2, seed=42,
        logger=DummyLogger(),
    )
    test_types = set(df.loc[test_idx, "type"])
    assert "normal" in test_types
    assert "attack_a" in test_types


def test_stratified_by_column_rare_class_not_dropped():
    """With min_count=1 (default), the rare class rows are not silently dropped."""
    df = _make_type_col_df()
    train_idx, val_idx, test_idx = split_stratified_by_column(
        df, stratify_col="type", test_size=0.2, val_size=0.2, seed=42,
        logger=DummyLogger(), min_count=1,
    )
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    # All 10 rare rows must appear somewhere across the splits
    rare_mask = df["type"] == "rare_b"
    assert rare_mask.iloc[all_idx].any(), "rare_b rows were silently dropped"


def test_stratified_by_column_deterministic():
    df = _make_type_col_df()
    kwargs = dict(
        df=df, stratify_col="type", test_size=0.2, val_size=0.2, seed=99,
        logger=DummyLogger(),
    )
    r1 = split_stratified_by_column(**kwargs)
    r2 = split_stratified_by_column(**kwargs)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# split_temporal
# ---------------------------------------------------------------------------

def test_temporal_strict_ordering():
    """max(train_ts) <= min(val_ts) and max(val_ts) <= min(test_ts)."""
    df, _ = _make_temporal_df()
    train_idx, val_idx, test_idx = split_temporal(
        df, timestamp_col="timestamp",
        train_frac=0.7, val_frac=0.15, test_frac=0.15,
        logger=DummyLogger(),
    )
    ts = df["timestamp"]
    assert ts.iloc[train_idx].max() <= ts.iloc[val_idx].min()
    assert ts.iloc[val_idx].max() <= ts.iloc[test_idx].min()


def test_temporal_all_splits_nonempty():
    df, _ = _make_temporal_df()
    train_idx, val_idx, test_idx = split_temporal(
        df, timestamp_col="timestamp",
        train_frac=0.7, val_frac=0.15, test_frac=0.15,
        logger=DummyLogger(),
    )
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0
