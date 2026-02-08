import numpy as np
import pandas as pd
import pytest

from hydra.data.split import check_group_disjointness, split_host


def test_host_split_disjoint():
    df = pd.DataFrame({"x": range(100), "group": [f"g{i}" for i in range(100)]})
    y = pd.Series([0, 1] * 50)

    train_idx, val_idx, test_idx = split_host(
        df,
        y,
        group_col="group",
        test_size=0.2,
        val_size=0.2,
        seed=42,
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
        check_group_disjointness(groups, np.array([0, 1]), np.array([2]), np.array([3]), DummyLogger())


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass
