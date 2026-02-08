import importlib.util
import os

import pytest

if os.environ.get("HYDRA_RUN_TORCH_TESTS") != "1":
    pytest.skip("Set HYDRA_RUN_TORCH_TESTS=1 to enable GNN tests.", allow_module_level=True)
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level=True)

from hydra.config import GraphConfig
from hydra.models.gnn import build_graph
from tests.conftest import make_graph_df


def test_graph_build_outputs():
    df = make_graph_df(n=120)
    graph = build_graph(
        df,
        label_col="label",
        src_col="src_ip",
        dst_col="dst_ip",
        timestamp_col="timestamp",
        cfg=GraphConfig(window_seconds=10),
    )

    assert graph.edge_attr.shape[0] > 0
    assert graph.edge_attr.shape[1] == len(graph.edge_feature_names)
    assert graph.edge_index.shape[0] == 2
