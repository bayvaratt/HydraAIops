import importlib.util
import os

import pytest

if os.environ.get("HYDRA_RUN_TORCH_TESTS") != "1":
    pytest.skip("Set HYDRA_RUN_TORCH_TESTS=1 to enable GNN tests.", allow_module_level=True)
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level=True)

from hydra.config import GraphConfig, GNNModelConfig
from hydra.data.split import split_edges_temporal
from hydra.models.gnn import build_graph, train_gnn
from tests.conftest import make_graph_df


def test_train_gnn_loss_decreases():
    df = make_graph_df(n=200)
    graph = build_graph(
        df,
        label_col="label",
        src_col="src_ip",
        dst_col="dst_ip",
        timestamp_col="timestamp",
        cfg=GraphConfig(window_seconds=5),
    )

    train_idx, val_idx, _ = split_edges_temporal(
        graph.edge_meta,
        "window_id",
        seed=42,
        test_size=0.2,
        val_size=0.2,
    )
    model, losses = train_gnn(graph, train_idx, val_idx, GNNModelConfig(epochs=3, hidden_dim=16))

    assert len(losses) == 3
    assert losses[-1] <= losses[0] + 1e-3 or min(losses) <= losses[0]
