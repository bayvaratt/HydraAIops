import importlib.util
import os

import pytest

if os.environ.get("HYDRA_RUN_TORCH_TESTS") != "1":
    pytest.skip("Set HYDRA_RUN_TORCH_TESTS=1 to enable GNN tests.", allow_module_level=True)
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level=True)

from hydra.config import GraphConfig, GNNModelConfig
from hydra.data.split import split_edges_temporal
from hydra.explain.gnn_explain import explain_edge
from hydra.eval.explainability import coverage
from hydra.models.gnn import build_graph, train_gnn
from tests.conftest import make_graph_df


def test_explain_gnn_outputs():
    df = make_graph_df(n=150)
    graph = build_graph(
        df,
        label_col="label",
        src_col="src_ip",
        dst_col="dst_ip",
        timestamp_col="timestamp",
        cfg=GraphConfig(window_seconds=5),
    )

    train_idx, val_idx, test_idx = split_edges_temporal(
        graph.edge_meta,
        "window_id",
        seed=42,
        test_size=0.2,
        val_size=0.2,
    )
    model, _ = train_gnn(graph, train_idx, val_idx, GNNModelConfig(epochs=2, hidden_dim=16))

    edge_id = int(test_idx[0])
    rec = explain_edge(
        model,
        graph,
        edge_id,
        model_name="gnn",
        feature_names=graph.edge_feature_names,
        feature_regime="behaviour_only",
        dataset_name="unit",
        seed=0,
        top_k=5,
    )

    assert len(rec.top_features) == 5
    assert rec.top_edges is not None
    cov = coverage([rec], [edge_id])
    assert cov > 0.8
