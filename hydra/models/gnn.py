from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from hydra.config import GraphConfig, GNNModelConfig

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import SAGEConv
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False


@dataclass
class GraphData:
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_label: torch.Tensor
    node_features: torch.Tensor
    edge_meta: pd.DataFrame
    node_mapping: Dict[str, int]
    edge_feature_names: List[str]

    def to(self, device: torch.device) -> "GraphData":
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.edge_label = self.edge_label.to(device)
        self.node_features = self.node_features.to(device)
        return self


class EdgeSAGEClassifier(torch.nn.Module):
    def __init__(self, node_in: int, edge_in: int, cfg: GNNModelConfig):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        hidden = cfg.hidden_dim
        self.convs.append(SAGEConv(node_in, hidden))
        for _ in range(cfg.num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2 + edge_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
        return self.edge_mlp(edge_input).squeeze(1)


class EdgeMLPClassifier(torch.nn.Module):
    def __init__(self, edge_in: int, cfg: GNNModelConfig):
        super().__init__()
        hidden = cfg.hidden_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.mlp(edge_attr).squeeze(1)


def build_graph(
    df: pd.DataFrame,
    label_col: str,
    src_col: str,
    dst_col: str,
    timestamp_col: Optional[str],
    cfg: GraphConfig,
) -> GraphData:
    df = df.copy()
    if src_col not in df.columns:
        fallback_src = _infer_src_col(df)
        if fallback_src is None:
            logger.warning("src_col '%s' missing; falling back to row index", src_col)
            df["_src_id"] = np.arange(len(df))
            src_col = "_src_id"
        else:
            logger.warning("src_col '%s' missing; using '%s'", src_col, fallback_src)
            src_col = fallback_src
    if dst_col not in df.columns:
        fallback_dst = _infer_dst_col(df)
        if fallback_dst is None:
            logger.warning("dst_col '%s' missing; falling back to row index", dst_col)
            df["_dst_id"] = np.arange(len(df))
            dst_col = "_dst_id"
        else:
            logger.warning("dst_col '%s' missing; using '%s'", dst_col, fallback_dst)
            dst_col = fallback_dst
    if timestamp_col and timestamp_col in df.columns:
        ts = pd.to_numeric(df[timestamp_col], errors="coerce").fillna(0).astype(int)
    else:
        logger.warning("Timestamp column missing; using row index as time")
        ts = pd.Series(np.arange(len(df)))
        timestamp_col = "timestamp"
        df[timestamp_col] = ts

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    window = (ts // max(cfg.window_seconds, 1)).astype(int)
    df["window_id"] = window

    numeric_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {label_col, timestamp_col, "window_id"}]

    grouped = (
        df.groupby(["window_id", src_col, dst_col])
        .agg(**{f"{k}_{cfg.edge_agg}": (k, cfg.edge_agg) for k in numeric_cols}, flow_count=("window_id", "size"))
        .reset_index()
    )

    label_grouped = (
        df.groupby(["window_id", src_col, dst_col])[label_col]
        .max()
        .reset_index()
        .rename(columns={label_col: "edge_label"})
    )

    edges = grouped.merge(label_grouped, on=["window_id", src_col, dst_col], how="left")
    if cfg.max_edges is not None and len(edges) > cfg.max_edges:
        edges = edges.sample(n=cfg.max_edges, random_state=42).reset_index(drop=True)

    nodes = pd.unique(edges[[src_col, dst_col]].values.ravel("K"))
    node_mapping = {str(node): idx for idx, node in enumerate(nodes)}

    src_ids = edges[src_col].astype(str).map(node_mapping).astype(int).to_numpy()
    dst_ids = edges[dst_col].astype(str).map(node_mapping).astype(int).to_numpy()
    edge_index = torch.tensor(np.vstack([src_ids, dst_ids]), dtype=torch.long)

    feature_cols = [c for c in edges.columns if c not in {"window_id", src_col, dst_col, "edge_label"}]
    edge_attr = torch.tensor(edges[feature_cols].to_numpy(dtype=float), dtype=torch.float32)
    edge_label = torch.tensor(edges["edge_label"].to_numpy(dtype=float), dtype=torch.float32)

    node_features = _build_node_features(len(nodes), edge_index, mode=cfg.node_feature_mode)

    edge_meta = edges[["window_id", src_col, dst_col, "edge_label"]].copy()
    edge_meta = edge_meta.rename(columns={src_col: "src", dst_col: "dst", "edge_label": "label"})
    edge_meta["edge_id"] = np.arange(len(edge_meta))

    return GraphData(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        node_features=node_features,
        edge_meta=edge_meta,
        node_mapping=node_mapping,
        edge_feature_names=feature_cols,
    )


def _infer_src_col(df: pd.DataFrame) -> Optional[str]:
    if "Protocol Type" in df.columns:
        return "Protocol Type"
    for col in df.columns:
        if "protocol" in str(col).lower():
            return col
    return None


def _infer_dst_col(df: pd.DataFrame) -> Optional[str]:
    if "app_proto" in df.columns:
        return "app_proto"
    app_cols = [c for c in ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC"] if c in df.columns]
    if app_cols:
        app_df = df[app_cols].fillna(0)
        has_any = app_df.eq(1).any(axis=1)
        app_name = app_df.eq(1).idxmax(axis=1)
        df["app_proto"] = app_name.where(has_any, "NONE")
        return "app_proto"
    return None


def _build_node_features(num_nodes: int, edge_index: torch.Tensor, mode: str = "degree") -> torch.Tensor:
    if mode == "zeros":
        return torch.zeros((num_nodes, 1), dtype=torch.float32)
    src, dst = edge_index
    out_deg = torch.bincount(src, minlength=num_nodes).float().unsqueeze(1)
    in_deg = torch.bincount(dst, minlength=num_nodes).float().unsqueeze(1)
    feat = torch.cat([out_deg, in_deg], dim=1)
    return feat


def build_model(edge_feat_dim: int, node_feat_dim: int, cfg: GNNModelConfig):
    if PYG_AVAILABLE:
        return EdgeSAGEClassifier(node_feat_dim, edge_feat_dim, cfg)
    logger.warning("torch_geometric not available; using edge-only MLP")
    return EdgeMLPClassifier(edge_feat_dim, cfg)


def train_gnn(
    data: GraphData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: GNNModelConfig,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, List[float]]:
    device = device or torch.device("cpu")
    data = data.to(device)

    model = build_model(data.edge_attr.shape[1], data.node_features.shape[1], cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    losses: List[float] = []
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    for _ in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.node_features, data.edge_index, data.edge_attr)
        loss = loss_fn(logits[train_idx_t], data.edge_label[train_idx_t])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if len(val_idx_t) > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(data.node_features, data.edge_index, data.edge_attr)
                _ = loss_fn(val_logits[val_idx_t], data.edge_label[val_idx_t]).item()

    return model, losses


def predict_proba(model: torch.nn.Module, data: GraphData) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(data.node_features, data.edge_index, data.edge_attr)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs
