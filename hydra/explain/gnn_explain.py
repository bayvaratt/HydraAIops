from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from hydra.explain.records import ExplanationRecord, default_metadata
from hydra.models.gnn import GraphData

logger = logging.getLogger(__name__)


def integrated_gradients_edge_features(
    model: torch.nn.Module,
    data: GraphData,
    edge_idx: int,
    steps: int = 20,
) -> np.ndarray:
    model.eval()
    baseline = torch.zeros_like(data.edge_attr)
    total_grad = torch.zeros_like(data.edge_attr)

    for alpha in torch.linspace(0, 1, steps):
        edge_attr = baseline + alpha * (data.edge_attr - baseline)
        edge_attr = edge_attr.clone().detach().requires_grad_(True)
        logits = model(data.node_features, data.edge_index, edge_attr)
        target = logits[edge_idx]
        grads = torch.autograd.grad(target, edge_attr, retain_graph=False)[0]
        total_grad += grads

    ig = (data.edge_attr - baseline) * total_grad / steps
    return ig[edge_idx].detach().cpu().numpy()


def edge_importance_by_removal(
    model: torch.nn.Module,
    data: GraphData,
    edge_idx: int,
    top_k: int = 10,
) -> List[dict]:
    model.eval()
    with torch.no_grad():
        base_logits = model(data.node_features, data.edge_index, data.edge_attr)
        base_score = torch.sigmoid(base_logits[edge_idx]).item()

    edge_index = data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    importances = []

    for i in range(num_edges):
        if i == edge_idx:
            continue
        mask = np.ones(num_edges, dtype=bool)
        mask[i] = False
        masked_edge_index = torch.tensor(edge_index[:, mask], dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=data.edge_attr.device)
        masked_edge_attr = data.edge_attr[mask_t]
        new_index = int(np.where(np.flatnonzero(mask) == edge_idx)[0][0])
        with torch.no_grad():
            logits = model(
                data.node_features,
                masked_edge_index.to(data.edge_index.device),
                masked_edge_attr,
            )
            score = torch.sigmoid(logits[new_index]).item()
        importances.append((i, base_score - score))

    importances.sort(key=lambda x: x[1], reverse=True)
    top = []
    for edge_id, importance in importances[:top_k]:
        meta = data.edge_meta.iloc[edge_id]
        top.append(
            {
                "edge_id": int(edge_id),
                "src": str(meta.get("src", "")),
                "dst": str(meta.get("dst", "")),
                "importance": float(importance),
            }
        )
    return top


def explain_edge(
    model: torch.nn.Module,
    data: GraphData,
    edge_idx: int,
    model_name: str,
    feature_names: List[str],
    feature_regime: str,
    dataset_name: str,
    seed: int,
    top_k: int = 10,
) -> ExplanationRecord:
    probs = torch.sigmoid(model(data.node_features, data.edge_index, data.edge_attr)).detach().cpu().numpy()
    ig = integrated_gradients_edge_features(model, data, edge_idx)
    top_features = []
    abs_ig = np.abs(ig)
    norm = abs_ig / (abs_ig.sum() + 1e-8)
    idx = np.argsort(abs_ig)[::-1][:top_k]
    for i in idx:
        direction = "positive" if ig[i] >= 0 else "negative"
        name = feature_names[i] if i < len(feature_names) else str(i)
        top_features.append(
            {
                "feature_name": name,
                "attribution": float(ig[i]),
                "direction": direction,
                "normalized_score": float(norm[i]),
            }
        )

    top_edges = edge_importance_by_removal(model, data, edge_idx, top_k=top_k)

    return ExplanationRecord(
        sample_id=int(edge_idx),
        model_name=model_name,
        pred_label=int(probs[edge_idx] >= 0.5),
        pred_proba=float(probs[edge_idx]),
        top_features=top_features,
        top_edges=top_edges,
        metadata=default_metadata(feature_regime, dataset_name, seed),
    )
