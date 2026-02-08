from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp

from hydra.explain.records import ExplanationRecord

logger = logging.getLogger(__name__)


def coverage(records: Sequence[ExplanationRecord], alert_ids: Sequence[int]) -> float:
    alert_ids = set(int(i) for i in alert_ids)
    explained = {int(r.sample_id) for r in records}
    if not alert_ids:
        return 0.0
    return float(len(explained & alert_ids) / len(alert_ids))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def stability_jaccard(records_by_seed: Dict[int, List[ExplanationRecord]], top_k: int = 10) -> Dict[str, float]:
    seeds = sorted(records_by_seed.keys())
    if len(seeds) < 2:
        return {"mean": 1.0, "std": 0.0}

    per_pair = []
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            map_a = {r.sample_id: [f["feature_name"] for f in r.top_features[:top_k]] for r in records_by_seed[seed_a]}
            map_b = {r.sample_id: [f["feature_name"] for f in r.top_features[:top_k]] for r in records_by_seed[seed_b]}
            common = set(map_a.keys()) & set(map_b.keys())
            if not common:
                continue
            scores = [jaccard(map_a[s], map_b[s]) for s in common]
            per_pair.append(np.mean(scores))

    if not per_pair:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(per_pair)), "std": float(np.std(per_pair))}


def fidelity_perturbation_tabular(
    model,
    X,
    feature_names: List[str],
    records: Sequence[ExplanationRecord],
    top_k: int = 10,
) -> Dict[str, float]:
    if not records:
        return {"mean_drop": 0.0, "mean_drop_random": 0.0}

    n_features = len(feature_names)
    drops = []
    drops_random = []
    rng = np.random.RandomState(42)

    for rec in records:
        idx = rec.sample_id
        top_features = [f["feature_name"] for f in rec.top_features[:top_k]]
        feature_idx = [feature_names.index(f) for f in top_features if f in feature_names]
        if not feature_idx:
            continue
        rand_idx = rng.choice(n_features, size=len(feature_idx), replace=False)

        row = X[idx]
        if sp.issparse(row):
            row = row.toarray().ravel()
        else:
            row = np.asarray(row).ravel()

        base_proba = _predict_proba_row(model, row)

        perturbed = row.copy()
        perturbed[feature_idx] = 0.0
        drop = base_proba - _predict_proba_row(model, perturbed)
        drops.append(drop)

        perturbed_rand = row.copy()
        perturbed_rand[rand_idx] = 0.0
        drop_rand = base_proba - _predict_proba_row(model, perturbed_rand)
        drops_random.append(drop_rand)

    return {
        "mean_drop": float(np.mean(drops) if drops else 0.0),
        "mean_drop_random": float(np.mean(drops_random) if drops_random else 0.0),
    }


def _predict_proba_row(model, row: np.ndarray) -> float:
    row_2d = row.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_2d)
        return float(proba[0, 1])
    if hasattr(model, "decision_function"):
        score = model.decision_function(row_2d)
        score = float(score[0])
        return float(1 / (1 + np.exp(-score)))
    pred = model.predict(row_2d)
    return float(pred[0])


def fidelity_gnn_edge_removal(
    model,
    data,
    records: Sequence[ExplanationRecord],
    top_k: int = 10,
) -> Dict[str, float]:
    if not records:
        return {"mean_drop": 0.0, "flip_rate": 0.0}

    import torch

    model.eval()
    with torch.no_grad():
        base_logits = model(data.node_features, data.edge_index, data.edge_attr)
        base_probs = torch.sigmoid(base_logits).cpu().numpy()

    drops = []
    flips = []
    edge_index = data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]

    for rec in records:
        edge_id = rec.sample_id
        top_edges = rec.top_edges or []
        remove_ids = [int(e["edge_id"]) for e in top_edges[:top_k] if "edge_id" in e]
        mask = np.ones(num_edges, dtype=bool)
        for rid in remove_ids:
            if 0 <= rid < num_edges:
                mask[rid] = False
        masked_edge_index = torch.tensor(edge_index[:, mask], dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=data.edge_attr.device)
        masked_edge_attr = data.edge_attr[mask_t]
        if edge_id not in np.flatnonzero(mask):
            continue
        new_index = int(np.where(np.flatnonzero(mask) == edge_id)[0][0])
        with torch.no_grad():
            logits = model(
                data.node_features,
                masked_edge_index.to(data.edge_index.device),
                masked_edge_attr,
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        drop = base_probs[edge_id] - probs[new_index]
        drops.append(drop)
        flips.append(int((base_probs[edge_id] >= 0.5) != (probs[new_index] >= 0.5)))

    return {
        "mean_drop": float(np.mean(drops) if drops else 0.0),
        "flip_rate": float(np.mean(flips) if flips else 0.0),
    }


def operational_plausibility(records: Sequence[ExplanationRecord], top_k: int = 10) -> Dict[str, int]:
    counter: Counter = Counter()
    for rec in records:
        for feat in rec.top_features[:top_k]:
            counter[feat["feature_name"]] += 1
    return dict(counter.most_common(10))
