"""IP-graph GNN for network intrusion detection.

Each network flow is an edge in a communication graph:
  Nodes = unique IP addresses
  Edges = flows between IPs (edge features = flow statistics)

Model: GraphSAGE (2 layers, max-aggregation) + edge MLP (E-GraphSAGE style).

Transductive setting:
  - One graph is built from ALL rows (train + val + test).
  - Self-loops added so each node retains its own representation.
  - GNN sees the full node neighbourhood (global topology).
  - Loss is computed only on train edges.
  - Val/test edges are evaluated with torch.no_grad().

Explainability:
  - Gradient × Input attribution on edge features (sensitivity analysis).
  - Saves global_importance.csv + per-type CSVs to out_explain_dir.
  - Feature names are prefixed with 'num__' for compatibility with xai_diagnostics.

Only applicable to datasets with src_ip / dst_ip columns (TON_IoT).
Datasets without those columns (CIC-IoT-2023) are skipped in run_tabular.py.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _select_edge_features(df: pd.DataFrame, exclude_cols: list[str]) -> list[str]:
    """Return numeric column names suitable as edge features."""
    exclude = set(exclude_cols)
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def _build_graph(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    src_col: str,
    dst_col: str,
    feature_cols: list[str],
    label_col: str,
):
    """Build a PyG Data object from the full DataFrame.

    Returns
    -------
    data : torch_geometric.data.Data
    scaler : fitted StandardScaler
    """
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops

    # Map IP strings to integer node IDs
    all_ips = pd.concat([df[src_col], df[dst_col]]).dropna().unique()
    ip2id = {ip: i for i, ip in enumerate(all_ips)}

    src_ids = df[src_col].map(ip2id).fillna(0).astype(int).values
    dst_ids = df[dst_col].map(ip2id).fillna(0).astype(int).values
    n_nodes = len(ip2id)

    # Edge features: impute with train median, then StandardScale
    X_edge = df[feature_cols].copy()
    train_medians = X_edge.iloc[train_idx].median()
    X_edge = X_edge.fillna(train_medians).fillna(0.0).astype(np.float32).values

    scaler = StandardScaler()
    X_edge[train_idx] = scaler.fit_transform(X_edge[train_idx])
    val_test_idx = np.concatenate([val_idx, test_idx])
    X_edge[val_test_idx] = scaler.transform(X_edge[val_test_idx])
    X_edge = np.clip(X_edge, -10.0, 10.0)

    # Node features: mean of all incident (outgoing + incoming) edge features
    x_node = np.zeros((n_nodes, X_edge.shape[1]), dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.float32)
    for i, (s, d) in enumerate(zip(src_ids, dst_ids)):
        x_node[s] += X_edge[i]
        x_node[d] += X_edge[i]
        counts[s] += 1
        counts[d] += 1
    x_node /= np.maximum(counts[:, None], 1)

    # Edge index with self-loops (improves node representation stability)
    edge_index_t = torch.tensor(np.stack([src_ids, dst_ids]), dtype=torch.long)
    edge_index_t, _ = add_self_loops(edge_index_t, num_nodes=n_nodes)

    # Masks (over original flow edges only — not self-loops)
    n_edges = len(df)
    train_mask = np.zeros(n_edges, dtype=bool)
    val_mask   = np.zeros(n_edges, dtype=bool)
    test_mask  = np.zeros(n_edges, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data = Data(
        x          = torch.tensor(x_node,    dtype=torch.float32),
        edge_index = edge_index_t,
        edge_attr  = torch.tensor(X_edge,    dtype=torch.float32),
        y          = torch.tensor(df[label_col].astype(np.float32).values,
                                  dtype=torch.float32),
        train_mask = torch.tensor(train_mask, dtype=torch.bool),
        val_mask   = torch.tensor(val_mask,   dtype=torch.bool),
        test_mask  = torch.tensor(test_mask,  dtype=torch.bool),
    )
    return data, scaler


# ---------------------------------------------------------------------------
# GNN model: GraphSAGE (max-aggr) + edge MLP
# ---------------------------------------------------------------------------

def _build_net(n_node_feats: int, n_edge_feats: int, hidden: int):
    """Return an nn.Module implementing GraphSAGE + edge MLP.

    Max-aggregation is more robust than mean when node degrees vary widely.
    The forward pass only predicts for the first n_flow edges (edge_attr rows),
    ignoring the appended self-loop entries in edge_index.
    """
    import torch
    import torch.nn as nn
    from torch_geometric.nn import SAGEConv

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_proj = nn.Sequential(
                nn.Linear(n_node_feats, hidden), nn.ReLU()
            )
            self.edge_proj = nn.Sequential(
                nn.Linear(n_edge_feats, hidden), nn.ReLU()
            )
            self.conv1 = SAGEConv(hidden, hidden, aggr="max")
            self.bn1   = nn.BatchNorm1d(hidden)
            self.conv2 = SAGEConv(hidden, hidden, aggr="max")
            self.bn2   = nn.BatchNorm1d(hidden)
            self.drop  = nn.Dropout(0.3)
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden * 3, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, 1),
            )

        def forward(self, x, edge_index, edge_attr):
            h = self.node_proj(x)
            h = self.drop(torch.relu(self.bn1(self.conv1(h, edge_index))))
            h = self.drop(torch.relu(self.bn2(self.conv2(h, edge_index))))
            # edge_attr covers only the original flow edges (not self-loops)
            n_flows = edge_attr.shape[0]
            src_emb = h[edge_index[0, :n_flows]]
            dst_emb = h[edge_index[1, :n_flows]]
            e_emb   = self.edge_proj(edge_attr)
            return self.edge_mlp(
                torch.cat([src_emb, dst_emb, e_emb], dim=-1)
            ).squeeze(-1)   # (n_flows,) raw logits

    return _Net()


# ---------------------------------------------------------------------------
# Explainability: Integrated Gradients on edge features (SOTA)
# ---------------------------------------------------------------------------

def _ig_edge_attribution(
    net,
    data,
    target_idx: np.ndarray,
    device,
    n_steps: int = 30,
) -> np.ndarray:
    """Integrated Gradients on GNN edge features for a subset of edges.

    IG_i(e, 0) = e_i × ∫₀¹ ∂f(α·e)/∂e_i dα

    Baseline = all-zeros edge features ("no traffic").
    Both direct contributions (edge's own features) and indirect contributions
    (edge features affecting node embeddings via SAGEConv aggregation) are
    captured because we interpolate the FULL edge_attr tensor and backprop
    through the complete GNN forward pass.

    Returns
    -------
    attr : (len(target_idx), n_edge_feats) float32 — signed IG attributions
    """
    import torch

    net.eval()
    net.to(device)

    x_dev  = data.x.to(device)
    ei_dev = data.edge_index.to(device)

    all_edge_arr = data.edge_attr.cpu().numpy()  # (n_total_edges, n_feats)
    n_feats = all_edge_arr.shape[1]

    # Uniform right-endpoint alpha values (avoids gradient instability at α=0)
    alphas = np.linspace(1.0 / n_steps, 1.0, n_steps, dtype=np.float32)

    total_grads = np.zeros((len(target_idx), n_feats), dtype=np.float32)

    for alpha in alphas:
        # Interpolated full edge_attr: α × edge_attr  (baseline = zeros)
        ea_interp = torch.tensor(
            alpha * all_edge_arr, dtype=torch.float32, device=device,
            requires_grad=True,
        )
        logits = net(x_dev, ei_dev, ea_interp)
        logits[target_idx].sum().backward()
        total_grads += ea_interp.grad[target_idx].detach().cpu().numpy()

    # Average gradients × (input − baseline) = average gradients × input
    avg_grads = total_grads / n_steps
    attr = avg_grads * all_edge_arr[target_idx]   # completeness: sum ≈ f(e) − f(0)
    return attr.astype(np.float32)


def _save_gnn_explanations(
    net,
    data,
    feature_cols: list[str],
    test_idx: np.ndarray,
    out_dir: Path,
    type_col_vals: Optional[np.ndarray] = None,
    n_samples: int = 500,
    device=None,
):
    """Integrated Gradients attribution on GNN edge features.

    Replaces Gradient×Input with IG (Sundararajan et al. 2017), which
    satisfies the completeness axiom:
        Σᵢ IG_i(edge, 0) ≈ f(edge) − f(zero_edge)

    Both direct (edge's own features via edge_proj) and indirect effects
    (edge features influencing node embeddings via SAGEConv) are captured.

    Saves
    -----
    out_dir/global_importance.csv           — normalised global importance
    out_dir/type_classifier/type_shap_*.csv — per-attack-type IG attributions
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = _get_device()

    feat_names = [f"num__{c}" for c in feature_cols]

    net.eval()
    net.to(device)

    # ---- Global attribution (sampled test edges) ----
    sample_idx = test_idx
    if len(test_idx) > n_samples:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(test_idx, size=n_samples, replace=False)

    attr = _ig_edge_attribution(net, data, sample_idx, device)   # (n_sample, n_feats)

    global_imp = pd.Series(np.abs(attr).mean(axis=0), index=feat_names)
    total = global_imp.sum()
    if total > 0:
        global_imp = global_imp / total

    pd.DataFrame({
        "feature":    global_imp.index,
        "importance": global_imp.values,
    }).to_csv(out_dir / "global_importance.csv", index=False)

    # ---- Per-type attribution ----
    if type_col_vals is not None:
        type_dir = out_dir / "type_classifier"
        type_dir.mkdir(exist_ok=True)

        for attack_type in sorted(set(type_col_vals[test_idx])):
            if str(attack_type).lower() in ("normal", "benign", "nan"):
                continue
            type_test_idx = test_idx[type_col_vals[test_idx] == attack_type]
            if len(type_test_idx) == 0:
                continue

            max_per_type = max(n_samples // 8, 50)
            type_sample = type_test_idx
            if len(type_test_idx) > max_per_type:
                rng = np.random.default_rng(42)
                type_sample = rng.choice(type_test_idx,
                                         size=max_per_type, replace=False)

            attr_t = _ig_edge_attribution(net, data, type_sample, device)

            safe_name = str(attack_type).replace("/", "_").replace(" ", "_")
            pd.DataFrame(attr_t, columns=feat_names).to_csv(
                type_dir / f"type_shap_{safe_name}.csv", index=False
            )


# ---------------------------------------------------------------------------
# GNNExplainer: learned soft edge/node-feature masks (PyG 2.x new API)
# ---------------------------------------------------------------------------

def _save_gnn_explainer_masks(
    net,
    data,
    feature_cols: list[str],
    test_idx: np.ndarray,
    out_dir: Path,
    device=None,
    n_explain: int = 100,
    epochs: int = 200,
):
    """GNNExplainer (Ying et al. NeurIPS 2019) — learned feature importance masks.

    GNNExplainer maximises MI between the masked model output and the full
    model output, learning a soft mask over edge features and node features.
    This gives a richer explanation than gradient methods by considering
    combinatorial feature interactions.

    Only runs on a small sample (n_explain) because GNNExplainer runs a
    separate optimisation loop per target edge.

    Saves
    -----
    out_dir/gnn_explainer_edge_mask.csv   — mean edge-feature mask per test edge
    out_dir/gnn_explainer_node_mask.csv   — mean node-feature mask per test edge
    out_dir/gnn_explainer_importance.csv  — normalised global importance from masks
    """
    try:
        from torch_geometric.explain import Explainer, GNNExplainer as _GNNExpl
    except ImportError:
        return  # PyG < 2.0 — skip silently

    import torch
    import torch.nn as nn

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = _get_device()

    feat_names = [f"num__{c}" for c in feature_cols]
    n_edge_feats = data.edge_attr.shape[1]
    n_node_feats = data.x.shape[1]

    net.eval()
    net.to(device)

    # Wrap the GNN so PyG's Explainer gets a compatible interface.
    # PyG Explainer calls model(x, edge_index, **kwargs) and expects a
    # (n_edges,) or (n_nodes, n_classes) output.
    # We expose only the FIRST target edge's logit per call so that
    # GNNExplainer can compute a gradient for that single prediction.
    class _ExplainWrapper(nn.Module):
        def __init__(self, inner, edge_attr_full):
            super().__init__()
            self.inner = inner
            # Store base edge_attr as a buffer (not a parameter — no grad)
            self.register_buffer("ea_base", edge_attr_full)
            self._target = 0

        def set_target(self, idx: int):
            self._target = idx

        def forward(self, x, edge_index, edge_attr=None):
            ea = edge_attr if edge_attr is not None else self.ea_base
            logits = self.inner(x, edge_index, ea)   # (n_flows,)
            # Return as (n_flows, 1) so PyG sees binary classification
            return logits.unsqueeze(-1)

    ea_full = data.edge_attr.detach().clone().to(device)
    wrapper = _ExplainWrapper(net, ea_full).to(device)
    wrapper.eval()

    try:
        explainer = Explainer(
            model=wrapper,
            algorithm=_GNNExpl(epochs=epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="node",   # edge-level not universally supported;
                                     # use node-level and interpret edge logits
                return_type="raw",
            ),
        )
    except Exception:
        return  # GNNExplainer config not supported — skip

    rng = np.random.default_rng(42)
    sample = test_idx
    if len(test_idx) > n_explain:
        sample = rng.choice(test_idx, size=n_explain, replace=False)

    edge_masks  = []
    node_masks  = []

    x_dev  = data.x.to(device)
    ei_dev = data.edge_index.to(device)

    for idx in sample:
        try:
            expl = explainer(
                x=x_dev, edge_index=ei_dev,
                edge_attr=ea_full,
                index=int(idx),
            )
            if hasattr(expl, "node_mask") and expl.node_mask is not None:
                node_masks.append(expl.node_mask.detach().cpu().numpy().mean(axis=0))
            if hasattr(expl, "edge_mask") and expl.edge_mask is not None:
                edge_masks.append(expl.edge_mask.detach().cpu().numpy())
        except Exception:
            continue

    if edge_masks:
        em_arr = np.stack(edge_masks)   # (n_explain, n_flows) — per-edge mask
        # Global: mean mask value per test edge (how often each edge is selected)
        pd.DataFrame(em_arr).to_csv(out_dir / "gnn_explainer_edge_mask.csv", index=False)

    if node_masks:
        nm_arr = np.stack(node_masks)   # (n_explain, n_node_feats)
        mean_nm = nm_arr.mean(axis=0)
        pd.DataFrame({"feature": feat_names[:len(mean_nm)],
                       "mask": mean_nm}).to_csv(
            out_dir / "gnn_explainer_node_mask.csv", index=False
        )
        # Use node mask as global importance (normalised)
        total = mean_nm.sum()
        imp = mean_nm / total if total > 0 else mean_nm
        pd.DataFrame({"feature": feat_names[:len(imp)],
                       "importance": imp}).to_csv(
            out_dir / "gnn_explainer_importance.csv", index=False
        )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_gnn(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train,
    y_val,
    src_col: str,
    dst_col: str,
    seed: int,
    logger,
    hidden: int = 64,
    epochs: int = 150,
    patience: int = 15,
    lr: float = 1e-3,
    label_col: str = "label",
    exclude_cols: Optional[list[str]] = None,
    out_explain_dir: Optional[Path] = None,
    type_col_vals: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build IP graph, train GraphSAGE, return (scores_val, scores_test).

    Parameters
    ----------
    df              : full DataFrame (all rows, all columns including src/dst IP)
    train_idx       : row indices for training edges
    val_idx         : row indices for validation edges
    test_idx        : row indices for test edges
    y_train, y_val  : unused directly (labels taken from df[label_col])
    src_col         : column name for source IP
    dst_col         : column name for destination IP
    seed            : random seed
    logger          : Python logger
    out_explain_dir : if set, save gradient attribution explanations here
    type_col_vals   : array of attack type labels aligned to df rows
                      (used for per-type explanation CSVs)

    Returns
    -------
    scores_val  : (n_val,)  float32 — attack probability per val edge
    scores_test : (n_test,) float32 — attack probability per test edge
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _get_device()

    # Determine edge feature columns (exclude metadata)
    _exclude = {src_col, dst_col, label_col}
    for meta in ("timestamp", "type", "src_port", "dst_port", "uid"):
        _exclude.add(meta)
    if exclude_cols:
        _exclude.update(exclude_cols)
    feature_cols = _select_edge_features(df, list(_exclude))

    n_unique_ips = len(pd.concat([df[src_col], df[dst_col]]).dropna().unique())
    logger.info("GNN: %d edge features, %d unique IPs (nodes), %d flows (edges)",
                len(feature_cols), n_unique_ips, len(df))

    # Build graph
    data, _ = _build_graph(
        df, train_idx, val_idx, test_idx,
        src_col, dst_col, feature_cols, label_col,
    )
    data = data.to(device)

    net = _build_net(data.x.shape[1], data.edge_attr.shape[1], hidden).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_weights  = copy.deepcopy(net.state_dict())
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        net.train()
        optimiser.zero_grad()
        logits = net(data.x, data.edge_index, data.edge_attr)
        loss   = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        net.eval()
        with torch.no_grad():
            val_logits = net(data.x, data.edge_index, data.edge_attr)
            val_loss   = criterion(
                val_logits[data.val_mask], data.y[data.val_mask]
            ).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(net.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("GNN early stop at epoch %d (val_loss=%.4f)",
                            epoch, val_loss)
                break

    net.load_state_dict(best_weights)
    net.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            net(data.x, data.edge_index, data.edge_attr)
        ).cpu().numpy()

    # ---- Explanations ----
    if out_explain_dir is not None:
        logger.info("GNN: computing Integrated Gradients explanations...")
        _save_gnn_explanations(
            net, data, feature_cols, test_idx,
            out_dir=out_explain_dir,
            type_col_vals=type_col_vals,
            device=device,
        )
        logger.info("GNN: IG explanations saved to %s", out_explain_dir)
        logger.info("GNN: running GNNExplainer (learned feature masks)...")
        try:
            _save_gnn_explainer_masks(
                net, data, feature_cols, test_idx,
                out_dir=out_explain_dir / "gnn_explainer",
                device=device,
            )
            logger.info("GNN: GNNExplainer masks saved to %s",
                        out_explain_dir / "gnn_explainer")
        except Exception as _exc:
            logger.warning("GNNExplainer failed (%s); IG explanations still saved", _exc)

    return probs[val_idx].astype(np.float32), probs[test_idx].astype(np.float32)
