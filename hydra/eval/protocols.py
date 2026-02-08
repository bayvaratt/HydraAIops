from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import random

from hydra.config import HydraConfig
from hydra.data.schema import run_smoke_checks
from hydra.data.preprocess import TabularPreprocessor
from hydra.data.split import split_tabular, split_edges_entity, split_edges_temporal, validate_group_disjointness
from hydra.eval.detection import pr_auc, find_threshold_at_recall, fpr_at_fixed_recall
from hydra.eval.explainability import (
    coverage,
    fidelity_gnn_edge_removal,
    fidelity_perturbation_tabular,
    operational_plausibility,
    stability_jaccard,
)
from hydra.explain.records import ExplanationRecord
from hydra.explain.shap_tabular import explain_logreg, explain_tree_shap, permutation_importance_global
from hydra.explain.gnn_explain import explain_edge
from hydra.models.baselines import MajorityClassBaseline, ThresholdBaseline
from hydra.models.tabular import get_tabular_models, predict_proba
from hydra.models.gnn import build_graph, train_gnn, predict_proba as gnn_predict_proba

logger = logging.getLogger(__name__)


def timestamped_dir(base: str) -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    out_dir = base_path / stamp
    suffix = 1
    while out_dir.exists():
        out_dir = base_path / f"{stamp}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir(parents=True)
    return str(out_dir)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _ensure_dense(model, X):
    if sp.issparse(X):
        name = model.__class__.__name__
        if name in {"RandomForestClassifier", "HistGradientBoostingClassifier"}:
            return X.toarray()
    return X


def _save_json(path: str, payload: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_jsonl(path: str, records: List[ExplanationRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")


def run_experiment_tabular(
    df: pd.DataFrame,
    dataset_name: str,
    cfg: HydraConfig,
    out_dir: str,
    label_col: str,
    type_col: Optional[str],
    feature_regime: str,
    timestamp_col: Optional[str] = None,
    split_strategy: Optional[str] = None,
    group_col: Optional[str] = None,
    model_names: Optional[List[str]] = None,
    seed: int = 42,
    label_permutation_probe: bool = False,
) -> Dict[str, Dict]:
    logger.info("Starting tabular experiment: %s", dataset_name)
    set_seed(seed)
    run_smoke_checks(
        df,
        dataset_name=dataset_name,
        label_col=label_col,
        type_col=type_col,
        out_dir=out_dir,
        high_cardinality_threshold=cfg.high_cardinality_threshold,
        high_cardinality_ratio=cfg.high_cardinality_ratio,
    )

    split_strategy = split_strategy or cfg.split_strategy
    train_df, val_df, test_df, split_idx = split_tabular(
        df,
        label_col=label_col,
        seed=seed,
        strategy=split_strategy,
        timestamp_col=timestamp_col,
        group_col=group_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        return_indices=True,
    )
    if split_strategy == "host":
        if not group_col or group_col not in df.columns:
            raise RuntimeError("Host-based split requires valid group_col; none provided.")
        validate_group_disjointness(split_idx, df[group_col], name=f"{dataset_name}:{group_col}")

    pre = TabularPreprocessor(
        feature_regime=feature_regime,
        label_col=label_col,
        type_col=type_col,
        high_cardinality_threshold=cfg.high_cardinality_threshold,
        high_cardinality_ratio=cfg.high_cardinality_ratio,
        categorical_top_k=cfg.categorical_top_k,
        enable_port_bucketing=cfg.enable_port_bucketing,
        port_top_n=cfg.port_top_n,
    )

    X_train = pre.fit_transform(train_df)
    X_val = pre.transform(val_df)
    X_test = pre.transform(test_df)

    y_train = train_df[label_col].astype(int).to_numpy()
    y_val = val_df[label_col].astype(int).to_numpy()
    y_test = test_df[label_col].astype(int).to_numpy()

    models = get_tabular_models(cfg.tabular_models)
    if model_names:
        missing = [name for name in model_names if name not in models]
        if missing:
            logger.warning("Requested models not available: %s", missing)
        models = {k: v for k, v in models.items() if k in model_names}

    metrics_by_model: Dict[str, Dict] = {}
    summary_rows = []

    baseline_train = train_df.drop(columns=[c for c in [label_col, type_col] if c], errors="ignore")
    baseline_val = val_df.drop(columns=[c for c in [label_col, type_col] if c], errors="ignore")
    baseline_test = test_df.drop(columns=[c for c in [label_col, type_col] if c], errors="ignore")

    def _warn_if_constant_scores(name: str, y_true: np.ndarray, scores: np.ndarray, pr: float) -> None:
        if np.var(scores) == 0:
            prevalence = float(np.mean(y_true))
            logger.warning(
                "%s: constant score detected; PR-AUC should be close to prevalence %.4f (got %.4f)",
                name,
                prevalence,
                pr,
            )

    def _warn_majority_baseline(y_true: np.ndarray, pr: float, scores: np.ndarray) -> None:
        prevalence = float(np.mean(y_true))
        tol = max(0.02, 0.1 * prevalence)
        if abs(pr - prevalence) > tol:
            logger.warning(
                "Majority baseline PR-AUC off prevalence: prevalence=%.4f pr_auc=%.4f tol=%.4f",
                prevalence,
                pr,
                tol,
            )
        try:
            from sklearn.metrics import roc_auc_score

            if len(np.unique(y_true)) > 1:
                roc = float(roc_auc_score(y_true, scores))
                if abs(roc - 0.5) > 0.05:
                    logger.warning("Majority baseline ROC-AUC not near 0.5: roc_auc=%.4f", roc)
        except Exception:
            logger.warning("Majority baseline ROC-AUC check skipped (missing sklearn or invalid labels).")

    majority = MajorityClassBaseline().fit(train_df[label_col])
    majority_val_scores = majority.predict_proba(baseline_val)[:, 1]
    majority_threshold = find_threshold_at_recall(y_val, majority_val_scores, cfg.recall_target)
    majority_test_scores = majority.predict_proba(baseline_test)[:, 1]
    majority_metrics = {
        "pr_auc": pr_auc(y_test, majority_test_scores),
        "threshold": float(majority_threshold),
        "fpr_at_recall": fpr_at_fixed_recall(y_test, majority_test_scores, cfg.recall_target, majority_threshold),
        "coverage": 0.0,
        "fidelity": {"mean_drop": 0.0, "mean_drop_random": 0.0},
        "operational_plausibility": {},
    }
    baseline_dir = Path(out_dir) / "baseline_majority"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _save_json(str(baseline_dir / "metrics.json"), majority_metrics)
    _save_jsonl(str(baseline_dir / "explanations.jsonl"), [])
    metrics_by_model["baseline_majority"] = majority_metrics
    summary_rows.append({"model": "baseline_majority", **majority_metrics})
    _warn_if_constant_scores("baseline_majority", y_test, majority_test_scores, majority_metrics["pr_auc"])
    _warn_majority_baseline(y_test, majority_metrics["pr_auc"], majority_test_scores)

    thresh_base = ThresholdBaseline().fit(baseline_train, train_df[label_col], cfg.recall_target)
    thresh_val_scores = thresh_base.predict_proba(baseline_val)[:, 1]
    thresh_threshold = find_threshold_at_recall(y_val, thresh_val_scores, cfg.recall_target)
    thresh_base.threshold = thresh_threshold
    thresh_test_scores = thresh_base.predict_proba(baseline_test)[:, 1]
    thresh_metrics = {
        "pr_auc": pr_auc(y_test, thresh_test_scores),
        "threshold": float(thresh_threshold),
        "fpr_at_recall": fpr_at_fixed_recall(y_test, thresh_test_scores, cfg.recall_target, thresh_threshold),
        "coverage": 0.0,
        "fidelity": {"mean_drop": 0.0, "mean_drop_random": 0.0},
        "operational_plausibility": {},
    }
    baseline_dir = Path(out_dir) / "baseline_threshold"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _save_json(str(baseline_dir / "metrics.json"), thresh_metrics)
    _save_jsonl(str(baseline_dir / "explanations.jsonl"), [])
    metrics_by_model["baseline_threshold"] = thresh_metrics
    summary_rows.append({"model": "baseline_threshold", **thresh_metrics})
    _warn_if_constant_scores("baseline_threshold", y_test, thresh_test_scores, thresh_metrics["pr_auc"])

    if label_permutation_probe:
        rng = np.random.RandomState(seed)
        y_train_shuffled = y_train.copy()
        rng.shuffle(y_train_shuffled)
        probe_models = get_tabular_models(cfg.tabular_models)
        probe_name = "lightgbm" if "lightgbm" in probe_models else "random_forest"
        probe_model = probe_models.get(probe_name)
        if probe_model is None:
            raise RuntimeError("Label permutation probe failed: no lightgbm or random_forest available.")
        probe_model = _clone_with_seed(probe_model, seed)

        X_train_m = _ensure_dense(probe_model, X_train)
        X_val_m = _ensure_dense(probe_model, X_val)
        X_test_m = _ensure_dense(probe_model, X_test)

        probe_model.fit(X_train_m, y_train_shuffled)
        probe_val_scores = predict_proba(probe_model, X_val_m)
        probe_threshold = find_threshold_at_recall(y_val, probe_val_scores, cfg.recall_target)
        probe_test_scores = predict_proba(probe_model, X_test_m)
        probe_pr = pr_auc(y_test, probe_test_scores)
        probe_metrics = {
            "pr_auc": probe_pr,
            "threshold": float(probe_threshold),
            "fpr_at_recall": fpr_at_fixed_recall(y_test, probe_test_scores, cfg.recall_target, probe_threshold),
            "coverage": 0.0,
            "fidelity": {"mean_drop": 0.0, "mean_drop_random": 0.0},
            "operational_plausibility": {},
        }
        _save_json(str(Path(out_dir) / "permutation_probe.json"), probe_metrics)
        summary_rows.append({"model": f"{probe_name}_label_shuffled", **probe_metrics})
        _warn_if_constant_scores(f"{probe_name}_label_shuffled", y_test, probe_test_scores, probe_pr)

        prevalence = float(np.mean(y_test))
        if probe_pr > prevalence + 0.10:
            raise RuntimeError(
                f"Label permutation probe failed: pr_auc={probe_pr:.4f} "
                f"prevalence={prevalence:.4f}. Likely leakage or evaluation bug."
            )

    for name, model in models.items():
        logger.info("Training model: %s", name)
        X_train_m = _ensure_dense(model, X_train)
        X_val_m = _ensure_dense(model, X_val)
        X_test_m = _ensure_dense(model, X_test)

        model.fit(X_train_m, y_train)

        val_scores = predict_proba(model, X_val_m)
        threshold = find_threshold_at_recall(y_val, val_scores, cfg.recall_target)

        test_scores = predict_proba(model, X_test_m)
        metrics = {
            "pr_auc": pr_auc(y_test, test_scores),
            "threshold": float(threshold),
            "fpr_at_recall": fpr_at_fixed_recall(y_test, test_scores, cfg.recall_target, threshold),
        }

        pred_pos = np.where(test_scores >= threshold)[0]
        alert_ids = pred_pos.tolist()
        explain_ids = alert_ids[: cfg.max_explain_samples]

        if name == "logreg":
            records = explain_logreg(
                model,
                X_test_m,
                pre.feature_names,
                explain_ids,
                test_scores,
                model_name=name,
                feature_regime=feature_regime,
                dataset_name=dataset_name,
                seed=seed,
                top_k=cfg.top_k_explanations,
            )
        else:
            X_test_shap = X_test_m.toarray() if sp.issparse(X_test_m) else X_test_m
            records = explain_tree_shap(
                model,
                X_test_shap,
                pre.feature_names,
                explain_ids,
                test_scores,
                model_name=name,
                feature_regime=feature_regime,
                dataset_name=dataset_name,
                seed=seed,
                top_k=cfg.top_k_explanations,
            )

        metrics["coverage"] = coverage(records, alert_ids)
        metrics["fidelity"] = fidelity_perturbation_tabular(
            model,
            X_test_m,
            pre.feature_names,
            records,
            top_k=cfg.top_k_explanations,
        )
        metrics["operational_plausibility"] = operational_plausibility(records, top_k=cfg.top_k_explanations)

        if len(cfg.seeds) > 1:
            records_by_seed: Dict[int, List[ExplanationRecord]] = {}
            for s in cfg.seeds:
                model_seeded = _clone_with_seed(model, s)
                model_seeded.fit(X_train_m, y_train)
                scores_seed = predict_proba(model_seeded, X_test_m)
                sample_ids = explain_ids[: cfg.stability_sample_size]
                if name == "logreg":
                    recs = explain_logreg(
                        model_seeded,
                        X_test_m,
                        pre.feature_names,
                        sample_ids,
                        scores_seed,
                        model_name=name,
                        feature_regime=feature_regime,
                        dataset_name=dataset_name,
                        seed=s,
                        top_k=cfg.stability_top_k,
                    )
                else:
                    X_test_shap = X_test_m.toarray() if sp.issparse(X_test_m) else X_test_m
                    recs = explain_tree_shap(
                        model_seeded,
                        X_test_shap,
                        pre.feature_names,
                        sample_ids,
                        scores_seed,
                        model_name=name,
                        feature_regime=feature_regime,
                        dataset_name=dataset_name,
                        seed=s,
                        top_k=cfg.stability_top_k,
                    )
                records_by_seed[s] = recs
            metrics["stability"] = stability_jaccard(records_by_seed, top_k=cfg.stability_top_k)

        model_dir = Path(out_dir) / name
        model_dir.mkdir(parents=True, exist_ok=True)
        _save_json(str(model_dir / "metrics.json"), metrics)
        _save_jsonl(str(model_dir / "explanations.jsonl"), records)
        perm = permutation_importance_global(model, X_val_m, y_val, pre.feature_names)
        _save_json(str(model_dir / "permutation_importance.json"), {"top_features": perm})

        summary_row = {"model": name, **metrics}
        summary_rows.append(summary_row)
        metrics_by_model[name] = metrics
        _warn_if_constant_scores(name, y_test, test_scores, metrics["pr_auc"])

    summary_path = Path(out_dir) / "metrics_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return metrics_by_model


def run_experiment_gnn(
    df: pd.DataFrame,
    dataset_name: str,
    cfg: HydraConfig,
    out_dir: str,
    label_col: str,
    feature_regime: str,
    src_col: str,
    dst_col: str,
    timestamp_col: Optional[str],
    seed: int = 42,
) -> Dict:
    logger.info("Starting GNN experiment: %s", dataset_name)
    set_seed(seed)

    run_smoke_checks(
        df,
        dataset_name=dataset_name,
        label_col=label_col,
        type_col=None,
        out_dir=out_dir,
        high_cardinality_threshold=cfg.high_cardinality_threshold,
        high_cardinality_ratio=cfg.high_cardinality_ratio,
    )

    graph = build_graph(
        df,
        label_col=label_col,
        src_col=src_col,
        dst_col=dst_col,
        timestamp_col=timestamp_col,
        cfg=cfg.graph,
    )

    if cfg.graph.split_strategy == "entity":
        train_idx, val_idx, test_idx = split_edges_entity(
            graph.edge_meta,
            src_col="src",
            dst_col="dst",
            seed=seed,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
        )
    else:
        train_idx, val_idx, test_idx = split_edges_temporal(
            graph.edge_meta,
            timestamp_col="window_id",
            seed=seed,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
        )

    model, losses = train_gnn(graph, train_idx, val_idx, cfg.gnn)
    scores = gnn_predict_proba(model, graph)
    if len(val_idx) > 0:
        threshold = find_threshold_at_recall(
            graph.edge_label[val_idx].cpu().numpy(),
            scores[val_idx],
            cfg.recall_target,
        )
    else:
        threshold = 0.5

    y_test = graph.edge_label[test_idx].cpu().numpy()
    metrics = {
        "pr_auc": pr_auc(y_test, scores[test_idx]),
        "threshold": float(threshold),
        "fpr_at_recall": fpr_at_fixed_recall(y_test, scores[test_idx], cfg.recall_target, threshold),
        "losses": losses,
    }

    pred_pos = np.where(scores[test_idx] >= threshold)[0]
    alert_ids = test_idx[pred_pos].tolist()
    explain_ids = alert_ids[: cfg.max_explain_samples]

    records: List[ExplanationRecord] = []
    for edge_id in explain_ids:
        rec = explain_edge(
            model,
            graph,
            edge_id,
            model_name="gnn",
            feature_names=graph.edge_feature_names,
            feature_regime=feature_regime,
            dataset_name=dataset_name,
            seed=seed,
            top_k=cfg.top_k_explanations,
        )
        records.append(rec)

    metrics["coverage"] = coverage(records, alert_ids)
    metrics["fidelity"] = fidelity_gnn_edge_removal(model, graph, records, top_k=cfg.top_k_explanations)
    metrics["operational_plausibility"] = operational_plausibility(records, top_k=cfg.top_k_explanations)

    _save_json(str(Path(out_dir) / "metrics.json"), metrics)
    _save_jsonl(str(Path(out_dir) / "explanations.jsonl"), records)

    return metrics


def _clone_with_seed(model, seed: int):
    from sklearn.base import clone

    try:
        new_model = clone(model)
    except Exception:
        new_model = model
    if hasattr(new_model, "random_state"):
        setattr(new_model, "random_state", seed)
    return new_model
