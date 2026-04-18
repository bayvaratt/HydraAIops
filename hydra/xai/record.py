"""XAI quality evaluation — five criteria.

All functions are model-agnostic.  They consume:
  - attributions : np.ndarray (n_samples, n_features) — pre-computed
                   attribution values (SHAP, IG, gradient×input, etc.)
  - predict_fn   : callable(X: np.ndarray) -> np.ndarray (n,)
                   Takes a dense preprocessed feature matrix, returns P(attack).
  - explain_fn   : callable(X: np.ndarray) -> np.ndarray (n_samples, n_features)
                   Takes dense preprocessed matrix, returns attribution matrix.

Criteria
--------
1. Faithfulness (Fidelity)
   Comprehensiveness: ablating the top-k attributed features drops P(attack).
   Sufficiency: the top-k features alone are sufficient to predict.
   Reference: DeYoung et al. (2020), ERASER benchmark.

2. Stability
   Small Gaussian noise added to inputs should not flip which features are
   considered most important (Alvarez-Melis & Jaakkola 2018).
   Measured via Spearman rank correlation + top-5 Jaccard overlap.

3. Simplicity
   How concentrated is the explanation?  Few features dominating is simpler
   (Ribeiro et al. 2016 — LIME; Bhatt et al. 2020).
   k90_frac: fraction of features needed to cover 90% of total |attribution|.
   gini_coeff: Gini coefficient of the mean |attribution| distribution.

4. Plausibility / Relevance
   Top features should align with expert-known important network features.
   RMA@k: fraction of model's top-k features that appear in the expert set.
   (Proxy for the Rank-Matching Accuracy metric; Adebayo et al. 2022.)

5. Usefulness / Timeliness
   Wall-clock milliseconds per sample explanation.

Entry point
-----------
    results = run_xai_eval(
        explain_fn, predict_fn, X_test_dense,
        feature_names, dataset_name, out_path, ...
    )
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

ExplainFn = Callable[[np.ndarray], np.ndarray]
PredictFn = Callable[[np.ndarray], np.ndarray]

# ---------------------------------------------------------------------------
# Expert feature sets for plausibility evaluation
# ---------------------------------------------------------------------------

# Substring matches against processed feature names (num__xxx, cat__xxx).
# Drawn from network intrusion detection literature; these are the
# universally-cited discriminative flow features.
EXPERT_FEATURES: dict[str, list[str]] = {
    "ton_iot": [
        "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
        "proto", "service", "conn_state",
    ],
    "ton_iot_dedup": [
        "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
        "proto", "service", "conn_state",
    ],
    "cic_iot2023": [
        "tot_size", "number", "iat", "variance",
        "header_length", "tcp", "udp", "icmp",
    ],
}


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------

def compute_faithfulness(
    predict_fn: PredictFn,
    X_dense: np.ndarray,
    attributions: np.ndarray,
    top_k_values: tuple = (5, 10),
    max_samples: int = 500,
    baseline: str = "zero",
) -> dict:
    """Comprehensiveness + sufficiency via feature ablation.

    Baseline for ablated features is configurable:
    - zero: set ablated features to 0.0 (baseline used historically)
    - mean: set ablated features to per-feature training mean
    - median: set ablated features to per-feature median
    - min/max: less commonly used, but available using min/max of the sample

    Returns
    -------
    dict with keys:
        comprehensiveness_k{k} : mean drop in P(attack) when top-k ablated
                                 (higher = more faithful; range ~[−1, 1])
        sufficiency_k{k}       : fraction of original confidence retained
                                 when using ONLY top-k features
                                 (higher = more faithful; range [0, 1])
    """
    n = min(len(X_dense), max_samples)
    X = X_dense[:n].copy()
    attr = attributions[:n]

    p_orig = np.clip(predict_fn(X), 0.0, 1.0)

    # Global top-k by mean |attribution| across samples
    # (same k features ablated/kept for all samples — this tests the
    # summary explanation, not per-sample, which is the relevant setting
    # for operational IDS deployment where one rule set is applied globally)
    mean_abs = np.abs(attr).mean(axis=0)

    if baseline == "zero":
        baseline_values = np.zeros(X.shape[1], dtype=np.float32)
    elif baseline == "mean":
        baseline_values = np.mean(X, axis=0)
    elif baseline == "median":
        baseline_values = np.median(X, axis=0)
    elif baseline == "min":
        baseline_values = np.min(X, axis=0)
    elif baseline == "max":
        baseline_values = np.max(X, axis=0)
    else:
        raise ValueError("Baseline must be one of 'zero', 'mean', 'median', 'min', 'max'.")

    result = {}
    for k in top_k_values:
        k = min(k, X.shape[1])
        top_idx = np.argsort(mean_abs)[::-1][:k]

        # Comprehensiveness: ablate top-k features
        X_ablated = X.copy()
        X_ablated[:, top_idx] = baseline_values[top_idx]
        p_ablated = np.clip(predict_fn(X_ablated), 0.0, 1.0)
        comprehensiveness = float(np.mean(p_orig - p_ablated))

        # Sufficiency: keep ONLY top-k features (replace others with baseline)
        X_sufficient = np.tile(baseline_values, (X.shape[0], 1))
        X_sufficient[:, top_idx] = X[:, top_idx]
        p_sufficient = np.clip(predict_fn(X_sufficient), 0.0, 1.0)
        # Fraction of original confidence retained
        sufficiency = float(np.mean(p_sufficient / (p_orig + 1e-8)))

        result[f"comprehensiveness_k{k}"] = round(comprehensiveness, 4)
        result[f"sufficiency_k{k}"] = round(min(sufficiency, 1.0), 4)
        if sufficiency > 1.0:
            result[f"sufficiency_unclamped_k{k}"] = round(sufficiency, 4)

    return result


# ---------------------------------------------------------------------------
# 2. Stability
# ---------------------------------------------------------------------------

def compute_stability(
    explain_fn: ExplainFn,
    X_dense: np.ndarray,
    n_samples: int = 50,
    n_perturb: int = 10,
    noise_std: float = 0.05,
    top_k: int = 5,
    random_state: int = 42,
) -> dict:
    """Spearman rank correlation + top-k Jaccard under small input noise.

    For each of n_samples test rows, add Gaussian noise n_perturb times,
    re-explain, and measure similarity between original and noisy attributions.

    noise_std = 0.05 corresponds to 5% of the typical feature range after
    median imputation — small enough that a faithful explanation should be
    stable.

    Returns
    -------
    dict with keys:
        mean_spearman_rank_corr : avg Spearman ρ of |attr| vectors [0,1] ↑ good
        mean_top{k}_jaccard     : avg Jaccard of top-k feature sets [0,1] ↑ good
        n_pairs                 : total (sample, perturb) pairs evaluated
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(random_state)
    n_total = len(X_dense)
    if n_total > n_samples:
        sel = rng.choice(n_total, size=n_samples, replace=False)
        X_probe = X_dense[sel]
    else:
        X_probe = X_dense.copy()
        n_samples = n_total

    spearman_scores: list[float] = []
    jaccard_scores:  list[float] = []

    for i in range(n_samples):
        x = X_probe[i : i + 1]  # (1, d)

        try:
            orig_attr = explain_fn(x)  # (1, d)
            if orig_attr is None or orig_attr.shape[1] == 0:
                continue
            orig_abs = np.abs(orig_attr[0])
        except Exception:
            continue

        orig_top = set(np.argsort(orig_abs)[::-1][:top_k].tolist())

        for _ in range(n_perturb):
            noise = rng.normal(0.0, noise_std, x.shape).astype(np.float32)
            x_noisy = x + noise

            try:
                noisy_attr = explain_fn(x_noisy)
                if noisy_attr is None:
                    continue
                noisy_abs = np.abs(noisy_attr[0])
            except Exception:
                continue

            # Spearman rank correlation on the full attribution vector
            if orig_abs.std() > 1e-8 and noisy_abs.std() > 1e-8:
                rho, _ = spearmanr(orig_abs, noisy_abs)
                if not np.isnan(rho):
                    spearman_scores.append(float(rho))

            # Top-k Jaccard
            noisy_top = set(np.argsort(noisy_abs)[::-1][:top_k].tolist())
            union = orig_top | noisy_top
            if union:
                jaccard_scores.append(len(orig_top & noisy_top) / len(union))

    return {
        "mean_spearman_rank_corr": round(float(np.mean(spearman_scores)), 4)
        if spearman_scores else float("nan"),
        f"mean_top{top_k}_jaccard": round(float(np.mean(jaccard_scores)), 4)
        if jaccard_scores else float("nan"),
        "n_pairs": len(spearman_scores),
    }


# ---------------------------------------------------------------------------
# 3. Simplicity
# ---------------------------------------------------------------------------

def compute_simplicity(attributions: np.ndarray) -> dict:
    """Concentration of the explanation: k90_frac + Gini coefficient.

    k90_frac
        Mean across samples of (min k s.t. top-k features cover ≥90% of
        total |attribution|) / n_features.  0 = one feature explains all;
        1 = all features needed equally.  Lower is simpler.

    gini_coeff
        Gini coefficient of the mean |attribution| distribution (0 = equal
        importance; 1 = single feature dominates).  Higher is simpler.

    Returns
    -------
    dict with keys: k90_frac, gini_coeff
    """
    n, d = attributions.shape
    abs_attr = np.abs(attributions)

    # k90 per sample
    k90_list = []
    for i in range(n):
        row = abs_attr[i]
        total = row.sum()
        if total < 1e-12:
            continue
        sorted_desc = np.sort(row)[::-1]
        cumsum = np.cumsum(sorted_desc)
        k = int(np.searchsorted(cumsum, 0.9 * total)) + 1
        k90_list.append(k / d)

    k90_frac = float(np.mean(k90_list)) if k90_list else float("nan")

    # Gini of mean |attribution|
    mean_abs = abs_attr.mean(axis=0)
    total = mean_abs.sum()
    if total < 1e-12:
        gini = float("nan")
    else:
        sorted_vals = np.sort(mean_abs / total)
        n_feats = len(sorted_vals)
        # Standard Gini formula
        idx = np.arange(1, n_feats + 1)
        gini = float((2 * (idx * sorted_vals).sum() / sorted_vals.sum() - (n_feats + 1)) / n_feats)

    return {
        "k90_frac": round(k90_frac, 4),
        "gini_coeff": round(max(0.0, gini), 4),
    }


# ---------------------------------------------------------------------------
# 4. Plausibility
# ---------------------------------------------------------------------------

def compute_plausibility(
    attributions: np.ndarray,
    feature_names: List[str],
    expert_feature_keywords: List[str],
    top_k: int = 10,
) -> dict:
    """RMA@k: fraction of model's top-k features matching expert set.

    Matching is substring-based (case-insensitive) against `feature_names`,
    which carry num__/cat__ prefixes from the ColumnTransformer.

    Returns
    -------
    dict with keys:
        rma_at_k            : fraction in [0, 1], higher = more plausible
        expert_features_found : list of matched feature names
        top_k_features      : list of model's top-k feature names
    """
    mean_abs = np.abs(attributions).mean(axis=0)
    top_k = min(top_k, len(feature_names))
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_features = [feature_names[i] for i in top_idx]

    matched = [
        f for f in top_features
        if any(kw.lower() in f.lower() for kw in expert_feature_keywords)
    ]

    return {
        "rma_at_k": round(len(matched) / top_k, 4) if top_k > 0 else float("nan"),
        "expert_features_found": matched,
        "top_k_features": top_features,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# 5. Timeliness
# ---------------------------------------------------------------------------

def compute_timeliness(
    explain_fn: ExplainFn,
    X_dense: np.ndarray,
    n_benchmark: int = 100,
    n_warmup: int = 1,
) -> dict:
    """Wall-clock ms per sample explanation.

    Runs `n_warmup` un-timed calls first (PyTorch JIT, thread pool warm-up),
    then times one call on `n_benchmark` samples.

    Returns
    -------
    dict with keys: ms_per_sample, total_ms, n_timed_samples
    """
    n = min(n_benchmark, len(X_dense))
    X_bench = X_dense[:n]

    for _ in range(n_warmup):
        try:
            explain_fn(X_bench)
        except Exception:
            pass

    t0 = time.perf_counter()
    try:
        explain_fn(X_bench)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except Exception:
        elapsed_ms = float("nan")

    ms_per = elapsed_ms / n if n > 0 else float("nan")
    return {
        "ms_per_sample": round(ms_per, 3),
        "total_ms": round(elapsed_ms, 1),
        "n_timed_samples": n,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_xai_eval(
    explain_fn: ExplainFn,
    predict_fn: PredictFn,
    X_test_dense: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
    out_path: Path,
    model_name: str = "unknown",
    max_faithfulness_samples: int = 500,
    faithfulness_baseline: str = "zero",
    stability_n_samples: int = 50,
    stability_n_perturb: int = 10,
    stability_noise_std: float = 0.05,
    plausibility_top_k: int = 10,
    timeliness_n: int = 100,
    logger=None,
) -> Optional[dict]:
    """Run all five XAI quality criteria and write results to `out_path`.

    Parameters
    ----------
    explain_fn          : callable(X_dense) -> (n, f) attribution array
    predict_fn          : callable(X_dense) -> (n,) P(attack)
    X_test_dense        : (N, f) dense float32, already preprocessed
    feature_names       : list of feature name strings (length f)
    dataset_name        : "ton_iot" | "cic_iot2023" (used for expert lookup)
    out_path            : where to write xai_eval.json
    model_name          : human-readable label for the JSON
    stability_n_samples : rows to use for stability probing (fewer for deep models)
    stability_n_perturb : noisy perturbations per row (fewer for deep models)

    Returns
    -------
    dict with all results, or None if attribution step failed.
    """
    def _log(msg: str):
        if logger is not None:
            logger.info(msg)

    n_test, n_feats = X_test_dense.shape
    _log(f"XAI eval [{model_name}]: n={n_test}, f={n_feats}")

    # Step 1: Compute attribution matrix
    _log(f"XAI eval [{model_name}]: computing attributions (cap {max_faithfulness_samples})...")
    attributions = explain_fn(
        X_test_dense[:max_faithfulness_samples]
        if len(X_test_dense) > max_faithfulness_samples
        else X_test_dense
    )
    if attributions is None or attributions.shape[1] != n_feats:
        if logger:
            logger.warning("XAI eval [%s]: attribution failed; skipping.", model_name)
        return None

    n_attr = attributions.shape[0]

    # Step 2: Faithfulness
    _log(f"XAI eval [{model_name}]: faithfulness...")
    try:
        faith = compute_faithfulness(
            predict_fn,
            X_test_dense[:n_attr],
            attributions,
            top_k_values=(5, 10),
            max_samples=max_faithfulness_samples,
            baseline=faithfulness_baseline,
        )
    except Exception as exc:
        faith = {"error": str(exc)}
        if logger:
            logger.warning("XAI eval [%s]: faithfulness failed: %s", model_name, exc)

    # Step 3: Stability
    _log(f"XAI eval [{model_name}]: stability (n_samples={stability_n_samples}, "
         f"n_perturb={stability_n_perturb}, noise_std={stability_noise_std})...")
    try:
        stab = compute_stability(
            explain_fn,
            X_test_dense,
            n_samples=stability_n_samples,
            n_perturb=stability_n_perturb,
            noise_std=stability_noise_std,
        )
    except Exception as exc:
        stab = {"error": str(exc)}
        if logger:
            logger.warning("XAI eval [%s]: stability failed: %s", model_name, exc)

    # Step 4: Simplicity
    _log(f"XAI eval [{model_name}]: simplicity...")
    try:
        simp = compute_simplicity(attributions)
    except Exception as exc:
        simp = {"error": str(exc)}

    # Step 5: Plausibility
    _log(f"XAI eval [{model_name}]: plausibility...")
    expert_kws = EXPERT_FEATURES.get(dataset_name.lower().replace("-", "_"), [])
    try:
        plaus = compute_plausibility(
            attributions, feature_names, expert_kws, top_k=plausibility_top_k
        )
    except Exception as exc:
        plaus = {"error": str(exc)}

    # Step 6: Timeliness
    _log(f"XAI eval [{model_name}]: timeliness...")
    try:
        timing = compute_timeliness(explain_fn, X_test_dense, n_benchmark=timeliness_n)
    except Exception as exc:
        timing = {"error": str(exc)}

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "n_test_samples": n_test,
        "n_features": n_feats,
        "n_attributed_samples": n_attr,
        "faithfulness": faith,
        "stability": stab,
        "simplicity": simp,
        "plausibility": plaus,
        "timeliness": timing,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    _log(f"XAI eval [{model_name}]: saved to {out_path}")
    return results


def _json_default(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return str(obj)
