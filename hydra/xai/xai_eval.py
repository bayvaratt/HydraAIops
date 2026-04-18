"""Five-criterion XAI evaluation module for the HydraAIops dissertation.

Implements the evaluation framework from Table 2.1:
    1. Faithfulness  — comprehensiveness & sufficiency
    2. Stability     — seed and noise robustness of rankings
    3. Simplicity    — k90 and Gini concentration
    4. Plausibility  — RMA@k against expert-defined reference sets
    5. Timeliness    — wall-clock explanation cost

Plus the composite OXS score and a convenience ``evaluate_explanations`` runner.
"""
from __future__ import annotations

import time
import warnings
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import spearmanr


# ═══════════════════════════════════════════════════════════════════════════
# Expert reference sets (Table B.1)
# ═══════════════════════════════════════════════════════════════════════════

EXPERT_REFERENCE_SETS: Dict[str, Set[str]] = {
    "dos": {"src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "duration", "conn_state"},
    "ddos": {"src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "duration", "conn_state"},
    "scanning": {"dst_pkts", "duration", "conn_state", "service", "proto"},
    "ransomware": {"src_bytes", "dst_bytes", "duration", "missed_bytes", "conn_state"},
    "backdoor": {"duration", "src_bytes", "dst_bytes", "conn_state", "proto"},
    "injection": {"src_bytes", "dst_bytes", "service", "conn_state", "proto"},
    "xss": {"src_bytes", "dst_bytes", "service", "conn_state", "duration"},
    "password": {"duration", "src_pkts", "dst_pkts", "conn_state", "proto"},
    "mitm": {"src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "conn_state"},
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Faithfulness
# ═══════════════════════════════════════════════════════════════════════════

def _baseline_values(X: np.ndarray, method: str = "median") -> np.ndarray:
    """Return a (n_features,) baseline vector from *X*.

    Parameters
    ----------
    X : (n_samples, n_features)
    method : ``"median"`` (default) or ``"mean"``
    """
    if method == "median":
        return np.median(X, axis=0)
    elif method == "mean":
        return np.mean(X, axis=0)
    raise ValueError(f"Unknown baseline method: {method!r}")


def _predicted_confidence(model_predict_proba: Callable, X: np.ndarray) -> np.ndarray:
    """Return the predicted probability of the *positive* (attack) class.

    Handles both binary (n_classes=2, take column 1) and single-column
    output (anomaly scores etc.).
    """
    proba = model_predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.ravel()


def compute_comprehensiveness(
    model_predict_proba: Callable,
    X: np.ndarray,
    shap_values: np.ndarray,
    k: int = 5,
    baseline: str = "median",
) -> float:
    """Faithfulness: remove top-*k* attributed features, measure mean confidence drop.

    For each sample the *k* features with the largest ``|shap_values|`` are
    replaced by dataset-wide baseline values.  A higher comprehensiveness
    score indicates more faithful explanations (removing important features
    degrades predictions more).

    Parameters
    ----------
    model_predict_proba : callable
        ``model.predict_proba`` or equivalent — takes ``(n, d)`` and returns
        ``(n, n_classes)`` or ``(n,)``.
    X : (n_samples, n_features) float array, preprocessed.
    shap_values : (n_samples, n_features) attribution array.
    k : number of features to remove per sample.
    baseline : ``"median"`` or ``"mean"``.

    Returns
    -------
    float
        Mean confidence drop across samples.  Range roughly [0, 1].
    """
    if X.shape[0] == 0:
        return 0.0

    k = min(k, X.shape[1])
    base = _baseline_values(X, method=baseline)
    p_orig = _predicted_confidence(model_predict_proba, X)

    X_pert = X.copy()
    abs_shap = np.abs(shap_values)
    # For each sample, replace top-k features with baseline
    top_k_idx = np.argsort(-abs_shap, axis=1)[:, :k]
    rows = np.arange(X.shape[0])[:, None]
    X_pert[rows, top_k_idx] = base[top_k_idx]

    p_pert = _predicted_confidence(model_predict_proba, X_pert)
    return float(np.mean(p_orig - p_pert))


def compute_sufficiency(
    model_predict_proba: Callable,
    X: np.ndarray,
    shap_values: np.ndarray,
    k: int = 5,
    baseline: str = "median",
) -> float:
    """Faithfulness: retain only top-*k* features, replace all others with baseline.

    A *lower* sufficiency score means the top-*k* features alone are
    sufficient to reproduce the original prediction (faithful explanation).

    Computed as ``mean(p_original − p_retained)``.

    Parameters
    ----------
    model_predict_proba : callable
    X : (n_samples, n_features)
    shap_values : (n_samples, n_features)
    k : features to *keep*.
    baseline : ``"median"`` or ``"mean"``.

    Returns
    -------
    float
        Mean confidence drop.  Lower = better sufficiency.
    """
    if X.shape[0] == 0:
        return 0.0

    k = min(k, X.shape[1])
    base = _baseline_values(X, method=baseline)
    p_orig = _predicted_confidence(model_predict_proba, X)

    # Start from all-baseline, then restore top-k per sample
    X_ret = np.tile(base, (X.shape[0], 1))
    abs_shap = np.abs(shap_values)
    top_k_idx = np.argsort(-abs_shap, axis=1)[:, :k]
    rows = np.arange(X.shape[0])[:, None]
    X_ret[rows, top_k_idx] = X[rows, top_k_idx]

    p_ret = _predicted_confidence(model_predict_proba, X_ret)
    return float(np.mean(p_orig - p_ret))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Stability
# ═══════════════════════════════════════════════════════════════════════════

def _mean_abs_shap_ranking(shap_values: np.ndarray) -> np.ndarray:
    """Return feature *ranks* (1-based) by descending mean |SHAP|.

    Ties are broken by feature index (stable sort).  The returned array has
    shape ``(n_features,)`` where ``ranking[j]`` is the rank of feature *j*.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    # argsort descending → feature indices ordered by importance
    order = np.argsort(-mean_abs, kind="stable")
    ranking = np.empty_like(order)
    ranking[order] = np.arange(1, len(order) + 1)
    return ranking


def compute_seed_stability(
    build_and_explain_fn: Callable[[int], np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_seeds: int = 5,
    k: int = 10,
) -> Dict[str, Any]:
    """Protocol S1: retrain with different seeds, compare feature rankings.

    ``build_and_explain_fn(seed)`` should build a model with the given random
    seed, fit on ``(X_train, y_train)``, explain ``X_test``, and return a
    ``(n_test, n_features)`` SHAP-value array.

    For every pair of seeds the Spearman rank correlation of the global
    mean-|SHAP| feature rankings is computed.

    Parameters
    ----------
    build_and_explain_fn : callable(seed) → ndarray
    X_train, y_train : training data (passed through for context;
        the callable decides how to use them).
    X_test : test data.
    n_seeds : number of random seeds.
    k : unused but reserved for top-*k* restricted ranking in future.

    Returns
    -------
    dict
        ``{'mean_rho', 'std_rho', 'n_pairs', 'all_rho'}``.
    """
    rankings: List[np.ndarray] = []
    for seed in range(n_seeds):
        sv = build_and_explain_fn(seed)
        rankings.append(_mean_abs_shap_ranking(sv))

    rhos: List[float] = []
    for (i, ri), (j, rj) in combinations(enumerate(rankings), 2):
        rho, _ = spearmanr(ri, rj)
        rhos.append(float(rho))

    if not rhos:
        return {"mean_rho": float("nan"), "std_rho": 0.0, "n_pairs": 0, "all_rho": []}

    return {
        "mean_rho": float(np.mean(rhos)),
        "std_rho": float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
        "n_pairs": len(rhos),
        "all_rho": rhos,
    }


def compute_noise_stability(
    explain_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    sigma_frac: float = 0.01,
    n_draws: int = 20,
    k: int = 10,
) -> Dict[str, Any]:
    """Protocol S2: add Gaussian noise to inputs, compare rankings.

    ``explain_fn(X)`` takes an ``(n, d)`` array and returns an ``(n, d)``
    SHAP-value array.

    The noise standard deviation per feature is ``sigma_frac * std(X, axis=0)``.

    Parameters
    ----------
    explain_fn : callable(X) → ndarray
    X_test : (n_samples, n_features)
    sigma_frac : fraction of per-feature std used as noise scale.
    n_draws : number of noisy copies.
    k : unused, reserved for top-*k* restricted ranking.

    Returns
    -------
    dict
        ``{'mean_rho', 'std_rho', 'n_draws'}``.
    """
    if X_test.shape[0] == 0:
        return {"mean_rho": float("nan"), "std_rho": 0.0, "n_draws": 0}

    sigma = sigma_frac * np.std(X_test, axis=0)
    # Zero-variance features get zero noise
    sigma = np.nan_to_num(sigma, nan=0.0)

    base_sv = explain_fn(X_test)
    base_rank = _mean_abs_shap_ranking(base_sv)

    rng = np.random.default_rng(seed=0)
    rhos: List[float] = []
    for _ in range(n_draws):
        noise = rng.normal(0.0, 1.0, size=X_test.shape) * sigma[None, :]
        X_noisy = X_test + noise
        sv_noisy = explain_fn(X_noisy)
        rank_noisy = _mean_abs_shap_ranking(sv_noisy)
        rho, _ = spearmanr(base_rank, rank_noisy)
        rhos.append(float(rho))

    return {
        "mean_rho": float(np.mean(rhos)),
        "std_rho": float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
        "n_draws": len(rhos),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Simplicity
# ═══════════════════════════════════════════════════════════════════════════

def compute_k90(shap_values: np.ndarray) -> float:
    """Minimum features for 90 % of total attribution mass, normalised by *d*.

    Steps:
        1. Compute mean |SHAP| per feature across samples.
        2. Sort descending.
        3. Find minimum *k* such that ``sum(top-k) >= 0.9 * total``.
        4. Return ``k / n_features``.

    Lower is better (more concentrated explanations).

    Parameters
    ----------
    shap_values : (n_samples, n_features)

    Returns
    -------
    float
        Ratio in (0, 1].  Returns 1.0 when all attributions are zero.
    """
    n_features = shap_values.shape[1]
    if n_features == 0:
        return 1.0

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total = mean_abs.sum()
    if total == 0.0:
        return 1.0

    sorted_desc = np.sort(mean_abs)[::-1]
    cumsum = np.cumsum(sorted_desc)
    threshold = 0.9 * total
    # argmax finds the *first* True index
    k = int(np.argmax(cumsum >= threshold)) + 1
    return k / n_features


def compute_gini(shap_values: np.ndarray) -> float:
    """Gini coefficient of the mean |SHAP| attribution distribution.

    A Gini of 1.0 means all attribution is concentrated on a single feature
    (maximum simplicity); 0.0 means perfectly uniform attribution (no
    concentration).

    Parameters
    ----------
    shap_values : (n_samples, n_features)

    Returns
    -------
    float
        Gini coefficient in [0, 1].  Returns 0.0 when all attributions are
        zero or there is only one feature.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    n = len(mean_abs)
    if n <= 1 or mean_abs.sum() == 0.0:
        return 0.0

    a = np.sort(mean_abs)  # ascending
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * a)) / (n * np.sum(a)) - (n + 1.0) / n)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Plausibility
# ═══════════════════════════════════════════════════════════════════════════

def _normalise_feature_name(name: str) -> str:
    """Lowercase, strip common sklearn column-transformer prefixes."""
    name = name.lower().strip()
    # Strip prefixes like "num__", "cat__", "remainder__"
    for prefix in ("num__", "cat__", "remainder__", "passthrough__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def compute_rma_at_k(
    top_k_features: List[str],
    reference_set: Set[str],
    k: int = 5,
) -> float:
    """Relevant-feature Match Accuracy at *k*.

    ``RMA@k = |top_k_features[:k] ∩ reference_set| / k``

    Parameters
    ----------
    top_k_features : ordered list of feature names (most important first).
    reference_set : expert-defined set of expected important features.
    k : number of top features to consider.

    Returns
    -------
    float
        Score in [0, 1].
    """
    if k <= 0:
        return 0.0
    top = [_normalise_feature_name(f) for f in top_k_features[:k]]
    ref = {_normalise_feature_name(f) for f in reference_set}
    matches = 0
    for f in top:
        # Exact match OR one-hot expansion of a reference feature
        # (e.g. "conn_state_s0" matches reference "conn_state")
        if f in ref or any(f.startswith(r + "_") for r in ref):
            matches += 1
    return matches / k


def compute_plausibility_scores(
    shap_values: np.ndarray,
    feature_names: List[str],
    attack_types: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    """Compute RMA@k for each attack type present in *attack_types*.

    Only attack types that appear in both the data and
    ``EXPERT_REFERENCE_SETS`` are scored.

    Parameters
    ----------
    shap_values : (n_samples, n_features)
    feature_names : length-*d* list of feature names.
    attack_types : (n_samples,) string labels.
    k : top-*k* features for RMA.

    Returns
    -------
    dict
        ``{attack_type: RMA@k}`` for each matched type.
    """
    results: Dict[str, float] = {}
    unique_types = np.unique(attack_types)

    # Build normalised lookup for reference sets
    ref_lookup: Dict[str, Set[str]] = {
        key.lower(): val for key, val in EXPERT_REFERENCE_SETS.items()
    }

    for atype in unique_types:
        atype_str = str(atype).lower().strip()
        if atype_str not in ref_lookup:
            continue

        mask = attack_types == atype
        sv_subset = shap_values[mask]
        if sv_subset.shape[0] == 0:
            continue

        mean_abs = np.mean(np.abs(sv_subset), axis=0)
        top_idx = np.argsort(-mean_abs)[:k]
        top_features = [feature_names[i] for i in top_idx]
        results[str(atype)] = compute_rma_at_k(
            top_features, ref_lookup[atype_str], k=k
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. Timeliness
# ═══════════════════════════════════════════════════════════════════════════

def measure_timeliness(
    explain_fn: Callable[[np.ndarray], np.ndarray],
    X_sample: np.ndarray,
    n_runs: int = 3,
) -> Dict[str, float]:
    """Measure wall-clock time to generate explanations.

    Parameters
    ----------
    explain_fn : callable(X) → ndarray
    X_sample : representative input batch.
    n_runs : number of timing repetitions.

    Returns
    -------
    dict
        ``{'mean_ms_per_sample', 'std_ms_per_sample', 'n_samples',
           'total_ms_mean'}``.
    """
    n_samples = X_sample.shape[0]
    if n_samples == 0:
        return {
            "mean_ms_per_sample": 0.0,
            "std_ms_per_sample": 0.0,
            "n_samples": 0,
            "total_ms_mean": 0.0,
        }

    times_ms: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = explain_fn(X_sample)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)

    per_sample = [t / n_samples for t in times_ms]
    return {
        "mean_ms_per_sample": float(np.mean(per_sample)),
        "std_ms_per_sample": float(np.std(per_sample, ddof=1)) if len(per_sample) > 1 else 0.0,
        "n_samples": n_samples,
        "total_ms_mean": float(np.mean(times_ms)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. OXS Composite Score
# ═══════════════════════════════════════════════════════════════════════════

def compute_oxs(
    faithfulness: float,
    stability: float,
    simplicity: float,
    plausibility: float,
    timeliness: float,
    weights: Optional[List[float]] = None,
) -> float:
    """Compute the OXS composite explanation quality score.

    All five inputs should be normalised to [0, 1] where **higher = better**.

    For timeliness the caller should invert the raw timing, e.g.::

        timeliness_norm = 1.0 / (1.0 + ms_per_sample / 1000.0)

    Parameters
    ----------
    faithfulness, stability, simplicity, plausibility, timeliness : float
        Normalised criterion scores.
    weights : optional list of 5 floats (default: equal 0.2 each).

    Returns
    -------
    float
        Weighted average in [0, 1].
    """
    scores = np.array([faithfulness, stability, simplicity, plausibility, timeliness])

    if weights is None:
        w = np.full(5, 0.2)
    else:
        if len(weights) != 5:
            raise ValueError(f"Expected 5 weights, got {len(weights)}")
        w = np.array(weights, dtype=np.float64)
        w_sum = w.sum()
        if w_sum == 0:
            raise ValueError("Weights must not sum to zero")
        w = w / w_sum  # normalise

    return float(np.dot(w, scores))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Full evaluation runner
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_explanations(
    model_predict_proba: Callable,
    explain_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    shap_values: np.ndarray,
    feature_names: List[str],
    attack_types: Optional[np.ndarray] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """Run all five XAI evaluation criteria and return a comprehensive results dict.

    Parameters
    ----------
    model_predict_proba : callable
        ``model.predict_proba`` — takes ``(n, d)``, returns ``(n, c)``.
    explain_fn : callable
        ``explain_fn(X)`` → ``(n, d)`` SHAP values.
    X_test : (n_samples, n_features) preprocessed test data.
    shap_values : (n_samples, n_features) precomputed attribution array.
    feature_names : length-*d* feature name list.
    attack_types : optional (n_samples,) string labels for plausibility.
    k : top-*k* for comprehensiveness, sufficiency, and plausibility.

    Returns
    -------
    dict
        Nested dictionary with keys ``faithfulness``, ``stability``,
        ``simplicity``, ``plausibility``, ``timeliness``, and ``oxs``.
    """
    results: Dict[str, Any] = {}

    # --- 1. Faithfulness ---
    comp = compute_comprehensiveness(model_predict_proba, X_test, shap_values, k=k)
    suff = compute_sufficiency(model_predict_proba, X_test, shap_values, k=k)
    results["faithfulness"] = {
        "comprehensiveness": comp,
        "sufficiency": suff,
    }

    # --- 2. Stability (noise only — seed requires retraining) ---
    noise_stab = compute_noise_stability(explain_fn, X_test, sigma_frac=0.01, n_draws=10)
    results["stability"] = {
        "noise": noise_stab,
    }

    # --- 3. Simplicity ---
    k90 = compute_k90(shap_values)
    gini = compute_gini(shap_values)
    results["simplicity"] = {
        "k90": k90,
        "gini": gini,
    }

    # --- 4. Plausibility ---
    if attack_types is not None and len(attack_types) > 0:
        plaus = compute_plausibility_scores(shap_values, feature_names, attack_types, k=k)
        plaus_mean = float(np.mean(list(plaus.values()))) if plaus else 0.0
    else:
        plaus = {}
        plaus_mean = 0.0
    results["plausibility"] = {
        "per_type_rma_at_k": plaus,
        "mean_rma_at_k": plaus_mean,
    }

    # --- 5. Timeliness ---
    # Use a subsample to keep timing affordable
    max_timing = min(200, X_test.shape[0])
    timing = measure_timeliness(explain_fn, X_test[:max_timing], n_runs=3)
    results["timeliness"] = timing

    # --- OXS composite ---
    # Normalise each criterion to [0,1] higher-is-better
    faith_norm = comp  # comprehensiveness already in ~[0,1]
    stab_norm = max(0.0, noise_stab.get("mean_rho", 0.0))
    simpl_norm = gini  # higher Gini = more concentrated = simpler
    plaus_norm = plaus_mean
    time_norm = 1.0 / (1.0 + timing["mean_ms_per_sample"] / 1000.0)

    oxs = compute_oxs(faith_norm, stab_norm, simpl_norm, plaus_norm, time_norm)
    results["oxs"] = {
        "score": oxs,
        "normalised_inputs": {
            "faithfulness": faith_norm,
            "stability": stab_norm,
            "simplicity": simpl_norm,
            "plausibility": plaus_norm,
            "timeliness": time_norm,
        },
    }

    return results
