from __future__ import annotations

import numpy as np


def _as_arrays(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    return y, s


def precision_recall_at_threshold(y_true, scores, threshold: float) -> tuple[float, float]:
    y, s = _as_arrays(y_true, scores)
    pred = s >= float(threshold)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(precision), float(recall)


def fpr_at_threshold(y_true, scores, threshold: float) -> float:
    y, s = _as_arrays(y_true, scores)
    pred = s >= float(threshold)
    fp = int(np.sum(pred & (y == 0)))
    n_neg = int(np.sum(y == 0))
    return float(fp / max(n_neg, 1))


def coverage_at_threshold(scores, threshold: float) -> float:
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return 0.0
    return float(np.mean(s >= float(threshold)))


def select_threshold_max_precision_at_recall(
    y_true,
    scores,
    recall_target: float,
    logger,
) -> tuple[float, bool]:
    y, s = _as_arrays(y_true, scores)
    if s.size == 0:
        raise ValueError("scores must not be empty")

    finite_scores = s[np.isfinite(s)]
    if finite_scores.size == 0:
        raise ValueError("scores must contain at least one finite value")

    thresholds = np.unique(finite_scores)[::-1]
    candidates: list[tuple[float, float, float]] = []
    best_fallback: tuple[float, float, float] | None = None

    for threshold in thresholds:
        precision, recall = precision_recall_at_threshold(y, s, threshold)
        candidate = (precision, recall, float(threshold))
        if best_fallback is None or (recall, precision, threshold) > (
            best_fallback[1],
            best_fallback[0],
            best_fallback[2],
        ):
            best_fallback = candidate
        if recall >= recall_target:
            candidates.append(candidate)

    if candidates:
        precision, recall, threshold = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        return threshold, True

    assert best_fallback is not None
    precision, recall, threshold = best_fallback
    logger.warning(
        "Could not meet recall target %.3f; using threshold %.6f with recall %.3f and precision %.3f",
        recall_target,
        threshold,
        recall,
        precision,
    )
    return threshold, False


def select_threshold_at_recall(y_true, scores, recall_target: float, logger) -> float:
    threshold, _ = select_threshold_max_precision_at_recall(y_true, scores, recall_target, logger)
    return threshold
