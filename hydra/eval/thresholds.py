from __future__ import annotations

import numpy as np


def select_threshold_at_recall(y_true, scores, target_recall, logger):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in validation labels; threshold selection is ill-defined")
        return float(np.nan)

    order = np.argsort(-scores)
    y_sorted = y_true[order]
    scores_sorted = scores[order]

    tp = 0
    total_pos = y_true.sum()
    best_threshold = scores_sorted[-1]

    for i, s in enumerate(scores_sorted):
        if y_sorted[i] == 1:
            tp += 1
        recall = tp / max(total_pos, 1)
        if recall >= target_recall:
            best_threshold = s
            break

    if tp / max(total_pos, 1) < target_recall:
        logger.warning("Target recall %.2f not reached; using lowest threshold", target_recall)
        best_threshold = scores_sorted[-1]

    return float(best_threshold)


def select_threshold_max_precision_at_recall(y_true, scores, target_recall, logger):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in validation labels; threshold selection is ill-defined")
        return float("nan"), False

    order = np.argsort(-scores)
    y_sorted = y_true[order]
    scores_sorted = scores[order]

    tp = 0
    fp = 0
    total_pos = y_true.sum()
    best_threshold = scores_sorted[-1]
    best_precision = -1.0
    best_recall = 0.0
    recall_target_met = False

    for i, s in enumerate(scores_sorted):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / max(total_pos, 1)
        precision = tp / max(tp + fp, 1)
        if recall >= target_recall:
            recall_target_met = True
            if precision > best_precision:
                best_precision = precision
                best_recall = recall
                best_threshold = s

    if not recall_target_met:
        logger.warning("Target recall %.2f not reached; using max-recall threshold", target_recall)
        tp = 0
        fp = 0
        best_recall = 0.0
        best_precision = -1.0
        best_threshold = scores_sorted[-1]
        for i, s in enumerate(scores_sorted):
            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            recall = tp / max(total_pos, 1)
            precision = tp / max(tp + fp, 1)
            if recall > best_recall or (recall == best_recall and precision > best_precision):
                best_recall = recall
                best_precision = precision
                best_threshold = s

    return float(best_threshold), recall_target_met


def fpr_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)

    fp = ((preds == 1) & (y_true == 0)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()
    if (fp + tn) == 0:
        return float("nan")
    return fp / (fp + tn)


def coverage_at_threshold(scores, threshold):
    scores = np.asarray(scores)
    return float((scores >= threshold).mean())


def precision_recall_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(precision), float(recall)


def precision_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    if (tp + fp) == 0:
        return float("nan")
    return tp / (tp + fp)
