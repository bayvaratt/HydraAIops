import numpy as np

from hydra.eval.detection import pr_auc, find_threshold_at_recall, fpr_at_fixed_recall


def test_detection_metrics():
    y_true = np.array([0, 1, 0, 1, 1])
    y_score = np.array([0.1, 0.8, 0.2, 0.7, 0.9])

    pr = pr_auc(y_true, y_score)
    assert 0.0 <= pr <= 1.0

    threshold = find_threshold_at_recall(y_true, y_score, recall_target=0.6)
    fpr = fpr_at_fixed_recall(y_true, y_score, recall_target=0.6, threshold=threshold)
    assert 0.0 <= fpr <= 1.0
