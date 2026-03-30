import numpy as np

from hydra.evaluation.thresholds import coverage_at_threshold, fpr_at_threshold, select_threshold_at_recall


def test_threshold_selection_and_fpr():
    y = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])

    class DummyLogger:
        def warning(self, *args, **kwargs):
            pass

    thr = select_threshold_at_recall(y, scores, 0.9, DummyLogger())
    fpr = fpr_at_threshold(y, scores, thr)
    cov = coverage_at_threshold(scores, thr)

    assert 0.0 <= fpr <= 1.0
    assert 0.0 <= cov <= 1.0
