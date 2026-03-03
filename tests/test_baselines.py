import numpy as np
import pandas as pd

from hydra.eval.metrics import compute_pr_auc
from hydra.models.baselines import baseline_majority_scores


def test_majority_baseline_pr_auc_matches_prevalence():
    y_train = pd.Series([1] * 30 + [0] * 70)
    y_test = pd.Series([1] * 20 + [0] * 80)
    scores = baseline_majority_scores(y_train, len(y_test))
    pr_auc = compute_pr_auc(y_test, scores)
    # PR-AUC of a constant-score classifier equals the *test* prevalence, not train.
    prevalence = y_test.mean()
    assert abs(pr_auc - prevalence) < 1e-6
