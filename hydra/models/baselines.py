from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hydra.eval.detection import find_threshold_at_recall

logger = logging.getLogger(__name__)


@dataclass
class MajorityClassBaseline:
    majority_class: int = 0
    prevalence: float = 0.0

    def fit(self, y: pd.Series) -> "MajorityClassBaseline":
        if len(y) == 0:
            self.prevalence = 0.0
            self.majority_class = 0
            return self
        self.prevalence = float(np.mean(y.astype(float)))
        self.majority_class = int(self.prevalence >= 0.5)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class, dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = float(self.prevalence)
        proba = np.zeros((len(X), 2), dtype=float)
        proba[:, 1] = p
        proba[:, 0] = 1.0 - p
        return proba


@dataclass
class ThresholdBaseline:
    threshold: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, recall_target: float = 0.9) -> "ThresholdBaseline":
        scores = self._score(X)
        self.threshold = find_threshold_at_recall(y.values, scores, recall_target)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._score(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return np.vstack([1 - scores, scores]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._score(X)
        thresh = self.threshold if self.threshold is not None else np.median(scores)
        return (scores >= thresh).astype(int)

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        def _col(name: str) -> pd.Series:
            if name not in X.columns:
                return pd.Series(0.0, index=X.index)
            return pd.to_numeric(X[name], errors="coerce").fillna(0.0)

        duration = _col("duration")
        bytes_sum = _col("src_bytes") + _col("dst_bytes")
        pkts_sum = _col("src_pkts") + _col("dst_pkts")
        scores = np.log1p(duration) + np.log1p(bytes_sum) + np.log1p(pkts_sum)
        return scores.to_numpy(dtype=float)
