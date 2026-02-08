from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ExplanationRecord:
    sample_id: int
    model_name: str
    pred_label: int
    pred_proba: float
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    top_edges: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "pred_label": int(self.pred_label),
            "pred_proba": float(self.pred_proba),
            "top_features": self.top_features,
            "top_edges": self.top_edges,
            "metadata": self.metadata,
        }
        return payload


def default_metadata(
    feature_regime: str,
    dataset_name: str,
    seed: int,
) -> Dict[str, Any]:
    return {
        "feature_regime": feature_regime,
        "dataset_name": dataset_name,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
