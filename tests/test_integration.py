"""Integration tests: verify end-to-end pipeline components work together."""
import numpy as np
import pandas as pd
import pytest


class TestPipelineIntegration:
    """Smoke tests for the main pipeline components chained together."""

    def test_preprocess_then_predict(self):
        """Preprocess → fit → predict chain works for all tabular models."""
        from hydra.data.preprocess import fit_preprocessor
        from hydra.models.tabular import build_logreg, build_random_forest, build_xgboost

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "feat_a": rng.random(200),
            "feat_b": rng.random(200),
            "feat_c": rng.random(200),
            "cat_x": rng.choice(["tcp", "udp", "icmp"], 200),
        })
        y = rng.integers(0, 2, 200)

        prep = fit_preprocessor(df, categorical_cols=["cat_x"], numeric_cols=["feat_a", "feat_b", "feat_c"])
        prep.fit(df)
        X = prep.transform(df)

        for builder in [build_logreg, build_random_forest, build_xgboost]:
            spec = builder(42)
            X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            spec.model.fit(X_dense, y)
            proba = spec.model.predict_proba(X_dense)
            assert proba.shape[0] == 200
            assert proba.shape[1] == 2

    def test_cross_dataset_alignment_preserves_rows(self):
        """Aligning both datasets preserves row count and produces identical columns."""
        from hydra.data.align import align_ton_iot, align_cic_iot2023, CANONICAL_FEATURES

        ton = pd.DataFrame({
            "src_pkts": [10, 20, 30], "dst_pkts": [5, 10, 15],
            "src_bytes": [100, 200, 300], "dst_bytes": [50, 100, 150],
            "duration": [1.0, 2.0, 3.0], "proto": ["tcp", "udp", "icmp"],
            "label": [0, 1, 1], "type": ["normal", "dos", "ddos"],
        })
        cic = pd.DataFrame({
            "number": [10, 20, 30], "tot_size": [100, 200, 300],
            "iat": [1.0, 2.0, 3.0], "tcp": [1, 0, 0], "udp": [0, 1, 0], "icmp": [0, 0, 1],
            "label": [0, 1, 1], "type": ["BENIGN", "DDoS", "DoS"],
        })
        a_ton = align_ton_iot(ton)
        a_cic = align_cic_iot2023(cic)

        assert len(a_ton) == 3
        assert len(a_cic) == 3
        assert list(a_ton.columns) == list(a_cic.columns)
        assert list(a_ton.columns)[:6] == CANONICAL_FEATURES

    def test_xai_eval_pipeline(self):
        """XAI eval functions work on real model output."""
        from hydra.models.tabular import build_logreg
        from hydra.xai.xai_eval import compute_comprehensiveness, compute_k90, compute_gini

        rng = np.random.default_rng(42)
        X = rng.random((100, 5)).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(int)

        spec = build_logreg(42)
        spec.model.fit(X, y)

        # Simulate SHAP values as coefficient * feature value
        coef = spec.model.coef_[0]
        shap_values = X * coef[np.newaxis, :]

        comp = compute_comprehensiveness(spec.model.predict_proba, X, shap_values, k=2)
        k90 = compute_k90(shap_values)
        gini = compute_gini(shap_values)

        assert isinstance(comp, float)
        assert 0.0 <= k90 <= 1.0
        assert 0.0 <= gini <= 1.0
