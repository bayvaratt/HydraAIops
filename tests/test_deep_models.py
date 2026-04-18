"""Tests for hydra.models.deep — CNN-LSTM classifier."""
import pytest
import numpy as np

# Skip all tests if torch not available
torch = pytest.importorskip("torch")

from hydra.models.deep import CNNLSTMClassifier, build_cnn_lstm


def test_cnn_lstm_binary_fit_predict():
    """Binary classification: fit on synthetic data, check output shapes."""
    rng = np.random.default_rng(42)
    X = rng.random((200, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=200)
    clf = CNNLSTMClassifier(hidden=16, epochs=3, batch_size=64, patience=2, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    preds = clf.predict(X)
    assert preds.shape == (200,)
    assert set(preds).issubset({0, 1})


def test_cnn_lstm_multiclass_fit_predict():
    """Multiclass classification: 4 classes."""
    rng = np.random.default_rng(42)
    X = rng.random((200, 10)).astype(np.float32)
    y = rng.integers(0, 4, size=200)
    clf = CNNLSTMClassifier(hidden=16, epochs=3, batch_size=64, patience=2, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (200, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_cnn_lstm_sklearn_interface():
    """get_params and set_params work correctly."""
    clf = CNNLSTMClassifier(hidden=32, epochs=10)
    params = clf.get_params()
    assert params["hidden"] == 32
    assert params["epochs"] == 10
    clf.set_params(hidden=64)
    assert clf.hidden == 64


def test_build_cnn_lstm_returns_modelspec():
    """build_cnn_lstm returns a ModelSpec with correct name."""
    from hydra.models.tabular import ModelSpec
    spec = build_cnn_lstm(42)
    assert isinstance(spec, ModelSpec)
    assert spec.name == "cnn_lstm"
    assert spec.backend == "torch"


def test_predict_proba_before_fit_raises():
    """predict_proba before fit should raise RuntimeError."""
    clf = CNNLSTMClassifier()
    with pytest.raises(RuntimeError, match="fit"):
        clf.predict_proba(np.zeros((5, 10), dtype=np.float32))


def test_sklearn_is_fitted():
    """__sklearn_is_fitted__ reflects fit state."""
    clf = CNNLSTMClassifier()
    assert not clf.__sklearn_is_fitted__()
    rng = np.random.default_rng(0)
    clf.fit(rng.random((100, 6)).astype(np.float32), rng.integers(0, 2, size=100))
    assert clf.__sklearn_is_fitted__()
