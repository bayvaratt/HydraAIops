"""Tests for hydra.xai.integrated_gradients — IG attribution."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra.xai.integrated_gradients import integrated_gradients_batch


def _make_simple_net():
    """Linear model: output = sum(x * weights). IG should recover weights."""
    import torch.nn as nn
    net = nn.Linear(5, 1, bias=False)
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]]))
    net.eval()
    return net


def test_ig_output_shape():
    net = _make_simple_net()
    X = np.random.default_rng(42).random((10, 5)).astype(np.float32)
    attr = integrated_gradients_batch(net, torch.device("cpu"), X, n_steps=20, max_explain=None)
    assert attr.shape == (10, 5)


def test_ig_completeness():
    """Sum of attributions should approximately equal f(x) - f(baseline)."""
    net = _make_simple_net()
    X = np.array([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    baseline = np.zeros(5, dtype=np.float32)
    attr = integrated_gradients_batch(net, torch.device("cpu"), X, baseline=baseline, n_steps=100, max_explain=None)

    with torch.no_grad():
        f_x = net(torch.tensor(X)).item()
        f_base = net(torch.tensor(baseline.reshape(1, -1))).item()

    assert abs(attr.sum() - (f_x - f_base)) < 0.1


def test_ig_zero_features_get_zero_attribution():
    """Features that are zero (= baseline) should get ~zero attribution."""
    net = _make_simple_net()
    X = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    attr = integrated_gradients_batch(net, torch.device("cpu"), X, n_steps=50, max_explain=None)
    # Features 1-4 are at baseline (0), so their attribution should be near 0
    assert np.abs(attr[0, 1:]).max() < 0.01


def test_ig_max_explain_subsamples():
    """When n > max_explain, output should be truncated."""
    net = _make_simple_net()
    X = np.random.default_rng(42).random((50, 5)).astype(np.float32)
    attr = integrated_gradients_batch(net, torch.device("cpu"), X, n_steps=10, max_explain=20)
    assert attr.shape == (20, 5)


def test_ig_attribution_signs():
    """Positive weights + positive input should yield positive attributions."""
    net = _make_simple_net()
    X = np.array([[2.0, 3.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    attr = integrated_gradients_batch(net, torch.device("cpu"), X, n_steps=50, max_explain=None)
    # Features 0-2 have positive weights and positive inputs → positive attr
    assert attr[0, 0] > 0
    assert attr[0, 1] > 0
    assert attr[0, 2] > 0
