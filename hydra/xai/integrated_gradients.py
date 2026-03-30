"""Integrated Gradients (Sundararajan et al. 2017) for PyTorch models.

IG_i(x, x') = (x_i - x'_i) × ∫₀¹ ∂f(x' + α(x − x'))/∂xᵢ dα

Approximated via a uniform Riemann sum with `n_steps` interpolation points.

Satisfies the completeness axiom:
    Σᵢ IG_i(x, x') ≈ f(x) − f(x')

This is the preferred attribution method for the CNN-LSTM in HydraAIops because:
  - MaxPool1d in the CNN-LSTM is non-differentiable at ties; IG averages the
    subgradient across the entire interpolation path, avoiding single-point
    artifacts that plain Gradient×Input suffers.
  - Completeness ensures that attribution magnitudes are proportional to the
    actual contribution to the output score, making them directly comparable
    across features.
  - Baseline = all-zeros represents "no network traffic" — a semantically
    meaningful uninformative reference for network intrusion detection.
    (Features are median-imputed, not standardised, so zero ≈ absent flow.)

Usage
-----
    from hydra.explain.integrated_gradients import integrated_gradients_batch
    attr = integrated_gradients_batch(model.net_, model.device_, X_dense)
    # attr: (n_samples, n_features) float32
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def integrated_gradients_batch(
    net,
    device,
    X_arr: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 50,
    batch_size: int = 32,
    max_explain: Optional[int] = 300,
) -> np.ndarray:
    """Compute Integrated Gradients for a batch of inputs.

    Parameters
    ----------
    net         : nn.Module — the raw PyTorch network (CNNLSTMClassifier.net_)
                  Must already be in eval mode; moved to `device` internally.
    device      : torch.device
    X_arr       : (n_samples, n_features) float32, already preprocessed.
    baseline    : (n_features,) baseline; defaults to all-zeros
                  (= absent-traffic reference in median-imputed feature space).
    n_steps     : Riemann sum steps.  50 is accurate; 20 is faster for
                  stability-probe calls where many noisy variants are needed.
    batch_size  : number of original samples per forward-backward chunk.
                  Each chunk expands to batch_size × n_steps inputs internally.
    max_explain : randomly subsample to this many rows if exceeded.

    Returns
    -------
    attr : (n_samples, n_features) float32
        Signed IG attributions.  Positive = pushes toward attack; negative =
        pushes toward normal.  Completeness: attr.sum(axis=1) ≈ f(X) − f(x').
    """
    import torch

    X_arr = np.asarray(X_arr, dtype=np.float32)
    n, d = X_arr.shape

    if max_explain is not None and n > max_explain:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_explain, replace=False)
        X_arr = X_arr[idx]
        n = max_explain

    if baseline is None:
        baseline = np.zeros(d, dtype=np.float32)
    else:
        baseline = np.asarray(baseline, dtype=np.float32).reshape(d)

    net.eval()
    net.to(device)

    # Wrap binary (1-D output) models so we always have a consistent scalar target
    import torch.nn as nn

    with torch.no_grad():
        probe = net(torch.zeros(1, d, dtype=torch.float32, device=device))
    is_binary = probe.dim() == 1 or (probe.dim() == 2 and probe.shape[1] == 1)

    baseline_t = torch.tensor(baseline, dtype=torch.float32, device=device)

    # Uniform alpha values: [1/n_steps, 2/n_steps, ..., 1]
    # Using right-endpoints of sub-intervals (equivalent accuracy to midpoints
    # for smooth functions; avoids gradient instability at alpha=0).
    alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=device)  # (steps,)

    total_attrs = np.zeros((n, d), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = torch.tensor(X_arr[start:end], dtype=torch.float32, device=device)
        bs = X_batch.shape[0]

        delta = X_batch - baseline_t[None, :]  # (bs, d)

        # Build (bs, n_steps, d) interpolated grid, then flatten to (bs*steps, d)
        X_interp = (
            baseline_t[None, None, :]
            + alphas[None, :, None] * delta[:, None, :]
        )  # (bs, steps, d)
        X_flat = X_interp.reshape(bs * n_steps, d)  # (bs*steps, d)
        X_flat = X_flat.detach().requires_grad_(True)

        out = net(X_flat)  # (bs*steps,) binary  OR  (bs*steps, n_classes) multiclass

        if is_binary:
            score = out.view(-1)  # (bs*steps,)
        else:
            score = out[:, -1]   # take last class (attack) for multiclass

        score.sum().backward()

        grads = X_flat.grad.detach().cpu().numpy()  # (bs*steps, d)

        # Reshape → average over steps → Riemann approximation of the integral
        avg_grads = grads.reshape(bs, n_steps, d).mean(axis=1)  # (bs, d)
        delta_np = (X_arr[start:end] - baseline)                # (bs, d)
        total_attrs[start:end] = avg_grads * delta_np

    return total_attrs
