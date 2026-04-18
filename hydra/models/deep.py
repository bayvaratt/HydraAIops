"""CNN-LSTM classifier for network flow tabular data.

Supports both binary detection (attack vs normal) and multiclass type
classification.  The number of output classes is detected automatically
from the training labels:
  - 2 classes  → BCEWithLogitsLoss + class-weighted pos_weight (binary)
  - >2 classes → CrossEntropyLoss + Linear head (multiclass)

The wrapper implements the sklearn estimator interface (fit / predict /
predict_proba) so it drops directly into the existing run_tabular.py
Pipeline and _two_stage_metrics.

Key improvements over v1:
  - BCEWithLogitsLoss with auto pos_weight to handle class imbalance
  - BatchNorm1d after Conv1d for better gradient flow / generalisation
  - Wider Conv1d kernel (5) to reduce sensitivity to column ordering
  - Cosine-annealing LR schedule
  - Longer defaults (epochs=50, patience=10)
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from hydra.models.tabular import ModelSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_dense(X) -> np.ndarray:
    """Convert sparse matrix or DataFrame to a dense float32 numpy array."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    if hasattr(X, "values"):
        X = X.values
    return np.asarray(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTorch module (supports binary and multiclass via n_classes)
# ---------------------------------------------------------------------------

def _build_net(n_features: int, hidden: int, n_classes: int):
    """Build CNN + LSTM nn.Module for tabular IDS data.

    Binary (n_classes==2): raw logits — BCEWithLogitsLoss handles sigmoid.
    Multiclass: raw logits — CrossEntropyLoss handles softmax.

    Architecture:
      Conv1d (local feature interactions) → LSTM (sequence context) →
      Global average pool → FC head

    Key design choices:
    - Two Conv1d layers with residual-style skip to deepen representation
    - Global average pooling after LSTM rather than only last hidden state —
      averages context across all positions, better gradient flow on long seqs
    - No BatchNorm (can break with small minority-class batches)
    - Dropout for regularisation
    """
    import torch
    import torch.nn as nn

    binary = (n_classes == 2)
    out_dim = 1 if binary else n_classes

    class _CNNLSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, out_dim),
            )
            # Kaiming init for conv/linear layers to break output symmetry
            # at initialisation — default PyTorch init can yield near-constant
            # forward passes that starve early gradients.
            for m in self.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):               # x: (batch, n_features)
            x = x.unsqueeze(1)              # (batch, 1, n_features)
            x = self.conv(x)               # (batch, hidden, n_features)
            # MaxPool halves sequence length → cuts LSTM memory in half.
            # Pad odd-length sequences so no feature is silently dropped.
            if x.size(2) % 2 == 1:
                x = torch.nn.functional.pad(x, (0, 1))  # zero-pad last position
            x = torch.nn.functional.max_pool1d(x, 2)  # (batch, hidden, ceil(n_features/2))
            x = x.permute(0, 2, 1)        # (batch, n_features//2, hidden)
            # Explicitly initialise LSTM hidden state to avoid PyTorch MPS
            # bug where internal zeros() receives device as a Tensor instead
            # of a torch.device (TypeError on MPS backend).
            batch = x.size(0)
            h0 = torch.zeros(1, batch, hidden, device=x.device, dtype=x.dtype)
            c0 = torch.zeros(1, batch, hidden, device=x.device, dtype=x.dtype)
            out_seq, _ = self.lstm(x, (h0, c0))  # (batch, n_features//2, hidden)
            # Global average pool: mean over all positions (not just last hidden).
            # Avoids vanishing gradient across 35-50 LSTM time steps.
            x = out_seq.mean(dim=1)        # (batch, hidden)
            out = self.head(x)             # (batch, out_dim)
            if binary:
                out = out.squeeze(-1)      # (batch,) — raw logits
            return out

    return _CNNLSTMNet()


# ---------------------------------------------------------------------------
# Sklearn wrapper
# ---------------------------------------------------------------------------

from sklearn.base import BaseEstimator, ClassifierMixin


class CNNLSTMClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible CNN-LSTM classifier (binary and multiclass).

    Parameters
    ----------
    hidden      : Conv1d / LSTM channel width.
    epochs      : Maximum training epochs.
    batch_size  : Mini-batch size.
    lr          : Adam learning rate.
    patience    : Early-stopping patience (epochs without val-loss improvement).
    val_frac    : Fraction of training data reserved for early stopping.
    random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        hidden: int = 64,
        epochs: int = 50,
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 10,
        val_frac: float = 0.1,
        random_state: int = 42,
    ):
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state

        # set after fit()
        self.net_: Optional[object] = None
        self.device_: Optional[object] = None
        self.n_features_: Optional[int] = None
        self.n_classes_: Optional[int] = None

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "CNNLSTMClassifier":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X = _to_dense(X)
        y_int = np.asarray(y, dtype=np.int64)
        n_classes = int(np.max(y_int) + 1)
        binary = (n_classes == 2)

        self.n_features_ = X.shape[1]
        self.n_classes_ = n_classes
        self.device_ = _get_device()
        device = self.device_

        # Internal train/val split for early stopping
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_frac, random_state=self.random_state
        )
        train_idx, val_idx = next(sss.split(X, y_int))
        X_tr, y_tr = X[train_idx], y_int[train_idx]
        X_val, y_val = X[val_idx], y_int[val_idx]

        net = _build_net(self.n_features_, self.hidden, n_classes).to(device)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr)

        if binary:
            n_pos = float((y_tr == 1).sum())
            n_neg = float((y_tr == 0).sum())
            pw = n_neg / max(n_pos, 1)
            # Always apply pos_weight to handle class imbalance in both
            # directions.  When attack is majority (pw < 1), this downweights
            # the positive class so the model cannot trivially minimise loss
            # by predicting all-attack.  Without this, the optimiser collapses
            # to constant predictions and ROC-AUC becomes NaN.
            pos_weight = torch.tensor([pw], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32)
        else:
            criterion = nn.CrossEntropyLoss()
            y_tr_t  = torch.tensor(y_tr,  dtype=torch.long)
            y_val_t = torch.tensor(y_val, dtype=torch.long)

        tr_ds  = TensorDataset(torch.tensor(X_tr,  dtype=torch.float32), y_tr_t)
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_val_t)

        tr_loader  = DataLoader(tr_ds,  batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs
        )

        best_val_loss = float("inf")
        best_weights  = copy.deepcopy(net.state_dict())
        no_improve    = 0

        for _ in range(1, self.epochs + 1):
            net.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                criterion(net(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimiser.step()
            scheduler.step()

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(net(xb), yb).item() * len(xb)
            val_loss /= len(val_idx)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights  = copy.deepcopy(net.state_dict())
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        net.load_state_dict(best_weights)
        net.eval()
        self.net_ = net
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if self.net_ is None:
            raise RuntimeError("Call fit() before predict_proba()")

        X = _to_dense(X)
        device = self.device_
        binary = (self.n_classes_ == 2)
        chunks = []

        self.net_.eval()
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.tensor(
                    X[start : start + self.batch_size], dtype=torch.float32
                ).to(device)
                out = self.net_(xb)
                if binary:
                    pos = torch.sigmoid(out).cpu().numpy()   # (batch,)
                    chunks.append(np.column_stack([1.0 - pos, pos]))
                else:
                    proba = F.softmax(out, dim=-1).cpu().numpy()  # (batch, n_classes)
                    chunks.append(proba)

        return np.concatenate(chunks, axis=0)   # (n, n_classes)

    def predict(self, X) -> np.ndarray:
        """Return hard class predictions (argmax of predict_proba)."""
        return np.argmax(self.predict_proba(X), axis=1)

    # sklearn interface
    def get_params(self, deep: bool = True) -> dict:
        return {
            "hidden": self.hidden,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "patience": self.patience,
            "val_frac": self.val_frac,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "CNNLSTMClassifier":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return self.net_ is not None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_cnn_lstm(random_state: int) -> ModelSpec:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for cnn_lstm. Install with: pip install torch"
        ) from exc
    return ModelSpec(name="cnn_lstm", backend="torch",
                     model=CNNLSTMClassifier(random_state=random_state))
