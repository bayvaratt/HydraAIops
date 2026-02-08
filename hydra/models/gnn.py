"""Optional GNN scaffold for communication-graph modeling.

This module is a placeholder by design. Implementing a full GNN pipeline
requires dataset-specific graph construction, temporal windowing, and a
training loop with deep learning dependencies.
"""


def train_gnn(*args, **kwargs):
    raise NotImplementedError(
        "GNN pipeline is not implemented yet. TODO: build graph constructor, "
        "define edge labels, and add a training loop."
    )
