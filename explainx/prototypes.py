"""Example-based explanation: prototypes and criticisms.

Sometimes the best explanation is *"here are the cases this model/data is built
around, and here are the ones it represents poorly."* Prototypes are the most
representative examples of the data distribution; criticisms are points in
regions the prototypes cover badly (outliers / under-represented cases). This is
the MMD-critic / ProtoDash family of methods (Kim et al., 2016) -- a callback to
the original explainX's Protodash, reimplemented dependency-light.

We use an RBF kernel over standardized features: prototypes are chosen greedily
to maximize closeness to the full dataset (high mean kernel similarity), and
criticisms are the points least well covered by the chosen prototypes.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _rbf_kernel(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    d2 = np.maximum(d2, 0)
    gamma = 1.0 / X.shape[1] if X.shape[1] else 1.0
    return np.exp(-gamma * d2)


def _standardize(X: pd.DataFrame) -> np.ndarray:
    numeric = X.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        numeric = pd.get_dummies(X)
    arr = numeric.to_numpy(dtype=float)
    std = arr.std(axis=0)
    std[std == 0] = 1.0
    return (arr - arr.mean(axis=0)) / std


def prototypes_and_criticisms(
    X: pd.DataFrame,
    n_prototypes: int = 5,
    n_criticisms: int = 3,
) -> tuple[list[int], list[int]]:
    Z = _standardize(X)
    K = _rbf_kernel(Z)
    n = K.shape[0]
    mean_sim = K.mean(axis=1)  # similarity of each point to the whole dataset

    # Greedy prototype selection: maximize coverage of the data distribution.
    selected: list[int] = []
    n_prototypes = min(n_prototypes, n)
    while len(selected) < n_prototypes:
        best_gain, best_idx = -np.inf, None
        for i in range(n):
            if i in selected:
                continue
            cand = selected + [i]
            # Objective: how well the prototype set covers the data on average.
            coverage = K[cand].max(axis=0).mean()
            if coverage > best_gain:
                best_gain, best_idx = coverage, i
        selected.append(int(best_idx))

    # Criticisms: points least covered by the chosen prototypes.
    if selected:
        covered = K[selected].max(axis=0)
    else:
        covered = mean_sim
    witness = mean_sim - covered  # high where data is dense but prototypes are far
    order = np.argsort(witness)[::-1]
    criticisms = [int(i) for i in order if i not in selected][: min(n_criticisms, n)]

    return selected, criticisms
