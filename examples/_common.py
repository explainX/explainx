"""Shared helpers for the framework examples (a small, slightly biased dataset)."""

import numpy as np
import pandas as pd


def loan_data(n=600, seed=0):
    """A tiny loan-approval dataset with a mild gender effect, for demos."""
    rng = np.random.RandomState(seed)
    gender = rng.randint(0, 2, n)
    income = rng.normal(50, 15, n)
    credit_score = rng.normal(650, 80, n)
    debt_ratio = rng.uniform(0, 1, n)
    approved = (
        0.04 * (income - 50) + 0.02 * (credit_score - 650) - 2.5 * debt_ratio
        + 1.0 * gender + rng.normal(0, 0.5, n) > 0
    ).astype(int)
    X = pd.DataFrame({"gender": gender, "income": income, "credit_score": credit_score, "debt_ratio": debt_ratio})
    return X, approved


def show(report, ex=None):
    """Print a compact view of an explainX report and a couple of modules."""
    print(report.summary)
    if ex is not None:
        top = ex.importance().top(3)
        print("\nTop features:", ", ".join(f"{f.feature}={f.importance:.3f}" for f in top))
        local = ex.explain(0, top_k=3)
        drivers = ", ".join(f"{c.feature} ({c.contribution:+.3f})" for c in local.contributions)
        print(f"Row 0 -> {local.prediction} via {local.method}: {drivers}")
