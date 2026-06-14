"""Data drift detection between a reference and a current dataset.

Models silently degrade when production data drifts away from training data, so
a go-to explainability toolkit needs monitoring too. We report, per feature:

* **PSI (Population Stability Index)** -- the standard industry drift score.
  Rules of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant.
* **Kolmogorov-Smirnov** statistic/p-value for numeric features (when SciPy is
  available) as a complementary distributional test.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .schema import DriftReport, DriftFeature

PSI_THRESHOLD = 0.25


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    reference = reference[~pd.isna(reference)]
    current = current[~pd.isna(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Quantile bins from the reference distribution.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = PSI_THRESHOLD,
) -> DriftReport:
    common = [c for c in reference.columns if c in current.columns]
    features: list[DriftFeature] = []
    n_drifted = 0

    try:
        from scipy.stats import ks_2samp
        has_scipy = True
    except Exception:
        has_scipy = False

    for col in common:
        ref, cur = reference[col], current[col]
        df = DriftFeature(feature=col)
        if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
            df.psi = population_stability_index(ref.to_numpy(dtype=float), cur.to_numpy(dtype=float))
            if has_scipy:
                stat, p = ks_2samp(ref.dropna(), cur.dropna())
                df.ks_statistic, df.ks_pvalue = float(stat), float(p)
        else:
            # Categorical PSI on value frequencies.
            ref_freq = ref.value_counts(normalize=True)
            cur_freq = cur.value_counts(normalize=True)
            cats = set(ref_freq.index) | set(cur_freq.index)
            rp = np.clip(np.array([ref_freq.get(c, 0) for c in cats]), 1e-6, None)
            cp = np.clip(np.array([cur_freq.get(c, 0) for c in cats]), 1e-6, None)
            df.psi = float(np.sum((cp - rp) * np.log(cp / rp)))

        df.drifted = df.psi is not None and df.psi > psi_threshold
        n_drifted += int(df.drifted)
        features.append(df)

    return DriftReport(
        n_features=len(common),
        n_drifted=n_drifted,
        psi_threshold=psi_threshold,
        drifted=n_drifted > 0,
        features=features,
    )
