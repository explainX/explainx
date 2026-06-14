"""Generate the example figures and text outputs shown in the README.

Runs the deliberately gender-biased loan model from the demo through every
explainability module and writes:

* PNG figures to ``docs/images/`` (what the dashboard renders for each module)
* the textual outputs to stdout (pasted into the README's example blocks)

Reproduce with::

    python docs/generate_examples.py
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from explainx import ModelExplainer, explain_model, detect_drift

IMG = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
ACCENT, NEG = "#4f8cff", "#ff6b6b"


def build():
    rng = np.random.RandomState(0)
    n = 800
    gender = rng.randint(0, 2, n)
    income = rng.normal(50, 15, n)
    credit_score = rng.normal(650, 80, n)
    debt_ratio = rng.uniform(0, 1, n)
    approved = (
        0.04 * (income - 50) + 0.02 * (credit_score - 650) - 2.5 * debt_ratio
        + 1.8 * gender + rng.normal(0, 0.5, n) > 0
    ).astype(int)
    X = pd.DataFrame({"gender": gender, "income": income, "credit_score": credit_score, "debt_ratio": debt_ratio})
    model = RandomForestClassifier(n_estimators=80, random_state=0).fit(X, approved)
    return model, X, approved


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    model, X, y = build()
    ex = ModelExplainer(model, X, y)

    # Global importance
    imp = ex.importance()
    feats = imp.features[::-1]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh([f.feature for f in feats], [f.importance for f in feats], color=ACCENT)
    ax.set_title(f"Global feature importance ({imp.method})")
    save(fig, "importance.png")

    # Local explanation
    local = ex.explain(0, top_k=4)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    cs = local.contributions[::-1]
    ax.barh([c.feature for c in cs], [c.contribution for c in cs],
            color=[ACCENT if c.contribution >= 0 else NEG for c in cs])
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_title(f"Local explanation · row 0 → {local.prediction} ({local.method})")
    save(fig, "local.png")

    # PDP + ALE
    pdp = ex.partial_dependence("credit_score")
    ale = ex.ale("credit_score")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(pdp.grid_values, pdp.average_prediction, color=ACCENT, lw=2)
    axes[0].set_title("Partial dependence · credit_score"); axes[0].set_xlabel("credit_score")
    axes[1].plot(ale.bin_edges, ale.ale, color="#9b59b6", lw=2)
    axes[1].set_title("ALE · credit_score"); axes[1].set_xlabel("credit_score")
    save(fig, "effects.png")

    # Fairness
    fr = ex.fairness("gender")
    groups = [str(g.group) for g in fr.groups]
    rates = [g.selection_rate for g in fr.groups]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(groups, rates, color=[NEG, ACCENT])
    ax.set_title(f"Selection rate by gender (disparate impact {fr.disparate_impact_ratio:.2f})")
    ax.set_xlabel("gender"); ax.set_ylabel("P(approved)")
    for i, r in enumerate(rates):
        ax.text(i, r + 0.01, f"{r:.0%}", ha="center")
    save(fig, "fairness.png")

    # Interactions
    inter = ex.interactions(top_k=5)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    labels = [f"{p.feature_a}×{p.feature_b}" for p in inter.pairs][::-1]
    ax.barh(labels, [p.strength for p in inter.pairs][::-1], color="#16a085")
    ax.set_title("Feature interactions (Friedman H)")
    save(fig, "interactions.png")

    # Conformal coverage
    n = len(X); k = n // 2
    conf = ex.conformal(X.iloc[:k], y[:k], X.iloc[k:], alpha=0.1, y_test=y[k:])
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(["target", "empirical"], [conf.coverage_target, conf.empirical_coverage], color=["#bbb", ACCENT])
    ax.set_ylim(0, 1.05); ax.set_title(f"Conformal coverage (avg set size {conf.average_set_size:.2f})")
    for i, v in enumerate([conf.coverage_target, conf.empirical_coverage]):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center")
    save(fig, "conformal.png")

    # Drift
    drift = detect_drift(X, X.assign(income=X.income + 25, debt_ratio=np.clip(X.debt_ratio + 0.15, 0, 1)))
    fig, ax = plt.subplots(figsize=(6, 2.6))
    fnames = [f.feature for f in drift.features]
    psis = [f.psi or 0 for f in drift.features]
    ax.barh(fnames, psis, color=[NEG if f.drifted else "#bbb" for f in drift.features])
    ax.axvline(0.25, color="#888", ls="--", lw=0.8)
    ax.set_title("Data drift (PSI; dashed = 0.25 threshold)")
    save(fig, "drift.png")

    # ---- text outputs for the README ----
    print("\n================ TEXT OUTPUTS ================\n")
    report = explain_model(model, X, y, sensitive_features=["gender"], n_local=2)
    print("### explain_model(...).summary\n")
    print(report.summary)

    print("\n### metrics\n", ex.metrics().to_dict()["metrics"])

    rej = int(np.where(model.predict(X) == 0)[0][0])
    cf = ex.recourse(rej, immutable_features=["gender"])
    print("\n### recourse (immutable: gender)")
    for c in cf.changes:
        print(f"  {c.feature}: {c.original_value} -> {c.counterfactual_value}")
    print(f"  => {cf.original_prediction} -> {cf.new_prediction}")

    a = ex.anchor(rej)
    print(f"\n### anchor\n  IF {' AND '.join(a.rules)} THEN {a.prediction} "
          f"(precision {a.precision:.2f}, coverage {a.coverage:.2f})")

    s = ex.surrogate(max_depth=3)
    print(f"\n### surrogate fidelity: {s.fidelity_metric}={s.fidelity:.3f}\n")
    print(s.rules_text[:400])

    q = ex.explanation_quality(0)
    print(f"\n### explanation quality: faithfulness={q.faithfulness}, stability={q.stability}")

    mit = ex.mitigate_bias("gender")
    print(f"\n### bias mitigation: parity gap {mit.parity_gap_before:.1%} -> {mit.parity_gap_after:.1%}")


if __name__ == "__main__":
    main()
