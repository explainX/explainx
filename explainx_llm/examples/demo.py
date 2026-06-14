"""End-to-end demo: train a loan-approval model, then explain it.

Run with::

    python -m explainx_llm.examples.demo

It deliberately bakes a gender bias into the data so you can see the fairness
checks fire and the natural-language summary recommend a fix -- exactly the
loop the project is designed for.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from explainx_llm import explain_model, ModelExplainer


def build_biased_loan_data(n: int = 800, seed: int = 0):
    rng = np.random.RandomState(seed)
    gender = rng.randint(0, 2, size=n)  # 0 = female, 1 = male
    income = rng.normal(50, 15, size=n)
    credit_score = rng.normal(650, 80, size=n)
    debt_ratio = rng.uniform(0, 1, size=n)

    # A genuine signal ... plus a thumb on the scale for gender == 1 (the bias).
    logit = (
        0.04 * (income - 50)
        + 0.02 * (credit_score - 650)
        - 2.5 * debt_ratio
        + 1.8 * gender  # <-- unfair advantage
    )
    approved = (logit + rng.normal(0, 0.5, size=n) > 0).astype(int)
    X = pd.DataFrame(
        {
            "gender": gender,
            "income": income,
            "credit_score": credit_score,
            "debt_ratio": debt_ratio,
        }
    )
    return X, approved


def main():
    X, y = build_biased_loan_data()
    model = RandomForestClassifier(n_estimators=80, random_state=0).fit(X, y)

    print("=" * 70)
    print("explainx-llm demo: loan-approval model")
    print("=" * 70)

    report = explain_model(model, X, y, sensitive_features=["gender"], n_local=2)
    print("\n--- NATURAL-LANGUAGE SUMMARY (for a human or LLM) ---\n")
    print(report.summary)

    print("\n--- COUNTERFACTUAL for a rejected applicant ---\n")
    rejected = int(np.where(model.predict(X) == 0)[0][0])
    cf = ModelExplainer(model, X, y).counterfactual(rejected)
    if cf.found:
        for ch in cf.changes:
            print(f"  {ch.feature}: {ch.original_value} -> {ch.counterfactual_value}")
        print(f"  => prediction flips {cf.original_prediction} -> {cf.new_prediction}")
    else:
        print("  No counterfactual found within the change budget.")

    print("\n(Full structured report is available via report.to_json())")


if __name__ == "__main__":
    main()
