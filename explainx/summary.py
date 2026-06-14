"""Render a structured report into a natural-language briefing.

The summary is written for an LLM agent (or a human stakeholder) to read and
act on: it states how well the model performs, what drives it, and whether any
bias was detected, then suggests concrete next steps for fixing training.
"""

from __future__ import annotations

from .schema import ExplanationReport


def build_summary(report: "ExplanationReport") -> str:
    lines: list[str] = []
    lines.append(
        f"Model `{report.model_name}` is a {report.problem_type} model evaluated on "
        f"{report.n_samples} samples across {report.n_features} features."
    )

    if report.metrics is not None and report.metrics.metrics:
        parts = [f"{k}={v:.3f}" for k, v in report.metrics.metrics.items()]
        lines.append("Performance: " + ", ".join(parts) + ".")

    if report.global_importance and report.global_importance.features:
        top = report.global_importance.top(5)
        named = ", ".join(f"{f.feature} ({f.importance:.3f})" for f in top)
        lines.append(
            f"The most influential features (via {report.global_importance.method}) are: {named}."
        )

    for local in report.local_explanations:
        top = local.contributions[:3]
        if not top:
            continue
        drivers = ", ".join(
            f"{c.feature}={c.value} ({'+' if c.contribution >= 0 else ''}{c.contribution:.3f})"
            for c in top
        )
        prob = (
            f" (probability {local.predicted_probability:.3f})"
            if local.predicted_probability is not None
            else ""
        )
        lines.append(
            f"Prediction for row {local.index} = {local.prediction}{prob}; "
            f"top drivers: {drivers}."
        )

    if report.surrogate is not None:
        s = report.surrogate
        lines.append(
            f"A depth-{s.max_depth} decision-tree surrogate reproduces the model with "
            f"{s.fidelity_metric}={s.fidelity:.3f} fidelity, giving an inspectable rule set."
        )

    if report.explanation_quality is not None:
        q = report.explanation_quality
        bits = []
        if q.faithfulness is not None:
            bits.append(f"faithfulness={q.faithfulness:.2f}")
        if q.stability is not None:
            bits.append(f"stability={q.stability:.2f}")
        if bits:
            lines.append(
                f"Explanation quality ({q.method}): " + ", ".join(bits) +
                " (higher is more trustworthy; ~1.0 is excellent)."
            )

    bias_found = False
    for fr in report.fairness:
        status = "BIAS DETECTED" if fr.biased else "no bias flagged"
        bias_found = bias_found or fr.biased
        detail = " ".join(fr.findings)
        lines.append(f"Fairness on `{fr.sensitive_feature}`: {status}. {detail}")

    # Actionable guidance tailored to what was found.
    recs: list[str] = []
    if bias_found:
        recs.append(
            "Address the detected bias before deployment: rebalance/ reweight the "
            "training data across the sensitive groups, consider removing or "
            "decorrelating proxy features, or apply a fairness constraint, then re-evaluate."
        )
    if report.metrics and report.metrics.problem_type == "classification":
        acc = report.metrics.metrics.get("accuracy")
        if acc is not None and acc < 0.7:
            recs.append(
                "Accuracy is modest -- revisit feature engineering, model choice, or "
                "hyperparameters before trusting the explanations."
            )
    if recs:
        lines.append("Recommended next steps: " + " ".join(recs))

    return "\n".join(lines)
