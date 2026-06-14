"""LLM narration: turn a structured report into plain language with Claude.

This is the part that makes explainX genuinely *LLM-native*. The engine produces
structured, faithful explanations (SHAP values, fairness metrics, counterfactuals,
conformal sets); this layer hands that evidence to Claude so it can:

* write a stakeholder-ready narrative of how the model behaves, or
* answer a specific question ("why was applicant 5 rejected, and what would
  change the decision?") grounded *only* in the computed evidence.

Keeping the numbers in the engine and the prose in the LLM avoids hallucinated
explanations: Claude narrates facts it is given rather than inventing them.

Requires the optional ``anthropic`` package and an ``ANTHROPIC_API_KEY``::

    pip install "explainx[llm]"
"""

from __future__ import annotations

from typing import Any, Optional
import json

from .schema import ExplanationReport

DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You are an ML explainability analyst. You are given a structured, "
    "machine-generated explainability report for a trained model (metrics, "
    "feature importances, per-prediction attributions, fairness diagnostics, "
    "counterfactuals, and uncertainty). Explain it in clear, plain language for "
    "the intended audience. Ground every statement in the numbers provided; do "
    "not invent findings the report does not contain. When the report flags "
    "bias or weak performance, say so plainly and suggest concrete next steps "
    "for fixing the training data or model. Be concise and lead with the takeaway."
)


def build_prompt(report: Any, question: Optional[str] = None, audience: str = "a data scientist") -> str:
    """Build the user prompt. Pure function -- testable without the API."""
    payload = report.to_dict() if hasattr(report, "to_dict") else report
    parts = [
        f"Audience: {audience}.",
        "Here is the structured explainability report as JSON:",
        "```json",
        json.dumps(payload, indent=2),
        "```",
    ]
    if question:
        parts.append(f"\nAnswer this specific question using only the report: {question}")
    else:
        parts.append(
            "\nWrite a briefing covering: how well the model performs, what drives "
            "its predictions, any fairness concerns, and recommended next steps."
        )
    return "\n".join(parts)


def narrate_report(
    report: ExplanationReport,
    question: Optional[str] = None,
    audience: str = "a data scientist",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    """Send the report to Claude and return a natural-language explanation.

    Raises ImportError if the ``anthropic`` package is not installed.
    """
    try:
        import anthropic
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "narrate_report requires the 'anthropic' package. "
            "Install it with `pip install \"explainx[llm]\"`."
        ) from exc

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(report, question, audience)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
