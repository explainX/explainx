"""Export an ExplanationReport to a self-contained HTML page.

A shareable artifact bridges the two audiences: stakeholders read the rendered
page, while the same file embeds the full structured JSON for any tool or agent.
No templating dependency -- just standard-library string building.
"""

from __future__ import annotations

import html
import json

from .schema import ExplanationReport


def _bar_table(rows: list[tuple[str, float]], max_rows: int = 12) -> str:
    if not rows:
        return "<p><em>n/a</em></p>"
    top = rows[:max_rows]
    peak = max((abs(v) for _, v in top), default=1.0) or 1.0
    out = ["<table class='bars'>"]
    for name, val in top:
        width = int(100 * abs(val) / peak)
        color = "#4f8" if val >= 0 else "#f86"
        out.append(
            f"<tr><td class='lbl'>{html.escape(str(name))}</td>"
            f"<td class='bar'><span style='width:{width}%;background:{color}'></span></td>"
            f"<td class='val'>{val:.4f}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


def report_to_html(report: ExplanationReport, title: str = "explainX-LLM report") -> str:
    r = report
    parts: list[str] = []
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(
        f"<p class='meta'>Model <b>{html.escape(r.model_name)}</b> &middot; "
        f"{html.escape(r.problem_type)} &middot; {r.n_samples} samples &middot; "
        f"{r.n_features} features</p>"
    )

    parts.append("<h2>Summary</h2>")
    parts.append("<pre class='summary'>" + html.escape(r.summary) + "</pre>")

    if r.metrics and r.metrics.metrics:
        parts.append("<h2>Metrics</h2><ul>")
        for k, v in r.metrics.metrics.items():
            parts.append(f"<li><b>{html.escape(k)}</b>: {v:.4f}</li>")
        parts.append("</ul>")

    if r.global_importance and r.global_importance.features:
        parts.append(f"<h2>Global importance <small>({html.escape(r.global_importance.method)})</small></h2>")
        parts.append(_bar_table([(f.feature, f.importance) for f in r.global_importance.features]))

    for local in r.local_explanations:
        parts.append(f"<h3>Prediction for row {local.index} = {html.escape(str(local.prediction))} <small>({html.escape(local.method)})</small></h3>")
        parts.append(_bar_table([(c.feature, c.contribution) for c in local.contributions]))

    for fr in r.fairness:
        badge = "BIAS DETECTED" if fr.biased else "no bias flagged"
        cls = "bad" if fr.biased else "good"
        parts.append(f"<h2>Fairness: {html.escape(fr.sensitive_feature)} <span class='badge {cls}'>{badge}</span></h2>")
        parts.append("<ul>" + "".join(f"<li>{html.escape(f)}</li>" for f in fr.findings) + "</ul>")

    if r.surrogate:
        parts.append(f"<h2>Surrogate rules <small>(fidelity {r.surrogate.fidelity:.3f})</small></h2>")
        parts.append("<pre class='rules'>" + html.escape(r.surrogate.rules_text) + "</pre>")

    # Embed the full structured payload for machine consumption.
    payload = json.dumps(report.to_dict())
    parts.append("<h2>Structured data</h2>")
    parts.append("<details><summary>Show JSON</summary><pre class='json'>"
                 + html.escape(json.dumps(report.to_dict(), indent=2)) + "</pre></details>")
    parts.append(f"<script type='application/json' id='explainx-report'>{html.escape(payload)}</script>")

    body = "\n".join(parts)
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{html.escape(title)}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1a1a2e;line-height:1.5}}
 h1{{margin-bottom:0}} .meta{{color:#667}} h2{{border-bottom:1px solid #eee;padding-bottom:.2rem;margin-top:2rem}}
 pre{{background:#0f1020;color:#e6e6f0;padding:1rem;border-radius:8px;overflow:auto;white-space:pre-wrap}}
 .summary{{background:#f5f6ff;color:#1a1a2e}}
 table.bars{{width:100%;border-collapse:collapse}} table.bars td{{padding:2px 6px;font-size:14px}}
 td.lbl{{white-space:nowrap}} td.bar{{width:60%}} td.bar span{{display:inline-block;height:12px;border-radius:3px}}
 td.val{{text-align:right;font-variant-numeric:tabular-nums;color:#556}}
 .badge{{font-size:12px;padding:2px 8px;border-radius:12px;color:#fff}} .badge.bad{{background:#e23}} .badge.good{{background:#2a4}}
</style></head><body>
{body}
</body></html>"""


def save_html(report: ExplanationReport, path: str, title: str = "explainX-LLM report") -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report_to_html(report, title=title))
    return path
