# explainX-LLM: LLM-native Explainable AI
<img src="explainx_logo.png" align="right" width="150"/>

**explainX-LLM** turns the explainX idea into a modern, **LLM-native** explainability
engine. Train any machine-learning model, then let a human *or an LLM agent*
inspect it: understand *why* a prediction was made, surface bias, find the
minimal change that flips a decision, and feed those insights back into training.

Where the original explainX rendered a human-only Plotly dashboard on a 2020
dependency stack, this rewrite returns **structured, machine-readable results**
(typed objects that serialize to JSON) plus a natural-language summary — usable
from plain Python *and* over the **Model Context Protocol (MCP)** so agents like
Claude can call it as tools while they build models.

> The goal: bring state-of-the-art explainability research into one place, with
> an interface designed for the era where LLMs train and debug models.

---

## Why

1. **Explain predictions** — global feature importance + per-prediction reasoning.
2. **Debug models** — counterfactuals and partial-dependence curves show what the
   model actually learned.
3. **Detect bias** — group-fairness metrics (disparate impact, demographic parity,
   equal opportunity) answer "is my model rejecting one group regardless of profile?"
4. **Build trust** — a plain-language summary for stakeholders and agents.
5. **Close the loop** — results are structured so an LLM can read them and decide
   how to fix the training data or model.

## What's inside

A unified API over the methods the XAI literature identifies as the most
deployed — SHAP, LIME, surrogate trees and counterfactuals — plus modern
additions (ALE, anchors) and a 2024–2025 research frontier most tools skip:
**quantifying whether an explanation can be trusted**.

| Capability | Method | Notes |
|---|---|---|
| Global importance | **SHAP** (auto) → permutation → intrinsic | SHAP used automatically when installed |
| Local explanation | **SHAP** (auto) → model-agnostic ablation | per-prediction signed contributions |
| Local surrogate | **LIME** (from scratch) | local weighted linear approximation |
| Sufficient rules | **Anchors** | high-precision IF-THEN rule per prediction |
| Counterfactuals | greedy model-agnostic search | smallest change that flips a decision |
| Global surrogate | **decision tree** + fidelity | inspectable glassbox rules + how faithful they are |
| Feature effects | **PDP** and **ALE** | ALE stays unbiased under feature correlation |
| Explanation quality | **faithfulness** + **stability** | does the explanation reflect the model, and is it robust? |
| Counterfactuals & **recourse** | greedy search with immutable / monotonic constraints | actionable "what to change" |
| **Uncertainty** | **conformal prediction** | distribution-free prediction sets / intervals with coverage guarantee |
| Fairness / bias | demographic parity, disparate impact (4/5 rule), equal opportunity | per sensitive attribute |
| Bias **mitigation** | post-processing per-group thresholds | detect → *fix* |
| **Interactions** | Friedman's H-statistic | which features matter *together* |
| Example-based | **prototypes & criticisms** (MMD) | representative vs. atypical cases |
| Metrics | classification + regression | accuracy/precision/recall/f1/auc, r²/mae/rmse |
| Monitoring | **data drift** (PSI + KS) | reference vs. current dataset |
| **LLM narration** | Claude (`claude-opus-4-8`) | plain-language briefings / Q&A grounded in the report |
| Reporting | **HTML export** + **CLI** | shareable artifact; no-code usage |

Works with any estimator following the scikit-learn `predict` / `predict_proba`
convention (scikit-learn, XGBoost, LightGBM, CatBoost, …).

## Install

```sh
pip install -e ".[all]"     # core + SHAP + MCP server
# or minimal:
pip install -e .            # core only (SHAP/MCP optional)
```

## Python API

```python
from explainx_llm import explain_model

report = explain_model(
    model, X_test, y_test,
    sensitive_features=["gender"],   # run bias analysis on these columns
    n_local=3,                       # explain a few individual predictions
)

print(report.summary)      # natural-language briefing for a human/LLM
report.to_dict()           # full structured result (JSON-ready)
report.to_json()
```

Need finer control? Use the stateful explainer:

```python
from explainx_llm import ModelExplainer

ex = ModelExplainer(model, X_test, y_test)
ex.metrics()                       # ModelMetrics
ex.importance()                    # GlobalImportance (SHAP when available)
ex.explain(index=0, top_k=5)       # LocalExplanation (SHAP/ablation)
ex.lime(index=0)                   # LocalExplanation (LIME)
ex.anchor(index=0)                 # Anchor: high-precision sufficient rule
ex.fairness("gender")              # FairnessReport
ex.counterfactual(index=0)         # Counterfactual: minimal flip
ex.recourse(index=0, immutable_features=["age", "gender"])  # actionable recourse
ex.surrogate()                     # SurrogateExplanation: glassbox tree + fidelity
ex.partial_dependence("income")    # PartialDependence curve
ex.ale("income")                   # ALEResult: correlation-robust effect
ex.explanation_quality(index=0)    # ExplanationQuality: faithfulness + stability
ex.conformal(X_cal, y_cal, X_test) # ConformalResult: guaranteed-coverage sets/intervals
ex.mitigate_bias("gender")         # MitigationResult: per-group thresholds that fix parity
ex.interactions(top_k=5)           # InteractionResult: Friedman H-statistic
ex.prototypes()                    # PrototypesResult: representative + atypical rows
```

### LLM narration (optional)

```python
from explainx_llm.narrate import narrate_report   # needs: pip install "explainx-llm[llm]"

report = explain_model(model, X_test, y_test, sensitive_features=["gender"])
print(narrate_report(report, question="Why was applicant 5 rejected, and what would change it?"))
```

The engine computes the evidence (SHAP, fairness, counterfactuals, conformal sets);
Claude narrates it. Numbers stay in the engine, prose comes from the LLM — so the
explanation is grounded, not hallucinated.

### Monitoring & reporting

```python
from explainx_llm import detect_drift, save_html

detect_drift(reference_df, current_df)   # DriftReport (PSI + KS per feature)
save_html(report, "report.html")         # shareable page; embeds the full JSON
```

### No-code CLI

```sh
explainx-llm report --model m.joblib --data d.csv --target y --sensitive gender --html out.html
explainx-llm bias   --model m.joblib --data d.csv --target y --sensitive gender
explainx-llm drift  --reference train.csv --current prod.csv
```

### Try the demo

```sh
python -m explainx_llm.examples.demo
```

It trains a deliberately gender-biased loan model and shows the fairness check
firing, plus a counterfactual that flips a rejection to an approval.

## Use it from an LLM agent (MCP)

Start the server (stdio transport):

```sh
explainx-mcp              # installed console script
# or:  python -m explainx_llm.mcp_server
```

Register it with an MCP client (e.g. Claude Desktop / Claude Code):

```json
{
  "mcpServers": {
    "explainx": { "command": "explainx-mcp" }
  }
}
```

The agent saves a fitted model and dataset to disk, then calls tools by path:

| Tool | Purpose |
|---|---|
| `explain_model` | Full report (metrics, importance, local, fairness, surrogate, quality) |
| `feature_importance` | Global importance ranking |
| `explain_prediction` | Why one row was predicted as it was (SHAP/ablation) |
| `lime_explain_prediction` | Local LIME explanation for one row |
| `anchor_rule` | High-precision sufficient rule for one row |
| `counterfactual` | Minimal change that flips a row's class |
| `surrogate_rules` | Glassbox decision-tree rules + fidelity |
| `check_bias` | Group-fairness analysis on a sensitive feature |
| `model_metrics` | Performance metrics |
| `partial_dependence` | Marginal effect curve for a feature |
| `accumulated_local_effects` | Correlation-robust effect curve (ALE) |
| `explanation_quality` | Faithfulness + stability of an explanation |
| `conformal_prediction` | Guaranteed-coverage prediction sets / intervals |
| `actionable_recourse` | Minimal flip respecting immutable features |
| `mitigate_bias` | Per-group thresholds that equalize selection rate |
| `feature_interactions_tool` | Strongest pairwise interactions (H-statistic) |
| `prototypes_and_criticisms_tool` | Representative + atypical rows |
| `detect_data_drift` | Distribution drift between two datasets |
| `html_report` | Write a shareable HTML report |

Each returns a JSON-ready dict the agent can reason over — e.g. read a
`disparate_impact_ratio` below 0.8, conclude the model is biased, and rebalance
the training data.

```python
# what the agent does first:
import joblib
joblib.dump(model, "model.joblib")
df.to_csv("data.csv", index=False)   # features + target column
# then it calls:  check_bias(model_path="model.joblib", data_path="data.csv",
#                            sensitive_feature="gender", target_column="approved")
```

## Tests

```sh
pytest          # or: python -m pytest explainx_llm/tests
```

## Migrating from legacy explainX

The 2020 Dash dashboard (`explain.py`, `main.py`, `lib/`) and its pinned,
no-longer-installable stack have been removed in favour of this engine. The new
import is `explainx_llm`; explanations are returned as data rather than rendered
as a web app, which is what makes them consumable by both humans and LLMs.

## License

[MIT](https://choosealicense.com/licenses/mit/)
