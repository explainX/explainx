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

| Capability | Method | Notes |
|---|---|---|
| Global importance | **SHAP** (auto) → permutation importance → intrinsic | SHAP used automatically when installed |
| Local explanation | **SHAP** (auto) → model-agnostic ablation | per-prediction signed contributions |
| Counterfactuals | greedy model-agnostic search | smallest change that flips a decision |
| Partial dependence | scikit-learn PDP → agnostic fallback | marginal effect of a feature |
| Fairness / bias | demographic parity, disparate impact (4/5 rule), equal opportunity | per sensitive attribute |
| Metrics | classification + regression | accuracy/precision/recall/f1/auc, r²/mae/rmse |

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
ex.importance()                    # GlobalImportance
ex.explain(index=0, top_k=5)       # LocalExplanation
ex.fairness("gender")              # FairnessReport
ex.counterfactual(index=0)         # Counterfactual: minimal flip
ex.partial_dependence("income")    # PartialDependence curve
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
| `explain_model` | Full report (metrics, importance, local, fairness) |
| `feature_importance` | Global importance ranking |
| `explain_prediction` | Why one row was predicted as it was |
| `counterfactual` | Minimal change that flips a row's class |
| `check_bias` | Group-fairness analysis on a sensitive feature |
| `model_metrics` | Performance metrics |
| `partial_dependence` | Marginal effect curve for a feature |

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
