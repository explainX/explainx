"""An MCP server exposing explainx-llm as agent-callable tools.

This lets an LLM agent (Claude, etc.) inspect a model it just trained: it saves
the fitted estimator and a dataset to disk, then calls these tools by path. Each
tool returns a JSON-ready dict the agent can read and reason over -- e.g. to
detect bias and decide how to fix the training.

Run it::

    python -m explainx_llm.mcp_server          # stdio transport

or register it in an MCP client config pointing at that command.

Tools
-----
explain_model        Full report: metrics, importance, local explanations, fairness.
feature_importance   Global feature importance.
explain_prediction   Why one row was predicted the way it was.
counterfactual       Minimal change that flips a row's predicted class.
check_bias           Group-fairness analysis on a sensitive feature.
model_metrics        Performance metrics.
partial_dependence   Marginal effect curve for one feature.
"""

from __future__ import annotations

from typing import Optional

from .io import load_model, load_xy
from .explainer import ModelExplainer, explain_model as _explain_model


def _make_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "The 'mcp' package is required to run the MCP server. "
            "Install it with `pip install mcp`."
        ) from exc

    mcp = FastMCP("explainx-llm")

    @mcp.tool()
    def explain_model(
        model_path: str,
        data_path: str,
        target_column: Optional[str] = None,
        sensitive_features: Optional[list[str]] = None,
        n_local: int = 3,
    ) -> dict:
        """Generate a full explainability report for a saved model and dataset.

        Args:
            model_path: Path to a joblib/pickle-serialized fitted estimator.
            data_path: Path to a CSV/Parquet/JSON dataset (features, optionally the target).
            target_column: Name of the ground-truth column, if present in the data.
            sensitive_features: Columns to run bias analysis on (e.g. ["gender"]).
            n_local: Number of individual predictions to explain.
        """
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        report = _explain_model(
            model, X, y, sensitive_features=sensitive_features, n_local=n_local
        )
        return report.to_dict()

    @mcp.tool()
    def feature_importance(
        model_path: str, data_path: str, target_column: Optional[str] = None
    ) -> dict:
        """Rank features by their global importance to the model."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return ModelExplainer(model, X, y).importance().to_dict()

    @mcp.tool()
    def explain_prediction(
        model_path: str,
        data_path: str,
        row_index: int,
        target_column: Optional[str] = None,
        top_k: int = 10,
    ) -> dict:
        """Explain why the model made its prediction for a single row."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return ModelExplainer(model, X, y).explain(row_index, top_k=top_k).to_dict()

    @mcp.tool()
    def counterfactual(
        model_path: str,
        data_path: str,
        row_index: int,
        target_column: Optional[str] = None,
        desired_class: Optional[str] = None,
        max_changes: int = 4,
    ) -> dict:
        """Find the smallest input change that flips a row's predicted class."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return (
            ModelExplainer(model, X, y)
            .counterfactual(row_index, desired_class=desired_class, max_changes=max_changes)
            .to_dict()
        )

    @mcp.tool()
    def check_bias(
        model_path: str,
        data_path: str,
        sensitive_feature: str,
        target_column: Optional[str] = None,
    ) -> dict:
        """Run group-fairness analysis on a sensitive feature (e.g. gender, race)."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return ModelExplainer(model, X, y).fairness(sensitive_feature).to_dict()

    @mcp.tool()
    def model_metrics(model_path: str, data_path: str, target_column: str) -> dict:
        """Compute standard performance metrics for the model."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return ModelExplainer(model, X, y).metrics().to_dict()

    @mcp.tool()
    def partial_dependence(
        model_path: str,
        data_path: str,
        feature: str,
        target_column: Optional[str] = None,
        grid_resolution: int = 20,
    ) -> dict:
        """Compute the marginal effect (partial dependence) curve for a feature."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return (
            ModelExplainer(model, X, y)
            .partial_dependence(feature, grid_resolution=grid_resolution)
            .to_dict()
        )

    @mcp.tool()
    def surrogate_rules(
        model_path: str, data_path: str, target_column: Optional[str] = None, max_depth: int = 4
    ) -> dict:
        """Fit a glassbox decision-tree surrogate and return its rules + fidelity."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return ModelExplainer(model, X, y).surrogate(max_depth=max_depth).to_dict()

    @mcp.tool()
    def lime_explain_prediction(
        model_path: str,
        data_path: str,
        row_index: int,
        target_column: Optional[str] = None,
        top_k: int = 10,
    ) -> dict:
        """Explain one prediction with LIME (local linear surrogate)."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return ModelExplainer(model, X, y).lime(row_index, top_k=top_k).to_dict()

    @mcp.tool()
    def anchor_rule(
        model_path: str,
        data_path: str,
        row_index: int,
        target_column: Optional[str] = None,
        precision_threshold: float = 0.95,
    ) -> dict:
        """Find a high-precision IF-THEN rule that anchors a row's prediction."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return (
            ModelExplainer(model, X, y)
            .anchor(row_index, precision_threshold=precision_threshold)
            .to_dict()
        )

    @mcp.tool()
    def accumulated_local_effects(
        model_path: str,
        data_path: str,
        feature: str,
        target_column: Optional[str] = None,
        n_bins: int = 10,
    ) -> dict:
        """Compute ALE for a numeric feature (correlation-robust effect curve)."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return ModelExplainer(model, X, y).ale(feature, n_bins=n_bins).to_dict()

    @mcp.tool()
    def explanation_quality(
        model_path: str,
        data_path: str,
        row_index: int = 0,
        target_column: Optional[str] = None,
    ) -> dict:
        """Score the faithfulness and stability of a local explanation."""
        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        return ModelExplainer(model, X, y).explanation_quality(row_index).to_dict()

    @mcp.tool()
    def detect_data_drift(reference_path: str, current_path: str, psi_threshold: float = 0.25) -> dict:
        """Detect distribution drift between a reference and a current dataset (PSI + KS)."""
        from .io import load_dataframe
        from .drift import detect_drift

        return detect_drift(
            load_dataframe(reference_path), load_dataframe(current_path), psi_threshold
        ).to_dict()

    @mcp.tool()
    def html_report(
        model_path: str,
        data_path: str,
        output_path: str,
        target_column: Optional[str] = None,
        sensitive_features: Optional[list[str]] = None,
    ) -> dict:
        """Generate a shareable HTML report and write it to output_path."""
        from .report_html import save_html

        model = load_model(model_path)
        X, y = load_xy(data_path, target_column)
        report = _explain_model(model, X, y, sensitive_features=sensitive_features)
        save_html(report, output_path)
        return {"written": output_path, "biased": any(f.biased for f in report.fairness)}

    return mcp


def main():
    _make_server().run()


if __name__ == "__main__":
    main()
