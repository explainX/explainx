"""Smoke test for the MCP server: build it and invoke each tool by path.

This mirrors how an agent uses the server -- save a fitted model and dataset to
disk, then call the tools with those paths and read structured dicts back.
"""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mcp")
import joblib  # noqa: E402


@pytest.fixture
def saved_artifacts(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(0)
    n = 300
    gender = rng.randint(0, 2, size=n)
    score = rng.normal(size=n)
    approve = ((score + 1.5 * gender) > 0).astype(int)
    df = pd.DataFrame({"gender": gender, "score": score, "approved": approve})

    X = df.drop(columns=["approved"])
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, df["approved"])

    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "data.csv"
    joblib.dump(model, model_path)
    df.to_csv(data_path, index=False)
    return str(model_path), str(data_path)


def _tool(server, name):
    """Resolve a registered FastMCP tool's underlying callable across versions."""
    manager = server._tool_manager
    tool = manager.get_tool(name) if hasattr(manager, "get_tool") else manager._tools[name]
    return tool.fn


def test_mcp_tools_roundtrip(saved_artifacts):
    from explainx.mcp_server import _make_server

    model_path, data_path = saved_artifacts
    server = _make_server()

    report = _tool(server, "explain_model")(
        model_path, data_path, target_column="approved", sensitive_features=["gender"]
    )
    assert report["problem_type"] == "classification"
    assert report["fairness"][0]["sensitive_feature"] == "gender"
    json.dumps(report)  # must be serializable

    imp = _tool(server, "feature_importance")(model_path, data_path, "approved")
    assert imp["features"]

    bias = _tool(server, "check_bias")(model_path, data_path, "gender", "approved")
    assert bias["biased"] is True

    metrics = _tool(server, "model_metrics")(model_path, data_path, "approved")
    assert "accuracy" in metrics["metrics"]

    local = _tool(server, "explain_prediction")(model_path, data_path, 0, "approved")
    assert local["contributions"]
