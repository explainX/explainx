"""Light tests for the dashboard module.

The Streamlit UI itself is exercised by booting the server; here we cover the
import surface and the pure data-shaping helpers (no Streamlit context needed).
"""

from explainx import dashboard


def test_dashboard_exposes_launcher():
    assert callable(dashboard.launch)
    assert callable(dashboard._app)


def test_importance_frame():
    from explainx.schema import FeatureImportance

    feats = [FeatureImportance("a", 0.5), FeatureImportance("b", 0.2)]
    frame = dashboard._importance_frame(feats)
    assert list(frame.index) == ["a", "b"]
    assert frame.loc["a", "importance"] == 0.5


def test_contrib_frame():
    from explainx.schema import FeatureContribution

    contribs = [FeatureContribution("a", 1.0, 0.3), FeatureContribution("b", 2.0, -0.1)]
    frame = dashboard._contrib_frame(contribs)
    assert list(frame.index) == ["a", "b"]
    assert frame.loc["b", "contribution"] == -0.1
