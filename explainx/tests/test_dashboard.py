"""Dashboard tests.

Beyond import/helper checks, this drives every per-module render function with a
headless Streamlit stub against a real (biased) model, asserting each path runs
without error and emits output. This is the programmatic equivalent of clicking
through every module in the UI.
"""

import contextlib

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from explainx import ModelExplainer, dashboard
from explainx.dashboard import Ctx, RENDERERS, VIEWS


class FakeSt:
    """A minimal headless stand-in for the streamlit module.

    Widget calls return sensible defaults; output calls are recorded so a test
    can assert that a render path actually produced something.
    """

    def __init__(self):
        self.outputs = []

    # --- output sinks (recorded) ---
    def _record(self, *a, **k):
        if a:
            self.outputs.append(a[0])

    title = caption = header = subheader = text = write = success = warning = _record
    bar_chart = line_chart = dataframe = json = metric = _record

    def info(self, *a, **k):
        # info is used for "pick a feature" guidance; record but don't count as data
        pass

    def set_page_config(self, *a, **k):
        pass

    def download_button(self, *a, **k):
        self.outputs.append("download")

    # --- widgets (return defaults) ---
    def file_uploader(self, *a, **k):
        return None

    def selectbox(self, label, options, *a, **k):
        return options[0]

    def multiselect(self, label, options=(), *a, **k):
        return []

    def radio(self, label, options, index=0, **k):
        return options[index]

    def number_input(self, label, mn=0, mx=0, value=0, **k):
        return value

    def slider(self, label, mn, mx, value, **k):
        return value

    def columns(self, n):
        return [self for _ in range(n if isinstance(n, int) else len(n))]

    @contextlib.contextmanager
    def spinner(self, *a, **k):
        yield

    def stop(self):
        raise RuntimeError("st.stop called")


@pytest.fixture
def ctx():
    rng = np.random.RandomState(0)
    n = 300
    gender = rng.randint(0, 2, n)
    income = rng.normal(50, 15, n)
    score = rng.normal(650, 80, n)
    approve = ((0.03 * (income - 50) + 0.02 * (score - 650) + 1.6 * gender + rng.normal(0, 0.5, n)) > 0).astype(int)
    df = pd.DataFrame({"gender": gender, "income": income, "credit_score": score, "approved": approve})
    X = df[["gender", "income", "credit_score"]]
    y = df["approved"].to_numpy()
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    return Ctx(model=model, df=df, X=X, y=y, ex=ModelExplainer(model, X, y), sensitive=["gender"])


def test_every_view_has_a_renderer():
    assert set(VIEWS) == set(RENDERERS)


@pytest.mark.parametrize("view", VIEWS)
def test_render_runs_without_error(view, ctx):
    st = FakeSt()
    RENDERERS[view](st, ctx)  # must not raise
    # Drift renders an info() prompt (no second file) — every other view emits output.
    if view != "Data drift":
        assert st.outputs, f"{view} produced no output"


def test_launch_is_callable():
    assert callable(dashboard.launch)
