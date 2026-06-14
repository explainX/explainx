"""Interactive Streamlit dashboard for the explainX engine.

Upload a fitted model and a dataset, then run any explainability module (or the
full report) and see the results as live tables and charts -- the human-facing
counterpart to the structured API and MCP server.

Launch it with the installed console script::

    explainx-dashboard

(which is a thin wrapper around ``streamlit run`` on this file). Requires the
optional dashboard dependency::

    pip install "explainx[dashboard]"
"""

from __future__ import annotations

import io
import os
import sys
import subprocess


# --------------------------------------------------------------------------- #
# Launcher (console-script entry point)
# --------------------------------------------------------------------------- #
def launch() -> None:
    """Start the Streamlit server on this file. Entry point for `explainx-dashboard`."""
    try:
        import streamlit  # noqa: F401
    except Exception:
        sys.stderr.write(
            "The dashboard needs Streamlit. Install it with:\n"
            '    pip install "explainx[dashboard]"\n'
        )
        raise SystemExit(1)
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__), *sys.argv[1:]]
        )
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_model(uploaded):
    import joblib

    try:
        return joblib.load(uploaded)
    except Exception:
        import pickle

        uploaded.seek(0)
        return pickle.load(uploaded)


def _load_data(uploaded):
    import pandas as pd

    name = uploaded.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    if name.endswith(".json"):
        return pd.read_json(uploaded)
    return pd.read_csv(uploaded)


def _importance_frame(features):
    import pandas as pd

    return pd.DataFrame(
        {"feature": [f.feature for f in features], "importance": [f.importance for f in features]}
    ).set_index("feature")


def _contrib_frame(contributions):
    import pandas as pd

    return pd.DataFrame(
        {
            "feature": [c.feature for c in contributions],
            "contribution": [c.contribution for c in contributions],
        }
    ).set_index("feature")


# --------------------------------------------------------------------------- #
# The app (executed by `streamlit run`)
# --------------------------------------------------------------------------- #
def _app() -> None:
    import streamlit as st
    import pandas as pd

    from explainx import ModelExplainer, explain_model, detect_drift, save_html, prototypes_and_criticisms

    st.set_page_config(page_title="explainX", layout="wide")
    st.title("explainX — model explainability dashboard")
    st.caption("Upload a fitted model + dataset, then explore any module or the full report.")

    with st.sidebar:
        st.header("Inputs")
        model_file = st.file_uploader("Model (.joblib / .pkl)", type=["joblib", "pkl", "pickle"])
        data_file = st.file_uploader("Dataset (.csv / .parquet / .json)", type=["csv", "parquet", "json"])

        if not model_file or not data_file:
            st.info("Upload a model and a dataset to begin.")
            st.stop()

        model = _load_model(model_file)
        df = _load_data(data_file)
        columns = list(df.columns)
        target = st.selectbox("Target column (optional)", ["(none)"] + columns)
        target = None if target == "(none)" else target
        feature_cols = [c for c in columns if c != target]
        sensitive = st.multiselect("Sensitive feature(s)", feature_cols)

        X = df[feature_cols] if target else df
        y = df[target].to_numpy() if target else None
        ex = ModelExplainer(model, X, y)
        st.success(f"{type(model).__name__} · {ex.problem_type} · {len(X)} rows · {X.shape[1]} features")

        view = st.radio(
            "Module",
            [
                "Full report", "Global importance", "Local explanation", "Counterfactual / recourse",
                "Feature effects (PDP / ALE)", "Interactions", "Fairness", "Bias mitigation",
                "Conformal uncertainty", "Prototypes & criticisms", "Explanation quality", "Data drift",
            ],
        )

    # ---- Full report ---------------------------------------------------- #
    if view == "Full report":
        with st.spinner("Computing full report..."):
            report = explain_model(model, X, y, sensitive_features=sensitive or None, n_local=3)
        st.subheader("Summary")
        st.text(report.summary)
        if report.global_importance:
            st.subheader("Global importance")
            st.bar_chart(_importance_frame(report.global_importance.features))
        for fr in report.fairness:
            tag = "🔴 BIAS DETECTED" if fr.biased else "🟢 no bias flagged"
            st.subheader(f"Fairness · {fr.sensitive_feature} — {tag}")
            for f in fr.findings:
                st.write("• " + f)
        from explainx.report_html import report_to_html

        c1, c2 = st.columns(2)
        c1.download_button("Download JSON", report.to_json(), "explainx_report.json", "application/json")
        c2.download_button("Download HTML", report_to_html(report), "explainx_report.html", "text/html")

    # ---- Global importance --------------------------------------------- #
    elif view == "Global importance":
        with st.spinner("Computing importance..."):
            imp = ex.importance()
        st.caption(f"method: {imp.method}")
        st.bar_chart(_importance_frame(imp.features))
        st.dataframe(_importance_frame(imp.features))

    # ---- Local explanation --------------------------------------------- #
    elif view == "Local explanation":
        row = st.number_input("Row index", 0, len(X) - 1, 0)
        method = st.selectbox("Method", ["SHAP / ablation", "LIME", "Anchor (rule)"])
        if method == "Anchor (rule)":
            a = ex.anchor(int(row))
            st.write(f"**IF** {' AND '.join(a.rules) or '(none)'}")
            st.write(f"**THEN** prediction = `{a.prediction}` (precision {a.precision:.2f}, coverage {a.coverage:.2f})")
        else:
            local = ex.lime(int(row)) if method == "LIME" else ex.explain(int(row))
            st.caption(f"method: {local.method} · prediction: {local.prediction}")
            st.bar_chart(_contrib_frame(local.contributions))
            st.dataframe(_contrib_frame(local.contributions))

    # ---- Counterfactual / recourse ------------------------------------- #
    elif view == "Counterfactual / recourse":
        row = st.number_input("Row index", 0, len(X) - 1, 0)
        immutable = st.multiselect("Immutable features (recourse)", list(X.columns))
        cf = ex.recourse(int(row), immutable_features=immutable or None)
        if cf.found:
            st.success(f"Prediction flips {cf.original_prediction} → {cf.new_prediction} with {cf.n_features_changed} change(s):")
            st.dataframe(pd.DataFrame([{"feature": c.feature, "from": c.original_value, "to": c.counterfactual_value} for c in cf.changes]))
        else:
            st.warning("No counterfactual found within the change budget.")

    # ---- Feature effects ----------------------------------------------- #
    elif view == "Feature effects (PDP / ALE)":
        numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        feat = st.selectbox("Feature", numeric or list(X.columns))
        kind = st.radio("Curve", ["PDP", "ALE"], horizontal=True)
        if kind == "PDP":
            r = ex.partial_dependence(feat)
            st.line_chart(pd.DataFrame({"prediction": r.average_prediction}, index=r.grid_values))
        else:
            r = ex.ale(feat)
            st.line_chart(pd.DataFrame({"ALE": r.ale}, index=r.bin_edges))

    # ---- Interactions -------------------------------------------------- #
    elif view == "Interactions":
        with st.spinner("Computing H-statistic..."):
            res = ex.interactions(top_k=8)
        st.dataframe(pd.DataFrame([{"pair": f"{p.feature_a} × {p.feature_b}", "strength": p.strength} for p in res.pairs]).set_index("pair"))

    # ---- Fairness ------------------------------------------------------ #
    elif view == "Fairness":
        if not sensitive:
            st.info("Pick at least one sensitive feature in the sidebar.")
        for s in sensitive:
            fr = ex.fairness(s)
            tag = "🔴 BIAS DETECTED" if fr.biased else "🟢 no bias flagged"
            st.subheader(f"{s} — {tag}")
            st.dataframe(pd.DataFrame([g.to_dict() for g in fr.groups]))
            for f in fr.findings:
                st.write("• " + f)

    # ---- Bias mitigation ----------------------------------------------- #
    elif view == "Bias mitigation":
        if not sensitive:
            st.info("Pick a sensitive feature in the sidebar.")
        else:
            res = ex.mitigate_bias(sensitive[0])
            st.metric("Parity gap", f"{res.parity_gap_after:.1%}", f"{(res.parity_gap_after - (res.parity_gap_before or 0)):.1%}")
            st.dataframe(pd.DataFrame([t.to_dict() for t in res.thresholds]))
            for n in res.notes:
                st.write("• " + n)

    # ---- Conformal ----------------------------------------------------- #
    elif view == "Conformal uncertainty":
        if y is None:
            st.info("Select a target column to enable conformal prediction.")
        else:
            alpha = st.slider("alpha (miscoverage)", 0.01, 0.5, 0.1)
            frac = st.slider("calibration fraction", 0.1, 0.9, 0.5)
            n = len(X); k = int(n * frac)
            res = ex.conformal(X.iloc[:k], y[:k], X.iloc[k:], alpha=alpha, y_test=y[k:])
            st.metric("Target coverage", f"{res.coverage_target:.0%}")
            if res.empirical_coverage is not None:
                st.metric("Empirical coverage", f"{res.empirical_coverage:.0%}")
            if res.average_set_size is not None:
                st.metric("Avg set size", f"{res.average_set_size:.2f}")
            if res.average_interval_width is not None:
                st.metric("Avg interval width", f"{res.average_interval_width:.3g}")

    # ---- Prototypes ---------------------------------------------------- #
    elif view == "Prototypes & criticisms":
        protos, crits = prototypes_and_criticisms(X)
        st.subheader("Prototypes (representative)")
        st.dataframe(X.iloc[protos])
        st.subheader("Criticisms (atypical)")
        st.dataframe(X.iloc[crits])

    # ---- Explanation quality ------------------------------------------- #
    elif view == "Explanation quality":
        row = st.number_input("Row index", 0, len(X) - 1, 0)
        q = ex.explanation_quality(int(row))
        c1, c2 = st.columns(2)
        c1.metric("Faithfulness", "n/a" if q.faithfulness is None else f"{q.faithfulness:.2f}")
        c2.metric("Stability", "n/a" if q.stability is None else f"{q.stability:.2f}")
        for n in q.notes:
            st.write("• " + n)

    # ---- Drift --------------------------------------------------------- #
    elif view == "Data drift":
        current_file = st.file_uploader("Current dataset to compare against the loaded one", type=["csv", "parquet", "json"], key="drift")
        if current_file:
            current = _load_data(current_file)
            report = detect_drift(df, current)
            tag = "🔴 drift detected" if report.drifted else "🟢 stable"
            st.subheader(f"{report.n_drifted}/{report.n_features} features drifted — {tag}")
            st.dataframe(pd.DataFrame([f.to_dict() for f in report.features]).set_index("feature"))


if __name__ == "__main__":
    _app()
