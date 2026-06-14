"""Command-line interface: explain a saved model without writing code.

    explainx report   --model m.joblib --data d.csv --target y --sensitive gender --html out.html
    explainx bias     --model m.joblib --data d.csv --target y --sensitive gender
    explainx explain  --model m.joblib --data d.csv --target y --row 0
    explainx drift     --reference train.csv --current prod.csv

Output is JSON on stdout (pipe it into `jq`, a file, or an agent), with an
optional rendered HTML report.
"""

from __future__ import annotations

import argparse
import json
import sys

from .io import load_model, load_xy, load_dataframe
from .explainer import ModelExplainer


def _common(p: argparse.ArgumentParser, target: bool = True):
    p.add_argument("--model", required=True, help="Path to joblib/pickle model.")
    p.add_argument("--data", required=True, help="Path to CSV/Parquet/JSON dataset.")
    if target:
        p.add_argument("--target", default=None, help="Name of the target column, if present.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="explainx", description="LLM-native model explainability.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="Full explainability report.")
    _common(p_report)
    p_report.add_argument("--sensitive", nargs="*", default=None, help="Sensitive feature columns.")
    p_report.add_argument("--n-local", type=int, default=3)
    p_report.add_argument("--html", default=None, help="Also write an HTML report to this path.")

    p_bias = sub.add_parser("bias", help="Fairness analysis on a sensitive feature.")
    _common(p_bias)
    p_bias.add_argument("--sensitive", required=True, help="Sensitive feature column.")

    p_explain = sub.add_parser("explain", help="Explain a single prediction.")
    _common(p_explain)
    p_explain.add_argument("--row", type=int, required=True)
    p_explain.add_argument("--method", choices=["auto", "lime"], default="auto")

    p_cf = sub.add_parser("counterfactual", help="Minimal change to flip a prediction.")
    _common(p_cf)
    p_cf.add_argument("--row", type=int, required=True)

    p_drift = sub.add_parser("drift", help="Detect drift between two datasets.")
    p_drift.add_argument("--reference", required=True)
    p_drift.add_argument("--current", required=True)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "drift":
        from .drift import detect_drift

        result = detect_drift(load_dataframe(args.reference), load_dataframe(args.current))
        print(result.to_json())
        return 0

    model = load_model(args.model)
    X, y = load_xy(args.data, getattr(args, "target", None))
    ex = ModelExplainer(model, X, y)

    if args.command == "report":
        report = ex.report(sensitive_features=args.sensitive, n_local=args.n_local)
        if args.html:
            from .report_html import save_html

            save_html(report, args.html)
            print(f"# HTML report written to {args.html}", file=sys.stderr)
        print(report.to_json())
    elif args.command == "bias":
        print(ex.fairness(args.sensitive).to_json())
    elif args.command == "explain":
        result = ex.lime(args.row) if args.method == "lime" else ex.explain(args.row)
        print(result.to_json())
    elif args.command == "counterfactual":
        print(ex.counterfactual(args.row).to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
