"""Loading helpers for the MCP boundary.

Over MCP a model can't be passed in memory, so tools take *paths*: a serialized
estimator (joblib or pickle) and a tabular dataset (CSV/Parquet/JSON). These
helpers load them and split off the target column when one is named.
"""

from __future__ import annotations

from typing import Any, Optional
import pandas as pd


def load_model(model_path: str) -> Any:
    try:
        import joblib

        return joblib.load(model_path)
    except Exception:
        import pickle

        with open(model_path, "rb") as fh:
            return pickle.load(fh)


def load_dataframe(data_path: str) -> pd.DataFrame:
    lower = data_path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(data_path)
    if lower.endswith(".json"):
        return pd.read_json(data_path)
    return pd.read_csv(data_path)


def load_xy(data_path: str, target_column: Optional[str] = None):
    """Return ``(X, y)`` splitting off ``target_column`` if it is present."""
    df = load_dataframe(data_path)
    if target_column and target_column in df.columns:
        y = df[target_column].to_numpy()
        X = df.drop(columns=[target_column])
        return X, y
    return df, None
