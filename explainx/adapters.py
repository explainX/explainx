"""Universal model adapter — make explainX work with any ML framework.

The engine speaks the scikit-learn convention (``predict`` / ``predict_proba`` /
``classes_``). Many libraries already follow it, so they work with **zero
wrapping**:

* scikit-learn (every estimator)
* XGBoost   — ``XGBClassifier`` / ``XGBRegressor``
* LightGBM  — ``LGBMClassifier`` / ``LGBMRegressor``
* CatBoost  — ``CatBoostClassifier`` / ``CatBoostRegressor``

For frameworks that *don't* expose that API, :func:`wrap_model` adapts them into
it: native XGBoost / LightGBM ``Booster`` objects, Keras / TensorFlow models,
PyTorch ``nn.Module``\\s, statsmodels results, or any custom ``predict_fn``.

``ModelExplainer`` calls :func:`wrap_model` automatically, so in practice you can
pass almost any trained model directly; reach for ``wrap_model`` explicitly only
when you need to pass a custom prediction function or override the task/classes.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence
import numpy as np


def _is_frame(X) -> bool:
    try:
        import pandas as pd

        return isinstance(X, (pd.DataFrame, pd.Series))
    except Exception:
        return False


def _to_numpy(X) -> np.ndarray:
    if _is_frame(X):
        return X.to_numpy()
    return np.asarray(X)


def is_sklearn_compatible(model: Any) -> bool:
    """True if the engine can use the model as-is (no wrapping needed)."""
    try:
        from sklearn.base import is_classifier, is_regressor

        if is_classifier(model) or is_regressor(model):
            return True
    except Exception:
        pass
    # CatBoost / other duck-typed estimators expose predict_proba or predict+classes_.
    if hasattr(model, "predict") and (hasattr(model, "predict_proba") or hasattr(model, "_estimator_type")):
        return True
    return False


def detect_framework(model: Any) -> str:
    module = type(model).__module__.split(".")[0]
    cls = type(model).__name__
    if module == "xgboost" and cls == "Booster":
        return "xgboost_booster"
    if module == "lightgbm" and cls == "Booster":
        return "lightgbm_booster"
    if module in ("keras", "tensorflow", "tf_keras"):
        return "keras"
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            return "torch"
    except Exception:
        pass
    if module == "statsmodels":
        return "statsmodels"
    if is_sklearn_compatible(model):
        return "sklearn"
    if callable(getattr(model, "predict", None)):
        return "sklearn"  # best-effort duck typing
    return "unknown"


class ModelAdapter:
    """Wraps an arbitrary model in the scikit-learn predict/predict_proba API."""

    def __init__(
        self,
        predict_fn: Callable,
        predict_proba_fn: Optional[Callable] = None,
        classes: Optional[Sequence] = None,
        problem_type: Optional[str] = None,
        name: str = "WrappedModel",
    ):
        self._predict_fn = predict_fn
        self._predict_proba_fn = predict_proba_fn
        self._name = name
        self.problem_type = problem_type
        if classes is not None:
            self.classes_ = np.asarray(classes)
        if predict_proba_fn is not None:
            # Expose predict_proba only when available, so detection works.
            self.predict_proba = self._predict_proba  # type: ignore[assignment]

    def predict(self, X):
        return np.asarray(self._predict_fn(X))

    def _predict_proba(self, X):
        return np.asarray(self._predict_proba_fn(X))

    def __repr__(self):
        return f"ModelAdapter({self._name})"


def _proba_to_classes(proba: np.ndarray):
    proba = np.asarray(proba)
    if proba.ndim == 1 or proba.shape[1] == 1:
        return np.array([0, 1])
    return np.arange(proba.shape[1])


def _from_proba_predict(proba: np.ndarray, classes: np.ndarray):
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return (proba >= 0.5).astype(int)
    if proba.shape[1] == 1:
        return (proba[:, 0] >= 0.5).astype(int)
    return classes[np.argmax(proba, axis=1)]


def _two_col(proba: np.ndarray) -> np.ndarray:
    """Normalize a 1-column / 1-D probability into a 2-column [neg, pos] matrix."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if proba.shape[1] == 1:
        return np.hstack([1 - proba, proba])
    return proba


def wrap_model(
    model: Any = None,
    *,
    predict_fn: Optional[Callable] = None,
    predict_proba_fn: Optional[Callable] = None,
    classes: Optional[Sequence] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
) -> Any:
    """Return a model usable by the engine, wrapping non-sklearn frameworks.

    sklearn-compatible models (incl. XGBoost/LightGBM/CatBoost sklearn wrappers)
    are returned unchanged. Pass ``predict_fn`` (+ optional ``predict_proba_fn``,
    ``classes``) to wrap a fully custom model. ``task`` is ``"classification"`` or
    ``"regression"`` and helps when it can't be inferred.
    """
    # 1. Fully custom prediction functions.
    if predict_fn is not None or predict_proba_fn is not None:
        if predict_proba_fn is not None and predict_fn is None:
            # Derive predict() by thresholding/argmaxing the probabilities.
            def _pred(X, _f=predict_proba_fn, _c=classes):
                p = np.asarray(_f(X))
                c = np.asarray(_c) if _c is not None else _proba_to_classes(p)
                return _from_proba_predict(p, c)
            return ModelAdapter(
                _pred, predict_proba_fn, classes=classes, problem_type="classification", name="custom"
            )
        return ModelAdapter(predict_fn, predict_proba_fn, classes=classes, problem_type=task, name="custom")

    if model is None:
        raise TypeError("wrap_model() needs a model, or a predict_fn / predict_proba_fn.")

    fw = framework or detect_framework(model)

    if fw in ("sklearn", "unknown"):
        if fw == "unknown":
            raise TypeError(
                f"Could not adapt model of type {type(model).__name__}. Pass predict_fn "
                "(and predict_proba_fn for classification) to wrap_model()."
            )
        return model  # already usable

    if fw == "xgboost_booster":
        import xgboost

        def raw(X):
            # Pass the DataFrame straight through so DMatrix keeps feature names
            # (the Booster validates names when it was trained with them).
            dm = xgboost.DMatrix(X if _is_frame(X) else _to_numpy(X))
            return np.asarray(model.predict(dm))
        return _wrap_raw_scores(raw, task, classes, name="xgboost.Booster")

    if fw == "lightgbm_booster":
        def raw(X):
            return np.asarray(model.predict(X if _is_frame(X) else _to_numpy(X)))
        return _wrap_raw_scores(raw, task, classes, name="lightgbm.Booster")

    if fw == "keras":
        def raw(X):
            return np.asarray(model.predict(_to_numpy(X), verbose=0))
        return _wrap_raw_scores(raw, task, classes, name="keras.Model")

    if fw == "torch":
        import torch

        def raw(X):
            model.eval()
            with torch.no_grad():
                out = model(torch.as_tensor(_to_numpy(X), dtype=torch.float32))
            return out.detach().cpu().numpy()
        return _wrap_raw_scores(raw, task, classes, name="torch.nn.Module")

    if fw == "statsmodels":
        def _pred(X):
            return np.asarray(model.predict(_to_numpy(X)))
        # statsmodels results are usually regression / single-prob outputs.
        if task == "classification":
            def _pp(X):
                return _two_col(model.predict(_to_numpy(X)))
            return ModelAdapter(
                lambda X: _from_proba_predict(model.predict(_to_numpy(X)), np.array([0, 1])),
                _pp, classes=classes or [0, 1], problem_type="classification", name="statsmodels",
            )
        return ModelAdapter(_pred, problem_type="regression", name="statsmodels")

    return model


def _wrap_raw_scores(raw_fn: Callable, task: Optional[str], classes, name: str) -> ModelAdapter:
    """Build an adapter from a function returning raw scores/probabilities.

    Heuristic when ``task`` is unset: a 2-D output (k>1 columns) is multiclass
    probabilities; a 1-D output in [0, 1] is binary probability; anything else is
    treated as a regression value.
    """
    sample = None

    def infer_task(p):
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] > 1:
            return "classification"
        flat = p.reshape(-1)
        if flat.size and flat.min() >= 0.0 and flat.max() <= 1.0:
            return "classification"
        return "regression"

    resolved_task = task

    def predict(X):
        nonlocal resolved_task
        p = raw_fn(X)
        if resolved_task is None:
            resolved_task = infer_task(p)
        if resolved_task == "regression":
            arr = np.asarray(p)
            return arr.reshape(-1) if arr.ndim > 1 and arr.shape[1] == 1 else arr
        cls = np.asarray(classes) if classes is not None else _proba_to_classes(p)
        return _from_proba_predict(p, cls)

    def predict_proba(X):
        return _two_col(raw_fn(X))

    if task == "regression":
        return ModelAdapter(predict, problem_type="regression", name=name)
    # Default to exposing predict_proba (classification); harmless if unused.
    return ModelAdapter(
        predict, predict_proba, classes=classes, problem_type=task or "classification", name=name
    )
