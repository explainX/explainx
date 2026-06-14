# explainX framework examples

Runnable, self-contained examples showing **explainX with every major ML
framework** — for humans to study and for LLM agents to learn the API from.

explainX speaks the scikit-learn convention (`predict` / `predict_proba`), so
many frameworks work **with no wrapping at all**. For the rest, `wrap_model()`
adapts them.

| Example | Framework | Wrapping needed? |
|---|---|---|
| [`01_sklearn.py`](01_sklearn.py) | scikit-learn | none |
| [`02_xgboost.py`](02_xgboost.py) | XGBoost (sklearn API **and** native `Booster`) | native Booster → `wrap_model` |
| [`03_lightgbm.py`](03_lightgbm.py) | LightGBM (sklearn API and native `Booster`) | native Booster → `wrap_model` |
| [`04_catboost.py`](04_catboost.py) | CatBoost | none |
| [`05_keras_tensorflow.py`](05_keras_tensorflow.py) | Keras / TensorFlow | `wrap_model` |
| [`06_pytorch.py`](06_pytorch.py) | PyTorch | `wrap_model` |
| [`07_statsmodels.py`](07_statsmodels.py) | statsmodels | `wrap_model(task=...)` |
| [`08_custom_predict_fn.py`](08_custom_predict_fn.py) | any model / API | `wrap_model(predict_fn=...)` |

Run any of them:

```sh
pip install "explainx[all]"      # plus the framework, e.g. pip install xgboost
python examples/02_xgboost.py
```

Each script trains a small model, then runs explainX — a full report plus a few
individual modules — and prints the structured results.

**Rule of thumb for agents:** if a model has `.predict` (and `.predict_proba`
for classification), pass it straight to `explain_model(model, X, y)`. Otherwise
wrap it: `explain_model(wrap_model(model, task="classification"), X, y)`, or for a
fully custom model `wrap_model(predict_fn=fn)` / `wrap_model(predict_proba_fn=fn)`.
