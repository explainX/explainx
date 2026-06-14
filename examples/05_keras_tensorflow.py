"""explainX with a Keras / TensorFlow model — wrap it with wrap_model.

A Keras model's .predict returns class probabilities but it has no
predict_proba, so wrap_model adapts it. For binary problems use a single
sigmoid output; explainX reads it as P(positive class).

    pip install tensorflow
    python examples/05_keras_tensorflow.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from explainx import ModelExplainer, wrap_model
from _common import loan_data

X, y = loan_data()
Xv = X.to_numpy().astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(Xv, y, epochs=15, batch_size=32, verbose=0)

# wrap_model detects Keras; task is inferred from the sigmoid output (classification).
ex = ModelExplainer(wrap_model(model, task="classification"), X, y)
print(ex.metrics().to_dict()["metrics"])
print("Top features:", [f.feature for f in ex.importance().top(3)])
local = ex.explain(0, top_k=3)
print(f"Row 0 -> {local.prediction} via {local.method}")
print(ex.fairness("gender").findings[0])
