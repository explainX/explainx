"""explainX with a PyTorch model — wrap the nn.Module with wrap_model.

wrap_model detects an nn.Module, runs it under torch.no_grad(), and reads the
output as class probabilities. Here we use a 2-logit + softmax head so the
output is a (n, 2) probability matrix.

    pip install torch
    python examples/06_pytorch.py
"""

import numpy as np
import torch
import torch.nn as nn

from explainx import ModelExplainer, wrap_model
from _common import loan_data

X, y = loan_data()
Xt = torch.tensor(X.to_numpy(), dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.long)


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 2), nn.Softmax(dim=1))

    def forward(self, x):
        return self.net(x)


model = Net(X.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
for _ in range(150):
    opt.zero_grad()
    loss = loss_fn(torch.log(model(Xt) + 1e-9), yt)
    loss.backward()
    opt.step()

# wrap_model detects torch.nn.Module; the (n, 2) softmax output -> classification.
ex = ModelExplainer(wrap_model(model, task="classification", classes=[0, 1]), X, y)
print(ex.metrics().to_dict()["metrics"])
print("Top features:", [f.feature for f in ex.importance().top(3)])
print(ex.fairness("gender").findings[0])
