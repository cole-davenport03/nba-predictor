"""Day 5: tiny MLP that predicts W/L from two stats.

Uses 20 hand-made games so you don't need to wire anything up.
Features: FG_PCT, AST.  Label: 1 if won, 0 if lost.
"""

import numpy as np
import torch
import torch.nn as nn


# --- data: 20 games, each row is [FG_PCT, AST, won?] ---
games = np.array([
    [0.52, 28, 1], [0.48, 25, 1], [0.55, 30, 1], [0.50, 24, 1], [0.49, 26, 1],
    [0.46, 22, 1], [0.51, 27, 1], [0.47, 23, 1], [0.53, 29, 1], [0.50, 25, 1],
    [0.41, 18, 0], [0.39, 20, 0], [0.43, 17, 0], [0.40, 19, 0], [0.42, 21, 0],
    [0.38, 16, 0], [0.44, 18, 0], [0.40, 20, 0], [0.42, 17, 0], [0.39, 19, 0],
], dtype=np.float32)

# numpy -> torch. X is features, y is labels (shape (n, 1) to match model output).
X = torch.from_numpy(games[:, :2])
y = torch.from_numpy(games[:, 2:3])


# --- the model: 2 inputs -> 8 hidden -> 1 output ---
# Each nn.Linear is the same w*x + b from Day 4. ReLU = max(0, x) between them.
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

#Goeffrey Hinton
#Yann LeCun


loss_fn = nn.BCEWithLogitsLoss()                       # binary classification loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# --- training loop: same 3 lines as Day 4 ---
for epoch in range(200):
    logits = model(X)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        preds = (logits > 0).float()                   # logit > 0  <=>  prob > 0.5
        acc = (preds == y).float().mean().item()
        print(f"epoch {epoch:3d}  loss={loss.item():.3f}  acc={acc:.2f}")

# --- final report ---
with torch.no_grad():
    final_logits = model(X)
    final_preds = (final_logits > 0).float()
    final_acc = (final_preds == y).float().mean().item()
print(f"final accuracy: {final_acc:.2f}")
