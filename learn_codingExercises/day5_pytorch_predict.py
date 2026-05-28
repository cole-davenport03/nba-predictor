"""Day 5: train a tiny PyTorch model on the features from day4.

This file imports your work from day4_numpy_features.py and trains a
classifier. The math is the same Linear/ReLU/Linear pattern you've seen,
just on real data.
"""
import numpy as np
import torch
import torch.nn as nn

from day4_numpy_features import fetch_games, build_features

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
games = fetch_games("OKC")
X_np, y_np = build_features(games)

# Standardize features so both columns have similar scale.
# (is_home is 0/1, avg_margin can be +/- 20. Without standardizing, the
# margin column dominates and training is slow.)
X_np = (X_np - X_np.mean(axis=0)) / (X_np.std(axis=0) + 1e-8)

# NumPy -> torch tensors. y_np[:, None] reshapes (n,) into (n, 1) so it
# matches the shape of the model's output.
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)[:, None]

# Train / val split: last 20% is validation.
n_val = len(X) // 5
X_train, X_val = X[:-n_val], X[-n_val:]
y_train, y_val = y[:-n_val], y[-n_val:]

# -------------------------------------------------------------------
# 2. Build the model
# -------------------------------------------------------------------
# Architecture: 2 inputs -> 8 hidden units -> 1 logit out.
# A "logit" is just a raw real number; BCEWithLogitsLoss handles the
# sigmoid internally.
#
# TODO: fill in the nn.Sequential with three pieces:
#   - nn.Linear(2, 8)
#   - nn.ReLU()
#   - nn.Linear(8, 1)
model = nn.Sequential(
    # your three lines here
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


# -------------------------------------------------------------------
# 3. Training loop
# -------------------------------------------------------------------
for epoch in range(200):
    # TODO: write the five canonical lines of a PyTorch training step.
    # In order:
    #   1) forward pass: logits = model(X_train)
    #   2) compute loss: loss = loss_fn(logits, y_train)
    #   3) clear old gradients: optimizer.zero_grad()
    #   4) compute new gradients: loss.backward()
    #   5) update parameters: optimizer.step()
    pass

    # Log progress every 40 epochs.
    if epoch % 40 == 0:
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = ((val_logits > 0).float() == y_val).float().mean().item()
        # TODO: print epoch, loss.item(), and val_acc in one line.


# -------------------------------------------------------------------
# 4. Final report
# -------------------------------------------------------------------
with torch.no_grad():
    final_acc = ((model(X_val) > 0).float() == y_val).float().mean().item()
print(f"final validation accuracy: {final_acc:.2f}")
