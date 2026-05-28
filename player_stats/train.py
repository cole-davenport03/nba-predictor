"""Train the masked-reconstruction MLP on the player game data.

Run:    python player_stats/train.py
Output: player_stats/trained_model.pt  (best checkpoint by test MAE)
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from masking import load_and_split, FEATURE_COLS
from model import build_model


MODEL_PATH = Path(__file__).parent / "trained_model.pt"


def main() -> None:
    # --- 1. Load data ---
    X_train, y_train, X_test, y_test, y_mean, y_std, df, test_idx = load_and_split()
    print(f"train: {tuple(X_train.shape)}   test: {tuple(X_test.shape)}\n")

    # --- 2. Build model + loss + optimizer ---
    model = build_model(n_features=len(FEATURE_COLS))
    loss_fn = nn.MSELoss()                                    # regression loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 3. Baseline: "always predict the league mean" ---
    # Our model has to beat this to be doing anything useful.
    truth_real = y_test.squeeze(1).numpy() * y_std + y_mean   # un-standardize
    baseline_mae = float(np.abs(truth_real - y_mean).mean())
    print(f"baseline MAE (always predict {y_mean:.2f}): {baseline_mae:.2f} PTS\n")

    # --- 4. Best-tracker (initialized ONCE, before the loop) ---
    best_mae = float("inf")
    best_epoch = -1

    # --- 5. Training loop ---
    epochs = 500
    for epoch in range(epochs):
        # the canonical 5 lines
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds_z = model(X_test).squeeze(1).numpy()
                preds_real = preds_z * y_std + y_mean
                mae = float(np.abs(preds_real - truth_real).mean())
            model.train()

            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), MODEL_PATH)

            star = " *" if is_best else ""
            print(f"epoch {epoch:3d}   train loss={loss.item():.4f}   test MAE={mae:.2f} PTS{star}")

    # --- 6. Final summary ---
    print(f"\nbest test MAE = {best_mae:.2f} PTS at epoch {best_epoch}")
    print(f"saved best model weights → {MODEL_PATH}")


if __name__ == "__main__":
    main()