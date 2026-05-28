"""Train the BERT-style transformer for masked reconstruction.

Each step: pick a random column from MASKABLE_COLS, mask it with the
[MASK] token, predict its value. IS_OT is context — always visible.

Run:    python random_masking/train.py
Output: random_masking/trained_model.pt
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from masking import load_and_split, apply_mask, PTS_IDX, N_MASKABLE, N_CONTEXT
from model import build_model


MODEL_PATH = Path(__file__).parent / "trained_model.pt"


def main() -> None:
    X_train_full, X_test_full, col_means, col_stds, df, test_idx = load_and_split()
    print(f"train: {tuple(X_train_full.shape)}   test: {tuple(X_test_full.shape)}\n")

    pts_mean = float(col_means[PTS_IDX])
    pts_std = float(col_stds[PTS_IDX])
    test_pts_real = X_test_full[:, PTS_IDX].numpy() * pts_std + pts_mean

    model = build_model(n_maskable=N_MASKABLE, n_context=N_CONTEXT)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    baseline_mae = float(np.abs(pts_mean - test_pts_real).mean())
    print(f"baseline MAE (always predict {pts_mean:.2f}): {baseline_mae:.2f} PTS\n")

    best_mae = float("inf")
    best_epoch = -1

    epochs = 500
    for epoch in range(epochs):
        batch_size = X_train_full.shape[0]
        # Random mask column per row — only from MASKABLE cols.
        mask_cols = torch.randint(0, N_MASKABLE, (batch_size,))

        values, mask_positions, target = apply_mask(X_train_full, mask_cols)

        preds = model(values, mask_positions)
        loss = loss_fn(preds, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                n_test = X_test_full.shape[0]
                pts_mask_cols = torch.full((n_test,), PTS_IDX, dtype=torch.long)
                values_t, mask_pos_t, _ = apply_mask(X_test_full, pts_mask_cols)
                preds_z = model(values_t, mask_pos_t).squeeze(1).numpy()
                preds_real = preds_z * pts_std + pts_mean
                mae = float(np.abs(preds_real - test_pts_real).mean())
            model.train()

            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), MODEL_PATH)

            star = " *" if is_best else ""
            print(f"epoch {epoch:3d}   train loss={loss.item():.4f}   test PTS MAE={mae:.2f}{star}")

    print(f"\nbest test PTS MAE = {best_mae:.2f} PTS at epoch {best_epoch}")
    print(f"saved best model weights → {MODEL_PATH}")


if __name__ == "__main__":
    main()