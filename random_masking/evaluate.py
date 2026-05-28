"""Evaluate the BERT-style transformer for masked reconstruction.

Default: mask PTS and predict it for SGA's 4 sample games.
Bonus: same SGA 5/18 game with different masked columns.

Run:     python random_masking/evaluate.py
Reads:   random_masking/trained_model.pt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from masking import (
    load_and_split, apply_mask,
    MASKABLE_COLS, CONTEXT_COLS, ALL_COLS,
    PTS_IDX, N_MASKABLE, N_CONTEXT,
)
from model import build_model


MODEL_PATH = Path(__file__).parent / "trained_model.pt"


def predict_one_game(model, raw_values, mask_col_name, col_means, col_stds):
    if mask_col_name not in MASKABLE_COLS:
        raise ValueError(f"{mask_col_name} is a context column and can't be masked")

    mask_idx = MASKABLE_COLS.index(mask_col_name)
    actual_raw = float(raw_values[mask_idx])

    row_z = (raw_values - col_means) / col_stds
    row_t = torch.from_numpy(row_z.astype(np.float32)).unsqueeze(0)
    mask_t = torch.tensor([mask_idx], dtype=torch.long)
    values, mask_positions, _ = apply_mask(row_t, mask_t)

    with torch.no_grad():
        pred_z = model(values, mask_positions).item()

    predicted_raw = pred_z * col_stds[mask_idx] + col_means[mask_idx]
    return predicted_raw, actual_raw, mask_idx


def report_game(model, df, test_idx, col_means, col_stds,
                player_name, date_str, mask_col="PTS"):
    mask = (df["PLAYER_NAME"] == player_name) & (df["GAME_DATE"] == pd.Timestamp(date_str))
    rows = df[mask]
    if rows.empty:
        print(f"{player_name} on {date_str} not found\n")
        return
    row = rows.iloc[0]
    df_idx = rows.index[0]
    split = "test" if df_idx in set(test_idx.tolist()) else "train"

    raw_values = row[ALL_COLS].to_numpy(dtype=np.float32)
    pred_raw, actual_raw, mask_idx = predict_one_game(
        model, raw_values, mask_col, col_means, col_stds
    )

    matchup = row.get("MATCHUP", "")
    print(f"=== {player_name}  {date_str}  {matchup} — predict masked {mask_col} ===")
    print(f"this game ended up in the {split.upper()} split")
    if split == "train":
        print("  (warning: model has seen this row during training — prediction is optimistic)")
    print()
    print(f"input stats (raw, with {mask_col} hidden):")
    for col, val in zip(ALL_COLS, raw_values):
        if col == mask_col:
            marker = "  ← MASKED"
        elif col in CONTEXT_COLS:
            marker = "  (context, always visible)"
        else:
            marker = ""
        if "_PCT" in col:
            print(f"  {col:<10} = {val:.3f}{marker}")
        else:
            print(f"  {col:<10} = {int(val)}{marker}")
    print()
    if mask_col.endswith("_PCT"):
        print(f"  predicted {mask_col}: {pred_raw:.3f}")
        print(f"  actual {mask_col}:    {actual_raw:.3f}")
    else:
        print(f"  predicted {mask_col}: {pred_raw:.2f}")
        print(f"  actual {mask_col}:    {actual_raw:.2f}")
    print(f"  error:        {pred_raw - actual_raw:+.2f}\n\n")


def main() -> None:
    X_train_full, X_test_full, col_means, col_stds, df, test_idx = load_and_split()

    model = build_model(n_maskable=N_MASKABLE, n_context=N_CONTEXT)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    pts_mean = float(col_means[PTS_IDX])
    pts_std = float(col_stds[PTS_IDX])
    test_pts_real = X_test_full[:, PTS_IDX].numpy() * pts_std + pts_mean

    n_test = X_test_full.shape[0]
    pts_mask_cols = torch.full((n_test,), PTS_IDX, dtype=torch.long)
    values_t, mask_pos_t, _ = apply_mask(X_test_full, pts_mask_cols)
    with torch.no_grad():
        preds_z = model(values_t, mask_pos_t).squeeze(1).numpy()
    preds_real = preds_z * pts_std + pts_mean
    overall_mae = float(np.abs(preds_real - test_pts_real).mean())
    print(f"overall test MAE on PTS-masked predictions: {overall_mae:.2f} PTS\n")

    sga = "Shai Gilgeous-Alexander"
    report_game(model, df, test_idx, col_means, col_stds, sga, "2026-05-18")
    report_game(model, df, test_idx, col_means, col_stds, sga, "2026-05-11")
    report_game(model, df, test_idx, col_means, col_stds, sga, "2026-05-09")
    report_game(model, df, test_idx, col_means, col_stds, sga, "2026-03-23")

    print("=" * 60)
    print("BONUS: same SGA 5/18 game, different masked columns")
    print("=" * 60 + "\n")
    for col in ["MIN", "AST", "REB", "FG_PCT", "PTS"]:
        report_game(model, df, test_idx, col_means, col_stds, sga, "2026-03-23", mask_col=col)


if __name__ == "__main__":
    main()