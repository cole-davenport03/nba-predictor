# MAE on test set + the SGA 5/18 demo
"""Evaluate the trained model: confirm test MAE + single-game prediction for SGA 5/18.

Run:     python player_stats/evaluate.py
Reads:   player_stats/trained_model.pt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from masking import load_and_split, FEATURE_COLS
from model import build_model


MODEL_PATH = Path(__file__).parent / "trained_model.pt"


def main() -> None:
    # --- 1. Re-load data with the same seed as train.py ---
    X_train, y_train, X_test, y_test, y_mean, y_std, df, test_idx = load_and_split()

    # --- 2. Load the best model checkpoint ---
    model = build_model(n_features=len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # --- 3. Confirm overall test MAE (should match what train.py printed) ---
    with torch.no_grad():
        preds_z = model(X_test).squeeze(1).numpy()
    preds_real = preds_z * y_std + y_mean
    truth_real = y_test.squeeze(1).numpy() * y_std + y_mean
    overall_mae = float(np.abs(preds_real - truth_real).mean())
    print(f"overall test MAE: {overall_mae:.2f} PTS")
    print("(should match the best test MAE that train.py reported)\n")

    # --- 4. Find SGA's 5/18 game in the dataframe ---
    sga_mask = (
        (df["PLAYER_NAME"] == "Shai Gilgeous-Alexander")
        & (df["GAME_DATE"] == pd.Timestamp("2026-05-18"))
    )
    sga_rows = df[sga_mask]
    if sga_rows.empty:
        print("SGA 5/18 game not in dataset, aborting demo")
        return

    sga_row = sga_rows.iloc[0]
    df_idx = sga_rows.index[0]
    split = "test" if df_idx in set(test_idx.tolist()) else "train"

    # --- 5. Standardize SGA's stats the same way the training data was ---
    # We re-compute X_mean / X_std the same way masking.py does. In a bigger
    # project you'd save these to disk during training and reload here.
    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0) + 1e-8

    sga_raw = sga_row[FEATURE_COLS].to_numpy(dtype=np.float32)
    sga_z = (sga_raw - X_mean) / X_std

    # --- 6. Predict ---
    x = torch.from_numpy(sga_z).unsqueeze(0)           # shape (1, 10)
    with torch.no_grad():
        pred_z = model(x).item()
    pred_real = pred_z * y_std + y_mean
    actual = int(sga_row["PTS"])

    # --- 7. Print the demo ---
    print("=== SGA ", sga_row["GAME_DATE"], "vs  — single-game prediction ===")
    print(f"this game ended up in the {split.upper()} split")
    if split == "train":
        print("  (warning: model has seen this row during training — prediction is optimistic)")
    print()
    print("input stats (raw):")
    for col, val in zip(FEATURE_COLS, sga_raw):
        if "_PCT" in col:
            print(f"  {col:<10} = {val:.3f}")
        else:
            print(f"  {col:<10} = {int(val)}")
    print()
    print(f"  predicted PTS: {pred_real:.2f}")
    print(f"  actual PTS:    {actual}")
    print(f"  error:         {pred_real - actual:+.2f}")

    print()
    print()

#SGA 5/11 game against the Lakers
# --- 4. Find SGA's 5/11 game in the dataframe ---
    sga_mask = (
        (df["PLAYER_NAME"] == "Shai Gilgeous-Alexander")
        & (df["GAME_DATE"] == pd.Timestamp("2026-05-11"))
    )
    sga_rows = df[sga_mask]
    if sga_rows.empty:
        print("SGA 5/11 game not in dataset, aborting demo")
        return

    sga_row = sga_rows.iloc[0]
    df_idx = sga_rows.index[0]
    split = "test" if df_idx in set(test_idx.tolist()) else "train"

    # --- 5. Standardize SGA's stats the same way the training data was ---
    # We re-compute X_mean / X_std the same way masking.py does. In a bigger
    # project you'd save these to disk during training and reload here.
    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0) + 1e-8

    sga_raw = sga_row[FEATURE_COLS].to_numpy(dtype=np.float32)
    sga_z = (sga_raw - X_mean) / X_std

    # --- 6. Predict ---
    x = torch.from_numpy(sga_z).unsqueeze(0)           # shape (1, 10)
    with torch.no_grad():
        pred_z = model(x).item()
    pred_real = pred_z * y_std + y_mean
    actual = int(sga_row["PTS"])

    # --- 7. Print the demo ---
    print("=== SGA ", sga_row["GAME_DATE"], "vs LAL — single-game prediction ===")
    print(f"this game ended up in the {split.upper()} split")
    if split == "train":
        print("  (warning: model has seen this row during training — prediction is optimistic)")
    print()
    print("input stats (raw):")
    for col, val in zip(FEATURE_COLS, sga_raw):
        if "_PCT" in col:
            print(f"  {col:<10} = {val:.3f}")
        else:
            print(f"  {col:<10} = {int(val)}")
    print()
    print(f"  predicted PTS: {pred_real:.2f}")
    print(f"  actual PTS:    {actual}")
    print(f"  error:         {pred_real - actual:+.2f}")

    print()
    print()


    # SGA 5/09 Game
    sga_mask = (
        (df["PLAYER_NAME"] == "Shai Gilgeous-Alexander")
        & (df["GAME_DATE"] == pd.Timestamp("2026-05-09"))
    )
    sga_rows = df[sga_mask]
    if sga_rows.empty:
        print("SGA 5/09 game not in dataset, aborting demo")
        return

    sga_row = sga_rows.iloc[0]
    df_idx = sga_rows.index[0]
    split = "test" if df_idx in set(test_idx.tolist()) else "train"

    # --- 5. Standardize SGA's stats the same way the training data was ---
    # We re-compute X_mean / X_std the same way masking.py does. In a bigger
    # project you'd save these to disk during training and reload here.
    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0) + 1e-8

    sga_raw = sga_row[FEATURE_COLS].to_numpy(dtype=np.float32)
    sga_z = (sga_raw - X_mean) / X_std

    # --- 6. Predict ---
    x = torch.from_numpy(sga_z).unsqueeze(0)           # shape (1, 10)
    with torch.no_grad():
        pred_z = model(x).item()
    pred_real = pred_z * y_std + y_mean
    actual = int(sga_row["PTS"])

    # --- 7. Print the demo ---
    print("=== SGA ", sga_row["GAME_DATE"], "vs LAL — single-game prediction ===")
    print(f"this game ended up in the {split.upper()} split")
    if split == "train":
        print("  (warning: model has seen this row during training — prediction is optimistic)")
    print()
    print("input stats (raw):")
    for col, val in zip(FEATURE_COLS, sga_raw):
        if "_PCT" in col:
            print(f"  {col:<10} = {val:.3f}")
        else:
            print(f"  {col:<10} = {int(val)}")
    print()
    print(f"  predicted PTS: {pred_real:.2f}")
    print(f"  actual PTS:    {actual}")
    print(f"  error:         {pred_real - actual:+.2f}")

    print()
    print()

    # SGA 3/23 Game
    sga_mask = (
        (df["PLAYER_NAME"] == "Shai Gilgeous-Alexander")
        & (df["GAME_DATE"] == pd.Timestamp("2026-03-23"))
    )
    sga_rows = df[sga_mask]
    if sga_rows.empty:
        print("SGA 3/23 game not in dataset, aborting demo")
        return

    sga_row = sga_rows.iloc[0]
    df_idx = sga_rows.index[0]
    split = "test" if df_idx in set(test_idx.tolist()) else "train"

    # --- 5. Standardize SGA's stats the same way the training data was ---
    # We re-compute X_mean / X_std the same way masking.py does. In a bigger
    # project you'd save these to disk during training and reload here.
    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0) + 1e-8

    sga_raw = sga_row[FEATURE_COLS].to_numpy(dtype=np.float32)
    sga_z = (sga_raw - X_mean) / X_std

    # --- 6. Predict ---
    x = torch.from_numpy(sga_z).unsqueeze(0)           # shape (1, 10)
    with torch.no_grad():
        pred_z = model(x).item()
    pred_real = pred_z * y_std + y_mean
    actual = int(sga_row["PTS"])

    # --- 7. Print the demo ---
    print("=== SGA ", sga_row["GAME_DATE"], "vs PHI — single-game prediction ===")
    print(f"this game ended up in the {split.upper()} split")
    if split == "train":
        print("  (warning: model has seen this row during training — prediction is optimistic)")
    print()
    print("input stats (raw):")
    for col, val in zip(FEATURE_COLS, sga_raw):
        if "_PCT" in col:
            print(f"  {col:<10} = {val:.3f}")
        else:
            print(f"  {col:<10} = {int(val)}")
    print()
    print(f"  predicted PTS: {pred_real:.2f}")
    print(f"  actual PTS:    {actual}")
    print(f"  error:         {pred_real - actual:+.2f}")

if __name__ == "__main__":
    main()