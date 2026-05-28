"""Collects predictions from both trained models for the dashboard.

This module rebuilds the eval logic from player_stats/evaluate.py and
random_masking/evaluate.py, but RETURNS structured data instead of printing.
Both player_stats/ and random_masking/ each define their own masking.py and
model.py modules — we load them by file path to keep the two sets isolated.

Also exposes:
    list_players(), list_dates_for(player)            -- for UI dropdowns
    retrain_player_stats(progress_cb), retrain_random_masking(progress_cb)
                                                       -- re-trains and saves
                                                          trained_model.pt
"""

from pathlib import Path
import importlib.util
import sys
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).parent


# --- helpers ----------------------------------------------------------------
def _load_module(unique_name: str, file_path: Path):
    """Load a Python file as a fresh module under a unique name."""
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


ps_masking = _load_module("ps_masking", ROOT / "player_stats" / "masking.py")
ps_model_mod = _load_module("ps_model_mod", ROOT / "player_stats" / "model.py")
rm_masking = _load_module("rm_masking", ROOT / "random_masking" / "masking.py")
rm_model_mod = _load_module("rm_model_mod", ROOT / "random_masking" / "model.py")


# Defaults — used when the dashboard first loads or when called from __main__.
DEFAULT_PLAYER = "Shai Gilgeous-Alexander"
DEFAULT_DATES = ["2026-05-18", "2026-05-11", "2026-05-09", "2026-03-23"]
DEFAULT_BONUS_COLS = ["MIN", "AST", "REB", "FG_PCT", "PTS"]


# --- data introspection (for the UI dropdowns) ------------------------------
def _read_df() -> pd.DataFrame:
    df = pd.read_csv(ps_masking.DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def list_players() -> list[str]:
    """All players that appear in the dataset, sorted alphabetically."""
    df = _read_df()
    return sorted(df["PLAYER_NAME"].dropna().unique().tolist())


def list_dates_for(player: str) -> list[str]:
    """All game dates for the given player, newest first, as ISO strings."""
    df = _read_df()
    dates = df.loc[df["PLAYER_NAME"] == player, "GAME_DATE"].sort_values(ascending=False)
    return [d.strftime("%Y-%m-%d") for d in dates]


# --- player_stats model -----------------------------------------------------
def get_player_stats_results(
    player: str = DEFAULT_PLAYER,
    dates: Iterable[str] = DEFAULT_DATES,
) -> dict:
    """Run the player_stats (PTS-only) transformer on the given games + test MAE."""
    X_train, y_train, X_test, y_test, y_mean, y_std, df, test_idx = ps_masking.load_and_split()

    model = ps_model_mod.build_model(n_features=len(ps_masking.FEATURE_COLS))
    model.load_state_dict(torch.load(ROOT / "player_stats" / "trained_model.pt"))
    model.eval()

    # Overall test MAE
    with torch.no_grad():
        preds_z = model(X_test).squeeze(1).numpy()
    preds_real = preds_z * y_std + y_mean
    truth_real = y_test.squeeze(1).numpy() * y_std + y_mean
    overall_mae = float(np.abs(preds_real - truth_real).mean())

    # Recompute the same feature standardisation evaluate.py uses
    X_all = df[ps_masking.FEATURE_COLS].to_numpy(dtype=np.float32)
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0) + 1e-8

    test_idx_set = set(test_idx.tolist())
    games = []
    for date_str in dates:
        mask = (df["PLAYER_NAME"] == player) & (df["GAME_DATE"] == pd.Timestamp(date_str))
        rows = df[mask]
        if rows.empty:
            continue
        row = rows.iloc[0]
        df_idx = rows.index[0]
        split = "test" if df_idx in test_idx_set else "train"

        raw = row[ps_masking.FEATURE_COLS].to_numpy(dtype=np.float32)
        z = (raw - X_mean) / X_std

        x = torch.from_numpy(z).unsqueeze(0)
        with torch.no_grad():
            pred_z = model(x).item()
        pred_pts = float(pred_z * y_std + y_mean)
        actual_pts = float(row["PTS"])

        games.append({
            "date": date_str,
            "player": player,
            "matchup": row.get("MATCHUP", ""),
            "split": split,
            "predicted": pred_pts,
            "actual": actual_pts,
            "error": pred_pts - actual_pts,
            "features": {col: float(val) for col, val in zip(ps_masking.FEATURE_COLS, raw)},
        })

    return {"overall_mae": overall_mae, "games": games, "player": player}


# --- random_masking model ---------------------------------------------------
def _predict_masked(model, raw_values, mask_col_name, col_means, col_stds):
    mask_idx = rm_masking.MASKABLE_COLS.index(mask_col_name)
    actual_raw = float(raw_values[mask_idx])
    row_z = (raw_values - col_means) / col_stds
    row_t = torch.from_numpy(row_z.astype(np.float32)).unsqueeze(0)
    mask_t = torch.tensor([mask_idx], dtype=torch.long)
    values, mask_positions, _ = rm_masking.apply_mask(row_t, mask_t)
    with torch.no_grad():
        pred_z = model(values, mask_positions).item()
    predicted_raw = float(pred_z * col_stds[mask_idx] + col_means[mask_idx])
    return predicted_raw, actual_raw


def get_random_masking_results(
    player: str = DEFAULT_PLAYER,
    dates: Iterable[str] = DEFAULT_DATES,
    bonus_date: str | None = None,
    bonus_cols: Iterable[str] = DEFAULT_BONUS_COLS,
) -> dict:
    """Run the masked-reconstruction transformer for the given games + bonus cols.

    bonus_date defaults to the last date in `dates` if not provided.
    """
    dates = list(dates)
    if bonus_date is None and dates:
        bonus_date = dates[-1]

    X_train_full, X_test_full, col_means, col_stds, df, test_idx = rm_masking.load_and_split()

    model = rm_model_mod.build_model(
        n_maskable=rm_masking.N_MASKABLE, n_context=rm_masking.N_CONTEXT
    )
    model.load_state_dict(torch.load(ROOT / "random_masking" / "trained_model.pt"))
    model.eval()

    # Overall test MAE on PTS-masked predictions
    pts_idx = rm_masking.PTS_IDX
    pts_mean = float(col_means[pts_idx])
    pts_std = float(col_stds[pts_idx])
    test_pts_real = X_test_full[:, pts_idx].numpy() * pts_std + pts_mean
    n_test = X_test_full.shape[0]
    pts_mask_cols = torch.full((n_test,), pts_idx, dtype=torch.long)
    values_t, mask_pos_t, _ = rm_masking.apply_mask(X_test_full, pts_mask_cols)
    with torch.no_grad():
        preds_z = model(values_t, mask_pos_t).squeeze(1).numpy()
    preds_real = preds_z * pts_std + pts_mean
    overall_mae = float(np.abs(preds_real - test_pts_real).mean())

    test_idx_set = set(test_idx.tolist())

    pts_games = []
    for date_str in dates:
        mask = (df["PLAYER_NAME"] == player) & (df["GAME_DATE"] == pd.Timestamp(date_str))
        rows = df[mask]
        if rows.empty:
            continue
        row = rows.iloc[0]
        df_idx = rows.index[0]
        split = "test" if df_idx in test_idx_set else "train"
        raw_values = row[rm_masking.ALL_COLS].to_numpy(dtype=np.float32)
        pred, actual = _predict_masked(model, raw_values, "PTS", col_means, col_stds)
        pts_games.append({
            "date": date_str,
            "player": player,
            "matchup": row.get("MATCHUP", ""),
            "split": split,
            "stat": "PTS",
            "predicted": pred,
            "actual": actual,
            "error": pred - actual,
        })

    # Bonus: same game, different masked columns
    bonus_games = []
    if bonus_date:
        mask = (df["PLAYER_NAME"] == player) & (df["GAME_DATE"] == pd.Timestamp(bonus_date))
        rows = df[mask]
        if not rows.empty:
            row = rows.iloc[0]
            df_idx = rows.index[0]
            split = "test" if df_idx in test_idx_set else "train"
            raw_values = row[rm_masking.ALL_COLS].to_numpy(dtype=np.float32)
            for col in bonus_cols:
                pred, actual = _predict_masked(model, raw_values, col, col_means, col_stds)
                bonus_games.append({
                    "date": bonus_date,
                    "player": player,
                    "matchup": row.get("MATCHUP", ""),
                    "split": split,
                    "stat": col,
                    "predicted": pred,
                    "actual": actual,
                    "error": pred - actual,
                })

    return {
        "overall_mae": overall_mae,
        "pts_games": pts_games,
        "bonus_games": bonus_games,
        "bonus_date": bonus_date,
        "player": player,
    }


# --- retraining -------------------------------------------------------------
# These mirror player_stats/train.py and random_masking/train.py, but yield
# progress via a callback so the Streamlit UI can show a live curve.
ProgressCB = Callable[[dict], None]


def retrain_player_stats(
    epochs: int = 500,
    log_every: int = 50,
    progress_cb: ProgressCB | None = None,
) -> dict:
    """Re-train the player_stats transformer. Mirrors player_stats/train.py."""
    X_train, y_train, X_test, y_test, y_mean, y_std, df, test_idx = ps_masking.load_and_split()

    model = ps_model_mod.build_model(n_features=len(ps_masking.FEATURE_COLS))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    truth_real = y_test.squeeze(1).numpy() * y_std + y_mean
    baseline_mae = float(np.abs(truth_real - y_mean).mean())

    best_mae = float("inf")
    best_epoch = -1
    history: list[dict] = []
    save_path = ROOT / "player_stats" / "trained_model.pt"

    for epoch in range(epochs):
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
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
                torch.save(model.state_dict(), save_path)

            history.append({"epoch": epoch, "train_loss": float(loss.item()), "test_mae": mae, "is_best": is_best})
            if progress_cb is not None:
                progress_cb({
                    "stage": "player_stats",
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": float(loss.item()),
                    "test_mae": mae,
                    "best_mae": best_mae,
                    "is_best": is_best,
                })

    return {
        "best_mae": best_mae,
        "best_epoch": best_epoch,
        "baseline_mae": baseline_mae,
        "history": history,
    }


def retrain_random_masking(
    epochs: int = 500,
    log_every: int = 50,
    progress_cb: ProgressCB | None = None,
) -> dict:
    """Re-train the random_masking transformer. Mirrors random_masking/train.py."""
    X_train_full, X_test_full, col_means, col_stds, df, test_idx = rm_masking.load_and_split()

    pts_idx = rm_masking.PTS_IDX
    pts_mean = float(col_means[pts_idx])
    pts_std = float(col_stds[pts_idx])
    test_pts_real = X_test_full[:, pts_idx].numpy() * pts_std + pts_mean

    model = rm_model_mod.build_model(
        n_maskable=rm_masking.N_MASKABLE, n_context=rm_masking.N_CONTEXT
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    baseline_mae = float(np.abs(pts_mean - test_pts_real).mean())

    best_mae = float("inf")
    best_epoch = -1
    history: list[dict] = []
    save_path = ROOT / "random_masking" / "trained_model.pt"

    for epoch in range(epochs):
        batch_size = X_train_full.shape[0]
        mask_cols = torch.randint(0, rm_masking.N_MASKABLE, (batch_size,))
        values, mask_positions, target = rm_masking.apply_mask(X_train_full, mask_cols)

        preds = model(values, mask_positions)
        loss = loss_fn(preds, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                n_test = X_test_full.shape[0]
                pts_mask_cols = torch.full((n_test,), pts_idx, dtype=torch.long)
                values_t, mask_pos_t, _ = rm_masking.apply_mask(X_test_full, pts_mask_cols)
                preds_z = model(values_t, mask_pos_t).squeeze(1).numpy()
                preds_real = preds_z * pts_std + pts_mean
                mae = float(np.abs(preds_real - test_pts_real).mean())
            model.train()

            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)

            history.append({"epoch": epoch, "train_loss": float(loss.item()), "test_mae": mae, "is_best": is_best})
            if progress_cb is not None:
                progress_cb({
                    "stage": "random_masking",
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": float(loss.item()),
                    "test_mae": mae,
                    "best_mae": best_mae,
                    "is_best": is_best,
                })

    return {
        "best_mae": best_mae,
        "best_epoch": best_epoch,
        "baseline_mae": baseline_mae,
        "history": history,
    }


if __name__ == "__main__":
    ps = get_player_stats_results()
    rm = get_random_masking_results()
    print(f"player_stats overall MAE: {ps['overall_mae']:.2f} PTS")
    for g in ps["games"]:
        print(f"  {g['date']}  pred={g['predicted']:.2f}  actual={g['actual']:.0f}  err={g['error']:+.2f}  [{g['split']}]")
    print(f"\nrandom_masking overall MAE (PTS): {rm['overall_mae']:.2f}")
    for g in rm["pts_games"]:
        print(f"  {g['date']}  pred={g['predicted']:.2f}  actual={g['actual']:.0f}  err={g['error']:+.2f}  [{g['split']}]")
    print(f"\nBonus (different masked columns on {rm['bonus_date']}):")
    for g in rm["bonus_games"]:
        fmt = "{:.3f}" if g["stat"].endswith("_PCT") else "{:.2f}"
        print(f"  {g['stat']:<8} pred=" + fmt.format(g['predicted'])
              + "  actual=" + fmt.format(g['actual'])
              + f"  err={g['error']:+.3f}")
    print(f"\nplayers available: {len(list_players())}")
    print(f"first 5 dates for {DEFAULT_PLAYER}: {list_dates_for(DEFAULT_PLAYER)[:5]}")
