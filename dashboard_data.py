"""Collects predictions from both trained models for the dashboard.

v2: player-aware. Each row now carries an integer player_id, and both models
have a learned per-player embedding added to every token. This gives the
model an "I'm looking at SGA" prior, which the v1 models couldn't have.

The original model files (model.py, masking.py, train.py) are unchanged.
We load them via importlib for the data-standardisation utilities and split,
and we load the new player-aware model classes from model_v2.py.

Weights for the player-aware models are saved alongside the originals:
    player_stats/trained_model_v2.pt
    random_masking/trained_model_v2.pt
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


# --- module loading ---------------------------------------------------------
def _load_module(unique_name: str, file_path: Path):
    """Load a Python file as a fresh module under a unique name."""
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


# v1 modules — we still use their data paths and shared constants
ps_masking = _load_module("ps_masking", ROOT / "player_stats" / "masking.py")
rm_masking = _load_module("rm_masking", ROOT / "random_masking" / "masking.py")

# v2 models — the new player-aware transformers
ps_model_v2 = _load_module("ps_model_v2", ROOT / "player_stats" / "model_v2.py")
rm_model_v2 = _load_module("rm_model_v2", ROOT / "random_masking" / "model_v2.py")


# Defaults
DEFAULT_PLAYER = "Shai Gilgeous-Alexander"
DEFAULT_DATES = ["2026-05-18", "2026-05-11", "2026-05-09", "2026-03-23"]
DEFAULT_BONUS_COLS = ["MIN", "AST", "REB", "FG_PCT", "PTS"]

PS_MODEL_PATH = ROOT / "player_stats" / "trained_model_v2.pt"
RM_MODEL_PATH = ROOT / "random_masking" / "trained_model_v2.pt"


# --- data introspection -----------------------------------------------------
def _read_df() -> pd.DataFrame:
    df = pd.read_csv(ps_masking.DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def _player_index() -> tuple[dict, int]:
    """name → int id, plus the count. Sorted so it's deterministic across runs."""
    df = _read_df()
    names = sorted(df["PLAYER_NAME"].dropna().unique().tolist())
    return {name: i for i, name in enumerate(names)}, len(names)


def list_players() -> list[str]:
    return sorted(_read_df()["PLAYER_NAME"].dropna().unique().tolist())


def list_dates_for(player: str) -> list[str]:
    df = _read_df()
    dates = df.loc[df["PLAYER_NAME"] == player, "GAME_DATE"].sort_values(ascending=False)
    return [d.strftime("%Y-%m-%d") for d in dates]


# --- data loading w/ player IDs --------------------------------------------
def _load_ps_with_players(seed: int = 0, test_frac: float = 0.2):
    """Mirror ps_masking.load_and_split but also return per-row player_ids."""
    df = _read_df()
    name_to_id, n_players = _player_index()

    X_np = df[ps_masking.FEATURE_COLS].to_numpy(dtype=np.float32)
    y_np = df[ps_masking.TARGET_COL].to_numpy(dtype=np.float32)
    pid_np = df["PLAYER_NAME"].map(name_to_id).to_numpy(dtype=np.int64)

    X_mean = X_np.mean(axis=0)
    X_std = X_np.std(axis=0) + 1e-8
    X_np = (X_np - X_mean) / X_std

    y_mean = float(y_np.mean())
    y_std = float(y_np.std()) + 1e-8
    y_np = (y_np - y_mean) / y_std

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_np))
    n_test = int(len(X_np) * test_frac)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return {
        "X_train": torch.from_numpy(X_np[train_idx]),
        "y_train": torch.from_numpy(y_np[train_idx]).unsqueeze(1),
        "pid_train": torch.from_numpy(pid_np[train_idx]),
        "X_test": torch.from_numpy(X_np[test_idx]),
        "y_test": torch.from_numpy(y_np[test_idx]).unsqueeze(1),
        "pid_test": torch.from_numpy(pid_np[test_idx]),
        "y_mean": y_mean, "y_std": y_std,
        "X_mean": X_mean, "X_std": X_std,
        "df": df, "test_idx": test_idx,
        "name_to_id": name_to_id, "n_players": n_players,
    }


def _load_rm_with_players(seed: int = 0, test_frac: float = 0.2):
    """Mirror rm_masking.load_and_split but also return per-row player_ids."""
    df = _read_df()
    name_to_id, n_players = _player_index()

    data_np = df[rm_masking.ALL_COLS].to_numpy(dtype=np.float32)
    pid_np = df["PLAYER_NAME"].map(name_to_id).to_numpy(dtype=np.int64)

    col_means = data_np.mean(axis=0)
    col_stds = data_np.std(axis=0) + 1e-8
    data_z = (data_np - col_means) / col_stds

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data_z))
    n_test = int(len(data_z) * test_frac)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return {
        "X_train": torch.from_numpy(data_z[train_idx]),
        "pid_train": torch.from_numpy(pid_np[train_idx]),
        "X_test": torch.from_numpy(data_z[test_idx]),
        "pid_test": torch.from_numpy(pid_np[test_idx]),
        "col_means": col_means, "col_stds": col_stds,
        "df": df, "test_idx": test_idx,
        "name_to_id": name_to_id, "n_players": n_players,
    }


# --- player_stats predictions (v2) -----------------------------------------
def get_player_stats_results(
    player: str = DEFAULT_PLAYER,
    dates: Iterable[str] = DEFAULT_DATES,
) -> dict:
    if not PS_MODEL_PATH.exists():
        return {"overall_mae": float("nan"), "games": [], "player": player,
                "needs_training": True}

    data = _load_ps_with_players()
    model = ps_model_v2.build_model(
        n_features=len(ps_masking.FEATURE_COLS),
        n_players=data["n_players"],
    )
    model.load_state_dict(torch.load(PS_MODEL_PATH))
    model.eval()

    # Overall test MAE
    with torch.no_grad():
        preds_z = model(data["X_test"], data["pid_test"]).squeeze(1).numpy()
    preds_real = preds_z * data["y_std"] + data["y_mean"]
    truth_real = data["y_test"].squeeze(1).numpy() * data["y_std"] + data["y_mean"]
    overall_mae = float(np.abs(preds_real - truth_real).mean())

    df = data["df"]
    name_to_id = data["name_to_id"]
    pid = name_to_id.get(player)
    if pid is None:
        return {"overall_mae": overall_mae, "games": [], "player": player}

    test_idx_set = set(data["test_idx"].tolist())
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
        z = (raw - data["X_mean"]) / data["X_std"]
        x = torch.from_numpy(z).unsqueeze(0)
        pid_t = torch.tensor([pid], dtype=torch.long)

        with torch.no_grad():
            pred_z = model(x, pid_t).item()
        pred_pts = float(pred_z * data["y_std"] + data["y_mean"])
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


# --- random_masking predictions (v2) ---------------------------------------
def _predict_masked_v2(model, raw_values, mask_col_name, col_means, col_stds, pid):
    mask_idx = rm_masking.MASKABLE_COLS.index(mask_col_name)
    actual_raw = float(raw_values[mask_idx])
    row_z = (raw_values - col_means) / col_stds
    row_t = torch.from_numpy(row_z.astype(np.float32)).unsqueeze(0)
    mask_t = torch.tensor([mask_idx], dtype=torch.long)
    pid_t = torch.tensor([pid], dtype=torch.long)
    with torch.no_grad():
        pred_z = model(row_t, mask_t, pid_t).item()
    predicted_raw = float(pred_z * col_stds[mask_idx] + col_means[mask_idx])
    return predicted_raw, actual_raw


def get_random_masking_results(
    player: str = DEFAULT_PLAYER,
    dates: Iterable[str] = DEFAULT_DATES,
    bonus_date: str | None = None,
    bonus_cols: Iterable[str] = DEFAULT_BONUS_COLS,
) -> dict:
    dates = list(dates)
    if bonus_date is None and dates:
        bonus_date = dates[-1]

    if not RM_MODEL_PATH.exists():
        return {"overall_mae": float("nan"), "pts_games": [], "bonus_games": [],
                "bonus_date": bonus_date, "player": player, "needs_training": True}

    data = _load_rm_with_players()
    model = rm_model_v2.build_model(
        n_maskable=rm_masking.N_MASKABLE,
        n_context=rm_masking.N_CONTEXT,
        n_players=data["n_players"],
    )
    model.load_state_dict(torch.load(RM_MODEL_PATH))
    model.eval()

    # Overall test MAE on PTS-masked predictions
    pts_idx = rm_masking.PTS_IDX
    pts_mean = float(data["col_means"][pts_idx])
    pts_std = float(data["col_stds"][pts_idx])
    test_pts_real = data["X_test"][:, pts_idx].numpy() * pts_std + pts_mean

    n_test = data["X_test"].shape[0]
    pts_mask_cols = torch.full((n_test,), pts_idx, dtype=torch.long)
    with torch.no_grad():
        preds_z = model(data["X_test"], pts_mask_cols, data["pid_test"]).squeeze(1).numpy()
    preds_real = preds_z * pts_std + pts_mean
    overall_mae = float(np.abs(preds_real - test_pts_real).mean())

    df = data["df"]
    name_to_id = data["name_to_id"]
    pid = name_to_id.get(player)
    if pid is None:
        return {"overall_mae": overall_mae, "pts_games": [], "bonus_games": [],
                "bonus_date": bonus_date, "player": player}

    test_idx_set = set(data["test_idx"].tolist())

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
        pred, actual = _predict_masked_v2(
            model, raw_values, "PTS", data["col_means"], data["col_stds"], pid,
        )
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
                pred, actual = _predict_masked_v2(
                    model, raw_values, col, data["col_means"], data["col_stds"], pid,
                )
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


# --- retraining (v2) --------------------------------------------------------
ProgressCB = Callable[[dict], None]


def retrain_player_stats(
    epochs: int = 800,
    log_every: int = 50,
    progress_cb: ProgressCB | None = None,
) -> dict:
    """Train the player-aware player_stats transformer from scratch."""
    data = _load_ps_with_players()

    model = ps_model_v2.build_model(
        n_features=len(ps_masking.FEATURE_COLS),
        n_players=data["n_players"],
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    truth_real = data["y_test"].squeeze(1).numpy() * data["y_std"] + data["y_mean"]
    baseline_mae = float(np.abs(truth_real - data["y_mean"]).mean())

    best_mae = float("inf")
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(epochs):
        preds = model(data["X_train"], data["pid_train"])
        loss = loss_fn(preds, data["y_train"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds_z = model(data["X_test"], data["pid_test"]).squeeze(1).numpy()
                preds_real = preds_z * data["y_std"] + data["y_mean"]
                mae = float(np.abs(preds_real - truth_real).mean())
            model.train()

            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), PS_MODEL_PATH)

            history.append({"epoch": epoch, "train_loss": float(loss.item()),
                            "test_mae": mae, "is_best": is_best})
            if progress_cb is not None:
                progress_cb({
                    "stage": "player_stats",
                    "epoch": epoch, "epochs": epochs,
                    "train_loss": float(loss.item()), "test_mae": mae,
                    "best_mae": best_mae, "is_best": is_best,
                })

    return {"best_mae": best_mae, "best_epoch": best_epoch,
            "baseline_mae": baseline_mae, "history": history}


def retrain_random_masking(
    epochs: int = 800,
    log_every: int = 50,
    progress_cb: ProgressCB | None = None,
) -> dict:
    """Train the player-aware random_masking transformer from scratch."""
    data = _load_rm_with_players()

    pts_idx = rm_masking.PTS_IDX
    pts_mean = float(data["col_means"][pts_idx])
    pts_std = float(data["col_stds"][pts_idx])
    test_pts_real = data["X_test"][:, pts_idx].numpy() * pts_std + pts_mean

    model = rm_model_v2.build_model(
        n_maskable=rm_masking.N_MASKABLE,
        n_context=rm_masking.N_CONTEXT,
        n_players=data["n_players"],
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    baseline_mae = float(np.abs(pts_mean - test_pts_real).mean())

    best_mae = float("inf")
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(epochs):
        batch_size = data["X_train"].shape[0]
        mask_cols = torch.randint(0, rm_masking.N_MASKABLE, (batch_size,))
        values, mask_positions, target = rm_masking.apply_mask(data["X_train"], mask_cols)

        preds = model(values, mask_positions, data["pid_train"])
        loss = loss_fn(preds, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                n_test = data["X_test"].shape[0]
                pts_mask_cols = torch.full((n_test,), pts_idx, dtype=torch.long)
                preds_z = model(data["X_test"], pts_mask_cols, data["pid_test"]).squeeze(1).numpy()
                preds_real = preds_z * pts_std + pts_mean
                mae = float(np.abs(preds_real - test_pts_real).mean())
            model.train()

            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), RM_MODEL_PATH)

            history.append({"epoch": epoch, "train_loss": float(loss.item()),
                            "test_mae": mae, "is_best": is_best})
            if progress_cb is not None:
                progress_cb({
                    "stage": "random_masking",
                    "epoch": epoch, "epochs": epochs,
                    "train_loss": float(loss.item()), "test_mae": mae,
                    "best_mae": best_mae, "is_best": is_best,
                })

    return {"best_mae": best_mae, "best_epoch": best_epoch,
            "baseline_mae": baseline_mae, "history": history}


if __name__ == "__main__":
    print(f"PS model path: {PS_MODEL_PATH}  exists={PS_MODEL_PATH.exists()}")
    print(f"RM model path: {RM_MODEL_PATH}  exists={RM_MODEL_PATH.exists()}")
    if not PS_MODEL_PATH.exists() or not RM_MODEL_PATH.exists():
        print("\nRun the dashboard and click \"Re-train both\", or call")
        print("  retrain_player_stats() and retrain_random_masking() from Python.")
    else:
        ps = get_player_stats_results()
        rm = get_random_masking_results()
        print(f"\nplayer_stats overall test MAE: {ps['overall_mae']:.2f} PTS")
        for g in ps["games"]:
            print(f"  {g['date']}  pred={g['predicted']:.2f}  actual={g['actual']:.0f}"
                  f"  err={g['error']:+.2f}  [{g['split']}]")
        print(f"\nrandom_masking overall test MAE (PTS): {rm['overall_mae']:.2f}")
        for g in rm["pts_games"]:
            print(f"  {g['date']}  pred={g['predicted']:.2f}  actual={g['actual']:.0f}"
                  f"  err={g['error']:+.2f}  [{g['split']}]")
