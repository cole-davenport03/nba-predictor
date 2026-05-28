"""Data loading + masking for transformer-based BERT-style masked reconstruction.

IS_OT is a CONTEXT column — always present, never masked.
The other 11 columns are MASKABLE.

This file's `apply_mask` is transformer-friendly: it returns the raw values
plus the position of the mask. The transformer handles the [MASK] token
substitution internally at the embedding stage (much cleaner than the
"zero the value + concat indicator" pattern we used with the MLP).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch


DATA_PATH = Path(__file__).parent.parent / "player_stats" / "data" / "players_games.csv"

MASKABLE_COLS = [
    "MIN", "FG_PCT", "FG3_PCT", "FT_PCT",
    "REB", "AST", "STL", "BLK", "TOV", "PF",
    "PTS",
]
CONTEXT_COLS = ["IS_OT"]
ALL_COLS = MASKABLE_COLS + CONTEXT_COLS

PTS_IDX = MASKABLE_COLS.index("PTS")
N_MASKABLE = len(MASKABLE_COLS)
N_CONTEXT = len(CONTEXT_COLS)
N_TOTAL = N_MASKABLE + N_CONTEXT


def load_and_split(test_frac: float = 0.2, seed: int = 0):
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    data_np = df[ALL_COLS].to_numpy(dtype=np.float32)

    col_means = data_np.mean(axis=0)
    col_stds = data_np.std(axis=0) + 1e-8
    data_z = (data_np - col_means) / col_stds

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data_z))
    n_test = int(len(data_z) * test_frac)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return (
        torch.from_numpy(data_z[train_idx]),
        torch.from_numpy(data_z[test_idx]),
        col_means,
        col_stds,
        df,
        test_idx,
    )


def apply_mask(X_full: torch.Tensor, mask_cols: torch.Tensor):
    """Transformer-friendly mask interface.

    Returns:
       values:         X_full unchanged — shape (batch, N_TOTAL)
       mask_positions: same as mask_cols — shape (batch,)
       target:         original value at masked position — shape (batch, 1)

    The model handles the [MASK] token substitution internally — we don't
    need to zero the input here.
    """
    batch_size = X_full.shape[0]
    row_idx = torch.arange(batch_size)
    target = X_full[row_idx, mask_cols].unsqueeze(1)
    return X_full, mask_cols, target


def main() -> None:
    X_train, X_test, col_means, col_stds, df, test_idx = load_and_split()
    print(f"X_train: {tuple(X_train.shape)}   X_test: {tuple(X_test.shape)}")
    print(f"\nmaskable cols ({N_MASKABLE}): {MASKABLE_COLS}")
    print(f"context cols ({N_CONTEXT}):  {CONTEXT_COLS}")
    print(f"PTS_IDX = {PTS_IDX}, N_TOTAL = {N_TOTAL}\n")
    for col, m, s in zip(ALL_COLS, col_means, col_stds):
        kind = "MASK" if col in MASKABLE_COLS else "CTX "
        print(f"  [{kind}] {col:<10} mean={m:.3f}  std={s:.3f}")


if __name__ == "__main__":
    main()