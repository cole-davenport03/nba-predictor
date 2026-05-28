 # the masking utility (one function)
"""Load and split player game data for the masked-reconstruction model.

v1: PTS is always the target column (always "masked"). Other columns are inputs.
v2 (later): randomize which column gets masked. Same dataset, different training loop.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch


DATA_PATH = Path(__file__).parent / "data" / "players_games.csv"

# The 10 input features. Order matters; index here = column index in X tensors.
FEATURE_COLS = [
    "MIN", "FG_PCT", "FG3_PCT", "FT_PCT",
    "REB", "AST", "STL", "BLK", "TOV", "PF",
    "IS_OT",                
]
TARGET_COL = "PTS"


def load_and_split(test_frac: float = 0.2, seed: int = 0):
    """Return (X_train, y_train, X_test, y_test, y_mean, y_std, df, test_idx).

    All tensors are standardized (mean 0, std 1). y_mean and y_std let you
    un-standardize predictions back to real PTS at eval time.
    """
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    X_np = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_np = df[TARGET_COL].to_numpy(dtype=np.float32)

    # Standardize features (per-column mean/std).
    X_mean = X_np.mean(axis=0)
    X_std = X_np.std(axis=0) + 1e-8
    X_np = (X_np - X_mean) / X_std

    # Standardize target (scalar mean/std for PTS).
    y_mean = float(y_np.mean())
    y_std = float(y_np.std()) + 1e-8
    y_np = (y_np - y_mean) / y_std

    # Shuffle and split.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_np))
    n_test = int(len(X_np) * test_frac)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return (
        torch.from_numpy(X_np[train_idx]),
        torch.from_numpy(y_np[train_idx]).unsqueeze(1),  # (n, 1) for MLP output shape
        torch.from_numpy(X_np[test_idx]),
        torch.from_numpy(y_np[test_idx]).unsqueeze(1),
        y_mean, y_std,
        df, test_idx,
    )


def main() -> None:
    X_train, y_train, X_test, y_test, y_mean, y_std, df, test_idx = load_and_split()
    print(f"X_train: {tuple(X_train.shape)}")
    print(f"y_train: {tuple(y_train.shape)}")
    print(f"X_test:  {tuple(X_test.shape)}")
    print(f"y_test:  {tuple(y_test.shape)}")
    print(f"\nfeature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"PTS mean = {y_mean:.2f}, std = {y_std:.2f}")
    print(f"\nfirst X_train row (standardized features):\n{X_train[0]}")
    print(f"first y_train value (standardized PTS): {y_train[0].item():.3f}")


if __name__ == "__main__":
    main()