"""Cleanliness + summary checks on the players_games.csv before modeling."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data" / "players_games.csv"


def main() -> None:
    df = pd.read_csv(DATA)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    numeric_cols = ["MIN", "FG_PCT", "FG3_PCT", "FT_PCT",
                    "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]

    print("=== shape & dtypes ===")
    print(f"rows: {len(df)}, cols: {len(df.columns)}")
    print(df.dtypes)

    print("\n=== missing values per column ===")
    print(df.isna().sum())

    print("\n=== numeric summary ===")
    print(df[numeric_cols].describe().round(2))

    print("\n=== correlations with PTS (sorted) ===")
    corr = df[numeric_cols].corr()["PTS"].drop("PTS").sort_values(ascending=False)
    print(corr.round(3))

    print("\n=== SGA spot-check ===")
    sga = df[df["PLAYER_NAME"] == "Shai Gilgeous-Alexander"]
    print(f"games: {len(sga)}, avg PTS: {sga['PTS'].mean():.2f}")
    sga_518 = sga[sga["GAME_DATE"] == pd.Timestamp("2026-05-18")]
    print("\n5/18 game:")
    print(sga_518[numeric_cols + ["MATCHUP", "WL"]].to_string(index=False))


if __name__ == "__main__":
    main()