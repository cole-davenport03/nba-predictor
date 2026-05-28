"""Fetch & clean per-game player stats for masked-reconstruction training.

Pulls regular season + playoff game logs from stats.nba.com (via nba_api)
for a hand-picked list of players, drops the columns that mechanically
compute PTS (so the model has to actually learn), and saves a single CSV.

Run:    python player_stats/data_pipeline.py
Output: player_stats/data/players_games.csv
"""

import time
from pathlib import Path

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEASON = "2025-26"

PLAYER_NAMES: list[str] = [
    # ── your existing 20 ──
    "Shai Gilgeous-Alexander", "Nikola Jokic", "Luka Doncic", "Jaylen Brown",
    "Giannis Antetokounmpo", "Kevin Durant", "Stephen Curry", "LeBron James",
    "Devin Booker", "Jalen Brunson", "Donovan Mitchell", "De'Aaron Fox",
    "Anthony Edwards", "Cade Cunningham", "Paolo Banchero", "Karl-Anthony Towns",
    "Isaiah Hartenstein", "Luguentz Dort", "Chet Holmgren", "Jalen Williams",

    # ── 30 more stars / high-usage starters ──
    "Jayson Tatum", "Joel Embiid", "Damian Lillard", "Kyrie Irving",
    "Anthony Davis", "Pascal Siakam", "Domantas Sabonis", "Jaren Jackson Jr.",
    "Bam Adebayo", "Trae Young", "Tyrese Haliburton", "Zion Williamson",
    "DeMar DeRozan", "Julius Randle", "Bradley Beal", "Jimmy Butler",
    "Mikal Bridges", "Franz Wagner", "Scottie Barnes", "Evan Mobley",
    "Alperen Sengun", "Victor Wembanyama", "Brandon Ingram", "Lauri Markkanen",
    "OG Anunoby", "RJ Barrett", "Tyler Herro", "CJ McCollum",
    "Fred VanVleet", "Khris Middleton",
]

# Columns we keep. Dropped ones (FGM, FGA, FG3M, FG3A, FTM, FTA) would let
# the model trivially recover PTS via PTS = 2*FGM + FG3M + FTM.
KEEP_COLS = [
    "GAME_DATE", "MATCHUP", "WL", "MIN",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "REB", "AST", "STL", "BLK", "TOV", "PF",
    "PTS",
]

OUT_PATH = Path(__file__).parent / "data" / "players_games.csv"
SLEEP_BETWEEN_CALLS = 0.6


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_player(name: str) -> pd.DataFrame:
    """Pull regular season + playoff game log for one player."""
    matches = players.find_players_by_full_name(name)
    if not matches:
        print(f"  WARN: no player found for '{name}', skipping")
        return pd.DataFrame()
    pid = matches[0]["id"]

    frames = []
    for season_type in ("Regular Season", "Playoffs"):
        log = playergamelog.PlayerGameLog(
            player_id=pid,
            season=SEASON,
            season_type_all_star=season_type,
        )
        df = log.get_data_frames()[0]
        if not df.empty:
            df["PLAYER_NAME"] = name
            df["SEASON_TYPE"] = season_type
            frames.append(df)
        time.sleep(SLEEP_BETWEEN_CALLS)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    all_frames: list[pd.DataFrame] = []
    for i, name in enumerate(PLAYER_NAMES, start=1):
        print(f"[{i:2d}/{len(PLAYER_NAMES)}] fetching {name}...")
        df = fetch_player(name)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("no data fetched, aborting")
        return

    full = pd.concat(all_frames, ignore_index=True)
    full = full[KEEP_COLS + ["PLAYER_NAME", "SEASON_TYPE"]]

    full["IS_OT"] = (full["MIN"] > 48).astype(float)

    # Convert GAME_DATE to real datetime so min/max/sorting work,
    # then sort each player's games oldest-to-newest for sanity.
    full["GAME_DATE"] = pd.to_datetime(full["GAME_DATE"])
    full = full.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_PATH, index=False)

    print(f"\nsaved {len(full)} rows → {OUT_PATH}")
    print(f"  unique players: {full['PLAYER_NAME'].nunique()}")
    print(f"  date range:     {full['GAME_DATE'].min().date()}  →  {full['GAME_DATE'].max().date()}")
    print(f"  cols:           {list(full.columns)}")

    # Sanity check: SGA's 5/18 game should be in there.
    sga_518 = full[
        (full["PLAYER_NAME"] == "Shai Gilgeous-Alexander")
        & (full["GAME_DATE"] == pd.Timestamp("2026-05-18"))
    ]
    if sga_518.empty:
        print("  WARN: SGA 5/18 game not found")
    else:
        print(f"  SGA 5/18 present, PTS = {int(sga_518['PTS'].iloc[0])}  (expect 24)")

    # Per-player row count (helps you spot anyone who returned no data).
    print("\nrows per player:")
    print(full["PLAYER_NAME"].value_counts().to_string())


if __name__ == "__main__":
    main()