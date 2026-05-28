"""
Embedded sample data for demo mode when the NBA stats API is unavailable.

Every team receives distinct, reproducible stats generated from a seeded RNG
based on the team name — so the same team always gets the same numbers, but
different teams produce genuinely different predictions.

Two teams (MIN, SAS) have hand-tuned 2025-26 playoff data for the featured matchup.
All others are generated within realistic NBA ranges.
"""

import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Hand-tuned data for the featured playoff matchup
# ---------------------------------------------------------------------------
_HAND_TUNED = {
    "Minnesota Timberwolves": {
        "season": dict(
            PTS=116.4, FGM=43.2, FGA=91.8, FG_PCT=0.470,
            FG3M=14.1, FG3A=38.6, FG3_PCT=0.365,
            FTM=15.9, FTA=20.8, FT_PCT=0.765,
            OREB=9.8, DREB=35.2, REB=45.0,
            AST=27.1, TOV=13.8, STL=8.4, BLK=5.9, PF=20.1, PLUS_MINUS=4.8,
            E_OFF_RATING=117.2, OFF_RATING=117.2,
            E_DEF_RATING=112.4, DEF_RATING=112.4,
            E_NET_RATING=4.8,   NET_RATING=4.8,
            PACE=99.3, E_PACE=99.3, W=56, L=26,
        ),
        "pts":  [121, 108, 119, 115, 124, 103, 118, 112, 126, 109],
        "wl":   ["W","L","W","W","W","L","W","W","W","L"],
        "opp":  ["SAS","SAS","SAS","DEN","OKC","OKC","DEN","LAL","LAL","GSW"],
        "clutch": dict(CLUTCH_W=22, CLUTCH_L=11, CLUTCH_PTS=8.4,
                       CLUTCH_PLUS_MINUS=2.1, CLUTCH_FG_PCT=0.461),
    },
    "San Antonio Spurs": {
        "season": dict(
            PTS=112.8, FGM=41.6, FGA=90.4, FG_PCT=0.460,
            FG3M=13.2, FG3A=37.1, FG3_PCT=0.356,
            FTM=16.4, FTA=21.9, FT_PCT=0.749,
            OREB=10.2, DREB=33.8, REB=44.0,
            AST=25.6, TOV=14.9, STL=7.6, BLK=6.8, PF=21.4, PLUS_MINUS=0.6,
            E_OFF_RATING=113.9, OFF_RATING=113.9,
            E_DEF_RATING=113.3, DEF_RATING=113.3,
            E_NET_RATING=0.6,   NET_RATING=0.6,
            PACE=100.8, E_PACE=100.8, W=47, L=35,
        ),
        "pts":  [108, 121, 101, 118, 105, 119, 110,  98, 115, 107],
        "wl":   ["L","W","L","W","L","W","W","L","W","L"],
        "opp":  ["MIN","MIN","MIN","MEM","MEM","HOU","HOU","MIN","DAL","DAL"],
        "clutch": dict(CLUTCH_W=16, CLUTCH_L=17, CLUTCH_PTS=7.9,
                       CLUTCH_PLUS_MINUS=-0.3, CLUTCH_FG_PCT=0.444),
    },
}


# ---------------------------------------------------------------------------
# Seeded generator for every other team
# ---------------------------------------------------------------------------
def _seed(team_name: str) -> int:
    return int(hashlib.md5(team_name.encode()).hexdigest()[:8], 16)


def _generate_season(team_name: str, team_id: int) -> dict:
    """Produce internally-consistent season averages for any team."""
    rng = np.random.RandomState(_seed(team_name))

    net_rtg  = float(rng.uniform(-9, 11))          # spread of real NBA teams
    off_rtg  = float(114.5 + net_rtg * 0.55 + rng.uniform(-1.5, 1.5))
    def_rtg  = off_rtg - net_rtg
    pace     = float(rng.uniform(97, 103))
    pts      = off_rtg * 0.97 + rng.uniform(-0.5, 0.5)
    fg_pct   = float(np.clip(0.465 + net_rtg * 0.0018 + rng.uniform(-0.014, 0.014), 0.430, 0.500))
    fg3_pct  = float(np.clip(0.360 + rng.uniform(-0.022, 0.022), 0.330, 0.395))
    ft_pct   = float(np.clip(0.770 + rng.uniform(-0.040, 0.040), 0.700, 0.840))
    ast      = float(rng.uniform(23, 30))
    reb      = float(rng.uniform(42, 47))
    tov      = float(rng.uniform(12, 16))
    stl      = float(rng.uniform(6.5, 9.5))
    blk      = float(rng.uniform(4.5, 7.5))
    wins     = int(np.clip(41 + net_rtg * 2.6, 15, 68))

    # Derive field goals from points + rates
    fga = round(pts / (2 * fg_pct + 0.001), 1)
    fgm = round(fga * fg_pct, 1)
    fg3a = round(fga * float(rng.uniform(0.38, 0.46)), 1)
    fg3m = round(fg3a * fg3_pct, 1)
    fta  = round(abs(pts - 2 * fgm - 3 * fg3m) / ft_pct + 1, 1)
    ftm  = round(fta * ft_pct, 1)
    oreb = round(reb * 0.22, 1)
    dreb = round(reb * 0.78, 1)

    return dict(
        TEAM_ID=team_id, TEAM_NAME=team_name,
        GP=82, W=wins, L=82 - wins,
        PTS=round(pts, 1),
        FGM=fgm, FGA=fga, FG_PCT=round(fg_pct, 3),
        FG3M=fg3m, FG3A=fg3a, FG3_PCT=round(fg3_pct, 3),
        FTM=ftm, FTA=fta, FT_PCT=round(ft_pct, 3),
        OREB=oreb, DREB=dreb, REB=round(reb, 1),
        AST=round(ast, 1), TOV=round(tov, 1),
        STL=round(stl, 1), BLK=round(blk, 1), PF=round(float(rng.uniform(18, 23)), 1),
        PLUS_MINUS=round(net_rtg, 1),
        E_OFF_RATING=round(off_rtg, 1), OFF_RATING=round(off_rtg, 1),
        E_DEF_RATING=round(def_rtg, 1), DEF_RATING=round(def_rtg, 1),
        E_NET_RATING=round(net_rtg, 1), NET_RATING=round(net_rtg, 1),
        PACE=round(pace, 1), E_PACE=round(pace, 1),
    )


def _generate_game_log(team_name: str, team_id: int, season_avgs: dict) -> pd.DataFrame:
    """Generate a 10-game log consistent with the team's season averages."""
    rng     = np.random.RandomState(_seed(team_name) + 1)
    base_pts = season_avgs["PTS"]
    base_fg  = season_avgs["FG_PCT"]
    base_fg3 = season_avgs["FG3_PCT"]
    base_reb = season_avgs["REB"]
    base_ast = season_avgs["AST"]
    base_tov = season_avgs["TOV"]
    win_rate = float(np.clip(0.3 + season_avgs["NET_RATING"] * 0.03, 0.15, 0.85))

    today  = datetime.today()
    rows   = []
    for i in range(10):
        pts   = round(float(base_pts + rng.uniform(-12, 12)), 0)
        fg    = round(float(np.clip(base_fg  + rng.uniform(-0.04, 0.04), 0.38, 0.55)), 3)
        fg3   = round(float(np.clip(base_fg3 + rng.uniform(-0.05, 0.05), 0.28, 0.46)), 3)
        reb   = round(float(base_reb + rng.uniform(-4, 4)), 0)
        ast   = round(float(base_ast + rng.uniform(-4, 4)), 0)
        tov   = round(float(max(base_tov + rng.uniform(-3, 3), 7)), 0)
        pm    = round(float(rng.uniform(-20, 20)), 0)
        wl    = "W" if rng.random() < win_rate else "L"
        fga   = round(pts / (2 * fg + 0.001), 0)
        fgm   = round(fga * fg, 0)
        fg3a  = round(fga * 0.42, 0)
        fg3m  = round(fg3a * fg3, 0)
        fta   = round(max(pts - 2 * fgm - 3 * fg3m, 0) / 0.77, 0)
        ftm   = round(fta * 0.77, 0)
        game_date = today - timedelta(days=2 * i + 1)
        rows.append(dict(
            GAME_DATE=game_date, MATCHUP=f"vs. OPP{i}", WL=wl,
            PTS=pts, FGM=fgm, FGA=fga, FG_PCT=fg,
            FG3M=fg3m, FG3A=fg3a, FG3_PCT=fg3,
            FTM=ftm, FTA=fta, FT_PCT=0.77,
            OREB=round(reb * 0.22, 0), DREB=round(reb * 0.78, 0), REB=reb,
            AST=ast, TOV=tov, STL=round(float(rng.uniform(6, 10)), 0),
            BLK=round(float(rng.uniform(4, 8)), 0), PF=round(float(rng.uniform(17, 24)), 0),
            PLUS_MINUS=pm,
        ))
    return pd.DataFrame(rows)


def _generate_clutch(team_name: str, season_avgs: dict) -> dict:
    rng    = np.random.RandomState(_seed(team_name) + 2)
    net    = season_avgs["NET_RATING"]
    c_pm   = round(float(net * 0.4 + rng.uniform(-2, 2)), 1)
    c_fg   = round(float(np.clip(0.45 + net * 0.003 + rng.uniform(-0.02, 0.02), 0.39, 0.52)), 3)
    c_w    = int(np.clip(18 + net * 1.5 + rng.uniform(-3, 3), 5, 38))
    return dict(CLUTCH_W=c_w, CLUTCH_L=38 - c_w,
                CLUTCH_PTS=round(float(rng.uniform(7, 10)), 1),
                CLUTCH_PLUS_MINUS=c_pm, CLUTCH_FG_PCT=c_fg)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_sample_data(team_name: str) -> dict:
    """
    Return a complete data dict matching the structure of fetch_all_team_data().
    Hand-tuned for MIN and SAS; seeded-generated for every other team.
    """
    # Resolve team info from the fetcher's local team index (no nba_api dependency)
    from data.fetcher import _TEAM_INDEX

    info = None
    name_lower = team_name.lower()
    for t in _TEAM_INDEX:
        if (name_lower in t["full_name"].lower()
                or name_lower in t["nickname"].lower()
                or name_lower in t["abbreviation"].lower()):
            info = t
            break
    if info is None:
        raise ValueError(f"Team not found: {team_name}")

    full_name = info["full_name"]
    team_id   = info["id"]

    # Check for hand-tuned entry (exact full name match)
    if full_name in _HAND_TUNED:
        entry       = _HAND_TUNED[full_name]
        season_avgs = {**entry["season"], "TEAM_ID": team_id, "TEAM_NAME": full_name}
        pts_list    = entry["pts"]
        wl_list     = entry["wl"]
        opp_list    = entry["opp"]
        clutch      = entry["clutch"]

        today = datetime.today()
        rows  = []
        for i, (pts, wl, opp) in enumerate(zip(pts_list, wl_list, opp_list)):
            fg  = season_avgs["FG_PCT"] + np.random.RandomState(i).uniform(-0.03, 0.03)
            fg3 = season_avgs["FG3_PCT"] + np.random.RandomState(i + 100).uniform(-0.04, 0.04)
            fga = round(pts / (2 * fg + 0.001), 0)
            fgm = round(fga * fg, 0)
            fg3a = round(fga * 0.42, 0)
            fg3m = round(fg3a * fg3, 0)
            fta  = round(max(pts - 2 * fgm - 3 * fg3m, 0) / 0.77, 0)
            ftm  = round(fta * 0.77, 0)
            reb  = round(season_avgs["REB"] + np.random.RandomState(i + 200).uniform(-4, 4), 0)
            rows.append(dict(
                GAME_DATE=today - timedelta(days=2 * i + 1),
                MATCHUP=f"{info['abbreviation']} vs. {opp}", WL=wl,
                PTS=pts, FGM=fgm, FGA=fga, FG_PCT=round(fg, 3),
                FG3M=fg3m, FG3A=fg3a, FG3_PCT=round(fg3, 3),
                FTM=ftm, FTA=fta, FT_PCT=0.77,
                OREB=round(reb * 0.22, 0), DREB=round(reb * 0.78, 0), REB=reb,
                AST=round(season_avgs["AST"] + np.random.RandomState(i+300).uniform(-3, 3), 0),
                TOV=round(season_avgs["TOV"] + np.random.RandomState(i+400).uniform(-2, 2), 0),
                STL=round(season_avgs["STL"], 0), BLK=round(season_avgs["BLK"], 0),
                PF=round(season_avgs["PF"], 0),
                PLUS_MINUS=round((1 if wl == "W" else -1) * np.random.RandomState(i+500).uniform(2, 18), 0),
            ))
        game_log = pd.DataFrame(rows)
    else:
        # Seeded generation for all other teams
        season_avgs = _generate_season(full_name, team_id)
        game_log    = _generate_game_log(full_name, team_id, season_avgs)
        clutch      = _generate_clutch(full_name, season_avgs)

    last10_log = game_log.head(10)
    numeric_cols = [
        "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "TOV", "STL", "BLK", "PF", "PLUS_MINUS",
    ]
    last10_avgs = {col: round(float(last10_log[col].mean()), 2)
                   for col in numeric_cols if col in last10_log.columns}
    last10_avgs["WIN_PCT_L10"] = round(float((last10_log["WL"] == "W").mean()), 3)
    last10_avgs["GAMES_IN_LAST10"] = len(last10_log)

    rest_days    = 1
    games_in_10d = int((last10_log["GAME_DATE"] >= datetime.today() - timedelta(days=10)).sum())

    return {
        "info":        info,
        "team_id":     team_id,
        "game_log":    game_log,
        "last10_log":  last10_log,
        "season_avgs": season_avgs,
        "last10_avgs": last10_avgs,
        "rest_days":   rest_days,
        "games_in_10d": games_in_10d,
        "clutch":      clutch,
        "_is_sample":  True,
    }
