"""
NBA data fetching using ESPN's public sports API.

ESPN's site.api.espn.com endpoints are undocumented but stable and don't
require auth. We fetch team season stats, recent box scores, and derive
advanced metrics (Off/Def/Net Rating, Pace) directly from box scores.

Falls back to embedded sample data if the API is unreachable.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from data.sample_data import get_sample_data

# ---------------------------------------------------------------------------
# ESPN endpoints
# ---------------------------------------------------------------------------
_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

# Browser-like UA; ESPN is permissive but we identify ourselves anyway.
NBA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

_REQUEST_TIMEOUT = 12
_RATE_DELAY = 0.15      # ESPN is fine with this
_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Team metadata (full_name -> dict with NBA stats id, ESPN id, abbreviation)
# NBA stats IDs preserved for compatibility with logo URLs in app.py.
# ---------------------------------------------------------------------------
_TEAMS = [
    # full_name,                       nba_id,  espn_id, espn_abbr, nba_abbr, nickname,      city
    ("Atlanta Hawks",                  1610612737,  1,  "ATL",  "ATL",  "Hawks",        "Atlanta"),
    ("Boston Celtics",                 1610612738,  2,  "BOS",  "BOS",  "Celtics",      "Boston"),
    ("New Orleans Pelicans",           1610612740,  3,  "NO",   "NOP",  "Pelicans",     "New Orleans"),
    ("Chicago Bulls",                  1610612741,  4,  "CHI",  "CHI",  "Bulls",        "Chicago"),
    ("Cleveland Cavaliers",            1610612739,  5,  "CLE",  "CLE",  "Cavaliers",    "Cleveland"),
    ("Dallas Mavericks",               1610612742,  6,  "DAL",  "DAL",  "Mavericks",    "Dallas"),
    ("Denver Nuggets",                 1610612743,  7,  "DEN",  "DEN",  "Nuggets",      "Denver"),
    ("Detroit Pistons",                1610612765,  8,  "DET",  "DET",  "Pistons",      "Detroit"),
    ("Golden State Warriors",          1610612744,  9,  "GS",   "GSW",  "Warriors",     "Golden State"),
    ("Houston Rockets",                1610612745,  10, "HOU",  "HOU",  "Rockets",      "Houston"),
    ("Indiana Pacers",                 1610612754,  11, "IND",  "IND",  "Pacers",       "Indiana"),
    ("LA Clippers",                    1610612746,  12, "LAC",  "LAC",  "Clippers",     "Los Angeles"),
    ("Los Angeles Lakers",             1610612747,  13, "LAL",  "LAL",  "Lakers",       "Los Angeles"),
    ("Miami Heat",                     1610612748,  14, "MIA",  "MIA",  "Heat",         "Miami"),
    ("Milwaukee Bucks",                1610612749,  15, "MIL",  "MIL",  "Bucks",        "Milwaukee"),
    ("Minnesota Timberwolves",         1610612750,  16, "MIN",  "MIN",  "Timberwolves", "Minnesota"),
    ("Brooklyn Nets",                  1610612751,  17, "BKN",  "BKN",  "Nets",         "Brooklyn"),
    ("New York Knicks",                1610612752,  18, "NY",   "NYK",  "Knicks",       "New York"),
    ("Orlando Magic",                  1610612753,  19, "ORL",  "ORL",  "Magic",        "Orlando"),
    ("Philadelphia 76ers",             1610612755,  20, "PHI",  "PHI",  "76ers",        "Philadelphia"),
    ("Phoenix Suns",                   1610612756,  21, "PHX",  "PHX",  "Suns",         "Phoenix"),
    ("Portland Trail Blazers",         1610612757,  22, "POR",  "POR",  "Trail Blazers","Portland"),
    ("Sacramento Kings",               1610612758,  23, "SAC",  "SAC",  "Kings",        "Sacramento"),
    ("San Antonio Spurs",              1610612759,  24, "SA",   "SAS",  "Spurs",        "San Antonio"),
    ("Oklahoma City Thunder",          1610612760,  25, "OKC",  "OKC",  "Thunder",      "Oklahoma City"),
    ("Utah Jazz",                      1610612762,  26, "UTAH", "UTA",  "Jazz",         "Utah"),
    ("Washington Wizards",             1610612764,  27, "WSH",  "WAS",  "Wizards",      "Washington"),
    ("Toronto Raptors",                1610612761,  28, "TOR",  "TOR",  "Raptors",      "Toronto"),
    ("Memphis Grizzlies",              1610612763,  29, "MEM",  "MEM",  "Grizzlies",    "Memphis"),
    ("Charlotte Hornets",              1610612766,  30, "CHA",  "CHA",  "Hornets",      "Charlotte"),
]

# Index for fast lookup
_TEAM_INDEX = []
for full_name, nba_id, espn_id, espn_abbr, nba_abbr, nickname, city in _TEAMS:
    _TEAM_INDEX.append({
        "id": nba_id,                # keep nba_api-style id for logo URL compatibility
        "full_name": full_name,
        "abbreviation": nba_abbr,    # NBA stats abbreviation (used by app.py for matchup parsing)
        "nickname": nickname,
        "city": city,
        "state": "",
        "year_founded": 0,
        "_espn_id": espn_id,
        "_espn_abbr": espn_abbr,
    })


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _http_get(url: str, params: Optional[dict] = None) -> dict:
    """GET with retries and rate-limiting."""
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            time.sleep(_RATE_DELAY)
            r = requests.get(url, params=params, headers=NBA_HEADERS,
                             timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(0.5 * attempt)
    raise last_exc


# ---------------------------------------------------------------------------
# Static team helpers
# ---------------------------------------------------------------------------
def get_team_info(team_name: str) -> dict:
    """Return team metadata (id, full_name, abbreviation, nickname, ...)."""
    needle = team_name.lower()
    for t in _TEAM_INDEX:
        if (needle in t["full_name"].lower()
                or needle in t["nickname"].lower()
                or needle == t["abbreviation"].lower()
                or needle == t["_espn_abbr"].lower()):
            return t
    raise ValueError(f"Team not found: {team_name}")


def get_team_id(team_name: str) -> int:
    return get_team_info(team_name)["id"]


# ---------------------------------------------------------------------------
# Season conversion helpers
# ---------------------------------------------------------------------------
def _season_end_year(season: str) -> int:
    """'2025-26' -> 2026."""
    if "-" in season:
        start = int(season.split("-")[0])
        return start + 1
    return int(season)


# ---------------------------------------------------------------------------
# Box score parsing
# ---------------------------------------------------------------------------
_STAT_NAME_MAP = {
    "fieldGoalsMade-fieldGoalsAttempted": "FG",
    "fieldGoalPct": "FG_PCT",
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": "FG3",
    "threePointFieldGoalPct": "FG3_PCT",
    "freeThrowsMade-freeThrowsAttempted": "FT",
    "freeThrowPct": "FT_PCT",
    "totalRebounds": "REB",
    "offensiveRebounds": "OREB",
    "defensiveRebounds": "DREB",
    "assists": "AST",
    "steals": "STL",
    "blocks": "BLK",
    "turnovers": "TOV",
    "fouls": "PF",
}


def _parse_made_attempted(s: str) -> tuple:
    """'43-91' -> (43, 91)"""
    try:
        a, b = s.split("-")
        return float(a), float(b)
    except Exception:
        return 0.0, 0.0


def _parse_team_box(team_stats: list) -> dict:
    """Convert ESPN boxscore team statistics list into a flat dict."""
    out = {}
    for stat in team_stats:
        name = stat.get("name")
        val = stat.get("displayValue", "")
        key = _STAT_NAME_MAP.get(name)
        if key is None:
            continue
        if key == "FG":
            m, a = _parse_made_attempted(val)
            out["FGM"], out["FGA"] = m, a
        elif key == "FG3":
            m, a = _parse_made_attempted(val)
            out["FG3M"], out["FG3A"] = m, a
        elif key == "FT":
            m, a = _parse_made_attempted(val)
            out["FTM"], out["FTA"] = m, a
        elif key.endswith("_PCT"):
            try:
                out[key] = float(val) / 100.0
            except ValueError:
                out[key] = 0.0
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = 0.0
    return out


def _possessions(box: dict) -> float:
    """Estimate possessions from a team box score."""
    fga = box.get("FGA", 0)
    fta = box.get("FTA", 0)
    oreb = box.get("OREB", 0)
    tov = box.get("TOV", 0)
    return max(fga - oreb + tov + 0.44 * fta, 1.0)


# ---------------------------------------------------------------------------
# Game log construction
# ---------------------------------------------------------------------------
def _fetch_schedule(espn_team_id: int, season_year: int) -> list:
    """Return list of completed events for a team in the given season (ending year)."""
    url = f"{_ESPN_BASE}/teams/{espn_team_id}/schedule"
    data = _http_get(url, params={"season": season_year})
    events = data.get("events", []) or []
    completed = []
    for ev in events:
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        status = comp.get("status", {}).get("type", {})
        if status.get("completed") or status.get("state") == "post":
            completed.append(ev)
    return completed


def _fetch_summary(event_id: str) -> dict:
    """Fetch a single game's summary (full boxscore)."""
    url = f"{_ESPN_BASE}/summary"
    return _http_get(url, params={"event": event_id})


def _summary_to_row(summary: dict, espn_team_id: int) -> Optional[dict]:
    """Convert a /summary response into a game log row for the given team."""
    try:
        comp = summary["header"]["competitions"][0]
        competitors = comp.get("competitors", [])
        # Identify which competitor is "our" team
        ours = others = None
        for c in competitors:
            if str(c.get("id")) == str(espn_team_id):
                ours = c
            else:
                others = c
        if ours is None or others is None:
            return None

        # Scores + W/L
        team_pts = float(ours.get("score", 0))
        opp_pts = float(others.get("score", 0))
        wl = "W" if team_pts > opp_pts else "L"
        is_home = ours.get("homeAway") == "home"
        opp_abbr = others.get("team", {}).get("abbreviation", "OPP")
        matchup_sep = "vs." if is_home else "@"
        matchup = f"{ours.get('team', {}).get('abbreviation', '')} {matchup_sep} {opp_abbr}"

        # Box score stats
        boxscore = summary.get("boxscore", {})
        team_box, opp_box = {}, {}
        for entry in boxscore.get("teams", []) or []:
            stats_dict = _parse_team_box(entry.get("statistics", []))
            if str(entry.get("team", {}).get("id")) == str(espn_team_id):
                team_box = stats_dict
            else:
                opp_box = stats_dict

        # Date
        date_str = comp.get("date") or summary.get("header", {}).get("competitions", [{}])[0].get("date", "")
        try:
            game_date = pd.to_datetime(date_str).tz_localize(None)
        except Exception:
            game_date = datetime.today()

        row = {
            "GAME_DATE": game_date,
            "MATCHUP": matchup,
            "WL": wl,
            "PTS": team_pts,
            "OPP_PTS": opp_pts,
            "PLUS_MINUS": team_pts - opp_pts,
            "_EVENT_ID": str(summary.get("header", {}).get("id", "")),
            "_TEAM_POSS": _possessions(team_box) if team_box else 0,
            "_OPP_POSS": _possessions(opp_box) if opp_box else 0,
        }
        # Copy box score numeric fields
        for k in ("FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                  "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
                  "AST", "STL", "BLK", "TOV", "PF"):
            row[k] = team_box.get(k, 0)
        return row
    except Exception:
        return None


def get_recent_game_log(team_info: dict, season: str = "2025-26",
                        last_n: int = 15) -> pd.DataFrame:
    """Return the team's recent completed games as a DataFrame (newest first)."""
    season_year = _season_end_year(season)
    events = _fetch_schedule(team_info["_espn_id"], season_year)

    # Fall back to previous season if current has no completed games yet
    if not events:
        events = _fetch_schedule(team_info["_espn_id"], season_year - 1)

    # Sort by date desc and take the most recent N
    def _ev_date(ev):
        try:
            return pd.to_datetime(ev["competitions"][0]["date"])
        except Exception:
            return pd.Timestamp.min
    events.sort(key=_ev_date, reverse=True)
    events = events[:last_n]

    rows = []
    for ev in events:
        summary = _fetch_summary(ev["id"])
        row = _summary_to_row(summary, team_info["_espn_id"])
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def get_last10_averages(game_log: pd.DataFrame) -> dict:
    if game_log.empty:
        return {}
    cols = ["PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
            "STL", "BLK", "TOV", "PF", "PLUS_MINUS"]
    out = {}
    for c in cols:
        if c in game_log.columns:
            out[c] = round(float(game_log[c].mean()), 2)
    out["WIN_PCT_L10"] = round(float((game_log["WL"] == "W").mean()), 3)
    out["GAMES_IN_LAST10"] = int(len(game_log))
    return out


def get_rest_days(game_log: pd.DataFrame) -> int:
    if game_log.empty:
        return 1
    last_date = pd.to_datetime(game_log["GAME_DATE"].iloc[0])
    delta = datetime.today() - last_date
    return max(int(delta.days), 0)


def get_games_in_last_n_days(game_log: pd.DataFrame, n: int = 10) -> int:
    if game_log.empty:
        return 0
    cutoff = datetime.today() - timedelta(days=n)
    return int((pd.to_datetime(game_log["GAME_DATE"]) >= cutoff).sum())


def _season_averages_from_log(team_info: dict, full_log: pd.DataFrame) -> dict:
    """Aggregate season-long averages and advanced ratings from the game log."""
    if full_log.empty:
        return {}

    n = len(full_log)
    wins = int((full_log["WL"] == "W").sum())
    losses = n - wins

    # Per-game averages
    avg_cols = ["PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
                "PLUS_MINUS"]
    avgs = {c: round(float(full_log[c].mean()), 2) for c in avg_cols if c in full_log.columns}

    # Recompute shooting percentages from totals (more accurate than averaging %s)
    def _pct(make_col, att_col):
        att = float(full_log[att_col].sum())
        if att <= 0:
            return 0.0
        return round(float(full_log[make_col].sum()) / att, 3)

    avgs["FG_PCT"]  = _pct("FGM", "FGA")
    avgs["FG3_PCT"] = _pct("FG3M", "FG3A")
    avgs["FT_PCT"]  = _pct("FTM", "FTA")

    # Advanced ratings from possessions
    team_poss_total = float(full_log["_TEAM_POSS"].sum()) if "_TEAM_POSS" in full_log else 0
    opp_poss_total  = float(full_log["_OPP_POSS"].sum())  if "_OPP_POSS" in full_log  else 0
    pts_total = float(full_log["PTS"].sum())
    opp_pts_total = float(full_log["OPP_PTS"].sum()) if "OPP_PTS" in full_log else 0

    if team_poss_total > 0:
        off_rtg = round(100.0 * pts_total / team_poss_total, 1)
    else:
        off_rtg = 0.0
    if opp_poss_total > 0:
        def_rtg = round(100.0 * opp_pts_total / opp_poss_total, 1)
    else:
        def_rtg = 0.0
    net_rtg = round(off_rtg - def_rtg, 1)
    pace    = round((team_poss_total + opp_poss_total) / max(n * 2, 1), 1) if n else 0.0

    avgs.update({
        "TEAM_ID": team_info["id"],
        "TEAM_NAME": team_info["full_name"],
        "GP": n,
        "W": wins,
        "L": losses,
        "OFF_RATING":   off_rtg, "E_OFF_RATING": off_rtg,
        "DEF_RATING":   def_rtg, "E_DEF_RATING": def_rtg,
        "NET_RATING":   net_rtg, "E_NET_RATING": net_rtg,
        "PACE":         pace,    "E_PACE":       pace,
    })
    return avgs


def _clutch_from_log(full_log: pd.DataFrame) -> dict:
    """Approximate clutch stats: games decided by <=5 points."""
    if full_log.empty or "PLUS_MINUS" not in full_log.columns:
        return {}
    close = full_log[full_log["PLUS_MINUS"].abs() <= 5]
    if close.empty:
        return {}
    wins = int((close["WL"] == "W").sum())
    losses = int((close["WL"] == "L").sum())
    return {
        "CLUTCH_W": wins,
        "CLUTCH_L": losses,
        "CLUTCH_PTS": round(float(close["PTS"].mean()), 1),
        "CLUTCH_PLUS_MINUS": round(float(close["PLUS_MINUS"].mean()), 1),
        "CLUTCH_FG_PCT": round(float(close["FG_PCT"].mean()), 3),
    }


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------
def get_head_to_head(team_id_1: int, team_id_2: int, season: str = "2025-26",
                     season_type: str = "Regular Season") -> pd.DataFrame:
    """Return head-to-head games for team_1 against team_2 this season."""
    info_1 = next((t for t in _TEAM_INDEX if t["id"] == team_id_1), None)
    info_2 = next((t for t in _TEAM_INDEX if t["id"] == team_id_2), None)
    if not info_1 or not info_2:
        return pd.DataFrame()
    try:
        log = get_recent_game_log(info_1, season=season, last_n=82)
    except Exception:
        return pd.DataFrame()
    if log.empty:
        return log
    opp_abbr_nba = info_2["abbreviation"]
    opp_abbr_espn = info_2["_espn_abbr"]
    mask = log["MATCHUP"].str.contains(opp_abbr_nba) | log["MATCHUP"].str.contains(opp_abbr_espn)
    return log[mask].sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Master fetch
# ---------------------------------------------------------------------------
def fetch_all_team_data(team_name: str, season: str = "2025-26") -> dict:
    """
    Master fetch — returns all data needed for the prediction model.
    Falls back to embedded sample data if ESPN's API is unreachable
    or returns no completed games.
    """
    info = get_team_info(team_name)

    try:
        print(f"  [{info['full_name']}] Fetching schedule + box scores from ESPN...")
        # 20 games is enough to compute meaningful season averages quickly.
        full_log = get_recent_game_log(info, season=season, last_n=20)

        if full_log.empty:
            raise RuntimeError("No completed games found for this season.")

        last10_log = full_log.head(10).reset_index(drop=True)

        print(f"  [{info['full_name']}] Computing season + L10 averages...")
        season_avgs = _season_averages_from_log(info, full_log)
        last10_avgs = get_last10_averages(last10_log)

        rest_days = get_rest_days(last10_log)
        games_in_10d = get_games_in_last_n_days(last10_log, n=10)
        clutch = _clutch_from_log(full_log)

        return {
            "info": info,
            "team_id": info["id"],
            "game_log": full_log,
            "last10_log": last10_log,
            "season_avgs": season_avgs,
            "last10_avgs": last10_avgs,
            "rest_days": rest_days,
            "games_in_10d": games_in_10d,
            "clutch": clutch,
            "_is_sample": False,
        }

    except Exception as exc:
        print(f"  [{info['full_name']}] ESPN API unavailable ({exc.__class__.__name__}: {exc}). Using sample data.")
        return get_sample_data(team_name)
