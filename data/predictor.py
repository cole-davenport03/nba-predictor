"""
Win probability predictor using a weighted statistical model.

Each factor is scored 0–100 (higher = better for that team).
Factors are weighted by predictive importance and combined into
a final win probability using a logistic transformation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Factor weights — tuned for playoff predictiveness
# ---------------------------------------------------------------------------
WEIGHTS = {
    "net_rating":          0.20,   # Best single predictor of team quality
    "off_rating_l10":      0.11,   # Recent offensive efficiency
    "def_rating_l10":      0.11,   # Recent defensive efficiency
    "win_pct_l10":         0.10,   # Recent form (last 10 games)
    "true_shooting_l10":   0.08,   # Shooting efficiency recently
    "rest_advantage":      0.07,   # Days of rest differential
    "pace_adjusted_pts":   0.06,   # Pace-adjusted scoring
    "turnover_rate":       0.06,   # Ball security (lower = better)
    "home_court":          0.05,   # Venue advantage (team home/away splits)
    "rebound_rate":        0.05,   # Rebounding dominance
    "clutch_plus_minus":   0.05,   # Performance in close games
    "assist_to_tov":       0.04,   # Ball movement quality
    "bench_contribution":  0.02,   # Depth factor
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1"

# League-average home court boost in win-probability space (~3-4% historically)
_BASE_HCA = 0.035


def _safe(val, fallback=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def _norm_score(team_val, opp_val, higher_is_better=True) -> float:
    """
    Convert a raw stat pair into a 0–100 score for the team.
    50 = equal, >50 = team has the edge.
    """
    diff = team_val - opp_val if higher_is_better else opp_val - team_val
    total = abs(team_val) + abs(opp_val)
    if total == 0:
        return 50.0
    raw = diff / total  # in [-1, 1]
    return round(50 + raw * 50, 2)


def compute_true_shooting(pts, fga, fta):
    """TS% = PTS / (2 * (FGA + 0.44 * FTA))"""
    denom = 2 * (fga + 0.44 * fta)
    return pts / denom if denom > 0 else 0.0


def compute_tov_rate(tov, fga, fta):
    """TOV% = TOV / (FGA + 0.44 * FTA + TOV)"""
    denom = fga + 0.44 * fta + tov
    return tov / denom if denom > 0 else 0.0


def compute_rebound_rate(team_reb, opp_reb):
    """Simple rebounding rate vs opponent."""
    total = team_reb + opp_reb
    return team_reb / total if total > 0 else 0.5


def _location_splits(team_data: dict) -> tuple:
    """
    Return (home_win_pct, away_win_pct) from a team's game log.
    Defaults to (0.5, 0.5) when data is missing or all-sample.
    """
    log = team_data.get("game_log")
    if log is None or len(log) == 0 or "MATCHUP" not in log.columns:
        return 0.5, 0.5
    # Sample data has no real home/away mix — skip to neutral.
    if team_data.get("_is_sample"):
        return 0.5, 0.5
    home_mask = log["MATCHUP"].str.contains("vs.", regex=False, na=False)
    away_mask = log["MATCHUP"].str.contains("@", regex=False, na=False)
    home = log[home_mask]
    away = log[away_mask]
    home_wp = float((home["WL"] == "W").mean()) if len(home) else 0.5
    away_wp = float((away["WL"] == "W").mean()) if len(away) else 0.5
    return home_wp, away_wp


def build_factor_scores(team_data: dict, opp_data: dict,
                        is_team_home: bool = True) -> dict:
    """
    Compute per-factor edge scores (0–100) for the home team vs opponent.
    Returns a dict of factor_name -> score.
    """
    t_sa = team_data["season_avgs"]
    o_sa = opp_data["season_avgs"]
    t_l10 = team_data["last10_avgs"]
    o_l10 = opp_data["last10_avgs"]
    t_clutch = team_data.get("clutch", {})
    o_clutch = opp_data.get("clutch", {})

    # --- Net Rating ---
    t_net = _safe(t_sa.get("E_NET_RATING") or t_sa.get("NET_RATING"))
    o_net = _safe(o_sa.get("E_NET_RATING") or o_sa.get("NET_RATING"))
    net_rating_score = _norm_score(t_net, o_net)

    # --- Off / Def rating last 10 ---
    # Approximate from last 10 game log (pts scored vs pts allowed)
    # We use season OFF_RATING & DEF_RATING adjusted by L10 performance
    t_off = _safe(t_sa.get("E_OFF_RATING") or t_sa.get("OFF_RATING"))
    o_off = _safe(o_sa.get("E_OFF_RATING") or o_sa.get("OFF_RATING"))
    t_def = _safe(t_sa.get("E_DEF_RATING") or t_sa.get("DEF_RATING"))
    o_def = _safe(o_sa.get("E_DEF_RATING") or o_sa.get("DEF_RATING"))

    # Blend season rating with L10 PTS as a proxy for recent form
    t_pts_l10 = _safe(t_l10.get("PTS", t_off))
    o_pts_l10 = _safe(o_l10.get("PTS", o_off))
    off_score = _norm_score(
        0.5 * t_off + 0.5 * t_pts_l10,
        0.5 * o_off + 0.5 * o_pts_l10,
    )
    # DEF rating: lower is better
    def_score = _norm_score(t_def, o_def, higher_is_better=False)

    # --- Win % last 10 ---
    t_wp = _safe(t_l10.get("WIN_PCT_L10", 0.5))
    o_wp = _safe(o_l10.get("WIN_PCT_L10", 0.5))
    win_pct_score = _norm_score(t_wp, o_wp)

    # --- True Shooting last 10 ---
    t_ts = compute_true_shooting(
        _safe(t_l10.get("PTS")), _safe(t_l10.get("FGA")), _safe(t_l10.get("FTA"))
    )
    o_ts = compute_true_shooting(
        _safe(o_l10.get("PTS")), _safe(o_l10.get("FGA")), _safe(o_l10.get("FTA"))
    )
    ts_score = _norm_score(t_ts, o_ts)

    # --- Rest advantage ---
    t_rest = _safe(team_data.get("rest_days", 1))
    o_rest = _safe(opp_data.get("rest_days", 1))
    rest_score = _norm_score(t_rest, o_rest)

    # --- Pace-adjusted points (PTS / pace * 100) ---
    t_pace = _safe(t_sa.get("PACE") or t_sa.get("E_PACE") or 100)
    o_pace = _safe(o_sa.get("PACE") or o_sa.get("E_PACE") or 100)
    t_pa_pts = (t_pts_l10 / t_pace * 100) if t_pace else t_pts_l10
    o_pa_pts = (o_pts_l10 / o_pace * 100) if o_pace else o_pts_l10
    pace_score = _norm_score(t_pa_pts, o_pa_pts)

    # --- Turnover rate (lower is better) ---
    t_tov_rate = compute_tov_rate(
        _safe(t_l10.get("TOV")), _safe(t_l10.get("FGA")), _safe(t_l10.get("FTA"))
    )
    o_tov_rate = compute_tov_rate(
        _safe(o_l10.get("TOV")), _safe(o_l10.get("FGA")), _safe(o_l10.get("FTA"))
    )
    tov_score = _norm_score(t_tov_rate, o_tov_rate, higher_is_better=False)

    # --- Rebound rate ---
    t_reb = _safe(t_l10.get("REB"))
    o_reb = _safe(o_l10.get("REB"))
    reb_score = _norm_score(t_reb, o_reb)

    # --- Clutch plus/minus ---
    t_cl = _safe(t_clutch.get("CLUTCH_PLUS_MINUS", 0))
    o_cl = _safe(o_clutch.get("CLUTCH_PLUS_MINUS", 0))
    clutch_score = _norm_score(t_cl, o_cl)

    # --- Assist to turnover ratio ---
    t_ast = _safe(t_l10.get("AST"))
    t_tov = max(_safe(t_l10.get("TOV")), 0.1)
    o_ast = _safe(o_l10.get("AST"))
    o_tov2 = max(_safe(o_l10.get("TOV")), 0.1)
    ast_tov_score = _norm_score(t_ast / t_tov, o_ast / o_tov2)

    # --- Bench contribution (placeholder — plus_minus as proxy) ---
    t_pm = _safe(t_l10.get("PLUS_MINUS"))
    o_pm = _safe(o_l10.get("PLUS_MINUS"))
    bench_score = _norm_score(t_pm, o_pm)

    # --- Home court advantage ---
    # Compare each team's actual record at their current venue, plus a small
    # league-average HCA boost (since 10-game samples are noisy).
    t_home_wp, t_away_wp = _location_splits(team_data)
    o_home_wp, o_away_wp = _location_splits(opp_data)
    if is_team_home:
        t_loc = t_home_wp + _BASE_HCA
        o_loc = o_away_wp
    else:
        t_loc = t_away_wp
        o_loc = o_home_wp + _BASE_HCA
    home_court_score = _norm_score(t_loc, o_loc)

    return {
        "net_rating":          net_rating_score,
        "off_rating_l10":      off_score,
        "def_rating_l10":      def_score,
        "win_pct_l10":         win_pct_score,
        "true_shooting_l10":   ts_score,
        "rest_advantage":      rest_score,
        "pace_adjusted_pts":   pace_score,
        "turnover_rate":       tov_score,
        "home_court":          home_court_score,
        "rebound_rate":        reb_score,
        "clutch_plus_minus":   clutch_score,
        "assist_to_tov":       ast_tov_score,
        "bench_contribution":  bench_score,
    }


def compute_win_probability(factor_scores: dict):
    """
    Combine weighted factor scores into win probabilities for team and opponent.
    Returns (team_win_prob, opp_win_prob) as floats in [0, 1].
    """
    weighted_sum = sum(
        WEIGHTS[factor] * score
        for factor, score in factor_scores.items()
        if factor in WEIGHTS
    )
    # weighted_sum is in [0, 100]; center at 50
    raw_log_odds = (weighted_sum - 50) * 0.08  # scale to logit range
    team_prob = 1 / (1 + np.exp(-raw_log_odds))
    return round(float(team_prob), 4), round(float(1 - team_prob), 4)


def predict_game(team_data: dict, opp_data: dict,
                 team_is_home: bool = True) -> dict:
    """
    Full prediction pipeline.
    Returns a results dict with factor scores, win probabilities, and narrative.

    `team_is_home`: True if team_data is the home team (default).
    """
    factor_scores = build_factor_scores(team_data, opp_data, is_team_home=team_is_home)
    team_prob, opp_prob = compute_win_probability(factor_scores)

    team_name = team_data["info"]["full_name"]
    opp_name = opp_data["info"]["full_name"]

    winner = team_name if team_prob >= opp_prob else opp_name
    win_pct = max(team_prob, opp_prob)
    confidence = (
        "High" if win_pct > 0.65
        else "Medium" if win_pct > 0.55
        else "Low (Toss-up)"
    )

    # Key edges — factors where team leads by > 5 points
    team_edges = [
        f for f, s in factor_scores.items() if s > 55
    ]
    opp_edges = [
        f for f, s in factor_scores.items() if s < 45
    ]

    reasoning = generate_reasoning(
        team_data, opp_data, factor_scores, winner, team_prob, opp_prob
    )

    return {
        "team_name": team_name,
        "opp_name": opp_name,
        "team_win_prob": team_prob,
        "opp_win_prob": opp_prob,
        "predicted_winner": winner,
        "confidence": confidence,
        "factor_scores": factor_scores,
        "team_edges": team_edges,
        "opp_edges": opp_edges,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Human-readable reasoning generator
# ---------------------------------------------------------------------------

_FACTOR_DESCRIPTIONS = {
    "net_rating": (
        "Net Rating",
        "overall team efficiency (points scored minus allowed per 100 possessions)",
    ),
    "off_rating_l10": (
        "Offensive Rating (last 10)",
        "points scored per 100 possessions over the last 10 games",
    ),
    "def_rating_l10": (
        "Defensive Rating (last 10)",
        "points allowed per 100 possessions over the last 10 games",
    ),
    "win_pct_l10": (
        "Win % (last 10 games)",
        "recent form — how often they've been winning lately",
    ),
    "true_shooting_l10": (
        "True Shooting % (last 10)",
        "shooting efficiency accounting for 2-pointers, 3-pointers, and free throws",
    ),
    "rest_advantage": (
        "Rest Advantage",
        "days of rest since their last game — fresher legs typically mean better performance",
    ),
    "pace_adjusted_pts": (
        "Pace-Adjusted Scoring",
        "offensive output normalised for pace of play",
    ),
    "turnover_rate": (
        "Ball Security",
        "turnover rate — protecting the ball reduces easy points for the opponent",
    ),
    "rebound_rate": (
        "Rebounding",
        "controlling the glass on both ends of the floor",
    ),
    "clutch_plus_minus": (
        "Clutch Performance",
        "point differential in close games (within 5, final 5 minutes)",
    ),
    "home_court": (
        "Home Court Advantage",
        "venue effect based on each team's home and road win % plus a league-average HCA boost",
    ),
    "assist_to_tov": (
        "Assist-to-Turnover Ratio",
        "ball movement quality — sharing the ball without giving it away",
    ),
    "bench_contribution": (
        "Bench Depth",
        "second-unit impact via overall plus/minus",
    ),
}


def generate_reasoning(team_data: dict, opp_data: dict, factor_scores: dict,
                        winner: str, team_prob: float, opp_prob: float) -> dict:
    """
    Build a structured reasoning dict explaining why the model picked the winner.
    Returns sections used by the UI to render a 'Why?' breakdown.
    """
    team_name = team_data["info"]["full_name"]
    opp_name  = opp_data["info"]["full_name"]
    winner_is_team = (winner == team_name)

    w_data  = team_data  if winner_is_team else opp_data
    l_data  = opp_data   if winner_is_team else team_data
    w_prob  = team_prob  if winner_is_team else opp_prob
    l_prob  = opp_prob   if winner_is_team else team_prob
    loser   = opp_name   if winner_is_team else team_name

    sa_w = w_data["season_avgs"]
    sa_l = l_data["season_avgs"]
    l10_w = w_data["last10_avgs"]
    l10_l = l_data["last10_avgs"]

    # ── Key advantages (factors where winner leads by ≥ 8 pts) ──────────────
    advantages = []
    for key, score in factor_scores.items():
        winner_score = score if winner_is_team else (100 - score)
        loser_score  = 100 - winner_score
        if winner_score - loser_score >= 8:
            label, desc = _FACTOR_DESCRIPTIONS.get(key, (key, ""))
            advantages.append({
                "factor": key,
                "label": label,
                "description": desc,
                "winner_score": round(winner_score, 1),
                "loser_score":  round(loser_score, 1),
                "margin": round(winner_score - loser_score, 1),
            })

    advantages.sort(key=lambda x: x["margin"], reverse=True)

    # ── Concerns for the winner (factors where they trail) ───────────────────
    concerns = []
    for key, score in factor_scores.items():
        winner_score = score if winner_is_team else (100 - score)
        if winner_score < 48:
            label, desc = _FACTOR_DESCRIPTIONS.get(key, (key, ""))
            concerns.append({
                "factor": key,
                "label": label,
                "description": desc,
                "winner_score": round(winner_score, 1),
                "loser_score":  round(100 - winner_score, 1),
            })

    # ── Headline stats comparison ─────────────────────────────────────────────
    def _r(v): return round(_safe(v), 1)

    stats_comparison = [
        {
            "label": "Points Per Game",
            "winner_val": _r(sa_w.get("PTS", 0)),
            "loser_val":  _r(sa_l.get("PTS", 0)),
            "unit": "pts",
            "higher_better": True,
        },
        {
            "label": "Offensive Rating",
            "winner_val": _r(sa_w.get("E_OFF_RATING", sa_w.get("OFF_RATING", 0))),
            "loser_val":  _r(sa_l.get("E_OFF_RATING", sa_l.get("OFF_RATING", 0))),
            "unit": "",
            "higher_better": True,
        },
        {
            "label": "Defensive Rating",
            "winner_val": _r(sa_w.get("E_DEF_RATING", sa_w.get("DEF_RATING", 0))),
            "loser_val":  _r(sa_l.get("E_DEF_RATING", sa_l.get("DEF_RATING", 0))),
            "unit": "",
            "higher_better": False,
        },
        {
            "label": "Net Rating",
            "winner_val": _r(sa_w.get("E_NET_RATING", sa_w.get("NET_RATING", 0))),
            "loser_val":  _r(sa_l.get("E_NET_RATING", sa_l.get("NET_RATING", 0))),
            "unit": "",
            "higher_better": True,
        },
        {
            "label": "Win % (Last 10)",
            "winner_val": round(_safe(l10_w.get("WIN_PCT_L10", 0)) * 100, 0),
            "loser_val":  round(_safe(l10_l.get("WIN_PCT_L10", 0)) * 100, 0),
            "unit": "%",
            "higher_better": True,
        },
        {
            "label": "FG% (Last 10)",
            "winner_val": round(_safe(l10_w.get("FG_PCT", 0)) * 100, 1),
            "loser_val":  round(_safe(l10_l.get("FG_PCT", 0)) * 100, 1),
            "unit": "%",
            "higher_better": True,
        },
        {
            "label": "3P% (Last 10)",
            "winner_val": round(_safe(l10_w.get("FG3_PCT", 0)) * 100, 1),
            "loser_val":  round(_safe(l10_l.get("FG3_PCT", 0)) * 100, 1),
            "unit": "%",
            "higher_better": True,
        },
        {
            "label": "Turnovers (Last 10)",
            "winner_val": _r(l10_w.get("TOV", 0)),
            "loser_val":  _r(l10_l.get("TOV", 0)),
            "unit": "/gm",
            "higher_better": False,
        },
        {
            "label": "Assists (Last 10)",
            "winner_val": _r(l10_w.get("AST", 0)),
            "loser_val":  _r(l10_l.get("AST", 0)),
            "unit": "/gm",
            "higher_better": True,
        },
        {
            "label": "Rebounds (Last 10)",
            "winner_val": _r(l10_w.get("REB", 0)),
            "loser_val":  _r(l10_l.get("REB", 0)),
            "unit": "/gm",
            "higher_better": True,
        },
        {
            "label": "Rest Days",
            "winner_val": float(w_data.get("rest_days", 1)),
            "loser_val":  float(l_data.get("rest_days", 1)),
            "unit": "d",
            "higher_better": True,
        },
    ]

    # ── One-sentence narrative ────────────────────────────────────────────────
    top3 = [a["label"] for a in advantages[:3]]
    if top3:
        top_str = ", ".join(top3[:-1]) + (" and " + top3[-1] if len(top3) > 1 else top3[0])
        narrative = (
            f"{winner} is projected to win with {w_prob*100:.1f}% confidence, "
            f"driven by superior {top_str}."
        )
    else:
        narrative = (
            f"{winner} holds a narrow edge over {loser} "
            f"({w_prob*100:.1f}% vs {l_prob*100:.1f}%) in a closely contested matchup."
        )

    if concerns:
        concern_labels = [c["label"] for c in concerns[:2]]
        narrative += (
            f" The main risk area is {' and '.join(concern_labels)}, "
            f"where {loser} has an advantage."
        )

    return {
        "winner": winner,
        "loser": loser,
        "winner_prob": w_prob,
        "loser_prob": l_prob,
        "advantages": advantages,
        "concerns": concerns,
        "stats_comparison": stats_comparison,
        "narrative": narrative,
    }
