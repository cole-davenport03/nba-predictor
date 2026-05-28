"""Day 4: build a feature matrix from real ESPN data using NumPy.

This file does NO machine learning. It just turns a list of games into two
NumPy arrays that day5_pytorch_predict.py will consume:

    X: shape (n, 2)   features per game: [is_home, avg_margin_prior_5]
    y: shape (n,)     label:             1.0 if the team won, else 0.0

You're using the same ESPN API as day1_team_log.py.
"""
import json
import urllib.request
import numpy as np

ESPN_IDS = {"LAL": 13, "BOS": 2, "DEN": 7, "GSW": 9, "OKC": 25, "MIN": 16}


def fetch_games(team_abbr: str, season: int = 2026) -> list[dict]:
    """Return completed games as a list of small dicts, oldest first.

    Each dict has: 'home' (bool), 'points_for' (int),
    'points_against' (int), 'won' (bool).
    """
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
        f"/teams/{ESPN_IDS[team_abbr]}/schedule?season={season}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    games: list[dict] = []
    for event in data["events"]:
        comp = event["competitions"][0]
        if not comp["status"]["type"]["completed"]:
            continue

        competitors = comp["competitors"]
        us = next(c for c in competitors if c["team"]["abbreviation"] == team_abbr)
        them = next(c for c in competitors if c["team"]["abbreviation"] != team_abbr)

        points_for = int(us["score"]["value"])
        points_against = int(them["score"]["value"])

        games.append({
            "home": us["homeAway"] == "home",
            "points_for": points_for,
            "points_against": points_against,
            "won": points_for > points_against,
        })
    return games


def build_features(games: list[dict], window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Convert the game list into NumPy arrays X (features) and y (labels)."""
    # 1-D NumPy array of margins, one per game, in order.
    margins = np.array(
        [g["points_for"] - g["points_against"] for g in games],
        dtype=np.float32,
    )

    rows: list[list[float]] = []
    labels: list[float] = []

    # Only games with at least `window` prior games can have features.
    for i in range(window, len(games)):
        avg_margin_prior = margins[i - window : i].mean()
        is_home = 1.0 if games[i]["home"] else 0.0
        rows.append([is_home, float(avg_margin_prior)])
        labels.append(1.0 if games[i]["won"] else 0.0)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y




def main() -> None:
    games = fetch_games("OKC")
    print(f"fetched {len(games)} completed games")

    X, y = build_features(games)
    print(f"X shape: {X.shape}   (expect (n, 2))")
    print(f"y shape: {y.shape}   (expect (n,))")
    print(f"first 3 rows of X:\n{X[:3]}")
    print(f"first 3 labels:   {y[:3]}")
    print(f"overall win rate: {y.mean():.2f}")
    print(f"overall home win rate: {y[X[:, 0] == 1].mean():.2f}")
    print(f"overall away win rate: {y[X[:, 0] == 0].mean():.2f}")
    margins = np.array([g["points_for"] - g["points_against"] for g in games])
    print(f"avg margin:       {margins.mean():+.2f}")


if __name__ == "__main__":
    main()
