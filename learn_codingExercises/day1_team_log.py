"""Day 1: pure-Python NBA team report."""
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
import numpy as np
import torch

ESPN_IDS = {"LAL": 13, "BOS": 2, "DEN": 7, "GSW": 9, "OKC": 25, "MIN": 16}
TEAM_NAMES = {
    "LAL": "Los Angeles Lakers", "BOS": "Boston Celtics",
    "DEN": "Denver Nuggets", "GSW": "Golden State Warriors",
    "OKC": "Oklahoma City Thunder", "MIN": "Minnesota Timberwolves",
    "WSH": "Washington Wizards", "PHX": "Phoenix Suns", "MIA": "Miami Heat",
}


@dataclass
class Game:
    date: datetime
    opponent: str
    home: bool
    points_for: int
    points_against: int

    @property
    def won(self) -> bool:
        return self.points_for > self.points_against
    
    @property
    def lost(self) -> bool:
        return self.points_for < self.points_against

    @property
    def margin(self) -> int:
        return self.points_for - self.points_against


# --- two interchangeable data sources ---

def load_games_from_urllib(team_abbr: str) -> list[Game]:
    """Source 1: fetch & parse JSON directly. This is the Day 1 learning exercise."""
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
        f"/teams/{ESPN_IDS[team_abbr]}/schedule?season=2025"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    games: list[Game] = []
    for event in data["events"]:
        comp = event["competitions"][0]
        if not comp["status"]["type"]["completed"]:
            continue

        competitors = comp["competitors"]
        us = next(c for c in competitors if c["team"]["abbreviation"] == team_abbr)
        them = next(c for c in competitors if c["team"]["abbreviation"] != team_abbr)

        games.append(Game(
            date=datetime.fromisoformat(event["date"].replace("Z", "+00:00")),
            opponent=them["team"]["abbreviation"],
            home=us["homeAway"] == "home",
            points_for=int(us["score"]["value"]),
            points_against=int(them["score"]["value"]),
        ))
    return games


def load_games_from_fetcher(team_name: str) -> list[Game]:
    """Source 2: reuse the project's existing fetcher. DataFrame -> list[Game]."""
    from data.fetcher import fetch_all_team_data
    data = fetch_all_team_data(team_name)
    df = data["game_log"]
    games = []
    for _, row in df.iterrows():
        matchup = row["MATCHUP"]
        home = "vs." in matchup
        opponent = matchup.split()[-1]
        games.append(Game(
            date=row["GAME_DATE"].to_pydatetime(),
            opponent=opponent,
            home=home,
            points_for=int(row["PTS"]),
            points_against=int(row["PTS"] - row["PLUS_MINUS"]),
        ))
    return games


# --- everything below is source-agnostic ---

def summarize(games: list[Game]) -> dict:
    wins = sum(1 for g in games if g.won)
    best = max(games, key=lambda g: g.margin)
    return {
        "wins": wins,
        "losses": len(games) - wins,
        "win_pct": wins / len(games),
        "avg_margin": mean(g.margin for g in games),
        "best_win_margin": best.margin,
        "best_win_opponent": best.opponent,
        "worst_loss_margin": best.margin if best.won else best.margin,
        "worst_loss_opponent": best.opponent if not best.won else best.opponent
    }

def in_slump(games: list[Game]) -> bool:
    """Example of a more complex summary metric: 3+ losses in a row."""
    current_streak = 0
    for g in games:
        if g.won:
            current_streak = 0
        else:
            current_streak += 1
            if current_streak >= 3:
                return True
    return False

def get_current_streak(games: list[Game]) -> int:
    if not games:
        return 0

    most_recent_won = games[0].won
    count = 0
    for g in games:
        if g.won != most_recent_won:
            break
        count += 1

    streak = count if most_recent_won else count
    print("Current Streak: W" if most_recent_won else "Current Streak: L", streak)
    return streak

def get_best_win(games: list[Game]) -> Game:
    """Helper function to find the best win."""
    wins = [g for g in games if g.won]
    return max(wins, key=lambda g: g.margin) if wins else None

def print_best_win(games: list[Game]) -> None:
    best_win = get_best_win(games)
    if best_win:
        print(f"Best win: +{best_win.margin} vs {best_win.opponent}")
    else:
        print("Best win: No wins found.")

def get_worst_loss(games: list[Game]) -> Game:
    """Helper function to find the worst loss."""
    losses = [g for g in games if not g.won]
    return min(losses, key=lambda g: g.margin) if losses else None

def print_worst_loss(games: list[Game]) -> None:
    worst_loss = get_worst_loss(games)
    if worst_loss:
        print(f"Worst loss: {worst_loss.margin} vs {worst_loss.opponent}")
    else:
        print("Worst loss: No losses found.")



def print_report(team: str, games: list[Game], s: dict) -> None:
    for g in games:
        date_str = g.date.strftime("%Y-%m-%d")
        home_away = "vs." if g.home else "@"
        result = "W" if g.won else "L"
        print(f"{date_str} {home_away} {g.opponent}: {result} {g.points_for}-{g.points_against}")
    print()

    print(f"{TEAM_NAMES[team]} — last {len(games)} games")
    print(f"Record: {s['wins']}-{s['losses']} ({s['win_pct']:.0%})")
    print(f"Avg margin: {s['avg_margin']:+.1f}")
    print_best_win(games)
    print_worst_loss(games)
    get_current_streak(games)
    print(f"In slump: {'Yes' if in_slump(games) else 'No'}")


def main() -> None:
    print()
    print("Pure Python Example:")
    print("-" * 40)
    team = "OKC" 
    games = load_games_from_urllib(team)        # swap to load_games_from_fetcher(TEAM_NAMES[team]) to compare
    games.sort(key=lambda g: g.date, reverse=True)
    games = games[:10]
    print_report(team, games, summarize(games))

    print()
    print("Numpy Example:")
    print("-" * 40)


if __name__ == "__main__":
    main()