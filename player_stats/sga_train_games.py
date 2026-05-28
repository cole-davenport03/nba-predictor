import pandas as pd
from masking import load_and_split

X_train, y_train, X_test, y_test, y_mean, y_std, df, test_idx = load_and_split()
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

test_idx_set = set(test_idx.tolist())
train_df = df[~df.index.isin(test_idx_set)]

# Pick any player you're curious about
player = "Shai Gilgeous-Alexander"   # or "Nikola Jokic", "Luka Doncic", etc.
games = train_df[train_df["PLAYER_NAME"] == player].sort_values("GAME_DATE")
print(games[["GAME_DATE", "MATCHUP", "MIN", "PTS"]].tail(15).to_string())