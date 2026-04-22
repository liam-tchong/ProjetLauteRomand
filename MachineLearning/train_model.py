import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

df = pd.read_csv("match_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

def get_rolling_stats(df, team, date, n=5):
    past = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < date)
    ].tail(n)
    if len(past) == 0:
        return {"form": 0.5, "goals_scored": 1.0, "goals_conceded": 1.0}
    wins, scored, conceded = 0, 0, 0
    for _, row in past.iterrows():
        if row["home_team"] == team:
            scored += row["home_goals"]
            conceded += row["away_goals"]
            if row["result"] == "H": wins += 1
        else:
            scored += row["away_goals"]
            conceded += row["home_goals"]
            if row["result"] == "A": wins += 1
    return {
        "form": wins / len(past),
        "goals_scored": scored / len(past),
        "goals_conceded": conceded / len(past),
    }

print("Building features...")
rows = []
for _, match in df.iterrows():
    h = get_rolling_stats(df, match["home_team"], match["date"])
    a = get_rolling_stats(df, match["away_team"], match["date"])
    rows.append([
        h["form"], h["goals_scored"], h["goals_conceded"],
        a["form"], a["goals_scored"], a["goals_conceded"],
        {"H": 0, "D": 1, "A": 2}[match["result"]]
    ])

features = pd.DataFrame(rows, columns=[
    "h_form", "h_scored", "h_conceded",
    "a_form", "a_scored", "a_conceded",
    "result"
])
features = features.dropna()

X = features.drop("result", axis=1)
y = features["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc:.2%}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
with open(out, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {out}")