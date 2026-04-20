import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

print("Loading data...")
df = pd.read_csv("MachineLearning/match_data.csv")

# Convert result to numbers: H=0 (home win), D=1 (draw), A=2 (away win)
df["result"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
df = df.dropna(subset=["result"])

# Sort by date so we can calculate rolling stats
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

def get_rolling_stats(df, team, date, n=5):
    """Get last n matches stats for a team before a given date."""
    past = df[
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team)) &
        (df["Date"] < date)
    ].tail(n)
    
    if len(past) == 0:
        return {"form": 0.5, "goals_scored": 1.0, "goals_conceded": 1.0}
    
    wins, goals_scored, goals_conceded = 0, 0, 0
    for _, row in past.iterrows():
        if row["HomeTeam"] == team:
            goals_scored    += row["FTHG"]
            goals_conceded  += row["FTAG"]
            if row["FTR"] == "H": wins += 1
        else:
            goals_scored    += row["FTAG"]
            goals_conceded  += row["FTHG"]
            if row["FTR"] == "A": wins += 1
    
    return {
        "form":            wins / len(past),
        "goals_scored":    goals_scored / len(past),
        "goals_conceded":  goals_conceded / len(past),
    }

print("Engineering features (this takes a minute)...")
rows = []
for _, match in df.iterrows():
    home = match["HomeTeam"]
    away = match["AwayTeam"]
    date = match["Date"]
    
    home_stats = get_rolling_stats(df, home, date)
    away_stats = get_rolling_stats(df, away, date)
    
    rows.append({
        "home_form":           home_stats["form"],
        "home_goals_scored":   home_stats["goals_scored"],
        "home_goals_conceded": home_stats["goals_conceded"],
        "away_form":           away_stats["form"],
        "away_goals_scored":   away_stats["goals_scored"],
        "away_goals_conceded": away_stats["goals_conceded"],
        "result":              match["result"],
    })

features_df = pd.DataFrame(rows)

X = features_df.drop("result", axis=1)
y = features_df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model accuracy: {acc:.2%}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to model.pkl")