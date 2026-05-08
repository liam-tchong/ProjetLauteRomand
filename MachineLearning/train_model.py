import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
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
        return {"form": 0.5, "goals_scored": 1.3, "goals_conceded": 1.3}
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
        {"H": 0, "D": 1, "A": 2}[match["result"]],
        match["home_goals"],
        match["away_goals"],
    ])

features = pd.DataFrame(rows, columns=[
    "h_form", "h_scored", "h_conceded",
    "a_form", "a_scored", "a_conceded",
    "result", "home_goals", "away_goals"
]).dropna()

X = features[["h_form", "h_scored", "h_conceded", "a_form", "a_scored", "a_conceded"]]
y_result = features["result"].astype(int)
y_hg = features["home_goals"].astype(float)
y_ag = features["away_goals"].astype(float)

X_train, X_test, yr_train, yr_test = train_test_split(X, y_result, test_size=0.2, random_state=42)
_, _, yh_train, yh_test = train_test_split(X, y_hg, test_size=0.2, random_state=42)
_, _, ya_train, ya_test = train_test_split(X, y_ag, test_size=0.2, random_state=42)

print("Training result classifier (GradientBoosting)...")
clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
clf.fit(X_train, yr_train)
acc = accuracy_score(yr_test, clf.predict(X_test))
print(f"  Accuracy: {acc:.2%}")

print("Training home goals model (Poisson regression)...")
home_model = PoissonRegressor(alpha=0.5, max_iter=500)
home_model.fit(X_train, yh_train)
mae_h = mean_absolute_error(yh_test, home_model.predict(X_test))
print(f"  MAE home goals: {mae_h:.3f}")

print("Training away goals model (Poisson regression)...")
away_model = PoissonRegressor(alpha=0.5, max_iter=500)
away_model.fit(X_train, ya_train)
mae_a = mean_absolute_error(ya_test, away_model.predict(X_test))
print(f"  MAE away goals: {mae_a:.3f}")

out_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
    pickle.dump(clf, f)
print("Classifier saved.")

with open(os.path.join(out_dir, "goals_model.pkl"), "wb") as f:
    pickle.dump({"home": home_model, "away": away_model}, f)
print("Goals models saved.")
print(f"\nAll models in {out_dir}")
