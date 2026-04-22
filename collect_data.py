import requests
import pandas as pd
import os
import time

API_KEY = "911605e549af4b759c5d7d2ffa977742"
HEADERS = {"X-Auth-Token": API_KEY}

LEAGUES = {
    "FL1": "Ligue 1",
    "PL":  "Premier League",
    "PD":  "La Liga",
    "SA":  "Serie A",
    "BL1": "Bundesliga",
}

SEASONS = [2023, 2024, 2025]

def fetch_matches(league_code, season):
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    r = requests.get(url, headers=HEADERS, params={"season": season}, timeout=10)
    r.raise_for_status()
    matches = r.json().get("matches", [])
    rows = []
    for m in matches:
        if m["status"] != "FINISHED":
            continue
        rows.append({
            "league":    LEAGUES[league_code],
            "season":    season,
            "date":      m["utcDate"][:10],
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
            "result":    "H" if m["score"]["fullTime"]["home"] > m["score"]["fullTime"]["away"]
                         else "D" if m["score"]["fullTime"]["home"] == m["score"]["fullTime"]["away"]
                         else "A",
        })
    return rows

all_rows = []
for league_code in LEAGUES:
    for season in SEASONS:
        print(f"Fetching {LEAGUES[league_code]} {season}...")
        try:
            rows = fetch_matches(league_code, season)
            print(f"  → {len(rows)} matches")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  → ERROR: {e}")
        time.sleep(6)  # API rate limit: 10 requests/minute on free tier

df = pd.DataFrame(all_rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "match_data.csv")
df.to_csv(out, index=False)
print(f"\nDone! {len(df)} matches saved to {out}")