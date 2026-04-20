import requests
import pandas as pd
import time

API_KEY = "324bb6df2fc7354452b00f0a4e82affey"
HEADERS = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key": API_KEY
}

all_matches = []

for season in [2022, 2023, 2024]:
    print(f"Fetching season {season}...")
    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS,
        params={"league": 61, "season": season, "status": "FT"},
        timeout=10
    )
    fixtures = r.json().get("response", [])
    print(f"Found {len(fixtures)} matches")
    
    for m in fixtures:
        home_goals = m["goals"]["home"]
        away_goals = m["goals"]["away"]
        if home_goals is None or away_goals is None:
            continue
        if home_goals > away_goals:   result = 0
        elif home_goals < away_goals: result = 2
        else:                          result = 1
        all_matches.append({
            "season":      season,
            "home_team":   m["teams"]["home"]["name"],
            "away_team":   m["teams"]["away"]["name"],
            "home_goals":  home_goals,
            "away_goals":  away_goals,
            "result":      result
        })
    time.sleep(1)

df = pd.DataFrame(all_matches)
df.to_csv("match_data.csv", index=False)
print(f"Done. Saved {len(df)} matches to match_data.csv")