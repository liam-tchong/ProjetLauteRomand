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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "match_data.csv")

# Load existing data so we only fetch what's missing
if os.path.exists(OUT):
    existing = pd.read_csv(OUT)
    existing_keys = set(zip(existing["date"], existing["home_team"], existing["away_team"]))
    print(f"Existing CSV: {len(existing)} matches, latest date: {existing['date'].max()}")
else:
    existing = pd.DataFrame()
    existing_keys = set()
    print("No existing CSV — full fetch.")


def fetch_matches(league_code, season):
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    r = requests.get(url, headers=HEADERS, params={"season": season, "status": "FINISHED"}, timeout=10)
    r.raise_for_status()
    matches = r.json().get("matches", [])
    rows = []
    skipped = 0
    for m in matches:
        d = m["utcDate"][:10]
        ht = m["homeTeam"]["name"]
        at = m["awayTeam"]["name"]
        hs = m["score"]["fullTime"].get("home")
        as_ = m["score"]["fullTime"].get("away")
        if hs is None or as_ is None:
            continue
        if (d, ht, at) in existing_keys:
            skipped += 1
            continue
        rows.append({
            "league":     LEAGUES[league_code],
            "season":     season,
            "date":       d,
            "home_team":  ht,
            "away_team":  at,
            "home_goals": hs,
            "away_goals": as_,
            "result":     "H" if hs > as_ else ("D" if hs == as_ else "A"),
        })
    print(f"  → {len(rows)} new, {skipped} already have")
    return rows


all_new = []
for league_code in LEAGUES:
    for season in SEASONS:
        print(f"Fetching {LEAGUES[league_code]} {season}…")
        try:
            rows = fetch_matches(league_code, season)
            all_new.extend(rows)
        except Exception as e:
            print(f"  → ERROR: {e}")
        time.sleep(6)

if all_new:
    new_df = pd.DataFrame(all_new)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["date", "home_team", "away_team"], inplace=True)
    combined.sort_values("date", inplace=True)
    combined.to_csv(OUT, index=False)
    print(f"\nDone! {len(combined)} total matches ({len(all_new)} new) saved to {OUT}")
else:
    print("\nNo new matches found — CSV is already up to date.")
