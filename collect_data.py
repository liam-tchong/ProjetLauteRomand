import pandas as pd
import requests
import io

print("Downloading Ligue 1 historical data...")

all_dfs = []

urls = {
    2022: "https://www.football-data.co.uk/mmz4281/2223/F1.csv",
    2023: "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
    2024: "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
}

for season, url in urls.items():
    print(f"Fetching season {season}...")
    r = requests.get(url, timeout=10)
    df = pd.read_csv(io.StringIO(r.text))
    df["season"] = season
    all_dfs.append(df)
    print(f"Found {len(df)} matches")

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv("match_data.csv", index=False)
print(f"Done. Saved {len(combined)} matches to match_data.csv")