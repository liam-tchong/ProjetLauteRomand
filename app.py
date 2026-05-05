import streamlit as st
import requests
import re
import anthropic
import time
import os
import pickle

_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MachineLearning", "model.pkl")
try:
    with open(_model_path, "rb") as _f:
        MATCH_MODEL = pickle.load(_f)
except Exception:
    MATCH_MODEL = None

st.set_page_config(page_title="The Football Classroom", layout="wide")

# ── API ───────────────────────────────────────────────────────────────────────
API_KEY = "911605e549af4b759c5d7d2ffa977742"
HEADERS = {"X-Auth-Token": API_KEY}

ANTHROPIC_API_KEY  = st.secrets.get("ANTHROPIC_API_KEY", "")
API_FOOTBALL_KEY   = st.secrets.get("API_FOOTBALL_KEY", "")
SQUAD_API_KEY      = st.secrets.get("SQUAD_API_KEY", "")

LEAGUES = {
    "Ligue 1":        {"code": "FL1",  "flag": "🇫🇷", "country": "France",  "color": "#1A56C4", "color_lt": "#E8F0FB"},
    "La Liga":        {"code": "PD",   "flag": "🇪🇸", "country": "Spain",   "color": "#F5C800", "color_lt": "#FFFBE6"},
    "Serie A":        {"code": "SA",   "flag": "🇮🇹", "country": "Italy",   "color": "#1E9E4A", "color_lt": "#E6F7EC"},
    "Premier League": {"code": "PL",   "flag": "🇬🇧", "country": "England", "color": "#E02020", "color_lt": "#FDEAED"},
    "Bundesliga":     {"code": "BL1",  "flag": "🇩🇪", "country": "Germany", "color": "#7B4A1E", "color_lt": "#F5EDE6"},
}

# api-sports.io team IDs (used for squad composition in Positions tab)
API_FOOTBALL_IDS = {
    "Paris Saint-Germain":    85,
    "Olympique de Marseille": 81,
    "Olympique Lyonnais":     80,
    "AS Monaco":              91,
    "LOSC Lille":             79,
    "RC Lens":               116,
    "OGC Nice":               84,
    "Stade Rennais":          94,
    "RC Strasbourg":          95,
    "Toulouse FC":            96,
    "Stade Brestois":        130,
    "FC Nantes":              83,
    "Angers SCO":             82,
    "Le Havre AC":          1006,
    "AJ Auxerre":             78,
    "FC Metz":               112,
    "Paris FC":              167,
    "FC Lorient":           1041,
}

TEAM_NAME_MAP = {
    "Paris Saint-Germain FC": "Paris Saint-Germain",
    "Racing Club de Lens": "RC Lens",
    "Lille OSC": "LOSC Lille",
    "Stade Rennais FC 1901": "Stade Rennais",
    "AS Monaco FC": "AS Monaco",
    "RC Strasbourg Alsace": "RC Strasbourg",
    "Stade Brestois 29": "Stade Brestois",
}

def _form_score(form_list):
    """Convert last-5 form into a weighted momentum score 0..1 (recent matches weigh more)."""
    if not form_list:
        return 0.5
    weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(form_list):]
    pts = {"W": 1.0, "D": 0.5, "L": 0.0}
    total_w = sum(weights)
    return sum(pts.get(r, 0.5) * w for r, w in zip(form_list, weights)) / total_w


def predict_expected_score(standings, home_team, away_team,
                            form_home=None, form_away=None,
                            extra_home=None, extra_away=None):
    """Predict expected goals for each team using attack vs defence averages + recent form.

    Uses a simplified Poisson-like approach: each team's expected goals =
    (their attack rate) × (opponent's defensive weakness) × (home/away adjustment) × (form factor)

    Returns dict with xg_home, xg_away, most_likely_score, alternatives.
    """
    if not standings:
        return None
    try:
        dh = standings.get(home_team, {})
        da = standings.get(away_team, {})
        if not dh or not da:
            return None

        played_h = dh.get("played", 1) or 1
        played_a = da.get("played", 1) or 1

        # Prefer recent (last 15) averages if available — more in tune with current form
        gf_h = (extra_home or {}).get("gf_avg_recent") or dh.get("goals_for", 0) / played_h
        ga_h = (extra_home or {}).get("ga_avg_recent") or dh.get("goals_against", 0) / played_h
        gf_a = (extra_away or {}).get("gf_avg_recent") or da.get("goals_for", 0) / played_a
        ga_a = (extra_away or {}).get("ga_avg_recent") or da.get("goals_against", 0) / played_a

        # League average ≈ 1.3 goals per team per match as baseline
        LEAGUE_AVG = 1.3

        # Attack strength = team attack / league avg; Defence weakness = team defence / league avg
        atk_strength_h = gf_h / LEAGUE_AVG if LEAGUE_AVG else 1
        def_weakness_a = ga_a / LEAGUE_AVG if LEAGUE_AVG else 1
        atk_strength_a = gf_a / LEAGUE_AVG if LEAGUE_AVG else 1
        def_weakness_h = ga_h / LEAGUE_AVG if LEAGUE_AVG else 1

        # Baseline expected goals
        xg_home = LEAGUE_AVG * atk_strength_h * def_weakness_a
        xg_away = LEAGUE_AVG * atk_strength_a * def_weakness_h

        # Home advantage: historically ~15% boost for home team
        xg_home *= 1.15

        # Recent form adjustment
        fh = _form_score(form_home) if form_home else 0.5
        fa = _form_score(form_away) if form_away else 0.5
        # Form factor: 0.85x for cold streak (form=0), 1.15x for hot streak (form=1)
        xg_home *= 0.85 + fh * 0.3
        xg_away *= 0.85 + fa * 0.3

        # Round to most likely integer score
        most_likely_h = max(0, round(xg_home))
        most_likely_a = max(0, round(xg_away))

        return {
            "xg_home": round(xg_home, 2),
            "xg_away": round(xg_away, 2),
            "most_likely_score": (most_likely_h, most_likely_a),
        }
    except Exception:
        return None


def predict_match(standings, home_team, away_team, form_home=None, form_away=None,
                  extra_home=None, extra_away=None):
    """Match prediction combining season stats + live form + tactical context.

    - standings: current season table (season-level data)
    - form_home / form_away: last 5 results as ['W','D','L',...] (live momentum)
    - extra_home / extra_away: extended stats dict with clean_sheets, gf_avg_recent, etc.

    Returns (probs, meta) where probs = [P(home_win), P(draw), P(away_win)]
    and meta is a dict with interpretable factors (momentum, form boost, etc.).
    """
    if MATCH_MODEL is None or not standings:
        return None, None
    try:
        dh = standings.get(home_team, {})
        da = standings.get(away_team, {})
        if not dh or not da:
            return None, None
        played_h = dh.get("played", 1) or 1
        played_a = da.get("played", 1) or 1

        # ── Season-level features (must match train_model.py column order) ──
        import pandas as _pd
        features = _pd.DataFrame([[
            dh.get("won", 0) / played_h,
            dh.get("goals_for", 0) / played_h,
            dh.get("goals_against", 0) / played_h,
            da.get("won", 0) / played_a,
            da.get("goals_for", 0) / played_a,
            da.get("goals_against", 0) / played_a,
        ]], columns=["h_form","h_scored","h_conceded","a_form","a_scored","a_conceded"])
        base_probs = MATCH_MODEL.predict_proba(features)[0]  # [H, D, A]

        # ── Live adjustments (actualité / current form) ──
        # 1) Recent form: last-5 momentum score (weights most recent matches heavier)
        fh = _form_score(form_home) if form_home else 0.5
        fa = _form_score(form_away) if form_away else 0.5

        # 2) Home/away record from last 15 (gets closer to live state than season avg)
        def _rate(rec):
            if not rec:
                return 0.5
            w, d, l = 0, 0, 0
            for p in rec.split():
                if p.endswith("W"): w = int(p[:-1])
                elif p.endswith("D"): d = int(p[:-1])
                elif p.endswith("L"): l = int(p[:-1])
            tot = w + d + l
            return (w + 0.5 * d) / tot if tot else 0.5
        hr = _rate((extra_home or {}).get("home_record")) if extra_home else 0.5
        ar = _rate((extra_away or {}).get("away_record")) if extra_away else 0.5

        # 3) Recent attacking/defensive momentum (last 15 matches)
        gf_h = (extra_home or {}).get("gf_avg_recent") or dh.get("goals_for", 0) / played_h
        ga_h = (extra_home or {}).get("ga_avg_recent") or dh.get("goals_against", 0) / played_h
        gf_a = (extra_away or {}).get("gf_avg_recent") or da.get("goals_for", 0) / played_a
        ga_a = (extra_away or {}).get("ga_avg_recent") or da.get("goals_against", 0) / played_a

        # Momentum delta: attack vs their likely opposition defence
        atk_h = gf_h - ga_a  # how much home side should outscore away
        atk_a = gf_a - ga_h
        momentum_delta = (atk_h - atk_a) * 0.04  # small adjustment, ~±8% max

        # 4) Form-based probability shift
        # Each 0.1 difference in form ≈ 4% shift toward the hotter team
        form_delta = (fh - fa) * 0.40  # ~±20% max shift for extreme form gaps
        venue_delta = (hr - ar) * 0.15  # home/away form

        # Combine — bounded adjustments so the season-trained model stays relevant
        total_shift = max(-0.25, min(0.25, form_delta + venue_delta + momentum_delta))

        # Apply shift: positive → home gains, negative → away gains. Draw is mostly stable.
        ph, pd_, pa = base_probs[0], base_probs[1], base_probs[2]
        if total_shift > 0:
            # take from away_win, give to home_win
            take = min(total_shift, pa * 0.6)
            ph += take; pa -= take
        else:
            take = min(-total_shift, ph * 0.6)
            pa += take; ph -= take
        # Re-normalise
        s = ph + pd_ + pa
        ph, pd_, pa = ph / s, pd_ / s, pa / s
        adjusted = [ph, pd_, pa]

        meta = {
            "momentum_home": fh,
            "momentum_away": fa,
            "home_venue_rate": hr,
            "away_venue_rate": ar,
            "form_delta": form_delta,
            "venue_delta": venue_delta,
            "momentum_delta": momentum_delta,
            "total_shift": total_shift,
            "base": list(base_probs),
            "confidence": max(adjusted) - sorted(adjusted)[-2],  # gap between top 2
        }
        return adjusted, meta
    except Exception:
        return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_standings(league_code):
    name_map = TEAM_NAME_MAP
    for attempt in range(4):
        try:
            r = requests.get(
                f"https://api.football-data.org/v4/competitions/{league_code}/standings",
                headers=HEADERS, timeout=10
            )
            if r.status_code == 429:
                time.sleep(7)
                continue
            r.raise_for_status()
            table = r.json()["standings"][0]["table"]
            result = {}
            for row in table:
                t = row["team"]
                name = name_map.get(t["name"], t["name"])
                result[name] = {
                    "id":            t["id"],
                    "crest":         t["crest"],
                    "short":         t.get("shortName") or name[:3].upper(),
                    "position":      row["position"],
                    "points":        row["points"],
                    "played":        row["playedGames"],
                    "won":           row["won"],
                    "draw":          row["draw"],
                    "lost":          row["lost"],
                    "goals_for":     row["goalsFor"],
                    "goals_against": row["goalsAgainst"],
                    "goal_diff":     row["goalDifference"],
                }
            return result
        except Exception:
            if attempt < 3:
                time.sleep(1.5)
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def generate_standings_summary(league_name, standings_tuple):
    """Generate a 3-4 sentence league summary via Claude Haiku."""
    if not ANTHROPIC_API_KEY or not standings_tuple:
        return ""
    teams = sorted(standings_tuple, key=lambda x: x[1])  # sort by position
    if len(teams) < 5:
        return ""
    n = len(teams)
    def fmt(t):
        return f"{t[0]} ({t[2]} pts)"
    top3 = teams[:3]
    cl4, cl5 = teams[3], teams[4]
    rel_border = teams[n - 4]
    rel_zone = teams[n - 3:]
    matchday = teams[0][3]
    context = (
        f"League: {league_name} — Matchday {matchday}\n"
        f"Leader: {fmt(top3[0])}\n"
        f"2nd: {fmt(top3[1])} (gap to 1st: {top3[0][2]-top3[1][2]} pts)\n"
        f"3rd: {fmt(top3[2])} (gap to 1st: {top3[0][2]-top3[2][2]} pts)\n"
        f"4th vs 5th (European border): {fmt(cl4)} vs {fmt(cl5)} — gap: {cl4[2]-cl5[2]} pts\n"
        f"Just above relegation: {fmt(rel_border)}\n"
        f"Relegation zone: {fmt(rel_zone[0])}, {fmt(rel_zone[1])}, {fmt(rel_zone[2])}\n"
        f"Gap between safety and drop zone: {rel_border[2]-rel_zone[0][2]} pts\n"
    )
    prompt = (
        "You are a sharp European football analyst writing for a quality sports publication. "
        "Based on the standings data below, respond in exactly this format — two lines, nothing else:\n"
        "TITLE: [a punchy 6-10 word headline capturing the key storyline of the season]\n"
        "BODY: [3-4 sentences of flowing analysis covering the title race, European spots battle, and relegation drama. Be specific with team names and points gaps. No bullet points.]\n\n"
        + context
    )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=260,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        title, body = "", raw
        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("TITLE:"):
                title = line[6:].strip()
            elif line.upper().startswith("BODY:"):
                body = line[5:].strip()
        return f"{title}|||{body}" if title else body
    except Exception:
        return ""

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_previous_standings(league_code):
    name_map = TEAM_NAME_MAP
    try:
        r = requests.get(
            f"https://api.football-data.org/v4/competitions/{league_code}/standings",
            headers=HEADERS,
            params={"season": 2024},
            timeout=10
        )
        r.raise_for_status()
        table = r.json()["standings"][0]["table"]
        return {
            name_map.get(row["team"]["name"], row["team"]["name"]): row["position"]
            for row in table
        }
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_schedule(league_code, date_from, date_to):
    """Fetch matches for a league between two dates (yyyy-mm-dd strings)."""
    for attempt in range(4):
        try:
            r = requests.get(
                f"https://api.football-data.org/v4/competitions/{league_code}/matches",
                headers=HEADERS,
                params={"dateFrom": date_from, "dateTo": date_to},
                timeout=10,
            )
            if r.status_code == 429:
                time.sleep(8)
                continue
            r.raise_for_status()
            matches = r.json().get("matches", [])
            if matches:
                return matches
            # Empty but valid response — return it (not an error)
            return []
        except Exception:
            time.sleep(3)
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_team_extended(team_id):
    """Last 15 finished matches: returns (form_5, extended_stats_dict)."""
    if not team_id:
        return [], {}
    try:
        r = requests.get(
            f"https://api.football-data.org/v4/teams/{team_id}/matches",
            headers=HEADERS,
            params={"status": "FINISHED", "limit": 15},
            timeout=10,
        )
        r.raise_for_status()
        matches = r.json().get("matches", [])

        form = []
        home_w = home_d = home_l = 0
        away_w = away_d = away_l = 0
        clean_sheets = 0
        gf_list = []
        ga_list = []

        for m in matches:
            home_id    = m["homeTeam"]["id"]
            hs = m["score"]["fullTime"]["home"]
            as_ = m["score"]["fullTime"]["away"]
            if hs is None or as_ is None:
                continue
            is_home = (home_id == team_id)
            gs = hs if is_home else as_
            gc = as_ if is_home else hs
            gf_list.append(gs)
            ga_list.append(gc)
            if gc == 0:
                clean_sheets += 1
            result = "W" if gs > gc else ("L" if gs < gc else "D")
            form.append(result)
            if is_home:
                if result == "W": home_w += 1
                elif result == "D": home_d += 1
                else: home_l += 1
            else:
                if result == "W": away_w += 1
                elif result == "D": away_d += 1
                else: away_l += 1

        n = len(gf_list)
        stats = {
            "home_record":   f"{home_w}W {home_d}D {home_l}L",
            "away_record":   f"{away_w}W {away_d}D {away_l}L",
            "clean_sheets":  clean_sheets,
            "gf_avg_recent": round(sum(gf_list) / n, 2) if n else None,
            "ga_avg_recent": round(sum(ga_list) / n, 2) if n else None,
            "win_pct":       round(form.count("W") / len(form) * 100) if form else None,
        }
        return form[-5:], stats
    except Exception:
        return [], {}


def fetch_team_form(team_id):
    """Kept for backward compat — returns just the 5-match form list."""
    form, _ = fetch_team_extended(team_id)
    return form


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_competition_scorers(league_code):
    name_map = TEAM_NAME_MAP
    try:
        r = requests.get(
            f"https://api.football-data.org/v4/competitions/{league_code}/scorers",
            headers=HEADERS,
            params={"limit": 50},
            timeout=10
        )
        r.raise_for_status()
        scorers_by_team = {}
        for s in r.json().get("scorers", []):
            raw_team = s["team"]["name"]
            team = name_map.get(raw_team, raw_team)
            player = s["player"]["name"]
            goals  = s.get("goals", 0)
            assists = s.get("assists") or 0
            scorers_by_team.setdefault(team, []).append((player, goals, assists))
        return scorers_by_team
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_api_football_stats(team_name, league_code="FL1"):
    """Advanced stats from API-Football: formation, passes, shots, clean sheets."""
    if league_code != "FL1" or not API_FOOTBALL_KEY:
        return {}
    team_id = API_FOOTBALL_IDS.get(team_name)
    if not team_id:
        return {}
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/teams/statistics",
            headers={"x-apisports-key": API_FOOTBALL_KEY},
            params={"league": 61, "season": 2025, "team": team_id},
            timeout=10
        )
        r.raise_for_status()
        data = r.json().get("response", {})
        if not data:
            return {}

        fixtures = data.get("fixtures", {})
        goals    = data.get("goals", {})
        lineups  = data.get("lineups", [])
        passes   = data.get("passes", {})
        shots    = data.get("shots", {})
        clean_sh = data.get("clean_sheet", {})
        failed   = data.get("failed_to_score", {})

        formation  = lineups[0]["formation"] if lineups else None
        gf_avg     = goals.get("for",     {}).get("average", {}).get("total")
        ga_avg     = goals.get("against", {}).get("average", {}).get("total")
        wins_home  = fixtures.get("wins",  {}).get("home", 0)
        wins_away  = fixtures.get("wins",  {}).get("away", 0)

        # Time slot when the team scores most
        minutes  = goals.get("for", {}).get("minute", {})
        top_slot = max(minutes, key=lambda k: (minutes[k].get("total") or 0)) if minutes else None

        # Passes & shots
        passes_pct     = passes.get("percentage")
        shots_total    = shots.get("total",  {}).get("total")
        shots_on       = shots.get("on",     {}).get("total")
        played_total   = fixtures.get("played", {}).get("total") or 1
        shots_pg       = round(shots_total / played_total, 1) if shots_total else None
        shots_on_pg    = round(shots_on    / played_total, 1) if shots_on    else None

        # Clean sheets & matches without scoring
        clean_sheets     = clean_sh.get("total")
        failed_to_score  = failed.get("total")

        return {
            "formation":       formation,
            "gf_avg":          gf_avg,
            "ga_avg":          ga_avg,
            "wins_home":       wins_home,
            "wins_away":       wins_away,
            "top_scoring_slot": top_slot,
            "passes_pct":      passes_pct,
            "shots_pg":        shots_pg,
            "shots_on_pg":     shots_on_pg,
            "clean_sheets":    clean_sheets,
            "failed_to_score": failed_to_score,
        }
    except Exception:
        return {}


# ── Squad composition via SQUAD_API_KEY ──────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_squad_composition(team_id):
    """Fetch squad players grouped by position using SQUAD_API_KEY."""
    if not SQUAD_API_KEY or not team_id:
        return {}
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/players/squads",
            headers={"x-apisports-key": SQUAD_API_KEY},
            params={"team": team_id},
            timeout=10,
        )
        r.raise_for_status()
        response = r.json().get("response", [])
        if not response:
            return {}
        players = response[0].get("players", [])
        by_position = {}
        for p in players:
            pos = p.get("position", "Unknown")
            by_position.setdefault(pos, []).append(p["name"])
        return by_position
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def generate_team_style(team_name, pts, played, won, draw, lost,
                        goals_for, goals_against, goal_diff, position,
                        prev_position,
                        form_tuple, key_scorers_tuple,
                        extra_formation, extra_gf_avg, extra_ga_avg,
                        extra_wins_home, extra_wins_away, extra_top_slot,
                        extra_passes_pct, extra_shots_pg, extra_shots_on_pg,
                        extra_clean_sheets, extra_failed_to_score,
                        home_record=None, away_record=None,
                        clean_sheets_recent=None, gf_avg_recent=None,
                        ga_avg_recent=None, win_pct=None):
    """Generates a 3-paragraph tactical analysis via Claude."""
    if not ANTHROPIC_API_KEY:
        return TEAM_STYLES.get(team_name, DEFAULT_STYLE)

    form_str = " → ".join(form_tuple) if form_tuple else "N/A"
    avg_gf   = round(goals_for      / max(played, 1), 2)
    avg_ga   = round(goals_against  / max(played, 1), 2)

    def ordinal(n):
        if 11 <= n % 100 <= 13: return f"{n}th"
        return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    prev_str = ordinal(prev_position) if prev_position else "N/A"
    ord_suffix = {1: "st", 2: "nd", 3: "rd"}.get(position % 10, "th") if (position and not (11 <= position % 100 <= 13)) else "th"
    pos_delta = ""
    if prev_position and position:
        diff = prev_position - position
        if diff > 0:   pos_delta = f" (↑ +{diff} vs last season)"
        elif diff < 0: pos_delta = f" (↓ {diff} vs last season)"
        else:           pos_delta = " (same position as last season)"

    stats_block = f"""Team: {team_name}
Current ranking: {position}{ord_suffix} place{pos_delta} — {pts} points in {played} matches
Previous season ranking (2024/25): {prev_str}
Record: {won}W / {draw}D / {lost}L
Goals scored: {goals_for} ({avg_gf}/match) | Goals conceded: {goals_against} ({avg_ga}/match)
Goal difference: {goal_diff:+}
Recent form (last 5 matches): {form_str}"""

    if key_scorers_tuple:
        scorers_str = ", ".join(f"{n} ({g} goals{', '+str(a)+' assists' if a else ''})"
                                for n, g, a in key_scorers_tuple)
        stats_block += f"\nKey players (scorers): {scorers_str}"
    if extra_formation:
        stats_block += f"\nMain formation: {extra_formation}"
    if extra_wins_home is not None:
        stats_block += f"\nHome / away wins: {extra_wins_home} / {extra_wins_away}"
    if extra_passes_pct:
        stats_block += f"\nPass accuracy: {extra_passes_pct}"
    if extra_shots_pg:
        stats_block += f"\nShots per match: {extra_shots_pg} (of which {extra_shots_on_pg} on target)"
    if extra_clean_sheets is not None:
        stats_block += f"\nClean sheets: {extra_clean_sheets}"
    if extra_failed_to_score is not None:
        stats_block += f"\nMatches without scoring: {extra_failed_to_score}"
    if extra_top_slot:
        stats_block += f"\nPeriod when team scores most: {extra_top_slot} min"
    if extra_gf_avg:
        stats_block += f"\nAverage goals for / against per match: {extra_gf_avg} / {extra_ga_avg}"
    if home_record:
        stats_block += f"\nHome record (last 15 matches): {home_record}"
    if away_record:
        stats_block += f"\nAway record (last 15 matches): {away_record}"
    if clean_sheets_recent is not None:
        stats_block += f"\nClean sheets (last 15 matches): {clean_sheets_recent}"
    if gf_avg_recent is not None:
        stats_block += f"\nAvg goals scored (last 15 matches): {gf_avg_recent}/match"
    if ga_avg_recent is not None:
        stats_block += f"\nAvg goals conceded (last 15 matches): {ga_avg_recent}/match"
    if win_pct is not None:
        stats_block += f"\nWin rate (last 15 matches): {win_pct}%"

    terms = ", ".join(TACTICAL_TERMS.keys())

    prompt = f"""You are explaining a football team to someone who has NEVER watched football before. Use simple, friendly, everyday language — like talking to a curious 12-year-old. No complicated words unless you explain them simply.

{stats_block}

Write EXACTLY 4 sections separated by "|||". Each section: MAX 2 short sentences. Be very brief. No titles, no numbers, no bullet points.

SECTION 1 — THE CLUB: Famous or not? Recent success or struggling? One sentence on their vibe.

SECTION 2 — HOW THEY PLAY (simple): 2-3 sentences. Do they attack or defend? Fast or patient? Use 2-3 terms from this list: {terms} — wrap each like <b>term</b> and explain it in simple words right after (e.g. "<b>pressing</b> (hunting the ball immediately when they lose it)").

SECTION 3 — HOW THEY PLAY (details): 3-4 sentences. Go a bit deeper. Name 1-2 real players and what they do. Use 3-4 terms from: {terms} — wrap each like <b>term</b> with a short simple explanation. Make the glossary terms feel natural in the sentences.

SECTION 4 — FUN FACT: One fun or surprising thing about this club in 1-2 sentences.

Reply with EXACTLY 4 sections separated by "|||", nothing else."""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        parts = [p.strip() for p in raw.split("|||")]
        while len(parts) < 4:
            parts.append("")
        return tuple(parts[:4])
    except Exception:
        fallback = TEAM_STYLES.get(team_name, DEFAULT_STYLE)
        return (fallback, "", "", "")


@st.cache_data(ttl=86400, show_spinner=False)
def generate_key_challenges(team_a, team_b, pts_a, pts_b, gf_a, gf_b, ga_a, ga_b):
    """Generates a short challenge paragraph (2-3 sentences) per team for this specific matchup."""
    if not ANTHROPIC_API_KEY:
        return (
            f"{team_a} must stay compact and limit space in behind. Defensive organisation will be key.",
            f"{team_b} must be clinical in the final third. Creating clear chances will decide the match.",
        )
    prompt = f"""You are a concise football analyst. Given these two teams and their season stats, write a short challenge for each team in this specific matchup.

{team_a}: {pts_a} pts, {gf_a} goals scored, {ga_a} goals conceded this season.
{team_b}: {pts_b} pts, {gf_b} goals scored, {ga_b} goals conceded this season.

For each team write 2 sentences maximum — the main tactical challenge they face and one concrete consequence if they fail to meet it. Be direct and specific to this matchup.

Format (reply with exactly these 2 blocks, nothing else):
{team_a}: <2 sentences>
{team_b}: <2 sentences>"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=180,
            messages=[{"role": "user", "content": prompt}]
        )
        lines = [l.strip() for l in msg.content[0].text.strip().split("\n") if l.strip()]
        def _extract(line, team):
            if ":" in line:
                return line.split(":", 1)[1].strip()
            return line.strip()
        challenge_a = _extract(lines[0], team_a) if len(lines) > 0 else "Stay compact and limit space in behind. Defensive organisation will be key."
        challenge_b = _extract(lines[1], team_b) if len(lines) > 1 else "Be clinical in the final third. Creating clear chances will decide the match."
        return challenge_a, challenge_b
    except Exception:
        return (
            f"Stay compact and limit space in behind. Defensive organisation will be key.",
            f"Be clinical in the final third. Creating clear chances will decide the match.",
        )


# ── Data ─────────────────────────────────────────────────────────────────────
TACTICAL_TERMS = {
    "pressing": {
        "definition":        "A collective tactic where players immediately pressure the ball carrier upon losing possession, aiming to quickly win the ball back.",
        "simple_explanation":"Instead of waiting, players run at the opponent to force mistakes and recover the ball fast.",
        "example":           "Liverpool press so high that opposition goalkeepers regularly make errors under pressure.",
        "animation_idea":    "Multiple players converging rapidly toward the ball carrier.",
    },
    "pivot": {
        "definition":        "A central striker who acts as a target man in the attacking phase, able to receive with their back to goal, hold the ball, and redistribute.",
        "simple_explanation":"A strong player who receives the ball with their back to goal and lays it off to teammates.",
        "example":           "Ibrahimovic was a perfect pivot: he controlled, shielded, and relaunched play effortlessly.",
        "animation_idea":    "Player receiving the ball with back to goal, then laying it off to arriving teammates.",
    },
    "false nine": {
        "definition":        "A centre-forward who drops into midfield rather than staying as a traditional striker, creating confusion in the opposition's defence.",
        "simple_explanation":"An attacker who drifts away from goal to confuse defenders who don't know whether to follow.",
        "example":           "Messi played as a false nine under Guardiola at Barcelona, leaving centre-backs disoriented.",
        "animation_idea":    "Watch F9 drop back — the CBs don't know whether to follow or stay. That gap they leave behind? That's exactly where the midfielders sprint into.",
    },
    "build-up play": {
        "definition":        "The phase of play when a team in possession tries to move the ball forward from defence into attack against an organized opposition.",
        "simple_explanation":"The team passes the ball from the back to the front in an organized, controlled way.",
        "example":           "A goalkeeper passes to a centre-back, who plays to the midfielder, who finds the striker.",
        "animation_idea":    "The ball travels step by step: GK → CB → CM → ST. No rush, no long ball — just controlled progression until the striker is in a dangerous spot.",
    },
    "through ball": {
        "definition":        "A pass played straight through or behind the opposition's defence to reach a teammate making a run.",
        "simple_explanation":"A pass that goes through the gaps in the defence to a teammate running behind the defenders.",
        "example":           "A midfielder slides a pass between two defenders for the striker to run onto and score.",
        "animation_idea":    "Ball piercing a defensive line toward a forward making a diagonal run behind.",
    },
    "switch of play": {
        "definition":        "Moving the ball quickly from one side of the pitch to the opposite flank to exploit space.",
        "simple_explanation":"Passing the ball all the way across the field to find a free teammate on the other side.",
        "example":           "A right back receives on the right and plays a long diagonal to the left winger who is unmarked.",
        "animation_idea":    "Ball traveling in a wide diagonal arc from one touchline to the other.",
    },
    "overlap": {
        "definition":        "A run made by an attacking player around the outside of a teammate who has the ball, forcing a defender to make a choice.",
        "simple_explanation":"A player runs around the outside of their own teammate to create a 2v1 situation on the wing.",
        "example":           "The right back sprints down the touchline past the right winger to receive the pass and cross.",
        "animation_idea":    "Player running in a curved arc around the outside of a teammate and a defender.",
    },
    "underlap": {
        "definition":        "A run made inside the ball carrier, cutting through the half-space rather than going around the outside.",
        "simple_explanation":"Instead of running outside, a player cuts inside their teammate to find space in the danger zone.",
        "example":           "As the winger stays wide, the fullback makes an underlapping run inside to arrive in the box.",
        "animation_idea":    "Player running diagonally inward through the space behind a wide player.",
    },
    "cross": {
        "definition":        "A delivery of the ball into the penalty area from a wide position, usually between the penalty box and the touchline.",
        "simple_explanation":"A player near the touchline kicks the ball into the box for a teammate to head or shoot.",
        "example":           "The left winger dribbles to the byline and whips a cross for the striker to head home.",
        "animation_idea":    "Ball curving from the wing into the penalty area toward the six-yard box.",
    },
    "final third": {
        "definition":        "The attacking portion of the pitch — the last 35 metres before the opponent's goal — where the most dangerous chances are created.",
        "simple_explanation":"The area of the pitch closest to the opponent's goal where teams try to create and score chances.",
        "example":           "A team struggles to create chances because they cannot enter the final third with the ball.",
        "animation_idea":    "Pitch divided into three horizontal zones, with the attacking third highlighted in green.",
    },
    "counter-attack": {
        "definition":        "An attacking manoeuvre in which a team suddenly transitions from defence to attack, overwhelming the opposition before they can reorganize.",
        "simple_explanation":"As soon as your team wins the ball, attack quickly before opponents get back in position.",
        "example":           "PSG wins the ball in midfield and three players sprint forward to score within seconds.",
        "animation_idea":    "Arrows showing rapid movement of players from their own half toward the opponent's goal.",
    },
    "high press": {
        "definition":        "A pressing tactic applied very high up the pitch, near the opponent's own goal, to win the ball back in a dangerous area.",
        "simple_explanation":"Attacking players press the opponents right in their own half, far from your goal.",
        "example":           "Dortmund presses Barcelona's centre-backs to prevent any clean build-up play from the back.",
        "animation_idea":    "Attacking players surrounding opposition defenders near the opponent's penalty area.",
    },
    "low block": {
        "definition":        "A defensive tactic in which a team retreats very deep in their own half with all players behind the ball to restrict space.",
        "simple_explanation":"The whole team drops back near their own goal to leave absolutely no space for the opponents.",
        "example":           "Atletico Madrid defends with 8–9 players behind the ball in a disciplined low block.",
        "animation_idea":    "All outfield players packed tightly inside their own half forming a compact defensive wall.",
    },
    "man marking": {
        "definition":        "A defensive system in which each player is responsible for tracking one specific opposing player across the pitch.",
        "simple_explanation":"Your job is to follow one specific opponent wherever they go on the pitch.",
        "example":           "A midfielder is assigned to shadow the opponent's playmaker throughout the entire match.",
        "animation_idea":    "Colored lines linking each defender directly to their assigned opponent.",
    },
    "zonal marking": {
        "definition":        "A defensive system in which each player is responsible for an area of the pitch rather than a specific opposing player.",
        "simple_explanation":"You guard a zone — whoever enters your zone is your responsibility, not a specific person.",
        "example":           "At set pieces, defenders stand in zones rather than following individual attackers.",
        "animation_idea":    "The pitch divided into colored zones, each with one defender positioned inside.",
    },
    "tackle": {
        "definition":        "A method of winning the ball from an opponent by using a leg to wrest possession, or a sliding challenge to knock the ball away.",
        "simple_explanation":"A defender reaches in with their foot or slides to take the ball from an attacker.",
        "example":           "A centre-back makes a perfectly timed sliding tackle to stop a striker through on goal.",
        "animation_idea":    "Player sliding toward a ball carrier with foot extending to reach the ball.",
    },
    "interception": {
        "definition":        "The act of cutting off a pass before it reaches its intended recipient, stealing possession mid-flight.",
        "simple_explanation":"Reading the game well enough to step into the path of a pass and steal the ball.",
        "example":           "A midfielder reads the pass and steps in to intercept before the striker can receive it.",
        "animation_idea":    "Player stepping into the path of a dotted passing line between two opponents.",
    },
    "counter-pressing": {
        "definition":        "Pressing applied immediately after losing the ball to quickly regain possession, before the opponent can settle.",
        "simple_explanation":"The moment you lose the ball, instantly surround the opponent before they can look up.",
        "example":           "After losing the ball, Dortmund's players immediately swarm the opponent to win it back within seconds.",
        "animation_idea":    "The moment the ball is lost — don't retreat, attack it. 3 players swarm the opponent before they can look up. You have roughly 6 seconds before they're organized.",
    },
    "transition": {
        "definition":        "The moment a team switches from attack to defence (negative transition) or from defence to attack (positive transition).",
        "simple_explanation":"The moment your team changes from defending to attacking — or the other way around.",
        "example":           "A team is caught in a bad transition when they lose the ball with players pushed too far forward.",
        "animation_idea":    "The most chaotic moment in football — the split second the ball changes hands. Attackers become defenders, defenders become attackers. Teams that win transitions win games.",
    },
    "formation": {
        "definition":        "The organized arrangement of players on the pitch, usually described numerically from defence to attack (e.g. 4-3-3).",
        "simple_explanation":"The shape your team uses — 4-3-3 means 4 defenders, 3 midfielders, 3 forwards.",
        "example":           "A 4-3-3 focuses on wing play; a 3-5-2 focuses on midfield control — different strengths.",
        "animation_idea":    "Player dots rearranging on the pitch to show different formation shapes.",
    },
    "shape": {
        "definition":        "The organized defensive or attacking structure of a team — how compact, wide, or deep their collective positioning is.",
        "simple_explanation":"How the team looks as a unit — compact and narrow, or stretched out wide.",
        "example":           "A team with good shape stays close together, making it very hard to find gaps to play through.",
        "animation_idea":    "Compact block of player dots shifting together as a unit across the pitch.",
    },
    "width": {
        "definition":        "The tactical use of the full side-to-side space of the pitch to stretch the opposition's defensive shape.",
        "simple_explanation":"Spreading players out to the touchlines to stretch defenders and open space in the center.",
        "example":           "Wingers stay wide to pull defenders apart and create gaps for midfielders running through.",
        "animation_idea":    "Players positioned near both touchlines stretching a compressed defensive block.",
    },
    "depth": {
        "definition":        "The tactical use of forward-backward space, with players staggered at different distances from goal to provide passing options.",
        "simple_explanation":"Having players at different heights on the pitch so there are short and long passing options.",
        "example":           "A striker stays high while a midfielder drops deep to offer both short and long pass options.",
        "animation_idea":    "Players staggered at different vertical distances creating multiple passing lanes.",
    },
    "half-space": {
        "definition":        "The channel between the central area and the wide zone, just inside the full-back — one of the most dangerous areas to receive the ball.",
        "simple_explanation":"The danger zone between the center and the wing — hard for defenders to cover.",
        "example":           "An attacking midfielder receives in the half-space, turns quickly, and shoots at goal.",
        "animation_idea":    "The yellow zones are a defender's nightmare — too central for the full-back, too wide for the centre-back. Nobody clearly owns that space. Receive there, turn, and you're in a dangerous position immediately.",
    },
    "lines": {
        "definition":        "The horizontal rows of players a team organizes in defence or midfield — breaking these lines is a key attacking objective.",
        "simple_explanation":"Defenders form a flat 'line' across the pitch; attackers try to play passes through or over it.",
        "example":           "A through ball breaks the defensive line and puts a striker one-on-one with the goalkeeper.",
        "animation_idea":    "Horizontal rows of defenders with arrows showing passes that break through each line.",
    },
    "tiki-taka": {
        "definition":        "A style of play characterised by short, quick passing, constant movement and maintaining possession to control the match.",
        "simple_explanation":"Keep the ball moving fast with short passes — never hold it long, always move after passing.",
        "example":           "Barcelona under Pep Guardiola used tiki-taka to dominate possession and exhaust opponents.",
        "animation_idea":    "Watch the ball hop between all 5 players — nobody holds it more than a second. The opponent has to chase it around the whole pitch until they're too tired to defend properly. It's not showboating, it's exhaustion as a strategy.",
    },
    "total football": {
        "definition":        "A tactical theory in which any outfield player can take over the role of any other player, requiring universal positional flexibility.",
        "simple_explanation":"Every player is comfortable in any position — if someone moves, a teammate fills the gap.",
        "example":           "The Netherlands team of the 1970s rotated positions fluidly under coach Rinus Michels.",
        "animation_idea":    "Watch the CB and ST swap positions entirely. If every player can be anyone, the opponent can't mark anyone specifically. There are no fixed roles — just players filling space intelligently.",
    },
    "positional play": {
        "definition":        "A tactical system focused on controlling the game through intelligent positioning, occupying key spaces to dominate the pitch.",
        "simple_explanation":"Players take up smart positions to control space and make the team impossible to press.",
        "example":           "Manchester City under Guardiola use positional play to maintain passing structures at all times.",
        "animation_idea":    "It's not just about having the ball — it's about where you are when you have it. Players occupy specific zones so there's always a passing option. The opponent can't press because there's always someone open.",
    },
    "overload": {
        "definition":        "Creating a numerical superiority of attackers over defenders in a specific zone of the pitch.",
        "simple_explanation":"Getting 3 players against 2 defenders in one area to create a free man.",
        "example":           "A team overloads the left flank with 3 attackers, pulling the defence over to leave the right side open.",
        "animation_idea":    "Three attacker dots surrounding two defender dots in one side of the pitch.",
    },
    "third man run": {
        "definition":        "When a team is attacking, a third player makes a run to become an alternative receiver beyond the initial passer and receiver.",
        "simple_explanation":"Two players exchange the ball while a third runs beyond them to receive it in space.",
        "example":           "A midfielder passes to a striker, the striker lays it off, and the midfielder continues their run to receive.",
        "animation_idea":    "P1 passes to P2, P2 lays it off — but watch P3 who never stopped running. Defenders track the ball, not the run. P3 is always free because nobody remembered to follow them.",
    },
    "line-breaking pass": {
        "definition":        "A pass that bypasses one or more lines of opposing players in a single movement, skipping midfield or the defensive line.",
        "simple_explanation":"A pass that jumps over a whole line of opponents in one go, instantly unlocking the defence.",
        "example":           "A centre-back plays a line-breaking pass into the feet of a forward between the midfield and defensive lines.",
        "animation_idea":    "One pass, entire defence beaten. Watch the ball travel straight through both lines of defenders — it skips midfield entirely and lands at P2's feet with the whole defence behind them.",
    },
}

TEAM_STYLES = {
    "Paris Saint-Germain": "PSG applies an intense <b>high press</b> to win the ball back in the opposition's half, backed by immediate <b>counter-pressing</b> on every loss. In <b>build-up play</b>, the team relies on <b>positional play</b> with midfielders infiltrating the <b>half-space</b> to create openings. A <b>false nine</b> drops between the <b>lines</b> to trigger <b>line-breaking pass</b>es toward runners in behind. Wingers provide <b>width</b> to stretch the defence, and in <b>transition</b> the collective speed is the main weapon.",

    "Olympique de Marseille": "Marseille impose aggressive collective <b>pressing</b> and immediate <b>counter-pressing</b> on every loss of possession. The defensive <b>shape</b> alternates between <b>man marking</b> on ball carriers and <b>zonal marking</b> across the channels. The offensive <b>transition</b> is their main weapon — the ball travels quickly to attackers via a <b>counter-attack</b>. A quick <b>switch of play</b> frees a winger on the weak side for a <b>cross</b> into the <b>final third</b>.",

    "AS Monaco": "Monaco builds patiently through <b>build-up play</b> from the back, using <b>depth</b> to progress in stages. Full-backs make constant <b>overlap</b>s to create overloads on the wings, while midfielders offer <b>underlap</b>s through the <b>half-space</b>. Well-timed <b>third man run</b>s disorganize defences after quick combinations, and a precise <b>through ball</b> breaks the <b>lines</b> to release attackers. On every loss, immediate <b>counter-pressing</b> aims to recover possession before the opposition can settle.",

    "LOSC Lille": "Lille are renowned for their collective <b>high press</b> and immediate <b>counter-pressing</b> — every loss triggers a group reaction. A 4-4-2 <b>formation</b> creates a tight <b>shape</b> that compresses the spaces between the opposition's <b>lines</b>. Quick <b>line-breaking pass</b>es release attackers in <b>transition</b> after high recoveries. An <b>overload</b> on the ball side creates the numerical advantage, before a <b>switch of play</b> finds the freed side to reach the <b>final third</b>.",

    "Olympique Lyonnais": "Lyon play a style close to <b>tiki-taka</b>, with rigorous <b>positional play</b> and careful <b>build-up play</b> from the defenders. Midfielders slip into the <b>half-space</b>s to receive between the <b>lines</b>, exploiting both <b>depth</b> and <b>width</b> simultaneously. An <b>overload</b> on the ball side is followed by a quick <b>switch of play</b> to exploit the freed space. Defensively, organised <b>zonal marking</b> and a disciplined <b>low block</b> protect space in their adaptable 4-3-3 <b>formation</b>.",

    "RC Lens": "Lens stand out with intense collective <b>pressing</b> and a well-organised <b>shape</b> in a 3-4-3 <b>formation</b>. The wing-backs provide huge <b>width</b>, combining <b>overlap</b>s and <b>cross</b>es toward the <b>pivot</b> in the box. On recovery, the team immediately launches a vertical <b>counter-attack</b>. Well-timed <b>tackle</b>s and <b>interception</b>s in midfield fuel the offensive <b>transition</b>, while <b>third man run</b>s find the free player in behind.",

    "OGC Nice": "Nice play structured <b>positional play</b> with methodical <b>build-up play</b> from the centre-backs. The defensive <b>shape</b> relies on strict <b>zonal marking</b> and an organised <b>low block</b> when opponents have the ball. <b>Line-breaking pass</b>es target the <b>final third</b> by exploiting <b>depth</b> and the <b>half-space</b>s. A quick <b>switch of play</b> after an <b>interception</b> releases a winger on the weak side for a <b>cross</b> into the box.",

    "Stade Rennais": "Rennes embrace a philosophy close to <b>total football</b>, with constant rotations across all positions. Their <b>positional play</b> demands great <b>depth</b> and <b>width</b> to occupy the full pitch simultaneously. Frequent <b>third man run</b>s disorganize opposition defences, while full-back <b>overlap</b>s create overloads on the flanks. In <b>transition</b>, immediate <b>counter-pressing</b> prevents clean restarts, before a <b>through ball</b> finds an attacker running into the <b>half-space</b>.",

    "RC Strasbourg": "Strasbourg favour a direct game with a high <b>pressing</b> approach to force opposition errors. Wingers constantly seek <b>width</b> to deliver <b>cross</b>es toward the <b>pivot</b> or late-arriving midfielders. Defensively, a compact <b>shape</b> with an organised <b>low block</b> absorbs pressure well. <b>Tackle</b>s and <b>interception</b>s in midfield fuel direct <b>counter-attack</b>s, taking advantage of space left behind in <b>transition</b>. A quick <b>switch of play</b> allows them to change the axis of attack.",

    "Toulouse FC": "Toulouse build patiently with clean <b>build-up play</b> and rigorous <b>positional play</b> from the back. Midfielders drift into the <b>half-space</b>s to receive between the opposition's <b>lines</b> and play <b>through ball</b>s to attackers. The <b>formation</b> is flexible — defensively, strict <b>zonal marking</b> covers the channels and central zones. <b>Line-breaking pass</b>es trigger runs in behind toward the <b>final third</b>, and in negative <b>transition</b> the team reorganises quickly.",

    "Stade Brestois": "Brest rely on a very compact <b>low block</b> that limits space in their own <b>final third</b> by tightening the <b>lines</b>. The defensive <b>shape</b> combines strict <b>man marking</b> on ball carriers with <b>zonal marking</b> across dangerous zones. Frequent <b>tackle</b>s and <b>interception</b>s fuel rapid <b>counter-attack</b>s on the flanks. In offensive <b>transition</b>, the <b>pivot</b> acts as a relay to distribute in <b>width</b> and exploit open space. Targeted <b>pressing</b> can surprise opponents in their own build-up.",

    "FC Nantes": "Nantes build around a <b>cross</b>-heavy game from the flanks, with full-backs making regular <b>overlap</b>s to deliver into the box. The <b>pivot</b> is central to the attacking system in the <b>final third</b>, acting as a target for late-arriving midfielders. Defensively, collective <b>pressing</b> covers the whole pitch with <b>man marking</b> targeting the opposition's key players. A quick <b>switch of play</b> frees the weak side, while robust <b>tackle</b>s feed offensive <b>transition</b>s.",

    "Angers SCO": "Angers employ a defensive <b>low block</b> that tightens space between the <b>lines</b> and denies the opposition room in the <b>final third</b>. The compact <b>shape</b> is built on organised <b>zonal marking</b> across the whole pitch. Well-placed <b>interception</b>s and clean <b>tackle</b>s trigger <b>counter-attack</b>s in <b>transition</b>. <b>Pressing</b> is targeted rather than systematic — used only when the opponent is under pressure. In possession, cautious <b>build-up play</b> keeps the ball and avoids risk.",

    "Le Havre AC": "Le Havre rely on a rigorous <b>low block</b> to protect their goal and limit space in behind. The defensive <b>shape</b> combines <b>man marking</b> on attackers with <b>zonal marking</b> across the channels, maintaining strong <b>depth</b> to reduce space behind the <b>lines</b>. In offensive <b>transition</b>, players quickly seek a <b>counter-attack</b> through direct play forward. Targeted collective <b>pressing</b> can be triggered to win back possession in midfield.",

    "AJ Auxerre": "Auxerre use a dynamic <b>high press</b> to win the ball high up the pitch, backed by immediate <b>counter-pressing</b>. In possession, they create <b>overload</b>s on one side before using a <b>switch of play</b> to exploit the freed space. <b>Line-breaking pass</b>es and <b>through ball</b>s quickly reach the opposition's <b>final third</b>. Well-coordinated <b>third man run</b>s break opposition <b>lines</b>, and clean <b>build-up play</b> from the back allows them to restart calmly.",

    "FC Metz": "Metz rely on a very compact <b>low block</b> to absorb pressure and protect space in their <b>final third</b>. The defensive <b>shape</b> features strict <b>man marking</b> to neutralise the opposition's key players. Well-placed <b>tackle</b>s and <b>interception</b>s feed direct <b>counter-attack</b>s in <b>transition</b>. <b>Pressing</b> is used selectively in priority recovery zones. Offensively, a <b>cross</b>-based game from the wings looks to exploit the <b>pivot</b> in the box.",

    "Paris FC": "Paris FC build their game on careful <b>build-up play</b>, with midfielders positioning in the <b>half-space</b>s to progress between opposition <b>lines</b>. Their <b>positional play</b> philosophy relies on great <b>width</b> and controlled <b>depth</b> to occupy all available space. <b>Line-breaking pass</b>es reach attackers in the <b>final third</b>, followed by <b>through ball</b>s to runners in behind. A compact <b>shape</b> and well-calibrated <b>pressing</b> maintain defensive organisation, with quick <b>transition</b> in both directions.",

    "FC Lorient": "Lorient rely on collective <b>pressing</b> to win the ball back in midfield, with a well-organised defensive <b>shape</b>. The <b>counter-attack</b> is their main offensive weapon — after an <b>interception</b> or <b>tackle</b>, the ball moves quickly to the attackers. A rapid <b>switch of play</b> exploits the space left by the opposition in <b>transition</b>. Full-back <b>overlap</b>s followed by <b>cross</b>es into the box are the favourite attacking combination. In deep defence, a solid <b>low block</b> protects space in the <b>final third</b>.",
}
DEFAULT_STYLE = "Playing style to be documented."
WATCH_COLORS = ["#7CC99A", "#F5D06E", "#F2827F"]

# ── Tactical pitch data ───────────────────────────────────────────────────────
# Each team: formation, color, style_tags, players [(x%, y%, label)],
# moves [(player_idx, to_x%, to_y%)], zones [(cx%, cy%, rx%, ry%, opacity)]
TEAM_TACTICS = {
    "Paris Saint-Germain": {
        "formation": "4-3-3", "color": "#004E9A",
        "style_tags": ["High Press", "False 9", "Possession"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,55,"LCM"),(50,50,"CM"),(72,55,"RCM"),
            (16,24,"LW"),(50,18,"CF"),(84,24,"RW"),
        ],
        "moves": [(1,12,42),(9,50,34),(5,22,38),(10,72,18)],
        "zones": [(50,22,35,16,0.18),(16,42,14,22,0.12),(84,42,14,22,0.12),(50,52,22,12,0.10)],
    },
    "Olympique de Marseille": {
        "formation": "4-2-3-1", "color": "#2490D8",
        "style_tags": ["High Press", "Aggressive", "Direct"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (20,42,"LAM"),(50,38,"CAM"),(80,42,"RAM"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(7,14,58),(9,86,58),(1,16,52)],
        "zones": [(50,18,30,14,0.18),(50,42,40,16,0.12),(50,62,30,12,0.10)],
    },
    "AS Monaco": {
        "formation": "4-4-2", "color": "#E5273D",
        "style_tags": ["Overlaps", "Fluid", "Counter"],
        "players": [
            (50,93,"GK"),(15,74,"LB"),(38,78,"CB"),(62,78,"CB"),(85,74,"RB"),
            (18,52,"LM"),(38,52,"CM"),(62,52,"CM"),(82,52,"RM"),
            (35,22,"ST"),(65,22,"ST"),
        ],
        "moves": [(1,10,40),(4,88,40),(7,72,32),(9,28,32)],
        "zones": [(14,44,12,28,0.14),(86,44,12,28,0.14),(50,22,32,16,0.16)],
    },
    "LOSC Lille": {
        "formation": "4-4-2", "color": "#CC0000",
        "style_tags": ["Collective Press", "Compact", "Physical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (18,54,"LM"),(38,54,"CM"),(62,54,"CM"),(82,54,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(9,38,34),(10,62,34),(5,32,38),(6,68,38)],
        "zones": [(50,54,42,14,0.14),(50,22,36,18,0.16),(50,72,50,14,0.10)],
    },
    "Olympique Lyonnais": {
        "formation": "4-3-3", "color": "#1357BE",
        "style_tags": ["Tiki-Taka", "Positional", "Technical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,26,"LW"),(50,20,"CF"),(84,26,"RW"),
        ],
        "moves": [(5,20,40),(7,80,40),(8,26,38),(10,74,38)],
        "zones": [(26,46,14,20,0.14),(74,46,14,20,0.14),(50,52,20,12,0.12),(50,22,38,16,0.14)],
    },
    "RC Lens": {
        "formation": "3-4-3", "color": "#D4AF37",
        "style_tags": ["Wing-Backs", "High Press", "Vertical"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,54,"LWB"),(35,54,"CM"),(65,54,"CM"),(90,54,"RWB"),
            (22,24,"LW"),(50,18,"CF"),(78,24,"RW"),
        ],
        "moves": [(4,8,34),(7,92,34),(8,16,34),(10,84,34)],
        "zones": [(10,40,10,30,0.16),(90,40,10,30,0.16),(50,20,36,16,0.16)],
    },
    "OGC Nice": {
        "formation": "4-3-3", "color": "#CC0000",
        "style_tags": ["Positional", "Structured", "Patient"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,58,"LCM"),(50,52,"CM"),(72,58,"RCM"),
            (18,28,"LW"),(50,22,"CF"),(82,28,"RW"),
        ],
        "moves": [(6,50,44),(9,50,32),(3,84,62),(1,14,62)],
        "zones": [(50,58,40,14,0.12),(50,76,55,12,0.10),(50,26,34,16,0.14)],
    },
    "Stade Rennais": {
        "formation": "4-3-3", "color": "#CC0000",
        "style_tags": ["Total Football", "Rotations", "Overlaps"],
        "players": [
            (50,93,"GK"),(16,74,"LB"),(36,78,"CB"),(64,78,"CB"),(84,74,"RB"),
            (26,56,"LCM"),(50,50,"CM"),(74,56,"RCM"),
            (16,26,"LW"),(50,20,"CF"),(84,26,"RW"),
        ],
        "moves": [(1,12,38),(8,28,38),(4,88,38),(10,72,38)],
        "zones": [(50,24,40,18,0.14),(16,44,14,24,0.12),(84,44,14,24,0.12)],
    },
    "RC Strasbourg": {
        "formation": "4-2-3-1", "color": "#003F8A",
        "style_tags": ["Direct", "Cross-Heavy", "Physical"],
        "players": [
            (50,93,"GK"),(16,74,"LB"),(38,78,"CB"),(62,78,"CB"),(84,74,"RB"),
            (36,62,"DM"),(64,62,"DM"),
            (16,40,"LW"),(50,36,"AM"),(84,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(1,10,42),(4,90,42),(7,12,52),(9,88,52)],
        "zones": [(14,42,12,28,0.14),(86,42,12,28,0.14),(50,20,36,16,0.16)],
    },
    "Toulouse FC": {
        "formation": "4-3-3", "color": "#7C1C6A",
        "style_tags": ["Patient Build-Up", "Technical", "Positional"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (28,57,"LCM"),(50,51,"CM"),(72,57,"RCM"),
            (18,27,"LW"),(50,21,"CF"),(82,27,"RW"),
        ],
        "moves": [(5,22,40),(7,78,40),(6,50,36),(9,50,30)],
        "zones": [(26,48,14,20,0.14),(74,48,14,20,0.14),(50,60,38,12,0.10),(50,24,36,16,0.14)],
    },
    "Stade Brestois": {
        "formation": "3-5-2", "color": "#CC0000",
        "style_tags": ["Low Block", "Counter", "Compact"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,58,"LWB"),(32,58,"CM"),(50,54,"CM"),(68,58,"CM"),(90,58,"RWB"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(4,12,42),(8,88,42),(9,40,34),(10,60,34)],
        "zones": [(50,62,50,16,0.14),(50,78,55,10,0.10),(50,24,32,14,0.12)],
    },
    "FC Nantes": {
        "formation": "4-4-2", "color": "#F0A500",
        "style_tags": ["Cross-Based", "Physical", "Flanks"],
        "players": [
            (50,93,"GK"),(15,74,"LB"),(38,78,"CB"),(62,78,"CB"),(85,74,"RB"),
            (15,52,"LM"),(38,52,"CM"),(62,52,"CM"),(85,52,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(1,10,36),(4,90,36),(5,10,42),(8,90,42)],
        "zones": [(12,46,12,30,0.14),(88,46,12,30,0.14),(50,20,36,16,0.14)],
    },
    "Angers SCO": {
        "formation": "4-4-2", "color": "#1A1A1A",
        "style_tags": ["Low Block", "Counter", "Defensive"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,56,"LM"),(38,56,"CM"),(62,56,"CM"),(82,56,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,40,34),(10,60,34),(5,32,44),(6,68,44)],
        "zones": [(50,60,50,16,0.14),(50,78,55,10,0.10),(50,24,34,14,0.10)],
    },
    "Le Havre AC": {
        "formation": "4-4-2", "color": "#0050A0",
        "style_tags": ["Low Block", "Defensive", "Compact"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(5,34,44),(6,66,44),(9,36,32),(10,64,32)],
        "zones": [(50,62,52,16,0.14),(50,78,55,10,0.12),(50,26,30,12,0.10)],
    },
    "AJ Auxerre": {
        "formation": "4-3-3", "color": "#0050A0",
        "style_tags": ["High Press", "Vertical", "Dynamic"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (18,26,"LW"),(50,20,"CF"),(82,26,"RW"),
        ],
        "moves": [(8,20,38),(9,50,32),(10,80,38),(6,50,36)],
        "zones": [(50,22,38,18,0.18),(50,50,38,14,0.10)],
    },
    "FC Metz": {
        "formation": "4-4-2", "color": "#8B0000",
        "style_tags": ["Low Block", "Cross-Based", "Physical"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(1,12,42),(4,88,42),(9,36,32),(10,64,32)],
        "zones": [(50,62,52,14,0.14),(12,48,12,26,0.12),(88,48,12,26,0.12),(50,26,34,14,0.12)],
    },
    "Paris FC": {
        "formation": "4-3-3", "color": "#003F8A",
        "style_tags": ["Build-Up", "Positional", "Technical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (18,26,"LW"),(50,20,"CF"),(82,26,"RW"),
        ],
        "moves": [(5,22,40),(7,78,40),(9,50,30),(6,50,36)],
        "zones": [(26,48,14,20,0.12),(74,48,14,20,0.12),(50,24,36,16,0.14)],
    },
    "FC Lorient": {
        "formation": "4-4-2", "color": "#FF6600",
        "style_tags": ["Counter-Attack", "Flanks", "Low Block"],
        "players": [
            (50,93,"GK"),(15,74,"LB"),(38,78,"CB"),(62,78,"CB"),(85,74,"RB"),
            (15,53,"LM"),(38,53,"CM"),(62,53,"CM"),(85,53,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(1,10,40),(4,90,40),(9,34,30),(10,66,30)],
        "zones": [(12,46,12,28,0.14),(88,46,12,28,0.14),(50,22,34,14,0.14)],
    },

    # ── Premier League ────────────────────────────────────────────────────────
    "Arsenal FC": {
        "formation": "4-3-3", "color": "#EF0107",
        "style_tags": ["High Press", "Possession", "Vertical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,55,"LCM"),(50,49,"CM"),(72,55,"RCM"),
            (16,24,"LW"),(50,18,"CF"),(84,24,"RW"),
        ],
        "moves": [(8,20,38),(9,50,32),(10,80,38),(5,20,40)],
        "zones": [(50,20,36,16,0.18),(50,50,38,14,0.12),(16,42,14,22,0.12),(84,42,14,22,0.12)],
    },
    "Manchester City FC": {
        "formation": "4-3-3", "color": "#6CABDD",
        "style_tags": ["Positional Play", "False 9", "Possession"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,55,"LCM"),(50,49,"CM"),(72,55,"RCM"),
            (16,24,"LW"),(50,18,"CF"),(84,24,"RW"),
        ],
        "moves": [(1,12,42),(9,50,34),(5,22,38),(7,80,40)],
        "zones": [(26,46,14,20,0.14),(74,46,14,20,0.14),(50,52,22,12,0.12),(50,20,38,16,0.16)],
    },
    "Liverpool FC": {
        "formation": "4-3-3", "color": "#C8102E",
        "style_tags": ["Gegenpressing", "High Press", "Vertical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,55,"LCM"),(50,49,"CM"),(72,55,"RCM"),
            (16,24,"LW"),(50,18,"CF"),(84,24,"RW"),
        ],
        "moves": [(9,50,30),(8,20,36),(10,80,36),(6,50,36)],
        "zones": [(50,20,38,16,0.18),(50,50,40,14,0.14),(16,40,14,20,0.12),(84,40,14,20,0.12)],
    },
    "Chelsea FC": {
        "formation": "4-2-3-1", "color": "#034694",
        "style_tags": ["Possession", "Structured", "Technical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (20,42,"LAM"),(50,38,"CAM"),(80,42,"RAM"),(50,18,"CF"),
        ],
        "moves": [(5,22,48),(7,78,48),(10,50,28),(3,84,62)],
        "zones": [(50,40,40,16,0.14),(50,18,30,14,0.16),(26,48,14,20,0.12),(74,48,14,20,0.12)],
    },
    "Tottenham Hotspur FC": {
        "formation": "4-3-3", "color": "#132257",
        "style_tags": ["Counter-Attack", "Vertical", "Press"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,50,30),(5,20,40),(7,80,40),(1,14,44)],
        "zones": [(50,22,34,16,0.16),(50,52,38,14,0.10),(16,42,12,20,0.12),(84,42,12,20,0.12)],
    },
    "Manchester United FC": {
        "formation": "4-2-3-1", "color": "#DA291C",
        "style_tags": ["Direct", "Counter", "Physical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (20,42,"LAM"),(50,38,"CAM"),(80,42,"RAM"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(1,14,50),(4,86,50),(7,78,54)],
        "zones": [(50,20,32,14,0.14),(50,62,50,16,0.12),(50,40,40,14,0.10)],
    },
    "Newcastle United FC": {
        "formation": "4-3-3", "color": "#241F20",
        "style_tags": ["Organized", "Counter", "Physical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,50,30),(1,12,44),(4,88,44),(10,76,20)],
        "zones": [(50,56,42,14,0.12),(50,22,34,14,0.14),(12,46,12,26,0.12),(88,46,12,26,0.12)],
    },
    "Aston Villa FC": {
        "formation": "4-2-3-1", "color": "#95BFE5",
        "style_tags": ["Pressing", "Possession", "Attacking"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(7,14,58),(9,86,58),(10,50,28),(5,28,50)],
        "zones": [(50,20,34,14,0.16),(50,40,42,16,0.12),(50,62,38,12,0.10)],
    },
    "Brighton & Hove Albion FC": {
        "formation": "4-2-3-1", "color": "#0057B8",
        "style_tags": ["Positional Play", "Technical", "Press"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(5,28,50),(7,72,50),(10,50,28),(6,60,52)],
        "zones": [(26,46,14,20,0.14),(74,46,14,20,0.14),(50,20,34,14,0.16),(50,62,38,12,0.10)],
    },
    "West Ham United FC": {
        "formation": "4-2-3-1", "color": "#7A263A",
        "style_tags": ["Direct", "Physical", "Counter"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(1,12,50),(4,88,50),(8,80,46)],
        "zones": [(50,22,32,14,0.14),(50,62,48,14,0.12),(12,44,12,26,0.12),(88,44,12,26,0.12)],
    },
    "Wolverhampton Wanderers FC": {
        "formation": "3-4-3", "color": "#FDB913",
        "style_tags": ["Counter-Attack", "Wing-Backs", "Compact"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,54,"LWB"),(35,54,"CM"),(65,54,"CM"),(90,54,"RWB"),
            (22,24,"LW"),(50,18,"CF"),(78,24,"RW"),
        ],
        "moves": [(4,8,34),(7,92,34),(9,40,32),(10,60,32)],
        "zones": [(10,40,10,30,0.16),(90,40,10,30,0.16),(50,22,34,14,0.14),(50,62,50,14,0.10)],
    },
    "Fulham FC": {
        "formation": "4-2-3-1", "color": "#CC0000",
        "style_tags": ["Structured", "Direct", "Physical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(1,12,46),(4,88,46),(9,86,42)],
        "zones": [(50,20,32,14,0.14),(50,62,48,14,0.12),(12,44,12,26,0.12)],
    },
    "AFC Bournemouth": {
        "formation": "4-4-2", "color": "#DA291C",
        "style_tags": ["Counter", "Physical", "Direct"],
        "players": [
            (50,93,"GK"),(15,74,"LB"),(38,78,"CB"),(62,78,"CB"),(85,74,"RB"),
            (15,52,"LM"),(38,52,"CM"),(62,52,"CM"),(85,52,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(9,36,32),(10,64,32),(1,10,40),(4,90,40)],
        "zones": [(50,60,50,16,0.12),(50,22,34,14,0.14),(12,46,12,28,0.12),(88,46,12,28,0.12)],
    },
    "Crystal Palace FC": {
        "formation": "4-3-3", "color": "#1B458F",
        "style_tags": ["Direct", "Counter", "Wide Play"],
        "players": [
            (50,93,"GK"),(15,74,"LB"),(38,78,"CB"),(62,78,"CB"),(85,74,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (14,24,"LW"),(50,19,"CF"),(86,24,"RW"),
        ],
        "moves": [(8,10,34),(10,90,34),(9,50,30),(1,10,42)],
        "zones": [(12,30,12,28,0.16),(88,30,12,28,0.16),(50,22,32,14,0.14),(50,60,48,14,0.10)],
    },
    "Brentford FC": {
        "formation": "4-3-3", "color": "#E30613",
        "style_tags": ["Set Pieces", "Direct", "Physical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,24,"LW"),(50,18,"CF"),(84,24,"RW"),
        ],
        "moves": [(9,50,28),(1,12,44),(4,88,44),(8,18,36)],
        "zones": [(50,20,34,14,0.14),(50,60,48,14,0.10),(12,46,12,26,0.10),(88,46,12,26,0.10)],
    },
    "Everton FC": {
        "formation": "4-4-2", "color": "#003399",
        "style_tags": ["Defensive", "Physical", "Compact"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,56,"LM"),(38,56,"CM"),(62,56,"CM"),(82,56,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,32,44),(6,68,44)],
        "zones": [(50,62,50,16,0.14),(50,78,55,10,0.12),(50,24,32,12,0.10)],
    },
    "Nottingham Forest FC": {
        "formation": "4-2-3-1", "color": "#DD0000",
        "style_tags": ["Low Block", "Counter", "Organized"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,63,"DM"),(65,63,"DM"),
            (18,42,"LW"),(50,38,"AM"),(82,42,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(5,32,50),(6,68,50),(1,14,52)],
        "zones": [(50,64,50,14,0.14),(50,78,55,10,0.10),(50,20,30,12,0.10)],
    },
    "Leicester City FC": {
        "formation": "4-2-3-1", "color": "#003090",
        "style_tags": ["Counter-Attack", "Organized", "Direct"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,63,"DM"),(65,63,"DM"),
            (18,42,"LW"),(50,38,"AM"),(82,42,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(9,50,30),(1,12,46),(4,88,46)],
        "zones": [(50,20,32,14,0.14),(50,62,50,14,0.12),(50,40,40,14,0.10)],
    },
    "Ipswich Town FC": {
        "formation": "4-2-3-1", "color": "#0044A9",
        "style_tags": ["Organized", "Physical", "Direct"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,63,"DM"),(65,63,"DM"),
            (18,42,"LW"),(50,38,"AM"),(82,42,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(1,12,48),(4,88,48),(5,28,52)],
        "zones": [(50,20,30,12,0.12),(50,64,50,14,0.12),(12,44,12,24,0.10),(88,44,12,24,0.10)],
    },
    "Southampton FC": {
        "formation": "4-4-2", "color": "#D71920",
        "style_tags": ["Organized", "Physical", "Compact"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,56,"LM"),(38,56,"CM"),(62,56,"CM"),(82,56,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,32,44),(6,68,44)],
        "zones": [(50,62,50,14,0.12),(50,78,55,10,0.10),(50,24,30,12,0.10)],
    },

    # ── La Liga ───────────────────────────────────────────────────────────────
    "Real Madrid CF": {
        "formation": "4-3-3", "color": "#00529F",
        "style_tags": ["Counter-Attack", "Possession", "Clinical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,55,"LCM"),(50,50,"CM"),(72,55,"RCM"),
            (16,24,"LW"),(50,18,"CF"),(84,24,"RW"),
        ],
        "moves": [(9,50,30),(8,18,36),(10,82,36),(5,20,40)],
        "zones": [(50,20,36,16,0.18),(16,40,14,22,0.12),(84,40,14,22,0.12),(50,52,22,12,0.10)],
    },
    "FC Barcelona": {
        "formation": "4-3-3", "color": "#A50044",
        "style_tags": ["Tiki-Taka", "Possession", "Press"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,55,"LCM"),(50,50,"CM"),(72,55,"RCM"),
            (16,24,"LW"),(50,18,"CF"),(84,24,"RW"),
        ],
        "moves": [(5,20,40),(7,80,40),(8,26,36),(10,74,36)],
        "zones": [(26,46,14,20,0.14),(74,46,14,20,0.14),(50,52,22,12,0.12),(50,20,38,16,0.16)],
    },
    "Club Atlético de Madrid": {
        "formation": "4-4-2", "color": "#CB3524",
        "style_tags": ["Low Block", "Counter", "Physical"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,56,"LM"),(38,56,"CM"),(62,56,"CM"),(82,56,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(9,40,32),(10,60,32),(5,34,44),(6,66,44)],
        "zones": [(50,62,52,16,0.14),(50,78,55,12,0.12),(50,22,34,12,0.10)],
    },
    "Athletic Club": {
        "formation": "4-2-3-1", "color": "#EE2523",
        "style_tags": ["High Press", "Physical", "Vertical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(7,14,56),(9,86,56),(6,60,50)],
        "zones": [(50,20,34,14,0.16),(50,42,42,16,0.12),(50,62,38,12,0.10)],
    },
    "Real Sociedad de Fútbol": {
        "formation": "4-3-3", "color": "#0066CC",
        "style_tags": ["Positional Play", "Technical", "Possession"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(5,22,40),(7,78,40),(6,50,36),(9,50,30)],
        "zones": [(26,48,14,20,0.14),(74,48,14,20,0.14),(50,24,36,16,0.14),(50,52,20,12,0.10)],
    },
    "Villarreal CF": {
        "formation": "4-3-3", "color": "#FDD30F",
        "style_tags": ["Positional Play", "Technical", "Counter"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,50,30),(5,20,40),(7,80,40),(1,12,44)],
        "zones": [(50,22,34,14,0.14),(26,46,14,20,0.12),(74,46,14,20,0.12),(50,52,20,12,0.10)],
    },
    "Real Betis Balompié": {
        "formation": "4-2-3-1", "color": "#00954C",
        "style_tags": ["Possession", "Technical", "Positional"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(5,28,50),(7,72,50),(10,50,28),(6,60,52)],
        "zones": [(26,46,14,20,0.14),(74,46,14,20,0.14),(50,20,34,14,0.16)],
    },
    "Sevilla FC": {
        "formation": "4-2-3-1", "color": "#D4161B",
        "style_tags": ["Organized", "Counter", "Physical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,63,"DM"),(65,63,"DM"),
            (18,42,"LW"),(50,38,"AM"),(82,42,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(1,14,52),(4,86,52),(5,28,52)],
        "zones": [(50,20,32,14,0.14),(50,62,48,14,0.12),(50,40,40,14,0.10)],
    },
    "Valencia CF": {
        "formation": "4-4-2", "color": "#FB5B1F",
        "style_tags": ["Counter", "Direct", "Compact"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,56,"LM"),(38,56,"CM"),(62,56,"CM"),(82,56,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(1,12,44),(4,88,44)],
        "zones": [(50,62,50,14,0.14),(12,46,12,28,0.12),(88,46,12,28,0.12),(50,22,32,12,0.10)],
    },
    "Girona FC": {
        "formation": "4-3-3", "color": "#9D1F2A",
        "style_tags": ["High Press", "Technical", "Vertical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(8,20,38),(9,50,32),(10,80,38),(6,50,36)],
        "zones": [(50,22,36,14,0.16),(50,50,38,14,0.10),(16,40,12,20,0.12)],
    },
    "CA Osasuna": {
        "formation": "4-4-2", "color": "#1B3E6C",
        "style_tags": ["Defensive", "Physical", "Compact"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,56,"LM"),(38,56,"CM"),(62,56,"CM"),(82,56,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,44),(6,66,44)],
        "zones": [(50,62,52,14,0.14),(50,78,55,10,0.10),(50,24,30,12,0.10)],
    },
    "Rayo Vallecano de Madrid": {
        "formation": "4-2-3-1", "color": "#CC0000",
        "style_tags": ["High Press", "Direct", "Physical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,63,"DM"),(65,63,"DM"),
            (18,42,"LW"),(50,38,"AM"),(82,42,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(7,14,58),(9,86,58),(1,16,52)],
        "zones": [(50,20,32,14,0.16),(50,42,42,16,0.12),(50,64,48,14,0.10)],
    },
    "RC Celta de Vigo": {
        "formation": "4-2-3-1", "color": "#75AADB",
        "style_tags": ["Technical", "Possession", "Attacking"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(5,28,50),(7,72,50),(10,50,28),(8,80,46)],
        "zones": [(50,20,34,14,0.14),(26,44,14,20,0.12),(74,44,14,20,0.12)],
    },
    "RCD Mallorca": {
        "formation": "4-4-2", "color": "#CC0000",
        "style_tags": ["Low Block", "Counter", "Physical"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,56,"LM"),(38,56,"CM"),(62,56,"CM"),(82,56,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,44),(6,66,44)],
        "zones": [(50,62,52,14,0.14),(50,78,55,10,0.10),(50,24,30,12,0.10)],
    },
    "UD Las Palmas": {
        "formation": "4-3-3", "color": "#F7C000",
        "style_tags": ["Technical", "Possession", "Attacking"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(5,22,40),(7,78,40),(9,50,30),(6,50,36)],
        "zones": [(26,46,14,20,0.12),(74,46,14,20,0.12),(50,22,34,14,0.14)],
    },
    "Deportivo Alavés": {
        "formation": "4-4-2", "color": "#1B3A6B",
        "style_tags": ["Defensive", "Physical", "Organized"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,56,"LM"),(38,56,"CM"),(62,56,"CM"),(82,56,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,44),(6,66,44)],
        "zones": [(50,62,52,14,0.12),(50,78,55,10,0.10),(50,24,28,12,0.10)],
    },
    "Getafe CF": {
        "formation": "4-4-2", "color": "#0055A4",
        "style_tags": ["Low Block", "Physical", "Counter"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,46),(6,66,46)],
        "zones": [(50,64,52,14,0.14),(50,78,55,10,0.12),(50,26,28,12,0.10)],
    },
    "Leganés": {
        "formation": "4-4-2", "color": "#005FA8",
        "style_tags": ["Defensive", "Compact", "Counter"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,46),(6,66,46)],
        "zones": [(50,64,52,14,0.14),(50,78,55,10,0.12),(50,26,28,12,0.10)],
    },
    "Real Valladolid CF": {
        "formation": "4-4-2", "color": "#7B1FA2",
        "style_tags": ["Defensive", "Compact", "Counter"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,46),(6,66,46)],
        "zones": [(50,64,52,14,0.12),(50,78,55,10,0.10),(50,26,28,12,0.10)],
    },
    "Espanyol": {
        "formation": "4-2-3-1", "color": "#003DA5",
        "style_tags": ["Organized", "Counter", "Physical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,63,"DM"),(65,63,"DM"),
            (18,42,"LW"),(50,38,"AM"),(82,42,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(1,14,50),(4,86,50),(5,30,52)],
        "zones": [(50,20,30,12,0.12),(50,62,48,14,0.12),(50,40,40,14,0.10)],
    },

    # ── Bundesliga ────────────────────────────────────────────────────────────
    "Bayer 04 Leverkusen": {
        "formation": "3-4-3", "color": "#E32221",
        "style_tags": ["Gegenpressing", "Possession", "Vertical"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,54,"LWB"),(35,54,"CM"),(65,54,"CM"),(90,54,"RWB"),
            (22,24,"LW"),(50,18,"CF"),(78,24,"RW"),
        ],
        "moves": [(4,8,34),(7,92,34),(9,50,30),(8,18,34)],
        "zones": [(10,40,10,30,0.16),(90,40,10,30,0.16),(50,20,36,16,0.18),(50,52,40,14,0.10)],
    },
    "FC Bayern München": {
        "formation": "4-2-3-1", "color": "#DC052D",
        "style_tags": ["High Press", "Possession", "Dominant"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(1,12,42),(4,88,42),(10,50,28),(7,14,58)],
        "zones": [(50,20,38,16,0.18),(16,42,14,22,0.12),(84,42,14,22,0.12),(50,40,40,14,0.12)],
    },
    "VfB Stuttgart": {
        "formation": "4-3-3", "color": "#E32221",
        "style_tags": ["High Press", "Attacking", "Vertical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(8,20,38),(9,50,30),(10,80,38),(5,22,40)],
        "zones": [(50,22,36,14,0.16),(50,50,38,14,0.10),(16,40,12,20,0.12),(84,40,12,20,0.12)],
    },
    "RB Leipzig": {
        "formation": "4-3-3", "color": "#DD0741",
        "style_tags": ["Gegenpressing", "Vertical", "High Press"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,55,"LCM"),(50,49,"CM"),(72,55,"RCM"),
            (16,24,"LW"),(50,18,"CF"),(84,24,"RW"),
        ],
        "moves": [(9,50,30),(8,20,36),(10,80,36),(6,50,36)],
        "zones": [(50,20,38,16,0.18),(50,50,40,14,0.14),(16,40,14,20,0.12),(84,40,14,20,0.12)],
    },
    "Borussia Dortmund": {
        "formation": "4-2-3-1", "color": "#FDE100",
        "style_tags": ["High Press", "Vertical", "Counter"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(7,14,56),(9,86,56),(1,14,44)],
        "zones": [(50,20,36,16,0.16),(16,42,14,22,0.12),(84,42,14,22,0.12),(50,62,38,12,0.10)],
    },
    "Eintracht Frankfurt": {
        "formation": "3-4-3", "color": "#E1000F",
        "style_tags": ["Gegenpressing", "Direct", "Physical"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,54,"LWB"),(35,54,"CM"),(65,54,"CM"),(90,54,"RWB"),
            (22,24,"LW"),(50,18,"CF"),(78,24,"RW"),
        ],
        "moves": [(4,8,34),(7,92,34),(9,50,30),(10,78,24)],
        "zones": [(10,40,10,30,0.14),(90,40,10,30,0.14),(50,20,34,16,0.16),(50,54,42,14,0.10)],
    },
    "TSG 1899 Hoffenheim": {
        "formation": "4-3-3", "color": "#1C63B7",
        "style_tags": ["Positional Play", "Technical", "Possession"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(5,22,40),(7,78,40),(6,50,36),(9,50,30)],
        "zones": [(26,48,14,20,0.12),(74,48,14,20,0.12),(50,22,34,14,0.14)],
    },
    "SV Werder Bremen": {
        "formation": "4-2-3-1", "color": "#1D7141",
        "style_tags": ["Pressing", "Technical", "Attacking"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(7,14,56),(9,86,56),(5,28,50)],
        "zones": [(50,20,34,14,0.16),(50,40,42,16,0.12),(50,62,40,12,0.10)],
    },
    "Sport-Club Freiburg": {
        "formation": "4-3-3", "color": "#CC0000",
        "style_tags": ["Organized", "Pressing", "Compact"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,38,34),(10,62,34),(5,32,40),(6,68,40)],
        "zones": [(50,54,42,14,0.14),(50,22,34,14,0.12),(50,76,52,12,0.10)],
    },
    "VfL Wolfsburg": {
        "formation": "4-2-3-1", "color": "#65B32E",
        "style_tags": ["Organized", "Counter", "Physical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,63,"DM"),(65,63,"DM"),
            (18,42,"LW"),(50,38,"AM"),(82,42,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(1,14,50),(4,86,50),(5,30,52)],
        "zones": [(50,20,30,12,0.12),(50,62,48,14,0.12),(50,40,40,14,0.10)],
    },
    "Borussia Mönchengladbach": {
        "formation": "4-3-3", "color": "#000000",
        "style_tags": ["Pressing", "Technical", "Attacking"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(8,20,38),(9,50,30),(10,80,38),(5,22,40)],
        "zones": [(50,22,36,14,0.14),(50,50,38,14,0.10),(16,42,12,20,0.10),(84,42,12,20,0.10)],
    },
    "FSV Mainz 05": {
        "formation": "4-4-2", "color": "#CC0000",
        "style_tags": ["Pressing", "Physical", "Compact"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,56,"LM"),(38,56,"CM"),(62,56,"CM"),(82,56,"RM"),
            (36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(9,38,34),(10,62,34),(5,32,44),(6,68,44)],
        "zones": [(50,54,42,14,0.14),(50,22,34,16,0.14),(50,72,50,12,0.10)],
    },
    "VfL Bochum 1848": {
        "formation": "4-4-2", "color": "#003F8A",
        "style_tags": ["Physical", "Defensive", "Direct"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,46),(6,66,46)],
        "zones": [(50,64,52,14,0.14),(50,78,55,10,0.10),(50,26,28,12,0.10)],
    },
    "1. FC Heidenheim 1846": {
        "formation": "4-4-2", "color": "#CC0000",
        "style_tags": ["Compact", "Counter", "Physical"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,46),(6,66,46)],
        "zones": [(50,64,52,14,0.12),(50,78,55,10,0.10),(50,26,28,12,0.10)],
    },
    "FC Augsburg": {
        "formation": "4-4-2", "color": "#007A5E",
        "style_tags": ["Physical", "Direct", "Counter"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(1,12,44),(4,88,44)],
        "zones": [(50,60,50,14,0.12),(12,46,12,28,0.12),(88,46,12,28,0.12),(50,26,30,12,0.10)],
    },
    "FC Union Berlin": {
        "formation": "3-5-2", "color": "#CC0000",
        "style_tags": ["Physical", "Set Pieces", "Counter"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,58,"LWB"),(32,58,"CM"),(50,54,"CM"),(68,58,"CM"),(90,58,"RWB"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(4,12,42),(8,88,42),(9,40,32),(10,60,32)],
        "zones": [(50,64,50,14,0.12),(50,78,55,10,0.10),(12,46,10,28,0.12),(88,46,10,28,0.12)],
    },
    "Hamburger SV": {
        "formation": "4-2-3-1", "color": "#009EE0",
        "style_tags": ["Organized", "Attacking", "Technical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(5,28,50),(7,72,50),(1,14,44)],
        "zones": [(50,20,34,14,0.14),(50,40,42,16,0.12),(50,62,40,12,0.10)],
    },
    "Werder Bremen": {
        "formation": "4-2-3-1", "color": "#1D7141",
        "style_tags": ["Pressing", "Technical", "Attacking"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(7,14,56),(9,86,56),(5,28,50)],
        "zones": [(50,20,34,14,0.14),(50,40,42,14,0.10),(50,62,40,12,0.10)],
    },

    # ── Serie A ───────────────────────────────────────────────────────────────
    "Inter Milan": {
        "formation": "3-5-2", "color": "#010E80",
        "style_tags": ["Organized", "Wing-Backs", "Counter"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,55,"LWB"),(32,55,"CM"),(50,51,"CM"),(68,55,"CM"),(90,55,"RWB"),
            (35,22,"ST"),(65,22,"ST"),
        ],
        "moves": [(4,8,36),(8,92,36),(9,38,30),(10,62,30)],
        "zones": [(10,42,10,28,0.16),(90,42,10,28,0.16),(50,22,36,16,0.16),(50,56,42,14,0.10)],
    },
    "Juventus FC": {
        "formation": "3-5-2", "color": "#000000",
        "style_tags": ["Organized", "Defensive", "Counter"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,56,"LWB"),(32,56,"CM"),(50,52,"CM"),(68,56,"CM"),(90,56,"RWB"),
            (35,22,"ST"),(65,22,"ST"),
        ],
        "moves": [(4,12,42),(8,88,42),(9,38,32),(10,62,32)],
        "zones": [(50,62,52,14,0.14),(50,78,55,10,0.12),(10,44,10,28,0.12),(90,44,10,28,0.12)],
    },
    "AC Milan": {
        "formation": "4-2-3-1", "color": "#FB090B",
        "style_tags": ["Organized", "Counter", "Technical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(1,12,46),(4,88,46),(7,14,56)],
        "zones": [(50,20,34,14,0.16),(50,62,48,14,0.12),(16,42,14,22,0.12),(84,42,14,22,0.12)],
    },
    "SSC Napoli": {
        "formation": "4-3-3", "color": "#12A0C3",
        "style_tags": ["Possession", "Pressing", "Technical"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,55,"LCM"),(50,50,"CM"),(72,55,"RCM"),
            (16,24,"LW"),(50,18,"CF"),(84,24,"RW"),
        ],
        "moves": [(5,22,40),(7,78,40),(9,50,30),(8,20,36)],
        "zones": [(26,46,14,20,0.14),(74,46,14,20,0.14),(50,20,38,16,0.16),(50,52,22,12,0.10)],
    },
    "AS Roma": {
        "formation": "3-4-2-1", "color": "#8B0000",
        "style_tags": ["Organized", "Pressing", "Technical"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,56,"LWB"),(35,56,"CM"),(65,56,"CM"),(90,56,"RWB"),
            (30,32,"SS"),(70,32,"SS"),(50,18,"CF"),
        ],
        "moves": [(4,8,38),(7,92,38),(10,50,28),(8,28,42)],
        "zones": [(10,44,10,28,0.14),(90,44,10,28,0.14),(50,20,34,16,0.16),(50,34,38,14,0.12)],
    },
    "Atalanta BC": {
        "formation": "3-4-3", "color": "#0B00A0",
        "style_tags": ["Aggressive Press", "Vertical", "Attacking"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,54,"LWB"),(35,54,"CM"),(65,54,"CM"),(90,54,"RWB"),
            (22,24,"LW"),(50,18,"CF"),(78,24,"RW"),
        ],
        "moves": [(4,8,34),(7,92,34),(9,50,28),(8,18,34)],
        "zones": [(10,40,10,30,0.16),(90,40,10,30,0.16),(50,20,36,16,0.18),(50,52,40,14,0.12)],
    },
    "SS Lazio": {
        "formation": "4-3-3", "color": "#87D0F5",
        "style_tags": ["Technical", "Counter", "Possession"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,50,30),(5,20,40),(7,80,40),(1,14,44)],
        "zones": [(50,22,34,14,0.14),(16,42,12,20,0.12),(84,42,12,20,0.12),(50,52,20,12,0.10)],
    },
    "ACF Fiorentina": {
        "formation": "4-2-3-1", "color": "#5E1888",
        "style_tags": ["Positional Play", "Technical", "Possession"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(5,28,50),(7,72,50),(10,50,28),(6,60,52)],
        "zones": [(26,46,14,20,0.14),(74,46,14,20,0.14),(50,20,34,14,0.14)],
    },
    "Torino FC": {
        "formation": "3-4-1-2", "color": "#8B1A1A",
        "style_tags": ["Physical", "Direct", "Organized"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,57,"LWB"),(35,57,"CM"),(65,57,"CM"),(90,57,"RWB"),
            (50,38,"AM"),(36,22,"ST"),(64,22,"ST"),
        ],
        "moves": [(4,12,44),(7,88,44),(9,38,30),(10,62,30)],
        "zones": [(50,62,52,14,0.14),(10,44,10,28,0.12),(90,44,10,28,0.12),(50,22,34,14,0.12)],
    },
    "Bologna FC 1909": {
        "formation": "4-2-3-1", "color": "#003FA0",
        "style_tags": ["Pressing", "Technical", "Attacking"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(7,14,56),(9,86,56),(5,28,50)],
        "zones": [(50,20,34,14,0.14),(50,40,42,14,0.12),(50,62,40,12,0.10)],
    },
    "AC Monza": {
        "formation": "4-4-2", "color": "#CC0000",
        "style_tags": ["Defensive", "Organized", "Counter"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,46),(6,66,46)],
        "zones": [(50,64,52,14,0.12),(50,78,55,10,0.10),(50,26,28,12,0.10)],
    },
    "Udinese Calcio": {
        "formation": "3-5-2", "color": "#1A1A1A",
        "style_tags": ["Physical", "Counter", "Compact"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,58,"LWB"),(32,58,"CM"),(50,54,"CM"),(68,58,"CM"),(90,58,"RWB"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(4,12,44),(8,88,44),(9,38,32),(10,62,32)],
        "zones": [(50,64,52,14,0.12),(50,78,55,10,0.10),(10,46,10,28,0.12),(90,46,10,28,0.12)],
    },
    "Cagliari Calcio": {
        "formation": "4-4-2", "color": "#CC0000",
        "style_tags": ["Defensive", "Counter", "Physical"],
        "players": [
            (50,93,"GK"),(18,76,"LB"),(38,79,"CB"),(62,79,"CB"),(82,76,"RB"),
            (18,57,"LM"),(38,57,"CM"),(62,57,"CM"),(82,57,"RM"),
            (36,24,"ST"),(64,24,"ST"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,46),(6,66,46)],
        "zones": [(50,64,52,14,0.12),(50,78,55,10,0.10),(50,26,28,12,0.10)],
    },
    "Genoa CFC": {
        "formation": "4-3-3", "color": "#CC0000",
        "style_tags": ["Physical", "Organized", "Counter"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,44),(6,66,44)],
        "zones": [(50,62,50,14,0.12),(50,78,55,10,0.10),(50,24,30,12,0.10)],
    },
    "Hellas Verona FC": {
        "formation": "4-3-3", "color": "#002147",
        "style_tags": ["Physical", "Counter", "Direct"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,38,32),(10,62,32),(1,12,44),(4,88,44)],
        "zones": [(50,62,50,14,0.12),(50,78,55,10,0.10),(50,24,28,12,0.10)],
    },
    "US Lecce": {
        "formation": "4-3-3", "color": "#F7C000",
        "style_tags": ["Defensive", "Counter", "Compact"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,44),(6,66,44)],
        "zones": [(50,62,50,14,0.12),(50,78,55,10,0.10),(50,24,28,12,0.10)],
    },
    "Empoli FC": {
        "formation": "3-4-3", "color": "#006CB5",
        "style_tags": ["Pressing", "Technical", "Compact"],
        "players": [
            (50,93,"GK"),(28,80,"CB"),(50,82,"CB"),(72,80,"CB"),
            (10,54,"LWB"),(35,54,"CM"),(65,54,"CM"),(90,54,"RWB"),
            (22,24,"LW"),(50,18,"CF"),(78,24,"RW"),
        ],
        "moves": [(4,12,42),(8,88,42),(9,40,32),(10,60,32)],
        "zones": [(50,62,50,14,0.12),(10,42,10,28,0.12),(90,42,10,28,0.12),(50,22,32,14,0.10)],
    },
    "Venezia FC": {
        "formation": "4-3-3", "color": "#1A1A1A",
        "style_tags": ["Organized", "Counter", "Compact"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,38,32),(10,62,32),(5,34,44),(6,66,44)],
        "zones": [(50,62,50,14,0.12),(50,78,55,10,0.10),(50,24,28,12,0.10)],
    },
    "Como 1907": {
        "formation": "4-2-3-1", "color": "#003FA0",
        "style_tags": ["Technical", "Possession", "Organized"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (35,62,"DM"),(65,62,"DM"),
            (18,40,"LW"),(50,36,"AM"),(82,40,"RW"),(50,18,"CF"),
        ],
        "moves": [(10,50,28),(5,28,50),(7,72,50),(1,14,44)],
        "zones": [(50,20,32,12,0.12),(50,40,42,14,0.10),(50,62,40,12,0.10)],
    },
    "Parma Calcio 1913": {
        "formation": "4-3-3", "color": "#FADA00",
        "style_tags": ["Organized", "Physical", "Direct"],
        "players": [
            (50,93,"GK"),(18,75,"LB"),(38,78,"CB"),(62,78,"CB"),(82,75,"RB"),
            (28,56,"LCM"),(50,50,"CM"),(72,56,"RCM"),
            (16,25,"LW"),(50,19,"CF"),(84,25,"RW"),
        ],
        "moves": [(9,38,32),(10,62,32),(1,12,44),(4,88,44)],
        "zones": [(50,62,50,14,0.12),(12,46,12,26,0.10),(88,46,12,26,0.10),(50,24,28,12,0.10)],
    },
}

# ── Style tag → TACTICAL_TERMS key (for clickable pills) ─────────────────────
STYLE_TAG_TO_TERM = {
    "high press":       "high press",
    "false 9":          "false nine",
    "false nine":       "false nine",
    "possession":       "positional play",
    "counter":          "counter-attack",
    "counter-attack":   "counter-attack",
    "low block":        "low block",
    "tiki-taka":        "tiki-taka",
    "positional":       "positional play",
    "positional play":  "positional play",
    "total football":   "total football",
    "overlaps":         "overlap",
    "overlap":          "overlap",
    "pressing":         "pressing",
    "collective press": "pressing",
    "counter-pressing": "counter-pressing",
    "gegenpressing":    "counter-pressing",
    "build-up":         "build-up play",
    "patient build-up": "build-up play",
    "width":            "width",
    "flanks":           "width",
    "cross-heavy":      "cross",
    "cross-based":      "cross",
    "wing-backs":       "overlap",
    "compact":          "shape",
    "shape":            "shape",
    "formation":        "formation",
    "half-space":       "half-space",
    "transition":       "transition",
    "man marking":      "man marking",
    "zonal marking":    "zonal marking",
    "lines":            "lines",
    "depth":            "depth",
    "overload":         "overload",
    "press":            "pressing",
}

# ── First style tag → one-sentence pitch description ─────────────────────────
STYLE_TAG_GUIDE = {
    "high press": [
        ("⬆️", "Forwards push up high", "They sprint toward the opponent's defenders and GK to block every pass."),
        ("🔄", "Midfielders follow up", "They cut off all passing lanes — nowhere safe to play."),
        ("⚡", "Force a mistake", "Defenders panic, misplace passes — ball stolen near their goal."),
        ("🎯", "Why it works", "You win the ball 25m from goal — one pass and you're shooting."),
    ],
    "low block": [
        ("🛡️", "Everyone drops deep", "All 10 outfield players form two tight lines in front of their own goal."),
        ("🧱", "Close every gap", "Shoulder to shoulder — no space to pass through."),
        ("⏳", "Let them have the ball", "The opponent passes sideways in frustration — nothing dangerous."),
        ("🎯", "Why it works", "One fast counter into all the space they left behind can win it."),
    ],
    "counter-attack": [
        ("🛡️", "Defend deep first", "Stay compact, let the opponent commit players forward."),
        ("🏃", "Win the ball, GO!", "2-3 fast players sprint forward while opponents are out of position."),
        ("📐", "Exploit the space", "Defenders outnumbered — attackers run into open space."),
        ("🎯", "Why it works", "The more they attack, the more space they leave behind."),
    ],
    "counter": [
        ("🛡️", "Stay patient", "Absorb pressure, wait for the right moment."),
        ("🏃", "Explode forward", "Fast wingers burst forward before defenders recover."),
        ("⚡", "Strike in seconds", "Ball won to shot on goal in under 10 seconds."),
        ("🎯", "Why it works", "Opponents leave huge gaps when they commit forward."),
    ],
    "possession": [
        ("🔵", "Keep ball moving", "Short, safe passes — always within the team."),
        ("🔄", "Move after passing", "Every player repositions to create new options."),
        ("😫", "Tire the opponent", "They chase endlessly, exhausting themselves."),
        ("🎯", "Why it works", "Tired defenders lose concentration — gaps appear."),
    ],
    "tiki-taka": [
        ("⚡", "Ultra-fast passes", "Ball moves every 1-2 seconds — always in motion."),
        ("🏃", "Constant movement", "Triangles of passing options appear everywhere."),
        ("😵", "Opponent can't follow", "Defenders chase shadows — ball is always gone."),
        ("🎯", "Why it works", "Exhaustion as strategy — they collapse by the 60th minute."),
    ],
    "positional play": [
        ("📍", "Occupy key zones", "Each player takes a precise position on the pitch."),
        ("🔺", "Create triangles", "Always 2+ short passes available from any position."),
        ("🧠", "Brain over legs", "Control through positioning, not sprinting."),
        ("🎯", "Why it works", "Always someone open — impossible to press effectively."),
    ],
    "positional": [
        ("📍", "Smart positioning", "Players take up precise spots across the field."),
        ("🔺", "Triangle passing", "Guarantees 2 safe options at all times."),
        ("🧠", "Outsmart, don't outrun", "Perfect positioning beats raw speed."),
        ("🎯", "Why it works", "The team becomes a machine that controls games."),
    ],
    "total football": [
        ("🔄", "Anyone plays anywhere", "Defender becomes attacker, midfielder plays CB — fluid."),
        ("🔀", "Constant rotation", "Positions swap — teammates cover each other."),
        ("🤯", "Confuse opponents", "Can't mark players who keep changing position."),
        ("🎯", "Why it works", "Unpredictable — impossible to plan against."),
    ],
    "overlaps": [
        ("🏃", "Full-back sprints forward", "Past the winger, toward the corner flag area."),
        ("2️⃣", "Create a 2v1", "Defender can't cover both players at once."),
        ("⚽", "Cross into the box", "Overlapping player delivers into the penalty area."),
        ("🎯", "Why it works", "Defender must choose — whoever is free gets the ball."),
    ],
    "wing-backs": [
        ("⬆️", "Wing-backs push high", "They act like wingers, sprinting up and down."),
        ("📐", "Stretch the pitch", "Force the opponent to spread out, opening the centre."),
        ("✈️", "Deliver from wide", "Cross the ball into the box for the strikers."),
        ("🎯", "Why it works", "3 CBs cover defence, wing-backs attack freely."),
    ],
    "false 9": [
        ("⬇️", "Striker drops deep", "Centre-forward drops into midfield to receive."),
        ("❓", "Defenders confused", "Follow? Stay? No good answer for the CBs."),
        ("🏃", "Midfielders burst in", "Sprint into the gap left behind the defence."),
        ("🎯", "Why it works", "Creates a dilemma defenders can never solve."),
    ],
    "collective press": [
        ("📢", "Trigger moment", "Bad touch or sideways pass signals everyone to press."),
        ("🏃", "Everyone presses at once", "All players rush toward the ball carrier."),
        ("🔒", "Trap the opponent", "Surrounded with no escape — turnover."),
        ("🎯", "Why it works", "No player can handle 4-5 closing in at once."),
    ],
    "pressing": [
        ("🏃", "Close down the carrier", "Sprint toward the ball to reduce thinking time."),
        ("🚧", "Block passing lanes", "Teammates cut off obvious options."),
        ("💥", "Win the ball back", "Forced mistake — stolen possession."),
        ("🎯", "Why it works", "Turns defence into attack instantly."),
    ],
    "gegenpressing": [
        ("❌", "Ball is lost", "But instead of retreating, react immediately."),
        ("🌀", "Swarm in 6 seconds", "3-4 players surround the ball carrier."),
        ("🔄", "Win it back instantly", "Opponent barely touched it — stolen."),
        ("🎯", "Why it works", "No time for them to organize a counter."),
    ],
    "build-up": [
        ("🧤", "Start from the GK", "Short pass to a centre-back, not a long kick."),
        ("📈", "Progress through midfield", "Each pass moves the ball forward one station."),
        ("🎯", "Reach the final third", "Controlled passing — not rushed, not risky."),
        ("✅", "Why it works", "Draws opponents out, opens space to exploit."),
    ],
    "patient build-up": [
        ("🧤", "Start from the back", "GK and CBs pass calmly, waiting for the opening."),
        ("⏳", "Wait for the gap", "Keep passing until a defender steps out of position."),
        ("⚡", "Strike when ready", "Vertical pass cuts through to the attackers."),
        ("🎯", "Why it works", "Patience creates openings — the opponent tires."),
    ],
    "direct": [
        ("⬆️", "Ball goes forward fast", "No sideways passes — forward as quickly as possible."),
        ("🏃", "Runners in behind", "Fast forwards make runs behind the defence."),
        ("💨", "Skip the midfield", "Defence to attack in one or two passes."),
        ("🎯", "Why it works", "No time for the opponent to set up a block."),
    ],
    "cross-heavy": [
        ("↔️", "Wingers stay wide", "Hug the touchline, create 1v1 situations."),
        ("🏃", "Reach the byline", "Dribble past the defender to the goal line."),
        ("✈️", "Deliver into the box", "Cross whipped in for headers and volleys."),
        ("🎯", "Why it works", "A perfect cross only needs one good header to score."),
    ],
    "cross-based": [
        ("↔️", "Wide players get ball", "Full-backs and wingers advance toward the byline."),
        ("🏃", "Full-back overlaps", "Extra option — 2v1 on the wing."),
        ("✈️", "Cross to target man", "Ball delivered to the tall striker in the box."),
        ("🎯", "Why it works", "Simple but effective — quality crosses = chances."),
    ],
    "physical": [
        ("💪", "Win the duels", "Strength and athleticism dominate 50/50s."),
        ("🏃", "Outwork the opponent", "Run harder, run longer — 90 minutes."),
        ("🎯", "Why it works", "Win more tackles and headers = control the game."),
    ],
    "organized": [
        ("📐", "Disciplined shape", "Every player knows exactly where to be."),
        ("🛡️", "Hard to break down", "Gaps kept small — very difficult to play through."),
        ("🎯", "Why it works", "Organisation beats individual quality."),
    ],
    "technical": [
        ("🎯", "Precise passing", "Excellent touch — keeps possession under pressure."),
        ("🧠", "Smart decisions", "Creative solutions through tight spaces."),
        ("✨", "Why it works", "When everyone controls perfectly, any style works."),
    ],
    "vertical": [
        ("⬆️", "Play forward first", "Instinct: move the ball toward goal ASAP."),
        ("🏃", "Runners in behind", "Constant forward runs offering vertical options."),
        ("🎯", "Why it works", "Every forward pass puts immediate pressure on defence."),
    ],
    "compact": [
        ("🧱", "Stay close together", "Only ~30 metres between deepest and highest player."),
        ("🔒", "No space to exploit", "Denies room between the lines."),
        ("🎯", "Why it works", "Like a wall — opponents pass around, never through."),
    ],
    "attacking": [
        ("⬆️", "Commit players forward", "Numerical advantage in the final third."),
        ("🎯", "Create chances", "Calculated risks to break defences."),
        ("✨", "Why it works", "More in the box = more chances to score."),
    ],
    "structured": [
        ("📐", "Clear game plan", "Specific roles and instructions for each player."),
        ("🔗", "Connected play", "Rehearsed patterns — nothing improvised."),
        ("🎯", "Why it works", "Predictable for teammates, unpredictable for opponents."),
    ],
    "fluid": [
        ("🔄", "Constant movement", "Players swap positions throughout the match."),
        ("🌊", "Unpredictable patterns", "Shape constantly changes."),
        ("🎯", "Why it works", "Defenders can't mark players who keep moving."),
    ],
    "set pieces": [
        ("📐", "Rehearsed routines", "Corners and free kicks with specific movements."),
        ("🎯", "Dangerous deliveries", "Set pieces = genuine goal-scoring opportunities."),
        ("✨", "Why it works", "One well-executed corner can win the game."),
    ],
    "defensive": [
        ("🛡️", "Defence first", "Every player contributes defensively."),
        ("🔒", "Protect the goal", "Compact, deep — minimal risk."),
        ("🎯", "Why it works", "If you don't concede, you can't lose."),
    ],
    "dominant": [
        ("👑", "Control every phase", "Dominate possession, territory, and chances."),
        ("💪", "Relentless intensity", "High press + constant attacking + suffocating."),
        ("🎯", "Why it works", "The opponent never gets comfortable."),
    ],
}

_TACTICS_NAME_MAP = {
    "FC Internazionale Milano": "Inter Milan",
    "SC Freiburg":              "Sport-Club Freiburg",
    "RCD Espanyol de Barcelona": "Espanyol",
}

def _hex_to_rgb(hx):
    """Hex → 'r,g,b' string for rgba()."""
    h = hx.lstrip('#')
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

def render_tactical_pitch_html(team_name):
    """Generate a premium animated SVG tactical pitch for a given team."""
    import math
    PAD, PW, PH = 14, 252, 360
    SW, SH = PW + 2 * PAD, PH + 2 * PAD

    def sx(p): return PAD + p / 100 * PW
    def sy(p): return PAD + p / 100 * PH

    t = TEAM_TACTICS.get(_TACTICS_NAME_MAP.get(team_name, team_name))
    if not t:
        return (f'<div style="background:#1a2e1a;border-radius:18px;height:260px;'
                f'display:flex;align-items:center;justify-content:center;">'
                f'<span style="color:rgba(255,255,255,.4);font-size:.78rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;">'
                f'No tactical data</span></div>')

    color = t["color"]
    slug = re.sub(r'[^a-z0-9]', '_', team_name.lower())
    cx, cy = PAD + PW // 2, PAD + PH // 2

    # ── SVG defs (pattern, gradients, filters, marker) ──
    stripe_h = 40
    defs = (
        f'<defs>'
        # Grass stripes
        f'<pattern id="g_{slug}" x="0" y="0" width="{PW}" height="{stripe_h}" patternUnits="userSpaceOnUse">'
        f'<rect x="0" y="0" width="{PW}" height="{stripe_h//2}" fill="rgba(0,0,0,0.045)"/>'
        f'</pattern>'
        # Pitch vignette
        f'<radialGradient id="vig_{slug}" cx="50%" cy="50%" r="70%">'
        f'<stop offset="0%" stop-color="transparent"/>'
        f'<stop offset="100%" stop-color="rgba(0,0,0,0.25)"/>'
        f'</radialGradient>'
        # Player drop shadow
        f'<filter id="pshadow_{slug}" x="-30%" y="-30%" width="160%" height="160%">'
        f'<feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="rgba(0,0,0,0.5)"/>'
        f'</filter>'
        # Player glow (for moving players)
        f'<filter id="pglow_{slug}" x="-40%" y="-40%" width="180%" height="180%">'
        f'<feGaussianBlur stdDeviation="3" result="blur"/>'
        f'<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f'</filter>'
        # Arrowhead marker
        f'<marker id="arr_{slug}" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 7 3.5, 0 7" fill="rgba(255,255,255,0.9)"/>'
        f'</marker>'
    )
    # Radial gradient per zone
    for i, (zcx, zcy, zrx, zry, op) in enumerate(t.get("zones", [])):
        defs += (
            f'<radialGradient id="hz_{slug}_{i}" cx="50%" cy="50%" r="50%">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity="{min(op*2.2, 0.55):.2f}"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
            f'</radialGradient>'
        )
    defs += '</defs>'

    # ── Pitch background ──
    pitch_bg = (
        f'<rect x="0" y="0" width="{SW}" height="{SH}" '
        f'fill="url(#g_{slug})" rx="0"/>'
        f'<rect x="0" y="0" width="{SW}" height="{SH}" '
        f'fill="url(#vig_{slug})"/>'
    )

    # ── Pitch markings ──
    m = (
        # Border
        f'<rect x="{PAD}" y="{PAD}" width="{PW}" height="{PH}" fill="none" '
        f'stroke="rgba(255,255,255,.55)" stroke-width="1.5"/>'
        # Center line
        f'<line x1="{PAD}" y1="{cy}" x2="{PAD+PW}" y2="{cy}" '
        f'stroke="rgba(255,255,255,.45)" stroke-width="1"/>'
        # Center circle
        f'<circle cx="{cx}" cy="{cy}" r="32" fill="none" '
        f'stroke="rgba(255,255,255,.4)" stroke-width="1"/>'
        # Center dot
        f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="rgba(255,255,255,.6)"/>'
        # Penalty box top
        f'<rect x="{PAD+58}" y="{PAD}" width="136" height="66" fill="rgba(255,255,255,.03)" '
        f'stroke="rgba(255,255,255,.38)" stroke-width="1"/>'
        # Small box top
        f'<rect x="{PAD+96}" y="{PAD}" width="60" height="23" fill="rgba(255,255,255,.02)" '
        f'stroke="rgba(255,255,255,.28)" stroke-width="1"/>'
        # Penalty spot top
        f'<circle cx="{cx}" cy="{PAD+52}" r="2" fill="rgba(255,255,255,.45)"/>'
        # Penalty arc top (D)
        f'<path d="M {PAD+80} {PAD+66} A 32 32 0 0 0 {PAD+172} {PAD+66}" fill="none" '
        f'stroke="rgba(255,255,255,.3)" stroke-width="1"/>'
        # Penalty box bottom
        f'<rect x="{PAD+58}" y="{PAD+PH-66}" width="136" height="66" fill="rgba(255,255,255,.03)" '
        f'stroke="rgba(255,255,255,.38)" stroke-width="1"/>'
        # Small box bottom
        f'<rect x="{PAD+96}" y="{PAD+PH-23}" width="60" height="23" fill="rgba(255,255,255,.02)" '
        f'stroke="rgba(255,255,255,.28)" stroke-width="1"/>'
        # Penalty spot bottom
        f'<circle cx="{cx}" cy="{PAD+PH-52}" r="2" fill="rgba(255,255,255,.45)"/>'
        # Penalty arc bottom
        f'<path d="M {PAD+80} {PAD+PH-66} A 32 32 0 0 1 {PAD+172} {PAD+PH-66}" fill="none" '
        f'stroke="rgba(255,255,255,.3)" stroke-width="1"/>'
        # Goals
        f'<rect x="{PAD+102}" y="{PAD-8}" width="48" height="8" fill="rgba(255,255,255,.12)" '
        f'stroke="rgba(255,255,255,.45)" stroke-width="1"/>'
        f'<rect x="{PAD+102}" y="{PAD+PH}" width="48" height="8" fill="rgba(255,255,255,.12)" '
        f'stroke="rgba(255,255,255,.45)" stroke-width="1"/>'
        # Corner arcs
        f'<path d="M {PAD} {PAD+9} A 9 9 0 0 1 {PAD+9} {PAD}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
        f'<path d="M {PAD+PW-9} {PAD} A 9 9 0 0 1 {PAD+PW} {PAD+9}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
        f'<path d="M {PAD} {PAD+PH-9} A 9 9 0 0 1 {PAD+9} {PAD+PH}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
        f'<path d="M {PAD+PW-9} {PAD+PH} A 9 9 0 0 1 {PAD+PW} {PAD+PH-9}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
    )

    # ── Heat zones (radial gradient ellipses) ──
    zones_svg = ""
    for i, (zcx, zcy, zrx, zry, op) in enumerate(t.get("zones", [])):
        zones_svg += (
            f'<ellipse cx="{sx(zcx):.1f}" cy="{sy(zcy):.1f}" '
            f'rx="{zrx/100*PW:.1f}" ry="{zry/100*PH:.1f}" '
            f'fill="url(#hz_{slug}_{i})" class="hz_{slug}"/>\n'
        )

    # ── Formation lines (connect players in the same tactical line) ──
    players = t["players"]
    lines_by_depth = {}
    for i, (ppx, ppy, abbr) in enumerate(players):
        bucket = round(ppy / 22) * 22  # group by ~22% bands
        lines_by_depth.setdefault(bucket, []).append(i)

    formation_lines_svg = ""
    for bucket, idxs in lines_by_depth.items():
        if len(idxs) < 2:
            continue
        sorted_i = sorted(idxs, key=lambda i: players[i][0])
        for j in range(len(sorted_i) - 1):
            i1, i2 = sorted_i[j], sorted_i[j + 1]
            x1, y1 = sx(players[i1][0]), sy(players[i1][1])
            x2, y2 = sx(players[i2][0]), sy(players[i2][1])
            formation_lines_svg += (
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="1.2" opacity="0.35" stroke-dasharray="3 4"/>'
            )

    # ── Movement arrows (animated dashed bezier paths) ──
    moves_dict = {m[0]: (m[1], m[2]) for m in t.get("moves", [])}
    arrows_svg = ""
    arrow_css = []
    for seq, (pi, (to_x, to_y)) in enumerate(moves_dict.items()):
        ppx, ppy, _ = players[pi]
        x1, y1 = sx(ppx), sy(ppy)
        x2, y2 = sx(to_x), sy(to_y)
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1
        # Curved control point (slight perpendicular offset)
        perp_x, perp_y = -dy / L, dx / L
        offset = min(L * 0.22, 18)
        cpx = (x1 + x2) / 2 + perp_x * offset
        cpy = (y1 + y2) / 2 + perp_y * offset
        path_d = f"M {x1:.1f} {y1:.1f} Q {cpx:.1f} {cpy:.1f} {x2:.1f} {y2:.1f}"
        path_len = int(L * 1.18) + 10
        an = f"aw_{slug}_{pi}"
        cls_a = f"ac_{slug}_{pi}"
        delay = seq * 1.8
        arrow_css.append(
            f"@keyframes {an}{{"
            f"0%,{int(delay/9*100)}%{{stroke-dashoffset:{path_len};opacity:0}}"
            f"{int((delay+0.5)/9*100)}%,{int((delay+2.2)/9*100)}%{{stroke-dashoffset:0;opacity:.92}}"
            f"{int((delay+2.8)/9*100)}%,100%{{stroke-dashoffset:-{path_len};opacity:0}}}}"
            f".{cls_a}{{stroke-dasharray:{path_len};stroke-dashoffset:{path_len};"
            f"animation:{an} 9s ease-in-out infinite;}}"
        )
        arrows_svg += (
            f'<path class="{cls_a}" d="{path_d}" fill="none" stroke="rgba(255,255,255,.9)" '
            f'stroke-width="2" marker-end="url(#arr_{slug})"/>\n'
        )

    # ── Players ──
    css_lines = [
        f"@keyframes hz_p_{slug}{{0%,100%{{opacity:.75}}50%{{opacity:1}}}}",
        f".hz_{slug}{{animation:hz_p_{slug} 4.5s ease-in-out infinite;}}",
    ] + arrow_css

    # Add hover glow CSS for clickable player bubbles
    css_lines.append(
        ".pitch-player-link{cursor:pointer;text-decoration:none;}"
        ".pitch-player-link:hover circle:first-of-type{filter:brightness(1.35);stroke:white;stroke-width:2.4;}"
        ".pitch-player-link:hover{opacity:.92;}"
    )

    players_svg = ""
    for i, (ppx, ppy, abbr) in enumerate(players):
        x0, y0 = sx(ppx), sy(ppy)
        cls = f"pl_{slug}_{i}"
        delay = f"{i * 0.22:.2f}s"
        is_mover = i in moves_dict
        filt = f'filter="url(#pglow_{slug})"' if is_mover else f'filter="url(#pshadow_{slug})"'
        if is_mover:
            tx, ty = moves_dict[i]
            ddx = sx(tx) - x0
            ddy = sy(ty) - y0
            an = f"pm_{slug}_{i}"
            css_lines.append(
                f"@keyframes {an}{{"
                f"0%,18%{{transform:translate(0px,0px)}}"
                f"38%,62%{{transform:translate({ddx:.1f}px,{ddy:.1f}px)}}"
                f"82%,100%{{transform:translate(0px,0px)}}}}"
                f".{cls}{{animation:{an} 9s ease-in-out infinite;animation-delay:{delay};}}"
            )
            ring = f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="15" fill="none" stroke="{color}" stroke-width="1" opacity="0.4" stroke-dasharray="3 3"/>'
        else:
            an = f"ps_{slug}_{i}"
            css_lines.append(
                f"@keyframes {an}{{0%,100%{{opacity:1}}50%{{opacity:.82}}}}"
                f".{cls}{{animation:{an} 3.8s ease-in-out infinite;animation-delay:{delay};}}"
            )
            ring = ""
        # Goalkeeper gets a distinct visual: gold fill, larger circle, GK badge
        is_gk = (abbr == "GK")
        player_fill   = "#F5C842" if is_gk else color
        player_r      = 13.5     if is_gk else 11.5
        stroke_color  = "rgba(255,255,255,.95)"
        stroke_w      = 2.2      if is_gk else 1.8
        text_fill     = "#1A1A2E" if is_gk else "white"
        font_sz       = 5.8      if is_gk else 6.2
        # GK gets an extra outer ring to stand out
        gk_ring = (
            f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{player_r+4.5}" '
            f'fill="none" stroke="#F5C842" stroke-width="1.4" opacity="0.5" stroke-dasharray="4 3"/>'
        ) if is_gk else ""

        pos_anchor = f"?nav=glossaire&pos={abbr.lower()}"
        players_svg += (
            f'<a href="{pos_anchor}" class="pitch-player-link">'
            f'<g class="{cls}" {filt}>'
            f'{ring}'
            f'{gk_ring}'
            f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{player_r}" fill="{player_fill}" stroke="{stroke_color}" stroke-width="{stroke_w}"/>'
            f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{player_r}" fill="rgba(255,255,255,.08)"/>'
            f'<text x="{x0:.1f}" y="{y0:.1f}" text-anchor="middle" dominant-baseline="central" '
            f'font-size="{font_sz}" font-weight="900" fill="{text_fill}" font-family="Nunito,sans-serif" letter-spacing="-.3">{abbr}</text>'
            f'</g>'
            f'</a>\n'
        )

    # ── Animated ball ──
    ball_svg = ""
    if t.get("moves"):
        first_move = t["moves"][0]
        bpi, bto_x, bto_y = first_move
        bppx, bppy, _ = players[bpi]
        bx0, by0 = sx(bppx), sy(bppy)
        bx1, by1 = sx(bto_x), sy(bto_y)
        bddx, bddy = bx1 - bx0, by1 - by0
        ball_cls = f"ball_{slug}"
        ball_glow_id = f"ballglow_{slug}"
        # Add glow filter for the ball
        defs = defs[:-len("</defs>")] + (
            f'<filter id="{ball_glow_id}" x="-60%" y="-60%" width="220%" height="220%">'
            f'<feGaussianBlur stdDeviation="4" result="blur"/>'
            f'<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
            f'</filter>'
        ) + "</defs>"
        css_lines.append(
            f"@keyframes ballm_{slug}{{"
            f"0%,15%{{transform:translate(0px,0px)}}"
            f"35%,65%{{transform:translate({bddx:.1f}px,{bddy:.1f}px)}}"
            f"85%,100%{{transform:translate(0px,0px)}}}}"
            f"@keyframes ballpulse_{slug}{{0%,100%{{opacity:1}}50%{{opacity:.75}}}}"
            f".{ball_cls}{{animation:ballm_{slug} 9s ease-in-out infinite,ballpulse_{slug} 1.2s ease-in-out infinite;animation-delay:.18s,0s;}}"
        )
        ball_svg = (
            f'<g class="{ball_cls}" filter="url(#{ball_glow_id})">'
            # Outer glow ring
            f'<circle cx="{bx0:.1f}" cy="{by0:.1f}" r="11" fill="rgba(255,255,180,.18)" stroke="rgba(255,255,100,.4)" stroke-width="1.5"/>'
            # Ball body
            f'<circle cx="{bx0:.1f}" cy="{by0:.1f}" r="7.5" fill="white" stroke="rgba(0,0,0,.5)" stroke-width="1.5"/>'
            # Ball pattern lines
            f'<circle cx="{bx0:.1f}" cy="{by0:.1f}" r="7.5" fill="none" stroke="rgba(0,0,0,.2)" stroke-width="3.5" stroke-dasharray="4.5 4"/>'
            # "BALL" label so it's unmistakably a ball
            f'</g>'
        )

    # ── Style pills (clickable if glossary term exists) ──
    def _make_pill(s):
        term_key = STYLE_TAG_TO_TERM.get(s.lower())
        inner = (
            f'<span style="display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .72rem;'
            f'border-radius:100px;background:rgba(255,255,255,.08);color:rgba(255,255,255,.75);'
            f'font-size:.6rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;'
            f'border:1px solid rgba(255,255,255,.12);margin:.15rem .15rem 0 0;'
            + (f'cursor:pointer;transition:background .15s;' if term_key else '') +
            f'"'
            + (f' onmouseover="this.style.background=\'rgba(255,255,255,.18)\'" onmouseout="this.style.background=\'rgba(255,255,255,.08)\'"' if term_key else '') +
            f'><span style="width:5px;height:5px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0"></span>'
            f'{s}'
            + (' <span style="opacity:.5;font-size:.55rem">→</span>' if term_key else '') +
            f'</span>'
        )
        if term_key:
            return f'<a href="?term={term_key}&from=main" style="text-decoration:none;">{inner}</a>'
        return inner

    pills = "".join(_make_pill(s) for s in t.get("style_tags", []))

    # ── Animated step-by-step guide (appears on the pitch, one at a time) ──
    first_tag = (t.get("style_tags") or [""])[0].lower()
    guide_steps = STYLE_TAG_GUIDE.get(first_tag, [])
    if not guide_steps:
        guide_steps = [("", "How they play", "Arrows show typical player movements and runs.")]

    action_steps = [s for s in guide_steps if s[1].lower() != "why it works"][:3]
    why_step = next((s for s in guide_steps if s[1].lower() == "why it works"), None)

    # Total animation cycle: 16s — 4 phases of 4s each
    # Each phase: 0.4s fade-in, 3.2s visible, 0.4s fade-out
    all_anim_steps = list(action_steps)
    if why_step:
        all_anim_steps.append(why_step)
    n_phases = len(all_anim_steps)
    cycle = n_phases * 4  # 4s per phase

    overlay_html = ""
    overlay_css = ""
    if action_steps:
        for idx, (_, title, desc, *_) in enumerate(all_anim_steps):
            is_result = (idx == n_phases - 1 and why_step)
            phase_start = idx * 4
            # CSS keyframe percentages
            t_in   = phase_start / cycle * 100
            t_vis  = (phase_start + 0.5) / cycle * 100
            t_hold = (phase_start + 3.5) / cycle * 100
            t_out  = (phase_start + 4.0) / cycle * 100
            anim_name = f"sg_{slug}_{idx}"
            overlay_css += (
                f"@keyframes {anim_name}{{"
                f"0%,{t_in:.1f}%{{opacity:0;transform:translateY(6px)}}"
                f"{t_vis:.1f}%,{t_hold:.1f}%{{opacity:1;transform:translateY(0)}}"
                f"{t_out:.1f}%,100%{{opacity:0;transform:translateY(-4px)}}}}"
                f".{anim_name}{{animation:{anim_name} {cycle}s ease-in-out infinite;}}"
            )

            if is_result:
                # Result line — accent style
                overlay_html += (
                    f'<div class="{anim_name}" style="position:absolute;top:12px;left:10px;right:10px;'
                    f'opacity:0;pointer-events:none;'
                    f'padding:.55rem .7rem;border-radius:10px;'
                    f'background:linear-gradient(135deg,rgba({_hex_to_rgb(color)},.22),rgba(0,0,0,.55));'
                    f'backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);'
                    f'border:1px solid rgba({_hex_to_rgb(color)},.3);">'
                    f'<div style="font-size:.72rem;font-weight:900;color:{color};'
                    f'letter-spacing:.06em;line-height:1.5;">'
                    f'Result</div>'
                    f'<div style="font-size:.7rem;font-weight:600;color:rgba(255,255,255,.75);'
                    f'line-height:1.5;margin-top:.15rem;">{desc}</div>'
                    f'</div>'
                )
            else:
                step_num = idx + 1
                overlay_html += (
                    f'<div class="{anim_name}" style="position:absolute;top:12px;left:10px;right:10px;'
                    f'opacity:0;pointer-events:none;'
                    f'padding:.5rem .65rem;border-radius:10px;'
                    f'background:linear-gradient(135deg,rgba(0,0,0,.6),rgba(0,0,0,.45));'
                    f'backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);'
                    f'border:1px solid rgba(255,255,255,.08);">'
                    f'<div style="display:flex;align-items:center;gap:.45rem;">'
                    f'<span style="min-width:19px;height:19px;border-radius:50%;'
                    f'background:rgba({_hex_to_rgb(color)},.3);border:1.5px solid {color};'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:.55rem;font-weight:900;color:white;flex-shrink:0;">{step_num}</span>'
                    f'<div style="font-size:.72rem;font-weight:900;color:rgba(255,255,255,.9);'
                    f'letter-spacing:.02em;line-height:1.3;">{title}</div>'
                    f'</div>'
                    f'<div style="font-size:.68rem;font-weight:600;color:rgba(255,255,255,.52);'
                    f'line-height:1.45;margin-top:.3rem;padding-left:2rem;">{desc}</div>'
                    f'</div>'
                )

        css_lines.append(overlay_css)

    # Scoped to #pitch_{slug} to avoid polluting Streamlit page SVG elements
    css_lines.insert(0, f"#pitch_{slug} g,#pitch_{slug} ellipse,#pitch_{slug} path,#pitch_{slug} circle{{transform-box:fill-box;transform-origin:center;}}")
    css_block = "<style>" + "".join(css_lines) + "</style>"

    svg = (
        f'<svg viewBox="0 0 {SW} {SH}" xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;width:100%;max-width:300px;margin:0 auto;background:#1e5c1e;">'
        f'{defs}{pitch_bg}{m}{zones_svg}{formation_lines_svg}{arrows_svg}{players_svg}{ball_svg}'
        f'</svg>'
    )

    formation_val = t["formation"]
    return (
        f'{css_block}'
        f'<div id="pitch_{slug}" style="background:#0F1C0F;border-radius:20px;overflow:hidden;'
        f'box-shadow:0 8px 32px rgba(0,0,0,.45),0 0 0 1px rgba(255,255,255,.06);">'
        # Header
        f'<div style="padding:.9rem 1.1rem .6rem;display:flex;align-items:center;'
        f'justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.07);">'
        f'<div>'
        f'<div style="font-size:.64rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase;'
        f'color:{color};margin-bottom:.18rem;">Team Analysis</div>'
        f'<div style="font-size:.95rem;font-weight:900;color:rgba(255,255,255,.92);letter-spacing:-.02em;">{team_name}</div>'
        f'</div>'
        f'<a href="?nav=glossaire&formation={formation_val}" style="text-decoration:none;">'
        f'<span style="background:{color};color:white;font-size:.72rem;font-weight:900;'
        f'padding:.3rem .9rem;border-radius:100px;letter-spacing:.06em;cursor:pointer;'
        f'box-shadow:0 2px 10px {color}66;transition:opacity .15s;" '
        f'onmouseover="this.style.opacity=\'0.8\'" onmouseout="this.style.opacity=\'1\'">'
        f'{t["formation"]}</span>'
        f'</a>'
        f'</div>'
        # SVG pitch + animated overlay
        f'<div style="position:relative;">{svg}{overlay_html}</div>'
        # Footer: pills only
        f'<div style="padding:.55rem 1rem .75rem;border-top:1px solid rgba(255,255,255,.06);">'
        f'<div style="font-size:.52rem;font-weight:800;letter-spacing:.16em;color:rgba(255,255,255,.25);'
        f'text-transform:uppercase;margin-bottom:.28rem;">Playing style · click a tag to learn more</div>'
        f'{pills}</div>'
        f'</div>'
    )

def render_term_animation_html(term):
    """Generate a premium animated SVG tactical illustration matching the tactical pitch card style."""
    # ── Dimensions (landscape pitch) ──
    PAD, PW, PH = 8, 264, 204
    SW, SH = PW + 2 * PAD, PH + 2 * PAD   # 280 × 220
    cx, cy = PAD + PW // 2, PAD + PH // 2  # 140, 110

    # ── Term metadata ──
    TERM_META = {
        "pressing":          ("#E5273D", "Pressing",   ["Collective", "Recovery", "Intensity"]),
        "pivot":             ("#A50044", "Attack",     ["Target Man", "Hold-Up", "Link"]),
        "false nine":        ("#A50044", "Attack",     ["Movement", "Confusion", "Space"]),
        "build-up play":     ("#1357BE", "Possession", ["Patience", "Structure", "Progression"]),
        "through ball":      ("#FFB800", "Attack",     ["Penetration", "Timing", "Space"]),
        "switch of play":    ("#FFB800", "Possession", ["Width", "Diagonal", "Overload"]),
        "overlap":           ("#00C875", "Width",      ["Full-Back", "2v1", "Crossing"]),
        "underlap":          ("#00C875", "Width",      ["Half-Space", "Inside Run", "1v1"]),
        "cross":             ("#FFB800", "Attack",     ["Delivery", "Width", "Box"]),
        "final third":       ("#E5273D", "Attack",     ["Danger Zone", "Chances", "Goals"]),
        "counter-attack":    ("#FFB800", "Transition", ["Speed", "Numbers", "Space"]),
        "high press":        ("#E5273D", "Pressing",   ["High Line", "Intensity", "Recovery"]),
        "low block":         ("#4a6fa5", "Defense",    ["Compact", "Deep", "Discipline"]),
        "man marking":       ("#4a6fa5", "Defense",    ["Individual", "Tracking", "Pressure"]),
        "zonal marking":     ("#4a6fa5", "Defense",    ["Zones", "Collective", "Structure"]),
        "tackle":            ("#4a6fa5", "Defense",    ["Duel", "Timing", "Recovery"]),
        "interception":      ("#4a6fa5", "Defense",    ["Reading", "Anticipation", "Steal"]),
        "counter-pressing":  ("#E5273D", "Pressing",   ["Immediate", "Swarming", "6-second"]),
        "transition":        ("#FFB800", "Transition", ["Switch", "Momentum", "Speed"]),
        "formation":         ("#6CABDD", "Structure",  ["Shape", "System", "Organization"]),
        "shape":             ("#6CABDD", "Structure",  ["Compactness", "Block", "Unit"]),
        "width":             ("#00C875", "Structure",  ["Stretch", "Flanks", "Space"]),
        "depth":             ("#6CABDD", "Structure",  ["Staggered", "Layers", "Options"]),
        "half-space":        ("#FFB800", "Structure",  ["Channel", "Danger", "Between Lines"]),
        "lines":             ("#4a6fa5", "Structure",  ["Horizontal", "Block", "Penetration"]),
        "tiki-taka":         ("#A50044", "Possession", ["Short Pass", "Movement", "Control"]),
        "total football":    ("#1357BE", "Possession", ["Fluid", "Universal", "Rotation"]),
        "positional play":   ("#1357BE", "Possession", ["Occupation", "Control", "Triangles"]),
        "overload":          ("#E5273D", "Attack",     ["Numerical", "2v1", "3v2"]),
        "third man run":     ("#00C875", "Attack",     ["Combination", "Movement", "Run"]),
        "line-breaking pass":("#FFB800", "Attack",     ["Penetration", "Lines", "Unlock"]),
    }
    color, category, style_tags = TERM_META.get(term, ("#FFB800", "Tactical", ["Concept"]))
    slug = re.sub(r"[^a-z0-9]", "_", term)

    # ── SVG Defs (identical structure to tactical pitch) ──
    stripe_h = 32
    defs = (
        f"<defs>"
        f'<pattern id="tg_{slug}" x="0" y="0" width="{PW}" height="{stripe_h}" patternUnits="userSpaceOnUse">'
        f'<rect x="0" y="0" width="{PW}" height="{stripe_h//2}" fill="rgba(0,0,0,0.045)"/>'
        f"</pattern>"
        f'<radialGradient id="tvig_{slug}" cx="50%" cy="50%" r="70%">'
        f'<stop offset="0%" stop-color="transparent"/>'
        f'<stop offset="100%" stop-color="rgba(0,0,0,0.28)"/>'
        f"</radialGradient>"
        f'<filter id="tpshadow_{slug}" x="-30%" y="-30%" width="160%" height="160%">'
        f'<feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="rgba(0,0,0,0.5)"/>'
        f"</filter>"
        f'<filter id="tpglow_{slug}" x="-40%" y="-40%" width="180%" height="180%">'
        f'<feGaussianBlur stdDeviation="3" result="blur"/>'
        f'<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f"</filter>"
        f'<marker id="tarr_{slug}" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'
        f'<polygon points="0 0,7 3.5,0 7" fill="rgba(255,255,255,0.9)"/></marker>'
        f'<marker id="tyarr_{slug}" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'
        f'<polygon points="0 0,7 3.5,0 7" fill="{color}"/></marker>'
        f'<marker id="tgarr_{slug}" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'
        f'<polygon points="0 0,7 3.5,0 7" fill="#00C875"/></marker>'
        f'<filter id="tbglow_{slug}" x="-60%" y="-60%" width="220%" height="220%">'
        f'<feGaussianBlur stdDeviation="3.5" result="blur"/>'
        f'<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f'</filter>'
        f"</defs>"
    )

    # ── Pitch background (stripes + vignette) ──
    pitch_bg = (
        f'<rect x="0" y="0" width="{SW}" height="{SH}" fill="url(#tg_{slug})"/>'
        f'<rect x="0" y="0" width="{SW}" height="{SH}" fill="url(#tvig_{slug})"/>'
    )

    # ── Pitch markings (scaled to landscape 280×220) ──
    # proportional to PW=264, PH=204
    pb_x  = PAD + int(58/252*PW)      # penalty box x offset from PAD ≈ 61
    pb_w  = int(136/252*PW)           # penalty box width ≈ 143
    pb_h  = int(66/360*PH)            # penalty box height ≈ 37
    sb_x  = PAD + int(96/252*PW)      # small box x ≈ 109
    sb_w  = int(60/252*PW)            # small box width ≈ 63
    sb_h  = int(23/360*PH)            # small box height ≈ 13
    ps_y  = PAD + int(52/360*PH)      # penalty spot y ≈ 37
    g_x   = PAD + int(102/252*PW)     # goal x ≈ 115
    g_w   = int(48/252*PW)            # goal width ≈ 50
    cc_r  = 18                        # center circle radius

    markings = (
        f'<rect x="{PAD}" y="{PAD}" width="{PW}" height="{PH}" fill="none" stroke="rgba(255,255,255,.55)" stroke-width="1.5"/>'
        f'<line x1="{PAD}" y1="{cy}" x2="{PAD+PW}" y2="{cy}" stroke="rgba(255,255,255,.45)" stroke-width="1"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{cc_r}" fill="none" stroke="rgba(255,255,255,.4)" stroke-width="1"/>'
        f'<circle cx="{cx}" cy="{cy}" r="3" fill="rgba(255,255,255,.6)"/>'
        # Penalty box top
        f'<rect x="{pb_x}" y="{PAD}" width="{pb_w}" height="{pb_h}" fill="rgba(255,255,255,.03)" stroke="rgba(255,255,255,.38)" stroke-width="1"/>'
        # Small box top
        f'<rect x="{sb_x}" y="{PAD}" width="{sb_w}" height="{sb_h}" fill="rgba(255,255,255,.02)" stroke="rgba(255,255,255,.28)" stroke-width="1"/>'
        # Penalty spot top
        f'<circle cx="{cx}" cy="{ps_y}" r="2" fill="rgba(255,255,255,.45)"/>'
        # Penalty arc top
        f'<path d="M {pb_x+8} {PAD+pb_h} A {cc_r} {cc_r} 0 0 0 {pb_x+pb_w-8} {PAD+pb_h}" fill="none" stroke="rgba(255,255,255,.3)" stroke-width="1"/>'
        # Penalty box bottom
        f'<rect x="{pb_x}" y="{PAD+PH-pb_h}" width="{pb_w}" height="{pb_h}" fill="rgba(255,255,255,.03)" stroke="rgba(255,255,255,.38)" stroke-width="1"/>'
        # Small box bottom
        f'<rect x="{sb_x}" y="{PAD+PH-sb_h}" width="{sb_w}" height="{sb_h}" fill="rgba(255,255,255,.02)" stroke="rgba(255,255,255,.28)" stroke-width="1"/>'
        # Penalty spot bottom
        f'<circle cx="{cx}" cy="{PAD+PH-ps_y+PAD}" r="2" fill="rgba(255,255,255,.45)"/>'
        # Penalty arc bottom
        f'<path d="M {pb_x+8} {PAD+PH-pb_h} A {cc_r} {cc_r} 0 0 1 {pb_x+pb_w-8} {PAD+PH-pb_h}" fill="none" stroke="rgba(255,255,255,.3)" stroke-width="1"/>'
        # Goals
        f'<rect x="{g_x}" y="{PAD-7}" width="{g_w}" height="7" fill="rgba(255,255,255,.12)" stroke="rgba(255,255,255,.45)" stroke-width="1"/>'
        f'<rect x="{g_x}" y="{PAD+PH}" width="{g_w}" height="7" fill="rgba(255,255,255,.12)" stroke="rgba(255,255,255,.45)" stroke-width="1"/>'
        # Corner arcs
        f'<path d="M {PAD} {PAD+8} A 8 8 0 0 1 {PAD+8} {PAD}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
        f'<path d="M {PAD+PW-8} {PAD} A 8 8 0 0 1 {PAD+PW} {PAD+8}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
        f'<path d="M {PAD} {PAD+PH-8} A 8 8 0 0 1 {PAD+8} {PAD+PH}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
        f'<path d="M {PAD+PW-8} {PAD+PH} A 8 8 0 0 1 {PAD+PW} {PAD+PH-8}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
    )

    # ── Helpers ──
    def pc(x, y, r=9, fill="#fff", stroke="rgba(255,255,255,.85)", sw=1.8, cls_name="", filt=""):
        """Player circle with label support."""
        fstr = f'filter="url(#{filt})"' if filt else ""
        return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{sw}" class="{cls_name}" {fstr}/>')

    def pl(x, y, label, fill, cls_name="", filt="", r=9):
        """Full player dot with text."""
        fstr = f'filter="url(#{filt})"' if filt else ""
        return (
            f'<g class="{cls_name}" {fstr}>'
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="rgba(255,255,255,.9)" stroke-width="1.8"/>'
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="rgba(255,255,255,.07)"/>'
            f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" '
            f'font-size="6" font-weight="900" fill="white" font-family="Nunito,sans-serif" letter-spacing="-.2">{label}</text>'
            f'</g>'
        )

    def arrow(x1, y1, x2, y2, col="rgba(255,255,255,.9)", sw=2, marker="tarr", dash="", cls_name=""):
        d_attr = f'stroke-dasharray="{dash}"' if dash else ""
        cls_attr = f'class="{cls_name}"' if cls_name else ""
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}" '
                f'{d_attr} {cls_attr} marker-end="url(#{marker}_{slug})"/>')

    def curve(x1, y1, cpx, cpy, x2, y2, col="rgba(255,255,255,.9)", sw=2, marker="tarr", dash="", cls_name="", path_len=120):
        d_attr = f'stroke-dasharray="{path_len}" stroke-dashoffset="{path_len}"' if dash == "anim" else (f'stroke-dasharray="{dash}"' if dash else "")
        cls_attr = f'class="{cls_name}"' if cls_name else ""
        return (f'<path d="M {x1} {y1} Q {cpx} {cpy} {x2} {y2}" fill="none" stroke="{col}" stroke-width="{sw}" '
                f'{d_attr} {cls_attr} marker-end="url(#{marker}_{slug})"/>')

    wa = f"url(#tarr_{slug})"
    wc = f"url(#tyarr_{slug})"
    wg = f"url(#tgarr_{slug})"
    gf = f"tpglow_{slug}"
    sf = f"tpshadow_{slug}"

    content = ""
    css = ""

    # ───────────────────────────────────────────────
    # Term-specific animations
    # (all coordinates in the 280×220 SVG space)
    # ───────────────────────────────────────────────
    if term == "pressing":
        css = (
            f"@keyframes pra_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(30px,-20px)}}}}"
            f"@keyframes prb_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(-4px,-28px)}}}}"
            f"@keyframes prc_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(-28px,-14px)}}}}"
            f".pra_{slug}{{animation:pra_{slug} 2.2s ease-in-out infinite}}"
            f".prb_{slug}{{animation:prb_{slug} 2.2s ease-in-out infinite;animation-delay:.3s}}"
            f".prc_{slug}{{animation:prc_{slug} 2.2s ease-in-out infinite;animation-delay:.55s}}"
        )
        content += pl(140, 95, "BC", "#FFB800", filt=sf)
        for cls2, x, y in [(f"pra_{slug}", 88, 125), (f"prb_{slug}", 140, 132), (f"prc_{slug}", 192, 125)]:
            content += pl(x, y, "", color, cls_name=cls2, filt=gf)
        content += curve(93, 118, 112, 104, 133, 100, col=f"{color}bb", sw=2, dash="5 3", marker="tyarr")
        content += curve(140, 124, 140, 112, 140, 104, col=f"{color}bb", sw=2, dash="5 3", marker="tyarr")
        content += curve(187, 118, 162, 104, 147, 100, col=f"{color}bb", sw=2, dash="5 3", marker="tyarr")

    elif term == "pivot":
        css = (
            f"@keyframes piva_{slug}{{0%,40%{{transform:translate(0,0)}}60%,100%{{transform:translate(-38px,-28px)}}}}"
            f"@keyframes pivb_{slug}{{0%,100%{{stroke-dashoffset:72;opacity:0}}45%{{stroke-dashoffset:0;opacity:.9}}65%,100%{{stroke-dashoffset:72;opacity:0}}}}"
            f".pivpa_{slug}{{animation:piva_{slug} 3s ease-in-out infinite}}"
            f".pivba_{slug}{{stroke-dasharray:72;animation:pivb_{slug} 3s ease-in-out infinite}}"
        )
        content += pl(140, 48, "DEF", "#4a6fa5", filt=sf)
        content += pl(140, 100, "PVT", color, filt=sf)
        content += f'<circle cx="154" cy="100" r="5" fill="#FFB800" filter="url(#{gf})"/>'
        content += pl(80, 146, "RUN", color, cls_name=f"pivpa_{slug}", filt=gf)
        content += curve(140, 109, 110, 122, 83, 140, col="#FFB800", sw=2.5, cls_name=f"pivba_{slug}", marker="tyarr", path_len=72)

    elif term == "false nine":
        css = (
            f"@keyframes fna_{slug}{{0%,100%{{transform:translate(0,0)}}45%,55%{{transform:translate(0,50px)}}}}"
            f"@keyframes fnb_{slug}{{0%,100%{{transform:translate(0,0)}}45%,55%{{transform:translate(-26px,-36px)}}}}"
            f"@keyframes fnc_{slug}{{0%,100%{{transform:translate(0,0)}}45%,55%{{transform:translate(26px,-36px)}}}}"
            f".fnpa_{slug}{{animation:fna_{slug} 3.5s ease-in-out infinite}}"
            f".fnpb_{slug}{{animation:fnb_{slug} 3.5s ease-in-out infinite;animation-delay:.4s}}"
            f".fnpc_{slug}{{animation:fnc_{slug} 3.5s ease-in-out infinite;animation-delay:.4s}}"
        )
        content += pl(100, 62, "CB", "#4a6fa5", filt=sf)
        content += pl(180, 62, "CB", "#4a6fa5", filt=sf)
        content += pl(140, 44, "F9", color, cls_name=f"fnpa_{slug}", filt=gf)
        content += pl(106, 102, "LM", color, cls_name=f"fnpb_{slug}", filt=gf)
        content += pl(174, 102, "RM", color, cls_name=f"fnpc_{slug}", filt=gf)

    elif term == "build-up play":
        # Players: GK(140,194), CB(108,154), CM(164,112), ST(140,50)
        # Ball follows: GK→CB→CM→ST, synced with pass animations
        css = (
            f"@keyframes bu1_{slug}{{0%{{stroke-dashoffset:80;opacity:0}}20%,38%{{stroke-dashoffset:0;opacity:.9}}55%,100%{{stroke-dashoffset:80;opacity:0}}}}"
            f"@keyframes bu2_{slug}{{0%,20%{{stroke-dashoffset:80;opacity:0}}38%,56%{{stroke-dashoffset:0;opacity:.9}}72%,100%{{stroke-dashoffset:80;opacity:0}}}}"
            f"@keyframes bu3_{slug}{{0%,38%{{stroke-dashoffset:80;opacity:0}}56%,74%{{stroke-dashoffset:0;opacity:.9}}90%,100%{{stroke-dashoffset:80;opacity:0}}}}"
            f".bua1_{slug}{{stroke-dasharray:80;animation:bu1_{slug} 3.2s ease-in-out infinite}}"
            f".bua2_{slug}{{stroke-dasharray:80;animation:bu2_{slug} 3.2s ease-in-out infinite}}"
            f".bua3_{slug}{{stroke-dasharray:80;animation:bu3_{slug} 3.2s ease-in-out infinite}}"
            # Ball follows each pass: starts at GK(140,194)
            f"@keyframes tbm_{slug}{{"
            f"0%,16%{{transform:translate(0px,0px)}}"         # at GK
            f"25%,46%{{transform:translate(-32px,-40px)}}"    # at CB(108,154): -32,-40
            f"55%,72%{{transform:translate(24px,-82px)}}"     # at CM(164,112): +24,-82
            f"80%,96%{{transform:translate(0px,-144px)}}"     # at ST(140,50): 0,-144
            f"100%{{transform:translate(0px,0px)}}}}"
            f"@keyframes tbp_{slug}{{0%,100%{{opacity:1}}50%{{opacity:.7}}}}"
            f".tb_{slug}{{animation:tbm_{slug} 3.2s ease-in-out infinite,tbp_{slug} .8s ease-in-out infinite;}}"
        )
        content += pl(140, 194, "GK", "#FFB800", filt=sf)
        content += pl(108, 154, "CB", "#4a6fa5", filt=sf)
        content += pl(164, 112, "CM", color, filt=sf)
        content += pl(140, 50, "ST", color, filt=gf)
        content += curve(140, 186, 124, 168, 113, 163, col="#00C875", sw=2.5, cls_name=f"bua1_{slug}", marker="tgarr", path_len=80)
        content += curve(115, 146, 138, 128, 157, 120, col="#00C875", sw=2.5, cls_name=f"bua2_{slug}", marker="tgarr", path_len=80)
        content += curve(164, 103, 152, 78, 145, 59, col="#00C875", sw=2.5, cls_name=f"bua3_{slug}", marker="tgarr", path_len=80)

    elif term == "through ball":
        css = (
            f"@keyframes tba_{slug}{{0%,100%{{stroke-dashoffset:145;opacity:0}}30%,70%{{stroke-dashoffset:0;opacity:.95}}}}"
            f".tba_{slug}{{stroke-dasharray:145;animation:tba_{slug} 2.5s ease-in-out infinite}}"
        )
        for x in [78, 115, 158, 196]:
            content += pl(x, 108, "DEF", "#4a6fa5", filt=sf)
        content += f'<rect x="{122}" y="{75}" width="36" height="66" fill="rgba(255,184,0,.12)" rx="4"/>'
        content += pl(94, 146, "P1", color, filt=sf)
        content += pl(196, 46, "P2", "#00C875", filt=gf)
        content += curve(98, 138, 140, 88, 190, 55, col="#FFB800", sw=2.5, cls_name=f"tba_{slug}", marker="tyarr", path_len=145)

    elif term == "switch of play":
        css = (
            f"@keyframes swa_{slug}{{0%,100%{{stroke-dashoffset:260;opacity:0}}30%,70%{{stroke-dashoffset:0;opacity:.9}}}}"
            f".swa_{slug}{{stroke-dasharray:260;animation:swa_{slug} 2.8s ease-in-out infinite}}"
        )
        content += pl(22, 110, "P1", color, filt=gf)
        content += pl(258, 110, "P2", color, filt=gf)
        content += f'<rect x="36" y="82" width="96" height="56" fill="rgba(255,92,92,.1)" rx="6" stroke="rgba(255,92,92,.3)" stroke-width="1"/>'
        content += f'<text x="84" y="113" text-anchor="middle" font-size="7.5" fill="rgba(255,92,92,.8)" font-family="Nunito,sans-serif">crowded</text>'
        content += f'<rect x="168" y="82" width="84" height="56" fill="rgba(0,200,117,.1)" rx="6" stroke="rgba(0,200,117,.3)" stroke-width="1"/>'
        content += f'<text x="210" y="113" text-anchor="middle" font-size="7.5" fill="rgba(0,200,117,.8)" font-family="Nunito,sans-serif">free!</text>'
        content += curve(28, 110, 140, 42, 252, 110, col="#FFB800", sw=3, cls_name=f"swa_{slug}", marker="tyarr", path_len=260)

    elif term == "overlap":
        css = (
            f"@keyframes ola_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(32px,-72px)}}}}"
            f".olpa_{slug}{{animation:ola_{slug} 2.8s ease-in-out infinite}}"
        )
        content += pl(182, 108, "W", color, filt=sf)
        content += f'<circle cx="196" cy="108" r="5" fill="#FFB800" filter="url(#{sf})"/>'
        content += pl(148, 148, "FB", color, cls_name=f"olpa_{slug}", filt=gf)
        content += f'<path d="M 150 140 Q 202 118 185 58" fill="none" stroke="rgba(255,255,255,.38)" stroke-width="1.5" stroke-dasharray="4 3"/>'
        content += pl(202, 78, "DEF", "#4a6fa5", filt=sf)

    elif term == "underlap":
        css = (
            f"@keyframes ula_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(-22px,-58px)}}}}"
            f".ulpa_{slug}{{animation:ula_{slug} 2.8s ease-in-out infinite}}"
        )
        content += pl(222, 80, "W", color, filt=sf)
        content += pl(190, 140, "FB", color, cls_name=f"ulpa_{slug}", filt=gf)
        content += f'<path d="M 190 132 Q 174 98 170 58" fill="none" stroke="rgba(255,255,255,.38)" stroke-width="1.5" stroke-dasharray="4 3"/>'
        content += f'<rect x="155" y="36" width="46" height="86" fill="rgba(255,184,0,.1)" rx="4" stroke="rgba(255,184,0,.3)" stroke-width="1"/>'
        content += f'<text x="178" y="80" text-anchor="middle" font-size="7" fill="rgba(255,184,0,.8)" font-family="Nunito,sans-serif" transform="rotate(-90 178 80)">H-SPACE</text>'

    elif term == "cross":
        css = (
            f"@keyframes cra_{slug}{{0%,100%{{stroke-dashoffset:155;opacity:0}}35%,65%{{stroke-dashoffset:0;opacity:.9}}}}"
            f".cra_{slug}{{stroke-dasharray:155;animation:cra_{slug} 2.5s ease-in-out infinite}}"
        )
        content += f'<rect x="88" y="{PAD}" width="104" height="68" fill="rgba(255,255,255,.03)" stroke="rgba(255,255,255,.22)" stroke-width="1"/>'
        content += pl(252, 145, "W", color, filt=sf)
        content += pl(140, 44, "ST", color, filt=gf)
        content += pl(168, 60, "CM", color, filt=sf)
        content += curve(246, 138, 196, 72, 152, 48, col="#FFB800", sw=2.5, cls_name=f"cra_{slug}", marker="tyarr", path_len=155)

    elif term == "final third":
        css = (
            f"@keyframes fta_{slug}{{0%,100%{{opacity:.38}}50%{{opacity:.78}}}}"
            f".ftzone_{slug}{{animation:fta_{slug} 2s ease-in-out infinite}}"
        )
        th = (PH) // 3
        content += f'<rect x="{PAD}" y="{PAD}" width="{PW}" height="{th}" fill="rgba(0,200,117,.32)" class="ftzone_{slug}"/>'
        content += f'<text x="{cx}" y="{PAD+th//2}" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="800" fill="rgba(255,255,255,.92)" font-family="Nunito,sans-serif">FINAL THIRD</text>'
        content += f'<rect x="{PAD}" y="{PAD+th}" width="{PW}" height="{th}" fill="rgba(255,255,255,.03)"/>'
        content += f'<text x="{cx}" y="{PAD+th+th//2}" text-anchor="middle" dominant-baseline="central" font-size="7.5" fill="rgba(255,255,255,.35)" font-family="Nunito,sans-serif">Midfield</text>'
        content += f'<rect x="{PAD}" y="{PAD+2*th}" width="{PW}" height="{th}" fill="rgba(255,255,255,.02)"/>'
        content += f'<text x="{cx}" y="{PAD+2*th+th//2}" text-anchor="middle" dominant-baseline="central" font-size="7.5" fill="rgba(255,255,255,.35)" font-family="Nunito,sans-serif">Defensive Third</text>'
        for px, py in [(118, PAD+22), (150, PAD+38), (174, PAD+22)]:
            content += pl(px, py, "", color, filt=gf, r=8)

    elif term == "counter-attack":
        css = (
            f"@keyframes ca1_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(0,-88px)}}}}"
            f"@keyframes ca2_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(-22px,-78px)}}}}"
            f"@keyframes ca3_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(22px,-74px)}}}}"
            f".cap1_{slug}{{animation:ca1_{slug} 2.2s ease-in-out infinite}}"
            f".cap2_{slug}{{animation:ca2_{slug} 2.2s ease-in-out infinite;animation-delay:.15s}}"
            f".cap3_{slug}{{animation:ca3_{slug} 2.2s ease-in-out infinite;animation-delay:.3s}}"
        )
        content += f'<circle cx="{cx}" cy="148" r="5" fill="#FFB800" filter="url(#{gf})"/>'
        for x, y, cls2 in [(140,152,"cap1"),(112,158,"cap2"),(168,158,"cap3")]:
            content += pl(x, y, "", color, cls_name=f"{cls2}_{slug}", filt=gf)
        for x in [98, 140, 182]:
            content += pl(x, 52, "DEF", "#4a6fa5", filt=sf)

    elif term == "high press":
        css = (
            f"@keyframes hpa_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(-16px,28px)}}}}"
            f"@keyframes hpb_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(0,32px)}}}}"
            f"@keyframes hpc_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(16px,28px)}}}}"
            f".hppa_{slug}{{animation:hpa_{slug} 2.2s ease-in-out infinite}}"
            f".hppb_{slug}{{animation:hpb_{slug} 2.2s ease-in-out infinite;animation-delay:.2s}}"
            f".hppc_{slug}{{animation:hpc_{slug} 2.2s ease-in-out infinite;animation-delay:.4s}}"
        )
        content += pl(98, 58, "DEF", "#4a6fa5", filt=sf)
        content += pl(140, 46, "GK", "#4a6fa5", filt=sf)
        content += pl(182, 58, "DEF", "#4a6fa5", filt=sf)
        content += pl(92, 26, "", color, cls_name=f"hppa_{slug}", filt=gf)
        content += pl(140, 18, "", color, cls_name=f"hppb_{slug}", filt=gf)
        content += pl(188, 26, "", color, cls_name=f"hppc_{slug}", filt=gf)

    elif term == "low block":
        css = (
            f"@keyframes lba_{slug}{{0%,100%{{transform:translateX(0)}}33%{{transform:translateX(18px)}}66%{{transform:translateX(-18px)}}}}"
            f".lbg_{slug}{{animation:lba_{slug} 3s ease-in-out infinite}}"
        )
        content += f'<g class="lbg_{slug}">'
        for x in [52, 96, 140, 184, 228]:
            content += pl(x, 162, "DEF", "#4a6fa5", filt=sf)
        for x in [70, 112, 154, 196]:
            content += pl(x, 136, "DEF", "#4a6fa5", filt=sf)
        content += f'</g>'
        content += pl(140, 72, "ATT", color, filt=gf)

    elif term == "man marking":
        css = (
            f"@keyframes mma_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(22px,-22px)}}}}"
            f".mmpa_{slug}{{animation:mma_{slug} 2.5s ease-in-out infinite}}"
            f".mmpb_{slug}{{animation:mma_{slug} 2.5s ease-in-out infinite;animation-delay:.25s}}"
            f".mmpc_{slug}{{animation:mma_{slug} 2.5s ease-in-out infinite;animation-delay:.5s}}"
        )
        atks = [(80, 58), (140, 70), (200, 54)]
        defs2 = [(80, 92), (140, 104), (200, 88)]
        for x, y in atks:
            content += pl(x, y, "ATT", color, filt=gf)
        for (x, y), cls2 in zip(defs2, [f"mmpa_{slug}", f"mmpb_{slug}", f"mmpc_{slug}"]):
            content += pl(x, y, "DEF", "#4a6fa5", cls_name=cls2, filt=sf)
        for (ax, ay), (dx, dy) in zip(atks, defs2):
            content += f'<line x1="{ax}" y1="{ay+10}" x2="{dx}" y2="{dy-10}" stroke="rgba(255,184,0,.6)" stroke-width="1.5" stroke-dasharray="3 2"/>'

    elif term == "zonal marking":
        css = (
            f"@keyframes zma_{slug}{{0%,100%{{opacity:.35}}50%{{opacity:.75}}}}"
            f".zma_{slug}{{animation:zma_{slug} 2s ease-in-out infinite}}"
            f".zmb_{slug}{{animation:zma_{slug} 2s ease-in-out infinite;animation-delay:.66s}}"
            f".zmc_{slug}{{animation:zma_{slug} 2s ease-in-out infinite;animation-delay:1.33s}}"
        )
        zw = PW // 3
        cols = ["rgba(0,200,117,.22)", "rgba(255,184,0,.18)", "rgba(242,130,127,.22)"]
        clss = [f"zma_{slug}", f"zmb_{slug}", f"zmc_{slug}"]
        for i, (col, cls2) in enumerate(zip(cols, clss)):
            content += f'<rect x="{PAD+i*zw}" y="{cy-36}" width="{zw}" height="82" fill="{col}" class="{cls2}"/>'
        for i, x in enumerate([PAD+zw//2, PAD+zw+zw//2, PAD+2*zw+zw//2]):
            content += pl(x, cy+28, "DEF", "#4a6fa5", filt=sf)
        content += pl(82, cy-12, "ATT", color, filt=gf)
        content += pl(168, cy-6, "ATT", color, filt=gf)

    elif term == "tackle":
        css = (
            f"@keyframes tka_{slug}{{0%,60%{{transform:translate(0,0) rotate(0deg)}}80%{{transform:translate(24px,-12px) rotate(-38deg)}}100%{{transform:translate(0,0) rotate(0deg)}}}}"
            f".tkpa_{slug}{{animation:tka_{slug} 2s ease-in-out infinite}}"
        )
        content += pl(164, 90, "ATT", color, filt=gf)
        content += f'<circle cx="178" cy="90" r="5" fill="#FFB800" filter="url(#{sf})"/>'
        content += pl(112, 112, "DEF", "#4a6fa5", cls_name=f"tkpa_{slug}", filt=sf)
        content += f'<path d="M 118 106 Q 140 97 160 94" fill="none" stroke="rgba(255,255,255,.3)" stroke-width="1.5" stroke-dasharray="3 2"/>'

    elif term == "interception":
        css = (
            f"@keyframes ica_{slug}{{0%,100%{{transform:translate(0,0)}}40%{{transform:translate(32px,-24px)}}60%{{transform:translate(32px,-24px)}}80%{{transform:translate(0,0)}}}}"
            f"@keyframes icba_{slug}{{0%,100%{{stroke-dashoffset:100;opacity:.5}}40%{{stroke-dashoffset:0;opacity:.9}}60%{{stroke-dashoffset:100;opacity:.3}}}}"
            f".icpa_{slug}{{animation:ica_{slug} 2.8s ease-in-out infinite}}"
            f".icba_{slug}{{stroke-dasharray:100;animation:icba_{slug} 2.8s ease-in-out infinite}}"
        )
        content += pl(55, 150, "P1", "#4a6fa5", filt=sf)
        content += pl(225, 52, "P2", "#4a6fa5", filt=sf)
        content += curve(63, 142, 130, 100, 175, 72, col="rgba(255,255,255,.5)", sw=2, cls_name=f"icba_{slug}", marker="tarr", path_len=100)
        content += pl(138, 102, "INT", "#00C875", cls_name=f"icpa_{slug}", filt=gf)

    elif term == "counter-pressing":
        css = (
            f"@keyframes cpa_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(16px,-20px)}}}}"
            f"@keyframes cpb_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(-4px,-26px)}}}}"
            f"@keyframes cpc_{slug}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(-20px,-16px)}}}}"
            f".cppa_{slug}{{animation:cpa_{slug} 1.8s ease-in-out infinite}}"
            f".cppb_{slug}{{animation:cpb_{slug} 1.8s ease-in-out infinite;animation-delay:.2s}}"
            f".cppc_{slug}{{animation:cpc_{slug} 1.8s ease-in-out infinite;animation-delay:.4s}}"
        )
        content += pl(140, 100, "BC", "#4a6fa5", filt=sf)
        content += f'<circle cx="155" cy="100" r="5" fill="#FFB800" filter="url(#{sf})"/>'
        content += pl(98, 118, "", color, cls_name=f"cppa_{slug}", filt=gf)
        content += pl(140, 130, "", color, cls_name=f"cppb_{slug}", filt=gf)
        content += pl(182, 118, "", color, cls_name=f"cppc_{slug}", filt=gf)

    elif term == "transition":
        css = (
            f"@keyframes tra_{slug}{{0%,40%{{transform:translate(0,0)}}60%,100%{{transform:translate(0,-76px)}}}}"
            f"@keyframes trb_{slug}{{0%,40%{{transform:translate(0,0)}}60%,100%{{transform:translate(0,76px)}}}}"
            f".trpa_{slug}{{animation:tra_{slug} 3s ease-in-out infinite}}"
            f".trpb_{slug}{{animation:trb_{slug} 3s ease-in-out infinite}}"
        )
        content += f'<g class="trpa_{slug}">'
        for x in [98, 140, 182]:
            content += pl(x, 140, "", color, filt=gf)
        content += f'</g><g class="trpb_{slug}">'
        for x in [98, 140, 182]:
            content += pl(x, 72, "DEF", "#4a6fa5", filt=sf)
        content += f'</g>'
        content += f'<circle cx="{cx}" cy="{cy}" r="5" fill="#FFB800" filter="url(#{gf})"/>'

    elif term == "formation":
        css = (
            f"@keyframes fma_{slug}{{0%,28%{{opacity:1}}38%,58%{{opacity:0}}68%,100%{{opacity:1}}}}"
            f"@keyframes fmb_{slug}{{0%,28%{{opacity:0}}38%,58%{{opacity:1}}68%,100%{{opacity:0}}}}"
            f".fm433_{slug}{{animation:fma_{slug} 3.5s ease-in-out infinite}}"
            f".fm442_{slug}{{animation:fmb_{slug} 3.5s ease-in-out infinite}}"
        )
        content += f'<g class="fm433_{slug}">'
        content += f'<text x="{cx}" y="30" text-anchor="middle" font-size="11" font-weight="900" fill="{color}" font-family="Nunito,sans-serif">4-3-3</text>'
        for x in [62, 100, 140, 180, 218]: content += pl(x, 66, "", "#4a6fa5", r=7)
        for x in [88, 130, 172]: content += pl(x, 106, "", color, r=7)
        for x in [74, 140, 206]: content += pl(x, 148, "", color, r=7)
        content += f'</g><g class="fm442_{slug}">'
        content += f'<text x="{cx}" y="30" text-anchor="middle" font-size="11" font-weight="900" fill="#00C875" font-family="Nunito,sans-serif">4-4-2</text>'
        for x in [62, 100, 140, 180, 218]: content += pl(x, 66, "", "#1357BE", r=7)
        for x in [62, 100, 140, 180, 218]: content += pl(x, 106, "", "#1357BE", r=7)
        for x in [104, 176]: content += pl(x, 148, "", "#1357BE", r=7)
        content += f'</g>'

    elif term == "shape":
        css = (
            f"@keyframes sha_{slug}{{0%,100%{{transform:translateX(0)}}33%{{transform:translateX(34px)}}66%{{transform:translateX(-24px)}}}}"
            f".shg_{slug}{{animation:sha_{slug} 3s ease-in-out infinite}}"
        )
        content += f'<g class="shg_{slug}">'
        for n, y in [(5, 162), (4, 130), (3, 98)]:
            spacing = 140 // (n - 1) if n > 1 else 0
            for j in range(n):
                x = 70 + j * spacing if n > 1 else cx
                content += pl(x, y, "", "#4a6fa5", r=8)
        content += f'</g>'

    elif term == "width":
        css = (
            f"@keyframes wda_{slug}{{0%,100%{{transform:translateX(0)}}50%{{transform:translateX(-18px)}}}}"
            f"@keyframes wdb_{slug}{{0%,100%{{transform:translateX(0)}}50%{{transform:translateX(18px)}}}}"
            f".wdpl_{slug}{{animation:wda_{slug} 2.5s ease-in-out infinite}}"
            f".wdpr_{slug}{{animation:wdb_{slug} 2.5s ease-in-out infinite}}"
        )
        content += pl(22, cx, "LW", color, cls_name=f"wdpl_{slug}", filt=gf)
        content += pl(258, cx, "RW", color, cls_name=f"wdpr_{slug}", filt=gf)
        for x in [96, 128, 152, 184]:
            content += pl(x, cx, "DEF", "#4a6fa5", filt=sf)
        content += arrow(32, cx, 86, cx, col="rgba(255,255,255,.35)", sw=1.5, marker="tarr", dash="4 3")
        content += arrow(248, cx, 194, cx, col="rgba(255,255,255,.35)", sw=1.5, marker="tarr", dash="4 3")

    elif term == "depth":
        css = (
            f"@keyframes dpa_{slug}{{0%,100%{{opacity:1}}50%{{opacity:.45}}}}"
            f".dpp_{slug}{{animation:dpa_{slug} 2s ease-in-out infinite}}"
            f".dpp2_{slug}{{animation:dpa_{slug} 2s ease-in-out infinite;animation-delay:.5s}}"
            f".dpp3_{slug}{{animation:dpa_{slug} 2s ease-in-out infinite;animation-delay:1s}}"
        )
        content += pl(cx, 30, "ST", color, cls_name=f"dpp_{slug}", filt=gf)
        content += pl(cx, 110, "CM", color, cls_name=f"dpp2_{slug}", filt=sf)
        content += pl(cx, 190, "CB", "#4a6fa5", cls_name=f"dpp3_{slug}", filt=sf)
        content += f'<line x1="{cx}" y1="39" x2="{cx}" y2="101" stroke="rgba(255,255,255,.22)" stroke-width="1.5" stroke-dasharray="4 3"/>'
        content += f'<line x1="{cx}" y1="119" x2="{cx}" y2="181" stroke="rgba(255,255,255,.22)" stroke-width="1.5" stroke-dasharray="4 3"/>'

    elif term == "half-space":
        css = (
            f"@keyframes hsa_{slug}{{0%,100%{{opacity:.35}}50%{{opacity:.78}}}}"
            f".hszl_{slug}{{animation:hsa_{slug} 2s ease-in-out infinite}}"
            f".hszr_{slug}{{animation:hsa_{slug} 2s ease-in-out infinite;animation-delay:1s}}"
        )
        zw = PW // 5
        hs1x = PAD + zw; hs2x = PAD + 3 * zw
        content += f'<rect x="{hs1x}" y="{PAD}" width="{zw}" height="{PH}" fill="rgba(255,184,0,.18)" class="hszl_{slug}"/>'
        content += f'<rect x="{hs2x}" y="{PAD}" width="{zw}" height="{PH}" fill="rgba(255,184,0,.18)" class="hszr_{slug}"/>'
        hs1cx = hs1x + zw // 2; hs2cx = hs2x + zw // 2
        content += f'<text x="{hs1cx}" y="{cy}" text-anchor="middle" font-size="6.5" fill="rgba(255,184,0,.85)" font-family="Nunito,sans-serif" transform="rotate(-90 {hs1cx} {cy})">HALF-SPACE</text>'
        content += f'<text x="{hs2cx}" y="{cy}" text-anchor="middle" font-size="6.5" fill="rgba(255,184,0,.85)" font-family="Nunito,sans-serif" transform="rotate(-90 {hs2cx} {cy})">HALF-SPACE</text>'
        content += pl(hs1cx, 58, "AM", color, filt=gf)
        content += pl(hs2cx, 58, "AM", color, filt=gf)

    elif term == "lines":
        css = (
            f"@keyframes lia_{slug}{{0%,100%{{stroke-dashoffset:145;opacity:0}}35%,65%{{stroke-dashoffset:0;opacity:.95}}}}"
            f".lia_{slug}{{stroke-dasharray:145;animation:lia_{slug} 2.5s ease-in-out infinite}}"
        )
        for x in [62, 104, 148, 192, 232]:
            content += pl(x, 110, "DEF", "#4a6fa5", filt=sf)
        content += f'<line x1="{PAD+4}" y1="110" x2="{PAD+PW-4}" y2="110" stroke="rgba(255,255,255,.15)" stroke-width="1"/>'
        content += curve(98, 162, 130, 110, 165, 58, col="#FFB800", sw=2.5, cls_name=f"lia_{slug}", marker="tyarr", path_len=145)
        content += pl(98, 170, "P1", color, filt=sf)
        content += pl(168, 52, "P2", color, filt=gf)

    elif term == "tiki-taka":
        pts = [(78, 110), (138, 72), (202, 110), (164, 152), (100, 140)]
        css = "".join([
            f"@keyframes tt{i}_{slug}{{0%,{i*20}%{{stroke-dashoffset:70;opacity:0}}{i*20+12}%,{i*20+24}%{{stroke-dashoffset:0;opacity:.92}}{i*20+36}%,100%{{stroke-dashoffset:70;opacity:0}}}}"
            f".tta{i}_{slug}{{stroke-dasharray:70;animation:tt{i}_{slug} 3.5s ease-in-out infinite}}"
            for i in range(5)
        ])
        # Multi-step ball: hops through all 5 players in sync with passes
        # pts[0]=(78,110) is origin → translate offsets to each player
        _tt_offsets = [(p[0]-78, p[1]-110) for p in pts] + [(0, 0)]
        css += (
            f"@keyframes tbm_{slug}{{"
            f"0%,8%{{transform:translate(0px,0px)}}"
            f"18%,28%{{transform:translate({_tt_offsets[1][0]}px,{_tt_offsets[1][1]}px)}}"
            f"38%,48%{{transform:translate({_tt_offsets[2][0]}px,{_tt_offsets[2][1]}px)}}"
            f"58%,68%{{transform:translate({_tt_offsets[3][0]}px,{_tt_offsets[3][1]}px)}}"
            f"78%,88%{{transform:translate({_tt_offsets[4][0]}px,{_tt_offsets[4][1]}px)}}"
            f"98%,100%{{transform:translate(0px,0px)}}}}"
            f"@keyframes tbp_{slug}{{0%,100%{{opacity:1}}50%{{opacity:.7}}}}"
            f".tb_{slug}{{animation:tbm_{slug} 3.5s ease-in-out infinite,tbp_{slug} .7s ease-in-out infinite;}}"
        )
        for x, y in pts:
            content += pl(x, y, "", color, filt=gf)
        pairs = [(0,1),(1,2),(2,3),(3,4),(4,0)]
        for i, (a, b) in enumerate(pairs):
            x1,y1=pts[a]; x2,y2=pts[b]
            content += f'<line class="tta{i}_{slug}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="rgba(255,255,255,.9)" stroke-width="2" marker-end="url(#tarr_{slug})"/>'

    elif term == "total football":
        css = (
            f"@keyframes tfa_{slug}{{0%,100%{{transform:translate(0,0)}}45%,55%{{transform:translate(104px,-42px)}}}}"
            f"@keyframes tfb_{slug}{{0%,100%{{transform:translate(0,0)}}45%,55%{{transform:translate(-104px,42px)}}}}"
            f".tfpa_{slug}{{animation:tfa_{slug} 3.5s ease-in-out infinite}}"
            f".tfpb_{slug}{{animation:tfb_{slug} 3.5s ease-in-out infinite}}"
        )
        content += pl(78, 110, "CB", color, cls_name=f"tfpa_{slug}", filt=gf)
        content += pl(182, 66, "ST", color, cls_name=f"tfpb_{slug}", filt=gf)
        content += f'<path d="M 88 106 Q 130 52 172 70" fill="none" stroke="rgba(255,255,255,.38)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#tarr_{slug})"/>'
        content += f'<path d="M 172 74 Q 130 118 88 114" fill="none" stroke="rgba(255,255,255,.38)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#tarr_{slug})"/>'

    elif term == "positional play":
        css = (
            f"@keyframes ppa_{slug}{{0%,100%{{opacity:.9}}50%{{opacity:.35}}}}"
            f".ppl_{slug}{{animation:ppa_{slug} 2.5s ease-in-out infinite}}"
        )
        pts = [(78, 150), (140, 150), (202, 150), (108, 104), (172, 104), (140, 56)]
        for x, y in pts:
            content += pl(x, y, "", color, filt=sf)
        for a, b in [(0,3),(1,3),(1,4),(2,4),(3,5),(4,5)]:
            x1,y1=pts[a]; x2,y2=pts[b]
            content += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="rgba(255,255,255,.2)" stroke-width="1" stroke-dasharray="3 3" class="ppl_{slug}"/>'

    elif term == "overload":
        css = (
            f"@keyframes ova_{slug}{{0%,100%{{stroke-dashoffset:95;opacity:0}}40%,60%{{stroke-dashoffset:0;opacity:.9}}}}"
            f".ovla_{slug}{{stroke-dasharray:95;animation:ova_{slug} 2.2s ease-in-out infinite}}"
        )
        for x, y in [(58, 80), (88, 60), (68, 114)]:
            content += pl(x, y, "ATT", color, filt=gf)
        for x, y in [(130, 76), (126, 104)]:
            content += pl(x, y, "DEF", "#4a6fa5", filt=sf)
        content += pl(222, 90, "FREE", color, filt=gf)
        content += curve(134, 88, 172, 88, 212, 90, col="#FFB800", sw=2.5, cls_name=f"ovla_{slug}", marker="tyarr", path_len=95)
        content += f'<text x="94" y="148" text-anchor="middle" font-size="8.5" font-weight="800" fill="{color}cc" font-family="Nunito,sans-serif">3 v 2</text>'

    elif term == "third man run":
        css = (
            f"@keyframes tmra_{slug}{{0%,100%{{stroke-dashoffset:68;opacity:0}}22%,48%{{stroke-dashoffset:0;opacity:.92}}58%,100%{{stroke-dashoffset:68;opacity:0}}}}"
            f"@keyframes tmrb_{slug}{{0%,100%{{transform:translate(0,0)}}40%,72%{{transform:translate(-22px,-70px)}}}}"
            f".tmraa_{slug}{{stroke-dasharray:68;animation:tmra_{slug} 3s ease-in-out infinite}}"
            f".tmrba_{slug}{{animation:tmrb_{slug} 3s ease-in-out infinite}}"
        )
        content += pl(78, 150, "P1", color, filt=sf)
        content += pl(152, 112, "P2", color, filt=sf)
        content += curve(87, 143, 116, 128, 144, 120, col="rgba(255,255,255,.82)", sw=2, cls_name=f"tmraa_{slug}", marker="tarr", path_len=68)
        content += pl(202, 150, "P3", color, cls_name=f"tmrba_{slug}", filt=gf)
        content += f'<path d="M 202 142 L 180 66" fill="none" stroke="rgba(255,92,92,.42)" stroke-width="1.5" stroke-dasharray="4 3"/>'

    elif term == "line-breaking pass":
        css = (
            f"@keyframes lbpa_{slug}{{0%,100%{{stroke-dashoffset:135;opacity:0}}35%,65%{{stroke-dashoffset:0;opacity:.95}}}}"
            f".lbpa_{slug}{{stroke-dasharray:135;animation:lbpa_{slug} 2.5s ease-in-out infinite}}"
        )
        for x in [62, 104, 148, 192, 232]:
            content += pl(x, 108, "DEF", "#4a6fa5", filt=sf)
        for x in [78, 120, 162, 202]:
            content += pl(x, 136, "DEF", "#4a6fa5", r=7, filt=sf)
        content += pl(cx, 170, "P1", color, filt=sf)
        content += pl(cx, 50, "P2", color, filt=gf)
        content += curve(cx, 162, cx, 110, cx, 59, col="#FFB800", sw=3, cls_name=f"lbpa_{slug}", marker="tyarr", path_len=135)

    else:
        content += f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" font-size="10" fill="rgba(255,255,255,.35)" font-family="Nunito,sans-serif">{term.upper()}</text>'

    # ── Consistent styled ball (same as tactical pitch) ──
    # Config: (bx, by) for static pulse, or (bx, by, ex, ey, dur) for moving
    _TERM_BALL = {
        "pressing":          (140, 95),
        "pivot":             (154, 100, 42, 118, "3s"),
        "false nine":        (140, 62),
        "build-up play":     (140, 194),  # CSS already set in term block above
        "through ball":      (94, 146, 196, 46, "2.5s"),
        "switch of play":    (22, 110, 258, 110, "2.8s"),
        "overlap":           (196, 108),
        "underlap":          (222, 80),
        "cross":             (246, 138, 152, 48, "2.5s"),
        "final third":       (140, 26),
        "counter-attack":    (140, 148, 140, 60, "2.2s"),
        "high press":        (140, 46),
        "low block":         (140, 72),
        "man marking":       (80, 58),
        "zonal marking":     (82, 98),
        "tackle":            (178, 90, 138, 106, "2s"),
        "interception":      (63, 142, 138, 102, "2.8s"),
        "counter-pressing":  (155, 100),
        "transition":        (140, 110),
        "formation":         (140, 110),
        "shape":             (140, 62),
        "width":             (140, 110),
        "depth":             (140, 110),
        "half-space":        (86, 58),
        "lines":             (98, 170, 168, 52, "2.5s"),
        "tiki-taka":         (78, 110),   # CSS already set in term block above
        "total football":    (140, 90),
        "positional play":   (140, 150),
        "overload":          (134, 88, 222, 90, "2.2s"),
        "third man run":     (78, 150, 152, 112, "3s"),
        "line-breaking pass":(140, 162, 140, 59, "2.5s"),
    }
    ball_cfg = _TERM_BALL.get(term, (cx, cy))
    bx, by = ball_cfg[0], ball_cfg[1]
    ball_cls = f"tb_{slug}"
    # Only generate ball CSS if the term block didn't already define it
    _ball_css_already_set = f".tb_{slug}" in css
    if not _ball_css_already_set:
        if len(ball_cfg) >= 5:
            ex, ey, bdur = ball_cfg[2], ball_cfg[3], ball_cfg[4]
            bdx, bdy = ex - bx, ey - by
            css += (
                f"@keyframes tbm_{slug}{{0%,15%{{transform:translate(0px,0px)}}35%,65%{{transform:translate({bdx:.0f}px,{bdy:.0f}px)}}85%,100%{{transform:translate(0px,0px)}}}}"
                f"@keyframes tbp_{slug}{{0%,100%{{opacity:1}}50%{{opacity:.7}}}}"
                f".{ball_cls}{{animation:tbm_{slug} {bdur} ease-in-out infinite,tbp_{slug} 1.2s ease-in-out infinite;}}"
            )
        else:
            css += (
                f"@keyframes tbp_{slug}{{0%,100%{{opacity:1}}50%{{opacity:.75}}}}"
                f".{ball_cls}{{animation:tbp_{slug} 1.2s ease-in-out infinite;}}"
            )
    content += (
        f'<g class="{ball_cls}" filter="url(#tbglow_{slug})">'
        f'<circle cx="{bx}" cy="{by}" r="10.5" fill="rgba(255,255,180,.18)" stroke="rgba(255,255,100,.38)" stroke-width="1.5"/>'
        f'<circle cx="{bx}" cy="{by}" r="7.5" fill="white" stroke="rgba(0,0,0,.5)" stroke-width="1.5"/>'
        f'<circle cx="{bx}" cy="{by}" r="7.5" fill="none" stroke="rgba(0,0,0,.2)" stroke-width="3.5" stroke-dasharray="4.5 4"/>'
        f'</g>'
    )

    # ── Guide steps from TACTICAL_TERMS ──
    term_data_g = TACTICAL_TERMS.get(term, {})
    guide_steps_g = term_data_g.get("guide_steps", []) if isinstance(term_data_g, dict) else []
    anim_idea = term_data_g.get("animation_idea", "") if isinstance(term_data_g, dict) else ""

    # Build on-pitch overlay for glossary too
    g_action = [s for s in guide_steps_g if s[1].lower() != "why it works"][:3]
    g_overlay = ""
    if g_action:
        gi = ""
        for idx2, (gicon, gtitle, gdesc, *_) in enumerate(g_action):
            gi += (
                f'<div style="display:flex;gap:.4rem;align-items:flex-start;">'
                f'<span style="min-width:15px;height:15px;border-radius:50%;'
                f'background:rgba({_hex_to_rgb(color)},.35);border:1.5px solid {color};'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:.45rem;font-weight:900;color:white;flex-shrink:0;">{idx2+1}</span>'
                f'<div>'
                f'<div style="font-size:.52rem;font-weight:900;color:rgba(255,255,255,.85);line-height:1.2;">{gicon} {gtitle}</div>'
                f'<div style="font-size:.48rem;font-weight:600;color:rgba(255,255,255,.42);line-height:1.3;margin-top:.05rem;">{gdesc}</div>'
                f'</div></div>'
            )
        g_why = next((s for s in guide_steps_g if s[1].lower() == "why it works"), None)
        g_why_html = ""
        if g_why:
            g_why_html = (
                f'<div style="display:flex;gap:.35rem;align-items:flex-start;margin-top:.2rem;'
                f'padding:.25rem .35rem;border-radius:6px;'
                f'background:linear-gradient(135deg,rgba({_hex_to_rgb(color)},.18),rgba({_hex_to_rgb(color)},.06));'
                f'border:1px solid rgba({_hex_to_rgb(color)},.25);">'
                f'<span style="font-size:.6rem;flex-shrink:0;">{g_why[0]}</span>'
                f'<div style="font-size:.48rem;font-weight:700;color:rgba(255,255,255,.65);line-height:1.3;">'
                f'<span style="color:{color};font-weight:900;">Result:</span> {g_why[2]}</div>'
                f'</div>'
            )
        g_overlay = (
            f'<div style="position:absolute;bottom:0;left:0;right:0;'
            f'background:linear-gradient(to top,rgba(15,28,15,.95) 0%,rgba(15,28,15,.85) 60%,rgba(30,92,30,.0) 100%);'
            f'padding:1.2rem .7rem .45rem;display:flex;flex-direction:column;gap:.3rem;">'
            f'{gi}{g_why_html}</div>'
        )
    elif anim_idea:
        g_overlay = (
            f'<div style="position:absolute;bottom:0;left:0;right:0;'
            f'background:linear-gradient(to top,rgba(15,28,15,.92) 0%,rgba(30,92,30,.0) 100%);'
            f'padding:1rem .7rem .5rem;">'
            f'<div style="font-size:.56rem;color:rgba(255,255,255,.5);font-weight:600;line-height:1.4;">⚽ {anim_idea}</div>'
            f'</div>'
        )

    # ── Style pills ──
    pills = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .72rem;'
        f'border-radius:100px;background:rgba(255,255,255,.08);color:rgba(255,255,255,.75);'
        f'font-size:.6rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;'
        f'border:1px solid rgba(255,255,255,.12);margin:.15rem .15rem 0 0">'
        f'<span style="width:5px;height:5px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0"></span>'
        f'{s}</span>'
        for s in style_tags
    )

    css_block = f"<style>{css}</style>"

    svg = (
        f'<svg viewBox="0 0 {SW} {SH}" xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;width:100%;background:#1e5c1e;">'
        f'{defs}{pitch_bg}{markings}{content}'
        f'</svg>'
    )

    return (
        f'{css_block}'
        f'<div style="background:#0F1C0F;border-radius:20px;overflow:hidden;'
        f'box-shadow:0 8px 32px rgba(0,0,0,.45),0 0 0 1px rgba(255,255,255,.06);">'
        f'<div style="padding:.9rem 1.1rem .6rem;display:flex;align-items:center;'
        f'justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.07);">'
        f'<div>'
        f'<div style="font-size:.64rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase;'
        f'color:{color};margin-bottom:.18rem;">Tactical Concept</div>'
        f'<div style="font-size:.95rem;font-weight:900;color:rgba(255,255,255,.92);letter-spacing:-.02em;">{term.capitalize()}</div>'
        f'</div>'
        f'<span style="background:{color};color:white;font-size:.72rem;font-weight:900;'
        f'padding:.3rem .9rem;border-radius:100px;letter-spacing:.06em;'
        f'box-shadow:0 2px 10px {color}66;">{category}</span>'
        f'</div>'
        # SVG + on-pitch overlay
        f'<div style="position:relative;">{svg}{g_overlay}</div>'
        # Footer — pills only
        f'<div style="padding:.55rem 1rem .75rem;border-top:1px solid rgba(255,255,255,.06);">'
        f'<div style="font-size:.52rem;font-weight:800;letter-spacing:.16em;color:rgba(255,255,255,.25);'
        f'text-transform:uppercase;margin-bottom:.28rem;">Key attributes</div>'
        f'{pills}</div>'
        f'</div>'
    )


def linkify_terms(text, source_page="main", ta=None, tb=None):
    """Replace <b>term</b> with a colored clickable link."""
    extra = ""
    if ta:
        extra += f"&ta={ta}"
    if tb:
        extra += f"&tb={tb}"
    for term in TACTICAL_TERMS:
        text = text.replace(
            f'<b>{term}</b>',
            f'<a href="?term={term}&from={source_page}{extra}" class="term-link">{term}</a>'
        )
    return text

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
:root {
  --bg:#FFFAF3; --beige:#FFE8C8;
  --green:#00C875; --green-lt:#CCFFE9; --green-dk:#007A47;
  --red:#FF5C5C; --red-lt:#FFE0E0; --red-dk:#CC1F1F;
  --yellow:#FFB800; --yellow-lt:#FFF3CC; --yellow-dk:#7A5500;
  --purple:#8B5CF6; --purple-lt:#EDE9FE;
  --dark:#1A1A2E; --mid:#5A5A7A; --white:#FFFFFF;
  --radius:22px; --shadow:0 4px 20px rgba(26,26,46,0.08); --shadow-lg:0 8px 32px rgba(26,26,46,0.14);
}
#MainMenu,header,footer{visibility:hidden;}
.block-container{padding-top:0!important;padding-bottom:3rem;max-width:1200px;margin:0 auto;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background-color:var(--bg)!important;font-family:'Nunito',sans-serif!important;}
[data-testid="stVerticalBlock"]{background:transparent;}
[data-testid="stMain"]::before{content:'';display:block;height:5px;background:linear-gradient(90deg,var(--green) 0%,var(--yellow) 40%,var(--red) 70%,var(--purple) 100%);border-radius:0 0 8px 8px;margin-bottom:1.5rem;}

/* Live dot */
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.75);}}
.live-badge{display:inline-flex;align-items:center;gap:.45rem;background:rgba(124,201,154,.12);border:1.5px solid rgba(124,201,154,.35);border-radius:100px;padding:.3rem .85rem;font-size:.68rem;font-weight:800;color:var(--green);letter-spacing:.1em;text-transform:uppercase;}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);flex-shrink:0;animation:pulse-dot 1.8s ease-in-out infinite;}

/* Header */
.app-header{background:var(--dark);background-image:radial-gradient(ellipse at 80% 50%,rgba(0,200,117,.12) 0%,transparent 60%);border-radius:var(--radius);padding:1.8rem 2.5rem;margin-bottom:.8rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;}
.app-title{font-size:clamp(2rem,4vw,3.4rem);font-weight:900;color:var(--white);letter-spacing:-.03em;line-height:1;}
.app-title span{color:var(--yellow);}
.app-sub{font-size:.75rem;font-weight:700;color:rgba(255,255,255,.4);letter-spacing:.15em;text-transform:uppercase;margin-top:.5rem;}
.app-header-right{display:flex;flex-direction:column;align-items:flex-end;gap:.8rem;}
.app-badges{display:flex;gap:.5rem;flex-wrap:wrap;justify-content:flex-end;}
.app-badge{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:100px;padding:.28rem .8rem;font-size:.7rem;font-weight:700;color:rgba(255,255,255,.5);}

/* ── Nav bar ── */
.nav-bar{display:flex;align-items:center;gap:.4rem;background:var(--white);border:2px solid var(--beige);border-radius:var(--radius);padding:.5rem .6rem;margin-bottom:1.5rem;box-shadow:var(--shadow);}
.nav-btn{flex:1;padding:.6rem 1rem;border-radius:14px;font-size:.78rem;font-weight:800;letter-spacing:.06em;text-align:center;cursor:pointer;border:none;font-family:'Nunito',sans-serif;transition:all .15s;background:transparent;color:var(--mid);}
.nav-btn:hover{background:var(--bg);color:var(--dark);}
.nav-btn.active{background:var(--dark);color:var(--white);}

/* Nav buttons via Streamlit — override all buttons inside nav container */
div[data-testid="stHorizontalBlock"] button[kind="secondary"]{
  background:transparent!important;color:var(--mid)!important;
  border:none!important;border-radius:14px!important;
  font-family:'Nunito',sans-serif!important;font-weight:800!important;
  font-size:.78rem!important;letter-spacing:.06em!important;
  transition:all .15s!important;width:100%!important;
}
div[data-testid="stHorizontalBlock"] button[kind="primary"]{
  background:var(--dark)!important;color:var(--white)!important;
  border:none!important;border-radius:14px!important;
  font-family:'Nunito',sans-serif!important;font-weight:800!important;
  font-size:.78rem!important;letter-spacing:.06em!important;
  width:100%!important;
}

/* Pill */
.pill{display:inline-block;padding:.3rem 1rem;border-radius:100px;font-size:.72rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;}
.pill-green{background:var(--green-lt);color:var(--green-dk);}
.pill-red{background:var(--red-lt);color:var(--red-dk);}
.pill-yellow{background:var(--yellow-lt);color:var(--yellow-dk);}

/* Divider */
.div{border:none;border-top:2px dashed var(--beige);margin:2rem 0;}

/* Section */
.sec-label{font-size:.68rem;font-weight:800;letter-spacing:.22em;text-transform:uppercase;color:var(--mid);margin-bottom:.3rem;}
.sec-title{font-size:1.55rem;font-weight:900;color:var(--dark);letter-spacing:-.02em;margin-bottom:1.2rem;}

/* Term link */
.term-link{color:var(--yellow-dk);font-weight:900;text-decoration:none;background:var(--yellow-lt);border-radius:6px;padding:0 5px;border-bottom:2px solid var(--yellow);transition:background .15s,color .15s;}
.term-link:hover{background:var(--yellow);color:var(--dark);}

/* Match card */
.match-card{background:var(--white);border-radius:var(--radius);border:2px solid var(--beige);padding:1.2rem 1.8rem;box-shadow:var(--shadow);margin-bottom:.5rem;display:flex;align-items:center;gap:1rem;}
.match-card-label{font-size:.7rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--mid);flex:1;}
.match-vs-badge{background:var(--dark);color:var(--white);font-size:.7rem;font-weight:900;letter-spacing:.12em;padding:.28rem .85rem;border-radius:100px;}

/* Team picker cards */
.team-pick-card{background:var(--white);border-radius:var(--radius) var(--radius) 0 0;border:2px solid var(--beige);border-bottom:none;padding:1rem 1.2rem .85rem;box-shadow:var(--shadow);position:relative;}
.team-pick-a{border-top:4px solid var(--green);}
.team-pick-b{border-top:4px solid var(--red);}
.team-pick-label{font-size:.6rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;display:block;margin-bottom:.5rem;}
.team-pick-label-a{color:var(--green-dk);}
.team-pick-label-b{color:var(--red-dk);}
.team-pick-info{display:flex;align-items:center;gap:.75rem;}
.team-pick-meta{display:flex;flex-direction:column;gap:.1rem;}
.team-pick-name{font-size:1.05rem;font-weight:900;color:var(--dark);letter-spacing:-.02em;line-height:1.2;}
.team-pick-stat{font-size:.72rem;font-weight:700;color:var(--mid);}
.vs-mid-pill{display:flex;align-items:center;justify-content:center;background:var(--dark);color:var(--white);font-size:.8rem;font-weight:900;letter-spacing:.06em;border-radius:100px;padding:.35rem .45rem;margin-top:2rem;}

/* Team card */
.team-card{background:var(--white);border-radius:var(--radius);overflow:hidden;border:2px solid var(--beige);box-shadow:var(--shadow);transition:box-shadow .2s,transform .2s;position:relative;}
.team-card::before{content:'';position:absolute;top:0;left:0;right:0;height:5px;}
.card-a::before{background:var(--green);}
.card-b::before{background:var(--red);}
.team-card:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px);}
.team-card-header{padding:1rem 1.5rem;font-size:.82rem;font-weight:900;letter-spacing:.06em;text-transform:uppercase;color:var(--dark);display:flex;align-items:center;gap:.6rem;}
.team-card-header .badge{margin-left:auto;font-size:.62rem;font-weight:800;letter-spacing:.1em;padding:.2rem .6rem;border-radius:100px;}
.team-card-body{padding:1.2rem 1.5rem 1rem;font-size:.92rem;line-height:1.9;color:var(--mid);font-weight:600;border-top:1px solid var(--beige);}
.style-summary{margin-bottom:.4rem;}
.style-details{margin-top:.8rem;border-top:1px solid var(--beige);padding-top:.8rem;}
details.style-acc summary{list-style:none;cursor:pointer;display:inline-flex;align-items:center;font-size:.72rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--dark);background:var(--beige);border-radius:100px;padding:.25rem .75rem;margin-top:.5rem;user-select:none;}
details.style-acc summary::-webkit-details-marker{display:none;}
details.style-acc summary::after{content:'+ More details';}
details.style-acc[open] summary::after{content:'− Less';}
details.style-acc .style-details{animation:fadeIn .2s ease;}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
.card-a .team-card-header{background:var(--green-lt);}
.card-a .badge{background:var(--green);color:#fff;}
.card-b .team-card-header{background:var(--red-lt);}
.card-b .badge{background:var(--red);color:#fff;}
.team-stats-row{display:flex;gap:.5rem;padding:.8rem 1.5rem 1rem;border-top:1px solid var(--beige);flex-wrap:wrap;}
.team-stat-box{background:var(--bg);border-radius:10px;padding:.4rem .7rem;text-align:center;flex:1;min-width:44px;}
.team-stat-box-num{font-size:1.05rem;font-weight:900;color:var(--dark);line-height:1.1;}
.team-stat-box-lbl{font-size:.58rem;font-weight:800;color:var(--mid);letter-spacing:.08em;text-transform:uppercase;}

/* Stat comparison */
.stat-cmp-card{background:var(--white);border-radius:var(--radius);border:2px solid var(--beige);overflow:hidden;box-shadow:var(--shadow);transition:box-shadow .2s,transform .2s;}
.stat-cmp-card:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px);}
.stat-cmp-header{padding:.85rem 1.2rem;font-size:.7rem;font-weight:900;text-transform:uppercase;letter-spacing:.12em;color:var(--dark);border-bottom:1px solid rgba(42,32,24,.06);display:flex;align-items:center;gap:.5rem;}
.stat-cmp-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.stat-cmp-hdr-1{background:var(--green-lt);} .stat-cmp-hdr-2{background:var(--yellow-lt);} .stat-cmp-hdr-3{background:var(--red-lt);}
.stat-cmp-dot-1{background:var(--green);} .stat-cmp-dot-2{background:var(--yellow);} .stat-cmp-dot-3{background:var(--red);}
.stat-cmp-body{padding:1rem 1.2rem 1.2rem;display:flex;flex-direction:column;gap:.85rem;}
.stat-cmp-row{display:flex;align-items:center;gap:.7rem;}
.stat-cmp-logo{width:22px;height:22px;object-fit:contain;flex-shrink:0;}
.stat-cmp-lbl{font-size:.72rem;font-weight:800;color:var(--dark);width:52px;min-width:52px;max-width:52px;letter-spacing:-.01em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.stat-bar-track{flex:1;background:var(--beige);border-radius:100px;height:10px;overflow:hidden;}
.stat-bar-fill{height:100%;border-radius:100px;transition:width .4s ease;}
.stat-bar-fill-a{background:var(--green);} .stat-bar-fill-b{background:var(--red);}
.stat-cmp-val{font-size:.8rem;font-weight:900;color:var(--dark);min-width:22px;text-align:right;}
.stat-cmp-foot{font-size:.6rem;font-weight:700;color:var(--mid);letter-spacing:.08em;text-transform:uppercase;margin-top:.4rem;border-top:1px solid var(--beige);padding-top:.6rem;}

/* Standings */
.standings-card{background:var(--white);border-radius:var(--radius);border:2px solid var(--beige);overflow:hidden;box-shadow:var(--shadow);}
.standings-header{padding:.9rem 1.4rem;background:var(--dark);}
.standings-header-title{font-size:.72rem;font-weight:900;text-transform:uppercase;letter-spacing:.14em;color:rgba(255,255,255,.5);}
.standings-row{display:flex;align-items:center;gap:.7rem;padding:.6rem 1.4rem;border-bottom:1px solid var(--beige);font-size:.82rem;transition:background .15s;}
.standings-row:last-child{border-bottom:none;}
.standings-row:hover{background:var(--bg);}
.standings-row.highlighted-a{background:var(--green-lt)!important;}
.standings-row.highlighted-b{background:var(--red-lt)!important;}
.standings-pos{font-size:.68rem;font-weight:900;color:var(--mid);min-width:18px;text-align:center;}
.standings-crest{width:20px;height:20px;object-fit:contain;flex-shrink:0;}
.standings-name{font-weight:800;color:var(--dark);flex:1;font-size:.82rem;}
.standings-pts{font-weight:900;color:var(--dark);min-width:26px;text-align:right;}
.standings-stat{font-weight:700;color:var(--mid);font-size:.75rem;min-width:22px;text-align:right;}
.standings-gd{font-weight:700;min-width:30px;text-align:right;font-size:.75rem;}
.standings-gd-pos{color:var(--green-dk);} .standings-gd-neg{color:var(--red-dk);} .standings-gd-neu{color:var(--mid);}
.standings-hdr-row{display:flex;align-items:center;gap:.7rem;padding:.5rem 1.4rem;background:var(--bg);font-size:.62rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--mid);}
.standings-summary{background:var(--white);border-radius:var(--radius);border:2px solid var(--beige);box-shadow:var(--shadow);padding:1rem 1.4rem;margin-top:.75rem;}
.standings-summary-title{font-size:.82rem;font-weight:900;color:var(--dark);margin-bottom:.45rem;letter-spacing:-.01em;}
.standings-summary p{margin:0;font-size:.85rem;font-weight:600;color:var(--mid);line-height:1.75;}

/* Glossaire */
.glos-card{background:var(--white);border-radius:var(--radius);border:2px solid var(--beige);overflow:hidden;box-shadow:var(--shadow);transition:box-shadow .2s,transform .2s;margin-bottom:.8rem;}
.glos-card:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px);}
.glos-card-header{padding:1.1rem 1.6rem;display:flex;align-items:center;gap:.8rem;}
.glos-card-icon{width:36px;height:36px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;}
.glos-card-term{font-size:1.15rem;font-weight:900;color:var(--dark);letter-spacing:-.02em;}
.glos-card-body{font-size:.95rem;font-weight:600;line-height:1.85;color:var(--mid);padding:.8rem 1.6rem 1.4rem;border-top:1px solid var(--beige);}

/* Watch card */
.watch-card{background:var(--dark);border-radius:var(--radius);padding:2rem 2.2rem;box-shadow:var(--shadow-lg);background-image:radial-gradient(ellipse at 0% 100%,rgba(245,208,110,.06) 0%,transparent 50%);}
.watch-header{display:flex;align-items:center;gap:.8rem;margin-bottom:1.6rem;padding-bottom:1.2rem;border-bottom:1px solid rgba(255,255,255,.06);}
.watch-icon{width:36px;height:36px;background:var(--yellow);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;}
.watch-title{font-size:.7rem;font-weight:900;text-transform:uppercase;letter-spacing:.18em;color:rgba(255,255,255,.4);}
.watch-subtitle{font-size:1.05rem;font-weight:800;color:var(--white);letter-spacing:-.01em;}
.watch-item{display:flex;gap:1rem;align-items:flex-start;padding:.9rem 0 .9rem .8rem;border-bottom:1px solid rgba(255,255,255,.05);border-left:2px solid transparent;transition:border-color .2s;}
.watch-item:hover{border-left-color:rgba(255,255,255,.15);}
.watch-item:last-child{border-bottom:none;padding-bottom:0;}
.watch-num{font-size:.65rem;font-weight:900;color:rgba(255,255,255,.18);min-width:1.2rem;margin-top:.2rem;}
.watch-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:.3rem;}
.watch-text{font-size:.95rem;font-weight:600;color:#C8BAA0;line-height:1.6;}
.watch-challenge-divider{height:1px;background:rgba(255,255,255,.06);margin:1rem 0 1.2rem;}
.watch-challenge-label{font-size:.62rem;font-weight:900;letter-spacing:.18em;text-transform:uppercase;color:rgba(255,255,255,.25);margin-bottom:.9rem;}
.watch-challenge-grid{display:flex;gap:.8rem;}
.watch-challenge-card{flex:1;border-radius:14px;padding:.9rem 1.1rem;display:flex;flex-direction:column;gap:.35rem;}
.watch-challenge-card-a{background:rgba(124,201,154,.1);border:1px solid rgba(124,201,154,.2);}
.watch-challenge-card-b{background:rgba(242,130,127,.1);border:1px solid rgba(242,130,127,.2);}
.watch-challenge-team{font-size:.62rem;font-weight:900;letter-spacing:.1em;text-transform:uppercase;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.watch-challenge-team-a{color:var(--green);}
.watch-challenge-team-b{color:var(--red);}
.watch-challenge-text{font-size:.88rem;font-weight:600;color:#C8BAA0;line-height:1.5;}

/* Terrain */
.terrain-wrap{background:#3B7A3B;background-image:repeating-linear-gradient(0deg,transparent 0px,transparent 30px,rgba(0,0,0,.05) 30px,rgba(0,0,0,.05) 60px);border-radius:var(--radius);height:300px;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;border:3px solid #2A5A2A;box-shadow:var(--shadow-lg);}
.terrain-wrap::after{content:'';position:absolute;left:4%;right:4%;top:50%;height:2px;background:rgba(255,255,255,.3);}
.terrain-center{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:90px;height:90px;border:2px solid rgba(255,255,255,.35);border-radius:50%;}
.terrain-center-dot{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:8px;height:8px;background:rgba(255,255,255,.5);border-radius:50%;}
.terrain-border{position:absolute;inset:10px;border:2px solid rgba(255,255,255,.25);border-radius:10px;}
.terrain-box-top{position:absolute;top:10px;left:50%;transform:translateX(-50%);width:40%;height:24%;border:2px solid rgba(255,255,255,.25);border-top:none;border-radius:0 0 8px 8px;}
.terrain-box-bot{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);width:40%;height:24%;border:2px solid rgba(255,255,255,.25);border-bottom:none;border-radius:8px 8px 0 0;}
.terrain-small-top{position:absolute;top:10px;left:50%;transform:translateX(-50%);width:18%;height:10%;border:2px solid rgba(255,255,255,.2);border-top:none;border-radius:0 0 4px 4px;}
.terrain-small-bot{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);width:18%;height:10%;border:2px solid rgba(255,255,255,.2);border-bottom:none;border-radius:4px 4px 0 0;}
.terrain-label{font-family:'Nunito',sans-serif;font-weight:800;font-size:.8rem;color:rgba(255,255,255,.85);letter-spacing:.12em;text-transform:uppercase;z-index:1;background:rgba(0,0,0,.3);padding:.5rem 1.4rem;border-radius:100px;backdrop-filter:blur(6px);border:1px solid rgba(255,255,255,.1);}

/* Definition page */
.def-hero{background:var(--dark);background-image:radial-gradient(ellipse at 100% 0%,rgba(245,208,110,.08) 0%,transparent 50%);border-radius:var(--radius);padding:2.5rem 2.5rem 3rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.def-hero::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,var(--yellow) 0%,var(--green) 100%);}
.def-title{font-size:clamp(2.8rem,5vw,4.5rem);font-weight:900;color:var(--white);letter-spacing:-.03em;line-height:1;margin:.8rem 0 .5rem;}
.def-category{font-size:.72rem;font-weight:800;color:rgba(255,255,255,.35);letter-spacing:.18em;text-transform:uppercase;}
.def-text{font-size:1.05rem;font-weight:600;line-height:1.9;color:var(--mid);background:var(--white);border-radius:var(--radius);padding:1.8rem 2rem;border:2px solid var(--beige);box-shadow:var(--shadow);margin-bottom:.5rem;}
.def-simple{background:var(--green-lt);border:2px solid var(--green);border-radius:var(--radius);padding:1.4rem 1.8rem;margin-bottom:.5rem;}
.def-example{background:var(--yellow-lt);border:2px solid var(--yellow);border-radius:var(--radius);padding:1.4rem 1.8rem;margin-bottom:.5rem;}
.def-simple p,.def-example p{font-size:1rem;font-weight:600;color:var(--dark);line-height:1.7;margin:.5rem 0 0;}
.def-tag{font-size:.65rem;font-weight:900;letter-spacing:.14em;text-transform:uppercase;padding:.25rem .8rem;border-radius:100px;display:inline-block;}
.def-tag-green{background:var(--green);color:var(--white);}
.def-tag-yellow{background:var(--yellow);color:var(--dark);}

/* Selectbox */
label[data-testid="stWidgetLabel"] p{font-family:'Nunito',sans-serif!important;font-weight:800!important;font-size:.72rem!important;text-transform:uppercase!important;letter-spacing:.1em!important;color:var(--mid)!important;}
[data-testid="stSelectbox"]>div>div{background:var(--white)!important;border:2px solid var(--beige)!important;border-radius:14px!important;font-family:'Nunito',sans-serif!important;font-weight:700!important;}

/* Back button */
button[kind="primary"],[data-testid="stBaseButton-primary"]{background:var(--dark)!important;color:var(--white)!important;border:none!important;border-radius:100px!important;font-family:'Nunito',sans-serif!important;font-weight:800!important;}
</style>
""", unsafe_allow_html=True)

# ── League init (must happen before standings fetch) ─────────────────────────
if "league" not in st.session_state:
    st.session_state.league = "Ligue 1"

# ── Load data ─────────────────────────────────────────────────────────────────
_league_code = LEAGUES[st.session_state.league]["code"]
standings = fetch_standings(_league_code)
ALL_TEAMS = sorted(standings.keys(), key=lambda n: standings[n]["position"]) if standings else []

# ── Check query params (term click from style cards) ──────────────────────────
qp = st.query_params
if "term" in qp and qp["term"] in TACTICAL_TERMS:
    st.session_state.prev_page = qp.get("from", "main")
    if "ta" in qp and qp["ta"] in (ALL_TEAMS or []):
        st.session_state.team_a = qp["ta"]
    if "tb" in qp and qp["tb"] in (ALL_TEAMS or []):
        st.session_state.team_b = qp["tb"]
    st.session_state.active_term = qp["term"]
    st.session_state.page = "definition"
    st.query_params.clear()
    st.rerun()

# ── Navigate to glossary position / formation anchor from pitch page ──────────
if "nav" in qp and qp["nav"] == "glossaire":
    st.session_state.page = "glossaire"
    if qp.get("pos"):
        st.session_state.glossaire_anchor = qp["pos"]
        st.session_state.glossaire_tab = 1   # open Positions tab directly
    elif qp.get("formation"):
        st.session_state.glossaire_anchor = qp["formation"]
        st.session_state.glossaire_tab = 2   # open Composition tab directly
    st.query_params.clear()
    st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("page","main"), ("prev_page","main"), ("active_term",None),
             ("glossaire_anchor", None), ("glossaire_tab", 0),
             ("team_a", ALL_TEAMS[0] if ALL_TEAMS else ""),
             ("team_b", ALL_TEAMS[1] if len(ALL_TEAMS)>1 else "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_crest_img(name, size=28):
    d = standings.get(name, {})
    url = d.get("crest","")
    if url:
        return f'<img src="{url}" width="{size}" height="{size}" style="object-fit:contain;vertical-align:middle">'
    return ""

def gd_class(gd):
    if gd > 0: return "standings-gd-pos"
    if gd < 0: return "standings-gd-neg"
    return "standings-gd-neu"

def bar_pct(v, max_v):
    if not max_v: return 0
    return round(min(v / max_v * 100, 100))

def watch_points(a, b):
    da, db = standings.get(a,{}), standings.get(b,{})
    if not da or not db:
        return ["Data not available."]*3
    played = da.get("played",1) or 1
    gf_a, gf_b = da["goals_for"], db["goals_for"]
    ga_a, ga_b = da["goals_against"], db["goals_against"]
    pts_a, pts_b = da["points"], db["points"]
    return [
        f"<b>{a if gf_a>=gf_b else b}</b> leads offensively: {gf_a} goals for {a} vs {gf_b} for {b} ({gf_a/played:.1f} vs {gf_b/played:.1f} per match).",
        f"Defensive solidity: <b>{a if ga_a<=ga_b else b}</b> concedes less ({ga_a} vs {ga_b} goals conceded). Gap of {abs(ga_a-ga_b)} goals.",
        f"In the standings, <b>{a if pts_a>=pts_b else b}</b> is ahead ({pts_a} pts vs {pts_b} pts). Goal difference: {da['goal_diff']:+} vs {db['goal_diff']:+}.",
    ]

GLOS_ICONS = ["⚡","🎯","🔄","💨","🛡️"]
GLOS_COLORS = ["var(--yellow-lt)","var(--green-lt)","var(--red-lt)","var(--beige)","var(--green-lt)"]

# ── Position cards data ────────────────────────────────────────────────────────
POSITIONS_DATA = {
    "GK":  {"name":"Goalkeeper",              "emoji":"🧤", "color":"var(--yellow-lt)", "border":"var(--yellow)",
            "desc":"The last line of defence, responsible for protecting the goal. The goalkeeper commands their box, organises the defence, and initiates build-up with short passes or long distribution. Modern keepers are increasingly involved as an extra outfield player during possession phases."},
    "CB":  {"name":"Centre-Back",             "emoji":"🛡️", "color":"var(--beige)",     "border":"#bbb",
            "desc":"Central defenders form the backbone of the defensive unit. They win aerial duels, read the game to intercept passes, and engage in one-on-one duels. Ball-playing centre-backs also carry the ball forward and contribute to build-up play from deep positions."},
    "LB":  {"name":"Left-Back",               "emoji":"↙️", "color":"var(--green-lt)",  "border":"var(--green)",
            "desc":"The left full-back defends against right-sided attacks and provides width in possession. In attacking systems, left-backs push high to create overloads on the flank, delivering crosses or cutting inside to combine with central midfielders in the half-space."},
    "RB":  {"name":"Right-Back",              "emoji":"↘️", "color":"var(--green-lt)",  "border":"var(--green)",
            "desc":"The right full-back mirrors the left-back on the opposite flank. Many modern right-backs invert to play as an extra midfielder when in possession, adding numerical superiority in central areas and contributing to the team's attacking combinations."},
    "CM":  {"name":"Central Midfielder",      "emoji":"⚙️", "color":"var(--yellow-lt)", "border":"var(--yellow)",
            "desc":"The engine of the team, linking defence and attack. Central midfielders must win the ball back, distribute it forward, and arrive into goal-scoring positions. Box-to-box midfielders cover every inch of the pitch in both defensive and offensive phases."},
    "LCM": {"name":"Left Central Midfielder", "emoji":"⬅️", "color":"var(--yellow-lt)", "border":"var(--yellow)",
            "desc":"In a 4-3-3, the left-sided central midfielder operates left of the pivot. They support the left flank, make late runs into the box, and offer width in midfield. Usually a more progressive, dynamic profile compared to the more defensive pivot."},
    "RCM": {"name":"Right Central Midfielder","emoji":"➡️", "color":"var(--yellow-lt)", "border":"var(--yellow)",
            "desc":"The right counterpart to the left central midfielder, operating right of the pivot. They provide a passing option across the pitch, press triggers on the right side when defending, and maintain midfield shape by balancing defensive and attacking duties."},
    "DM":  {"name":"Defensive Midfielder",    "emoji":"🔒", "color":"var(--red-lt)",    "border":"var(--red)",
            "desc":"Sitting in front of the defence, the DM screens the backline, intercepts passes, and breaks up attacks. In possession they act as the first distributor, calmly recycling the ball. Their positioning and reading of the game prevent opponents from playing through the middle."},
    "AM":  {"name":"Attacking Midfielder",    "emoji":"🎯", "color":"var(--green-lt)",  "border":"var(--green)",
            "desc":"Operating between the opposition's midfield and defence, the AM is the team's creative hub. They orchestrate attacks, deliver key passes, and arrive in the box to score or assist. Often referred to as the 'number 10', they are typically the most technically gifted player."},
    "LM":  {"name":"Left Midfielder",         "emoji":"⚡", "color":"var(--green-lt)",  "border":"var(--green)",
            "desc":"In a flat 4-4-2, the left midfielder covers the entire left channel — tracking back to defend and pushing forward to create. They combine with the left-back on overlap runs and deliver crosses into the box, offering both width and defensive cover throughout the match."},
    "RM":  {"name":"Right Midfielder",        "emoji":"⚡", "color":"var(--green-lt)",  "border":"var(--green)",
            "desc":"The right counterpart in a four-midfielder system. The right midfielder patrols the right flank, providing defensive cover before transitioning into attack to support the forward line with crosses, cut-backs, and combinations from deep wide positions."},
    "LW":  {"name":"Left Winger",             "emoji":"💨", "color":"var(--red-lt)",    "border":"var(--red)",
            "desc":"Wingers use pace and skill to beat defenders on the flank. A natural left winger hugs the touchline to deliver crosses, while an inverted left winger cuts inside onto their stronger right foot to shoot or play combinations through the half-space in behind."},
    "RW":  {"name":"Right Winger",            "emoji":"💨", "color":"var(--red-lt)",    "border":"var(--red)",
            "desc":"The right winger operates symmetrically to the left. Traditional right wingers deliver crosses from the byline, while inverted right wingers cut inside onto their left foot to shoot or combine centrally. Width from wingers stretches defensive blocks and creates space."},
    "ST":  {"name":"Striker",                 "emoji":"⚽", "color":"var(--red-lt)",    "border":"var(--red)",
            "desc":"The primary goal-scorer, positioned at the top of the formation. Strikers combine movement to create space, hold-up play to bring teammates into attacks, and clinical finishing. In a two-striker system, they work as a pair — one holding, one running in behind."},
    "CF":  {"name":"Centre Forward",          "emoji":"🔥", "color":"var(--red-lt)",    "border":"var(--red)",
            "desc":"Similar to a striker but with greater freedom to drop deep and combine. The centre forward links midfield and attack, creates space for wingers, and acts as the focal point of the attack. In pressing systems, the CF leads the press aggressively from the front."},
}

# ── Formation cards data ───────────────────────────────────────────────────────
FORMATIONS_DATA = {
    "4-3-3":   {"emoji":"📐", "color":"var(--yellow-lt)", "border":"var(--yellow)",
                "desc":"Four defenders, three midfielders, and three forwards. The midfield trio typically includes a defensive pivot and two dynamic central midfielders. The three forwards provide width and a central striker, making it ideal for high-pressing and possession-based styles."},
    "4-4-2":   {"emoji":"🔲", "color":"var(--beige)",     "border":"#bbb",
                "desc":"A classic formation with two banks of four topped by a strike partnership. The flat midfield four provides strong defensive coverage and wide presence, while the two strikers work together as a pair. Compact and hard to break down, it excels in counter-attacking football."},
    "4-2-3-1": {"emoji":"🏗️", "color":"var(--green-lt)",  "border":"var(--green)",
                "desc":"A double pivot of two defensive midfielders shields the defence, with an attacking trio behind a lone striker. The two DMs give security and allow the number 10 to roam freely. Very versatile — can defend deep or press high depending on the team's philosophy."},
    "3-4-3":   {"emoji":"🔺", "color":"var(--red-lt)",    "border":"var(--red)",
                "desc":"Three central defenders supported by two aggressive wing-backs who provide width in both phases. Three midfielders control the centre and three forwards create constant offensive pressure. Requires wing-backs with exceptional stamina to cover the full length of the flanks."},
    "3-5-2":   {"emoji":"⬛", "color":"var(--beige)",     "border":"#bbb",
                "desc":"Three central defenders and five midfielders — including two wing-backs — create a compact, controlled shape. The five-man midfield dominates the centre, while two strikers work in tandem. Excellent for controlling possession and pressing, with wing-backs providing all the width."},
}


# ══════════════════════════════════════════════════════════════════════════════
# HEADER (shown on all pages except definition)
# ══════════════════════════════════════════════════════════════════════════════
def render_header():
    league = st.session_state.get("league", "Ligue 1")
    league_info = LEAGUES.get(league, {})
    flag = league_info.get("flag", "⚽")
    n_teams = len(ALL_TEAMS)
    st.markdown(f"""
<div class="app-header">
<div><div class="app-title">The Football <span>Classroom</span></div><div class="app-sub">Tactical analysis · 5 Leagues</div></div>
<div class="app-header-right">
<div class="live-badge"><span class="live-dot"></span>Live</div>
<div class="app-badges"><span class="app-badge">{flag} {league}</span><span class="app-badge">2025/26</span><span class="app-badge">{n_teams} teams</span></div>
</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
def render_nav():
    page = st.session_state.page
    st.markdown('<div style="background:var(--white);border:2px solid var(--beige);border-radius:22px;padding:.5rem .6rem;margin-bottom:1.5rem;box-shadow:0 4px 20px rgba(42,32,24,0.08);display:flex;gap:.4rem">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        t = "primary" if page == "main" else "secondary"
        if st.button("⚽  Analysis", type=t, use_container_width=True, key="nav_main"):
            st.session_state.page = "main"; st.rerun()
    with c2:
        t = "primary" if page == "classement" else "secondary"
        if st.button("📊  Standings", type=t, use_container_width=True, key="nav_class"):
            st.session_state.page = "classement"; st.rerun()
    with c3:
        t = "primary" if page == "schedule" else "secondary"
        if st.button("📅  Schedule", type=t, use_container_width=True, key="nav_sched"):
            st.session_state.page = "schedule"; st.rerun()
    with c4:
        t = "primary" if page == "regles" else "secondary"
        if st.button("📋  Rules", type=t, use_container_width=True, key="nav_regles"):
            st.session_state.page = "regles"; st.rerun()
    with c5:
        t = "primary" if page == "glossaire" else "secondary"
        if st.button("📖  Glossary", type=t, use_container_width=True, key="nav_glos"):
            st.session_state.page = "glossaire"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE DÉFINITION
# ══════════════════════════════════════════════════════════════════════════════
def page_definition():
    term = st.session_state.active_term
    if not term or term not in TACTICAL_TERMS:
        st.session_state.page = st.session_state.get("prev_page", "main")
        st.rerun()
        return
    term_data = TACTICAL_TERMS.get(term, {})
    definition      = term_data.get("definition",        "Definition coming soon.") if isinstance(term_data, dict) else term_data
    simple          = term_data.get("simple_explanation", "") if isinstance(term_data, dict) else ""
    example         = term_data.get("example",            "") if isinstance(term_data, dict) else ""

    if st.button("← Back", type="primary", key="back"):
        st.session_state.page = st.session_state.get("prev_page", "main")
        st.session_state.active_term = None
        st.rerun()

    st.markdown(f"""<div class="def-hero"><span class="pill pill-yellow">Tactical term</span><div class="def-title">{term.capitalize()}</div><div class="def-category">Glossary · Ligue 1</div></div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="def-text">{definition}</div>', unsafe_allow_html=True)
    if simple:
        st.markdown(f'<div class="def-simple"><span class="def-tag def-tag-green">💡 In simple terms</span><p>{simple}</p></div>', unsafe_allow_html=True)
    if example:
        st.markdown(f'<div class="def-example"><span class="def-tag def-tag-yellow">⚽ Example</span><p>{example}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Visualization</div><div class="sec-title">Tactical illustration</div>', unsafe_allow_html=True)
    st.markdown(render_term_animation_html(term), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE GLOSSAIRE
# ══════════════════════════════════════════════════════════════════════════════
def page_glossaire():
    # ── Consume anchor set when navigating from pitch page ────────────────────
    anchor = st.session_state.get("glossaire_anchor") or None
    st.session_state.glossaire_anchor = None

    st.markdown('<div class="sec-label">Vocabulary</div><div class="sec-title">Tactical Glossary</div>', unsafe_allow_html=True)

    # ── Tab switcher (same button pattern as nav bar) ─────────────────────────
    active_tab = st.session_state.get("glossaire_tab", 0)
    st.markdown('<div style="background:var(--white);border:2px solid var(--beige);border-radius:22px;padding:.5rem .6rem;margin-bottom:1.5rem;box-shadow:0 4px 20px rgba(42,32,24,0.08);display:flex;gap:.4rem">', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        if st.button("📖  Tactics", type="primary" if active_tab == 0 else "secondary", use_container_width=True, key="glos_tab0"):
            st.session_state.glossaire_tab = 0; st.rerun()
    with tc2:
        if st.button("🧍  Positions", type="primary" if active_tab == 1 else "secondary", use_container_width=True, key="glos_tab1"):
            st.session_state.glossaire_tab = 1; st.rerun()
    with tc3:
        if st.button("📐  Composition", type="primary" if active_tab == 2 else "secondary", use_container_width=True, key="glos_tab2"):
            st.session_state.glossaire_tab = 2; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 0: Tactics ────────────────────────────────────────────────────────
    if active_tab == 0:
        for i, (term, term_data) in enumerate(TACTICAL_TERMS.items()):
            definition = term_data.get("definition", "") if isinstance(term_data, dict) else term_data
            icon = GLOS_ICONS[i % len(GLOS_ICONS)]
            bg   = GLOS_COLORS[i % len(GLOS_COLORS)]
            st.markdown(
                f'<a href="?term={term}&from=glossaire" style="text-decoration:none;color:inherit">'
                f'<div class="glos-card">'
                f'<div class="glos-card-header">'
                f'<div class="glos-card-icon" style="background:{bg}">{icon}</div>'
                f'<span class="glos-card-term">{term.capitalize()}</span>'
                f'<span class="pill pill-yellow" style="margin-left:auto">Tactical →</span>'
                f'</div>'
                f'<div class="glos-card-body">{definition}</div>'
                f'</div>'
                f'</a>',
                unsafe_allow_html=True,
            )

    # ── Tab 1: Positions ──────────────────────────────────────────────────────
    elif active_tab == 1:
        team_a = st.session_state.get("team_a", "")
        team_b = st.session_state.get("team_b", "")

        id_a = API_FOOTBALL_IDS.get(team_a)
        id_b = API_FOOTBALL_IDS.get(team_b)
        squad_a = fetch_squad_composition(id_a) if id_a else {}
        squad_b = fetch_squad_composition(id_b) if id_b else {}

        def _get_players(squad, abbr):
            if abbr == "GK":
                return squad.get("Goalkeeper", [])[:2]
            elif abbr in ("CB", "LB", "RB"):
                return squad.get("Defender", [])[:3]
            elif abbr in ("CM", "DM", "AM", "LM", "RM", "LCM", "RCM"):
                return squad.get("Midfielder", [])[:3]
            elif abbr in ("LW", "RW", "ST", "CF"):
                return squad.get("Attacker", [])[:3]
            return []

        for abbr, pos in POSITIONS_DATA.items():
            players_a = _get_players(squad_a, abbr)
            players_b = _get_players(squad_b, abbr)

            players_html = ""
            if players_a and team_a:
                players_html += (
                    f'<div style="margin-top:.55rem;font-size:.72rem;font-weight:700;color:var(--mid);">'
                    f'<span style="font-weight:900;color:var(--dark);">{team_a}</span>: {" · ".join(players_a)}</div>'
                )
            if players_b and team_b:
                players_html += (
                    f'<div style="margin-top:.3rem;font-size:.72rem;font-weight:700;color:var(--mid);">'
                    f'<span style="font-weight:900;color:var(--dark);">{team_b}</span>: {" · ".join(players_b)}</div>'
                )

            st.markdown(
                f'<div id="position-{abbr.lower()}" class="glos-card" style="border-color:{pos["border"]};scroll-margin-top:80px;">'
                f'<div class="glos-card-header">'
                f'<div class="glos-card-icon" style="background:{pos["color"]}">{pos["emoji"]}</div>'
                f'<span class="glos-card-term">{abbr}</span>'
                f'<span style="margin-left:.5rem;font-size:.85rem;font-weight:700;color:var(--mid);">{pos["name"]}</span>'
                f'<span class="pill pill-yellow" style="margin-left:auto">Position</span>'
                f'</div>'
                f'<div class="glos-card-body">{pos["desc"]}{players_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Scroll to specific position card if arriving from pitch
        if anchor and anchor.upper() in POSITIONS_DATA:
            st.markdown(
                f'<script>setTimeout(function(){{var el=document.getElementById("position-{anchor.lower()}");'
                f'if(el)el.scrollIntoView({{behavior:"smooth",block:"center"}});}},300);</script>',
                unsafe_allow_html=True,
            )

    # ── Tab 2: Composition (Formations) ───────────────────────────────────────
    elif active_tab == 2:
        for formation, fdata in FORMATIONS_DATA.items():
            fkey = formation.replace("-", "").replace(".", "")
            st.markdown(
                f'<div id="formation-{fkey}" class="glos-card" style="border-color:{fdata["border"]};scroll-margin-top:80px;">'
                f'<div class="glos-card-header">'
                f'<div class="glos-card-icon" style="background:{fdata["color"]}">{fdata["emoji"]}</div>'
                f'<span class="glos-card-term">{formation}</span>'
                f'<span class="pill pill-yellow" style="margin-left:auto">Formation</span>'
                f'</div>'
                f'<div class="glos-card-body">{fdata["desc"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Scroll to specific formation card if arriving from pitch
        if anchor:
            fkey = anchor.replace("-", "").replace(".", "")
            st.markdown(
                f'<script>setTimeout(function(){{var el=document.getElementById("formation-{fkey}");'
                f'if(el)el.scrollIntoView({{behavior:"smooth",block:"center"}});}},300);</script>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CLASSEMENT
# ══════════════════════════════════════════════════════════════════════════════
def page_classement():
    selected_league = st.session_state.get("league", "Ligue 1")
    team_a = st.session_state.team_a
    team_b = st.session_state.team_b

    st.markdown('<div class="sec-label">5 Leagues</div><div class="sec-title">Standings 2025/26</div>', unsafe_allow_html=True)

    tab_labels = [f"{LEAGUES[l]['flag']} {l}" for l in LEAGUES]
    tabs = st.tabs(tab_labels)

    for (league_name, league_info), tab in zip(LEAGUES.items(), tabs):
        with tab:
            league_standings = fetch_standings(league_info["code"])
            league_teams = sorted(league_standings.keys(), key=lambda n: league_standings[n]["position"]) if league_standings else []
            if not league_standings:
                st.markdown("<p>Data not available.</p>", unsafe_allow_html=True)
                continue

            # Only highlight team_a / team_b when viewing the currently selected league
            def _row_cls(name):
                if league_name != selected_league:
                    return ""
                if name == team_a: return "highlighted-a"
                if name == team_b: return "highlighted-b"
                return ""

            first_team_data = league_standings[league_teams[0]] if league_teams else {}

            hdr = '<div class="standings-hdr-row"><span class="standings-pos">#</span><span style="width:20px"></span><span style="flex:1">Team</span><span class="standings-stat">P</span><span class="standings-stat">W</span><span class="standings-stat">D</span><span class="standings-stat">L</span><span class="standings-gd">GD</span><span class="standings-pts">Pts</span></div>'

            rows = ""
            for name in league_teams:
                d = league_standings[name]
                gd = d["goal_diff"]
                gd_str = f"{gd:+}" if gd != 0 else "0"
                cls = _row_cls(name)
                img = f'<img class="standings-crest" src="{d["crest"]}">' if d.get("crest") else '<span style="width:20px"></span>'
                rows += f'<div class="standings-row {cls}"><span class="standings-pos">{d["position"]}</span>{img}<span class="standings-name">{name}</span><span class="standings-stat">{d["played"]}</span><span class="standings-stat">{d["won"]}</span><span class="standings-stat">{d["draw"]}</span><span class="standings-stat">{d["lost"]}</span><span class="standings-gd {gd_class(gd)}">{gd_str}</span><span class="standings-pts">{d["points"]}</span></div>'

            st.markdown(
                f'<div class="standings-card">'
                f'<div class="standings-header"><span class="standings-header-title">Full standings — Matchday {first_team_data.get("played","?")}</span></div>'
                f'{hdr}{rows}'
                f'</div>',
                unsafe_allow_html=True
            )

            # ── AI standings summary ──
            standings_tuple = tuple(
                (name, d["position"], d["points"], d["played"], d["won"], d["goal_diff"])
                for name, d in league_standings.items()
            )
            with st.spinner("Generating league summary…"):
                summary = generate_standings_summary(league_name, standings_tuple)
            if summary:
                parts = summary.split("|||", 1)
                s_title = parts[0] if len(parts) > 1 else ""
                s_body  = parts[1] if len(parts) > 1 else parts[0]
                title_html = f'<div class="standings-summary-title">📊 {s_title}</div>' if s_title else ""
                st.markdown(
                    f'<div class="standings-summary">'
                    f'{title_html}'
                    f'<p>{s_body}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
def page_main():
    team_a = st.session_state.team_a
    team_b = st.session_state.team_b

    # ── League selector ──
    league_names = list(LEAGUES.keys())
    cur_league = st.session_state.get("league", "Ligue 1")
    cur_league_idx = league_names.index(cur_league) if cur_league in league_names else 0
    selected_league = st.selectbox(
        "Championship",
        league_names,
        index=cur_league_idx,
        key="sel_league",
        format_func=lambda l: f"{LEAGUES[l]['flag']}  {l}",
    )
    if selected_league != st.session_state.league:
        st.session_state.league = selected_league
        for k in ["team_a", "team_b"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # ── Integrated VS team selector ──
    col_a, col_mid, col_b = st.columns([10, 1, 10])

    with col_a:
        da_pre = standings.get(team_a, {})
        img_pre_a = get_crest_img(team_a, 44)
        st.markdown(
            f'<div class="team-pick-card team-pick-a">'
            f'<span class="team-pick-label team-pick-label-a">Team A</span>'
            f'<div class="team-pick-info">{img_pre_a}'
            f'<div class="team-pick-meta">'
            f'<div class="team-pick-name">{team_a}</div>'
            f'<div class="team-pick-stat">#{da_pre.get("position","—")} · {da_pre.get("points","—")} pts</div>'
            f'</div></div></div>',
            unsafe_allow_html=True
        )
        idx_a = ALL_TEAMS.index(team_a) if team_a in ALL_TEAMS else 0
        new_a = st.selectbox("Team A", ALL_TEAMS, index=idx_a, key="sel_a", label_visibility="collapsed")
        if new_a != st.session_state.team_a:
            st.session_state.team_a = new_a
            st.rerun()

    with col_mid:
        st.markdown('<div class="vs-mid-pill">VS</div>', unsafe_allow_html=True)

    with col_b:
        remaining = [t for t in ALL_TEAMS if t != st.session_state.team_a]
        if st.session_state.team_b not in remaining:
            st.session_state.team_b = remaining[0] if remaining else ""
        db_pre = standings.get(team_b, {})
        img_pre_b = get_crest_img(team_b, 44)
        st.markdown(
            f'<div class="team-pick-card team-pick-b">'
            f'<span class="team-pick-label team-pick-label-b">Team B</span>'
            f'<div class="team-pick-info">{img_pre_b}'
            f'<div class="team-pick-meta">'
            f'<div class="team-pick-name">{team_b}</div>'
            f'<div class="team-pick-stat">#{db_pre.get("position","—")} · {db_pre.get("points","—")} pts</div>'
            f'</div></div></div>',
            unsafe_allow_html=True
        )
        idx_b = remaining.index(st.session_state.team_b) if st.session_state.team_b in remaining else 0
        new_b = st.selectbox("Team B", remaining, index=idx_b, key="sel_b", label_visibility="collapsed")
        if new_b != st.session_state.team_b:
            st.session_state.team_b = new_b
            st.rerun()

    team_a, team_b = st.session_state.team_a, st.session_state.team_b
    da, db = standings.get(team_a, {}), standings.get(team_b, {})
    crest_a, crest_b = da.get("crest",""), db.get("crest","")

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # Fetch enriched data for both teams
    form_a, ext_a = fetch_team_extended(da.get("id"))
    form_b, ext_b = fetch_team_extended(db.get("id"))
    form_a, form_b = tuple(form_a), tuple(form_b)
    extra_a       = fetch_api_football_stats(team_a, _league_code)
    extra_b       = fetch_api_football_stats(team_b, _league_code)
    all_scorers   = fetch_competition_scorers(_league_code)
    scorers_a     = tuple(all_scorers.get(team_a, [])[:3])
    scorers_b     = tuple(all_scorers.get(team_b, [])[:3])
    prev_standings = fetch_previous_standings(_league_code)
    prev_pos_a    = prev_standings.get(team_a)
    prev_pos_b    = prev_standings.get(team_b)

    def _build_team_card_html(team_name, badge_label, hdr_bg, cards_tuple, form_tuple, stats_dict, crest_url):
        """CSS :target carousel — each panel owns its nav so prev/next arrows always point to the right card."""
        import re as _re
        slug = _re.sub(r'[^a-z0-9]', '_', team_name.lower())
        pill_style = {"W": "background:#CCFFE9;color:#007A47", "D": "background:#FFF3CC;color:#7A5500", "L": "background:#FFE0E0;color:#CC1F1F"}
        form_html = ""
        if form_tuple:
            pills = "".join(
                f'<span style="{pill_style.get(r,"background:#eee;color:#333")};padding:.1rem .4rem;border-radius:5px;font-size:.7rem;font-weight:900;margin-right:.18rem">{r}</span>'
                for r in form_tuple
            )
            form_html = (
                f'<div style="font-size:.6rem;font-weight:800;color:#5A5A7A;letter-spacing:.1em;'
                f'text-transform:uppercase;margin-bottom:.5rem;display:flex;align-items:center;flex-wrap:wrap;gap:.15rem">'
                f'Recent form &nbsp;{pills}</div>'
            )
        card_defs = [
            ("🏆", "The Club",      cards_tuple[0] if len(cards_tuple) > 0 else ""),
            ("⚽", "How They Play", cards_tuple[1] if len(cards_tuple) > 1 else ""),
            ("🎯", "Tactics",       cards_tuple[2] if len(cards_tuple) > 2 else ""),
            ("⭐", "Fun Fact",      cards_tuple[3] if len(cards_tuple) > 3 else ""),
        ]
        arr_style = (
            "display:inline-flex;align-items:center;justify-content:center;"
            "width:28px;height:28px;border-radius:50%;border:2px solid #FFE8C8;"
            "font-size:.9rem;color:#1A1A2E;text-decoration:none;cursor:pointer;"
            "background:none;transition:background .15s;flex-shrink:0"
        )
        dot_base = "display:inline-block;width:7px;height:7px;border-radius:50%;background:#FFE8C8;margin:0 3px;transition:all .2s;text-decoration:none"
        dot_active = "display:inline-block;width:18px;height:7px;border-radius:4px;background:#1A1A2E;margin:0 3px;transition:all .2s;text-decoration:none"
        panels_html = ""
        for i, (icon, label, text) in enumerate(card_defs):
            body = text or "—"
            for term in TACTICAL_TERMS:
                url = f"?term={term}&from=main&ta={team_a}&tb={team_b}"
                body = body.replace(
                    f"<b>{term}</b>",
                    f'<a href="{url}" style="color:#8B5CF6;font-weight:800;text-decoration:underline dotted 2px">{term}</a>'
                )
            prev_id = f"p-{slug}-{(i-1)%4}"
            next_id = f"p-{slug}-{(i+1)%4}"
            dots = "".join(
                f'<a href="#p-{slug}-{j}" style="{dot_active if j==i else dot_base}"></a>'
                for j in range(4)
            )
            nav = (
                f'<div style="display:flex;align-items:center;justify-content:center;gap:.5rem;padding:.45rem 0 .2rem">'
                f'<a href="#{prev_id}" style="{arr_style}">&#8592;</a>'
                f'<span style="display:flex;align-items:center">{dots}</span>'
                f'<a href="#{next_id}" style="{arr_style}">&#8594;</a>'
                f'</div>'
            )
            panels_html += (
                f'<div id="p-{slug}-{i}" class="tc-panel-{slug}">'
                f'<div style="height:210px;overflow-y:auto;padding:.3rem .1rem">'
                f'<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem">'
                f'<span style="font-size:1.6rem;line-height:1">{icon}</span>'
                f'<span style="font-size:.62rem;font-weight:900;text-transform:uppercase;letter-spacing:.13em;'
                f'color:#5A5A7A;background:#FFE8C8;padding:.18rem .7rem;border-radius:100px">{label}</span>'
                f'</div>'
                f'<div style="font-size:1.05rem;font-weight:700;line-height:1.7;color:#1A1A2E">{body}</div>'
                f'</div>'
                f'{nav}'
                f'</div>'
            )
        stats_html = "".join(
            f'<div style="flex:1;text-align:center">'
            f'<div style="font-size:1.1rem;font-weight:900;color:#1A1A2E">{v}</div>'
            f'<div style="font-size:.57rem;font-weight:800;color:#5A5A7A;text-transform:uppercase;letter-spacing:.08em">{l}</div>'
            f'</div>'
            for v, l in [(stats_dict.get("points","—"),"Pts"),(stats_dict.get("won","—"),"W"),
                         (stats_dict.get("draw","—"),"D"),(stats_dict.get("lost","—"),"L"),
                         (stats_dict.get("goals_for","—"),"Goals")]
        )
        crest_tag = (
            f'<img src="{crest_url}" style="width:20px;height:20px;object-fit:contain;margin-right:.45rem;vertical-align:middle" onerror="this.style.display=\'none\'">'
            if crest_url else ""
        )
        # :target shows the navigated panel; :has() shows panel 0 when none is targeted (modern browsers)
        css = (
            f'<style>'
            f'.tc-panel-{slug}{{display:none}}'
            f'.tc-panel-{slug}:target{{display:block}}'
            f'.tc-wrap-{slug}:not(:has(.tc-panel-{slug}:target)) #p-{slug}-0{{display:block}}'
            f'</style>'
        )
        return (
            f'{css}'
            f'<div class="tc-wrap-{slug}" style="background:#fff;border-radius:16px;border:2px solid #FFE8C8;'
            f'overflow:hidden;box-shadow:0 4px 20px rgba(42,32,24,.08);margin-bottom:.5rem">'
            f'<div style="background:{hdr_bg};padding:.7rem 1rem;font-size:.8rem;font-weight:900;'
            f'letter-spacing:.05em;text-transform:uppercase;color:#1A1A2E;display:flex;align-items:center">'
            f'{crest_tag}{team_name}'
            f'<span style="margin-left:auto;font-size:.6rem;font-weight:800;letter-spacing:.1em;'
            f'padding:.18rem .6rem;border-radius:100px;background:rgba(0,0,0,.08)">{badge_label}</span>'
            f'</div>'
            f'<div style="padding:.75rem .9rem .2rem">'
            f'{form_html}'
            f'{panels_html}'
            f'</div>'
            f'<div style="display:flex;border-top:1px solid #FFE8C8;padding:.55rem .4rem .45rem">{stats_html}</div>'
            f'</div>'
        )

    # ── AI Analysis (Playing Style) ──
    st.markdown('<div class="sec-label">AI Analysis</div><div class="sec-title">Playing Style</div>', unsafe_allow_html=True)

    with st.spinner("Generating AI analysis…"):
        style_a_raw = generate_team_style(
            team_a,
            da.get("points",0), da.get("played",1), da.get("won",0), da.get("draw",0), da.get("lost",0),
            da.get("goals_for",0), da.get("goals_against",0), da.get("goal_diff",0), da.get("position",0),
            prev_pos_a,
            form_a, scorers_a,
            extra_a.get("formation"), extra_a.get("gf_avg"), extra_a.get("ga_avg"),
            extra_a.get("wins_home"), extra_a.get("wins_away"), extra_a.get("top_scoring_slot"),
            extra_a.get("passes_pct"), extra_a.get("shots_pg"), extra_a.get("shots_on_pg"),
            extra_a.get("clean_sheets"), extra_a.get("failed_to_score"),
            home_record=ext_a.get("home_record"), away_record=ext_a.get("away_record"),
            clean_sheets_recent=ext_a.get("clean_sheets"), gf_avg_recent=ext_a.get("gf_avg_recent"),
            ga_avg_recent=ext_a.get("ga_avg_recent"), win_pct=ext_a.get("win_pct"),
        )
        style_b_raw = generate_team_style(
            team_b,
            db.get("points",0), db.get("played",1), db.get("won",0), db.get("draw",0), db.get("lost",0),
            db.get("goals_for",0), db.get("goals_against",0), db.get("goal_diff",0), db.get("position",0),
            prev_pos_b,
            form_b, scorers_b,
            extra_b.get("formation"), extra_b.get("gf_avg"), extra_b.get("ga_avg"),
            extra_b.get("wins_home"), extra_b.get("wins_away"), extra_b.get("top_scoring_slot"),
            extra_b.get("passes_pct"), extra_b.get("shots_pg"), extra_b.get("shots_on_pg"),
            extra_b.get("clean_sheets"), extra_b.get("failed_to_score"),
            home_record=ext_b.get("home_record"), away_record=ext_b.get("away_record"),
            clean_sheets_recent=ext_b.get("clean_sheets"), gf_avg_recent=ext_b.get("gf_avg_recent"),
            ga_avg_recent=ext_b.get("ga_avg_recent"), win_pct=ext_b.get("win_pct"),
        )

    # Handle legacy cache returning a plain string
    if isinstance(style_a_raw, str):
        style_a_raw = (style_a_raw, "", "", "")
    if isinstance(style_b_raw, str):
        style_b_raw = (style_b_raw, "", "", "")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            _build_team_card_html(team_a, "Team A", "#CCFFE9", style_a_raw, form_a, da, da.get("crest","")),
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            _build_team_card_html(team_b, "Team B", "#FFE0E0", style_b_raw, form_b, db, db.get("crest","")),
            unsafe_allow_html=True
        )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Terrain ──
    st.markdown('<div class="sec-label">Tactics</div><div class="sec-title">Tactical Pitch</div>', unsafe_allow_html=True)
    pitch_col_a, pitch_col_b = st.columns(2)
    with pitch_col_a:
        st.markdown(render_tactical_pitch_html(team_a), unsafe_allow_html=True)
    with pitch_col_b:
        st.markdown(render_tactical_pitch_html(team_b), unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)


    # ── ML Prediction Card ──
    st.markdown('<div class="sec-label">Machine Learning</div><div class="sec-title">Match Prediction</div>', unsafe_allow_html=True)

    ml_probs, ml_meta = predict_match(
        standings, team_a, team_b,
        form_home=list(form_a), form_away=list(form_b),
        extra_home=ext_a, extra_away=ext_b,
    )
    ml_xg = predict_expected_score(
        standings, team_a, team_b,
        form_home=list(form_a), form_away=list(form_b),
        extra_home=ext_a, extra_away=ext_b,
    )

    if ml_probs is not None and ml_meta is not None:
        hw = int(ml_probs[0] * 100)
        dr = int(ml_probs[1] * 100)
        aw = int(ml_probs[2] * 100)
        shift = ml_meta.get("total_shift", 0)
        shift_pct = abs(int(shift * 100))
        if shift > 0.05:
            momentum_msg = f"<strong>{team_a}</strong> is in better form (+{shift_pct}% boost)"
            momentum_col = "#00C875"
        elif shift < -0.05:
            momentum_msg = f"<strong>{team_b}</strong> is in better form (+{shift_pct}% boost)"
            momentum_col = "#F2827F"
        else:
            momentum_msg = "Both teams in balanced form"
            momentum_col = "#5A5A7A"

        conf = ml_meta.get("confidence", 0)
        if conf > 0.20:
            conf_label = "High confidence"
            conf_col = "#00C875"
        elif conf > 0.10:
            conf_label = "Medium confidence"
            conf_col = "#FFB800"
        else:
            conf_label = "Low confidence · close match"
            conf_col = "#F2827F"

        def _form_pills_main(form_list):
            if not form_list:
                return '<span style="color:var(--mid);font-size:.75rem">—</span>'
            color_map = {"W": "#00C875", "D": "#FFB800", "L": "#FF5C5C"}
            return "".join(
                f'<span style="display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:6px;background:{color_map.get(r,"#888")};color:white;font-size:.65rem;font-weight:900;margin-right:3px">{r}</span>'
                for r in form_list
            )

        xg_block = ""
        if ml_xg:
            mls = ml_xg["most_likely_score"]
            xg_block = (
                f'<div style="flex:1;background:var(--bg);border-radius:14px;padding:1rem 1.2rem;text-align:center;">'
                f'<div style="font-size:.58rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--mid);margin-bottom:.4rem;">Expected Score</div>'
                f'<div style="font-size:2rem;font-weight:900;color:var(--dark);letter-spacing:.05em;line-height:1;">{mls[0]} <span style="color:var(--mid);font-size:1.2rem;">–</span> {mls[1]}</div>'
                f'<div style="font-size:.62rem;color:var(--mid);font-weight:700;margin-top:.4rem;">xG {ml_xg["xg_home"]} · {ml_xg["xg_away"]}</div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:var(--white);border-radius:var(--radius);border:2px solid var(--beige);box-shadow:var(--shadow);padding:1.4rem 1.6rem;">'
            # Top row: probability bar
            f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem;">'
            f'<span style="font-size:.62rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--mid);">Win probability</span>'
            f'<span style="background:{conf_col};color:white;padding:.15rem .55rem;border-radius:100px;font-size:.58rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase;">{conf_label}</span>'
            f'</div>'
            f'<div style="display:flex;height:30px;border-radius:10px;overflow:hidden;box-shadow:inset 0 1px 3px rgba(0,0,0,.08);">'
            f'<div style="width:{hw}%;background:linear-gradient(90deg,#00C875,#00A862);display:flex;align-items:center;justify-content:center;color:white;font-weight:900;font-size:.78rem;">{hw}%</div>'
            f'<div style="width:{dr}%;background:linear-gradient(90deg,#FFB800,#F5A500);display:flex;align-items:center;justify-content:center;color:white;font-weight:900;font-size:.78rem;">{dr}%</div>'
            f'<div style="width:{aw}%;background:linear-gradient(90deg,#FF5C5C,#E63F3F);display:flex;align-items:center;justify-content:center;color:white;font-weight:900;font-size:.78rem;">{aw}%</div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-top:.4rem;font-size:.7rem;font-weight:800;">'
            f'<span style="color:#00A862;">{team_a} win</span>'
            f'<span style="color:#F5A500;">Draw</span>'
            f'<span style="color:#E63F3F;">{team_b} win</span>'
            f'</div>'
            # Middle row: expected score + recent form
            f'<div style="display:flex;gap:1rem;margin-top:1.2rem;">'
            f'{xg_block}'
            f'<div style="flex:1.5;background:var(--bg);border-radius:14px;padding:1rem 1.2rem;">'
            f'<div style="font-size:.58rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--mid);margin-bottom:.5rem;">Recent Form · Last 5</div>'
            f'<div style="display:flex;align-items:center;justify-content:space-between;">'
            f'<div style="display:flex;flex-direction:column;gap:.3rem;">'
            f'<span style="font-size:.72rem;font-weight:800;color:var(--dark);">{team_a}</span>'
            f'<div>{_form_pills_main(form_a)}</div>'
            f'</div>'
            f'<div style="display:flex;flex-direction:column;gap:.3rem;align-items:flex-end;">'
            f'<span style="font-size:.72rem;font-weight:800;color:var(--dark);">{team_b}</span>'
            f'<div>{_form_pills_main(form_b)}</div>'
            f'</div>'
            f'</div></div></div>'
            # Bottom: momentum insight
            f'<div style="margin-top:1rem;padding:.8rem 1rem;background:rgba({",".join(str(int(momentum_col[i:i+2],16)) for i in (1,3,5))},.08);border-left:3px solid {momentum_col};border-radius:8px;">'
            f'<div style="font-size:.58rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:{momentum_col};margin-bottom:.2rem;">📊 Model Insight</div>'
            f'<div style="font-size:.82rem;font-weight:600;color:var(--dark);line-height:1.5;">{momentum_msg}. The model adjusted its season-based prediction using live form data from each team\'s last 5 matches and recent home/away records.</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Stats comparison ──
    st.markdown('<div class="sec-label">Statistics</div><div class="sec-title">Head to head</div>', unsafe_allow_html=True)

    all_gf  = [v["goals_for"]     for v in standings.values()] or [1]
    all_ga  = [v["goals_against"] for v in standings.values()] or [1]
    all_pts = [v["points"]        for v in standings.values()] or [1]
    max_gf, max_ga, max_pts = max(all_gf), max(all_ga), max(all_pts)

    short_a = da.get("short", team_a[:3].upper())
    short_b = db.get("short", team_b[:3].upper())
    gf_a, gf_b   = da.get("goals_for",0), db.get("goals_for",0)
    ga_a, ga_b   = da.get("goals_against",0), db.get("goals_against",0)
    pts_a, pts_b = da.get("points",0), db.get("points",0)
    played_a = da.get("played",1) or 1

    def cmp_card(hdr_cls, dot_cls, title, val_a, val_b, max_v, foot, inverted=False):
        pa = bar_pct(val_a, max_v); pb = bar_pct(val_b, max_v)
        if inverted: pa, pb = 100-pa, 100-pb
        ia = f'<img class="stat-cmp-logo" src="{crest_a}">' if crest_a else ""
        ib = f'<img class="stat-cmp-logo" src="{crest_b}">' if crest_b else ""
        return (f'<div class="stat-cmp-card"><div class="stat-cmp-header {hdr_cls}"><span class="stat-cmp-dot {dot_cls}"></span>{title}</div>'
                f'<div class="stat-cmp-body">'
                f'<div class="stat-cmp-row">{ia}<span class="stat-cmp-lbl">{short_a}</span><div class="stat-bar-track"><div class="stat-bar-fill stat-bar-fill-a" style="width:{pa}%"></div></div><span class="stat-cmp-val">{val_a}</span></div>'
                f'<div class="stat-cmp-row">{ib}<span class="stat-cmp-lbl">{short_b}</span><div class="stat-bar-track"><div class="stat-bar-fill stat-bar-fill-b" style="width:{pb}%"></div></div><span class="stat-cmp-val">{val_b}</span></div>'
                f'<div class="stat-cmp-foot">{foot}</div>'
                f'</div></div>')

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(cmp_card("stat-cmp-hdr-1","stat-cmp-dot-1","Offensive efficiency",gf_a,gf_b,max_gf,f"Avg. {gf_a/played_a:.1f} vs {gf_b/played_a:.1f} goals / match"), unsafe_allow_html=True)
    with r2:
        st.markdown(cmp_card("stat-cmp-hdr-2","stat-cmp-dot-2","Points in standings",pts_a,pts_b,max_pts,f"#{da.get('position','—')} vs #{db.get('position','—')} in the standings"), unsafe_allow_html=True)
    with r3:
        st.markdown(cmp_card("stat-cmp-hdr-3","stat-cmp-dot-3","Defensive solidity",ga_a,ga_b,max_ga,"Goals conceded — lower is better",inverted=True), unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Watch card ──
    points = watch_points(team_a, team_b)
    challenge_a, challenge_b = generate_key_challenges(
        team_a, team_b,
        da.get("points",0), db.get("points",0),
        da.get("goals_for",0), db.get("goals_for",0),
        da.get("goals_against",0), db.get("goals_against",0),
    )
    st.markdown(
        f'<div class="watch-card">'
        f'<div class="watch-header"><div class="watch-icon">👁</div><div><div class="watch-title">Key points to watch</div><div class="watch-subtitle">{team_a} vs {team_b}</div></div></div>'
        f'<div class="watch-item"><div class="watch-num">01</div><div class="watch-dot" style="background:{WATCH_COLORS[0]}"></div><div class="watch-text">{points[0]}</div></div>'
        f'<div class="watch-item"><div class="watch-num">02</div><div class="watch-dot" style="background:{WATCH_COLORS[1]}"></div><div class="watch-text">{points[1]}</div></div>'
        f'<div class="watch-item"><div class="watch-num">03</div><div class="watch-dot" style="background:{WATCH_COLORS[2]}"></div><div class="watch-text">{points[2]}</div></div>'
        f'<div class="watch-challenge-divider"></div>'
        f'<div class="watch-challenge-label">⚡ Key challenge for each team</div>'
        f'<div class="watch-challenge-grid">'
        f'<div class="watch-challenge-card watch-challenge-card-a"><span class="watch-challenge-team watch-challenge-team-a">{team_a}</span><span class="watch-challenge-text">{challenge_a}</span></div>'
        f'<div class="watch-challenge-card watch-challenge-card-b"><span class="watch-challenge-team watch-challenge-team-b">{team_b}</span><span class="watch-challenge-text">{challenge_b}</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
def page_schedule():
    from datetime import datetime, timedelta, timezone
    from collections import defaultdict

    st.markdown('<div class="sec-label">5 Leagues</div><div class="sec-title">Match Schedule</div>', unsafe_allow_html=True)

    # ── League selector (single selection) ──
    if "sched_league" not in st.session_state:
        st.session_state.sched_league = "Ligue 1"

    btn_cols = st.columns(len(LEAGUES))
    for col, (lname, linfo) in zip(btn_cols, LEAGUES.items()):
        with col:
            active = lname == st.session_state.sched_league
            if st.button(f"{linfo['flag']} {lname}", key=f"sched_btn_{lname}",
                         type="primary" if active else "secondary",
                         use_container_width=True):
                st.session_state.sched_league = lname
                st.rerun()

    selected = {st.session_state.sched_league}

    now       = datetime.now(timezone.utc)
    date_from = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    date_to   = (now + timedelta(days=14)).strftime("%Y-%m-%d")

    all_standings = {}
    all_matches = []
    with st.spinner("Loading schedule…"):
        for lname in selected:
            linfo = LEAGUES[lname]
            all_standings[lname] = fetch_standings(linfo["code"])
            matches = fetch_schedule(linfo["code"], date_from, date_to)
            for m in matches:
                all_matches.append({
                    "league":   lname,
                    "color":    linfo.get("color", "#888"),
                    "utcDate":  m["utcDate"],
                    "status":   m["status"],
                    "matchday": m.get("matchday", ""),
                    "home":     m["homeTeam"]["name"],
                    "away":     m["awayTeam"]["name"],
                    "score_h":  m["score"]["fullTime"].get("home"),
                    "score_a":  m["score"]["fullTime"].get("away"),
                })

    if not all_matches:
        st.markdown('<p style="color:var(--mid);text-align:center;padding:2rem 0">No matches found for this period.</p>', unsafe_allow_html=True)
        return

    all_matches.sort(key=lambda m: m["utcDate"])

    by_date = defaultdict(list)
    for m in all_matches:
        dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        local_dt = dt.astimezone()
        m["local_time"] = local_dt.strftime("%H:%M")
        by_date[local_dt.strftime("%A %d %B %Y")].append(m)

    STATUS_MAP = {
        "FINISHED":  ("FIN",  "sched-status-fin"),
        "IN_PLAY":   ("LIVE", "sched-status-live"),
        "PAUSED":    ("HT",   "sched-status-live"),
        "TIMED":     ("",     "sched-status-sched"),
        "SCHEDULED": ("",     "sched-status-sched"),
        "POSTPONED": ("PPD",  "sched-status-sched"),
        "CANCELLED": ("CAN",  "sched-status-sched"),
    }

    # Inject CSS for match cards
    st.markdown("""<style>
.sched-date-group{margin-bottom:1.4rem;}
.sched-date-label{font-size:.68rem;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:var(--mid);margin-bottom:.55rem;padding-left:.2rem;}
.sched-match{display:flex;align-items:center;gap:.7rem;background:var(--white);border-radius:14px;padding:.65rem 1rem;margin-bottom:.45rem;box-shadow:0 2px 8px rgba(26,26,46,.06);border-left:4px solid #ccc;}
.sched-league-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.sched-teams{flex:1;font-size:.88rem;font-weight:700;color:var(--dark);}
.sched-score{font-size:.88rem;font-weight:900;color:var(--dark);min-width:42px;text-align:center;}
.sched-status{font-size:.62rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;padding:.18rem .55rem;border-radius:100px;white-space:nowrap;}
.sched-status-fin{background:var(--green-lt);color:var(--green-dk);}
.sched-status-live{background:#FFE0E0;color:var(--red-dk);animation:pulse 1.2s ease-in-out infinite;}
.sched-status-sched{background:var(--beige);color:var(--mid);}
.sched-matchday{font-size:.62rem;font-weight:700;color:var(--mid);white-space:nowrap;}
.sched-pred{margin-top:.3rem;}
.sched-pred-bar{display:flex;height:4px;border-radius:4px;overflow:hidden;margin-bottom:.15rem;}
.sched-pred-labels{display:flex;justify-content:space-between;font-size:.58rem;font-weight:700;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
</style>""", unsafe_allow_html=True)

    html = ""
    for date_label, matches in by_date.items():
        html += f'<div class="sched-date-group"><div class="sched-date-label">{date_label}</div>'
        for m in matches:
            status_raw = m["status"]
            status_txt, status_cls = STATUS_MAP.get(status_raw, (status_raw[:3], "sched-status-sched"))

            if status_raw == "FINISHED" and m["score_h"] is not None:
                score_html = f'<div class="sched-score">{m["score_h"]} – {m["score_a"]}</div>'
            elif status_raw in ("IN_PLAY", "PAUSED"):
                score_html = f'<div class="sched-score">{m["score_h"]} – {m["score_a"]}</div>'
            else:
                score_html = f'<div class="sched-score" style="color:var(--mid);font-size:.8rem">{m["local_time"]}</div>'

            if status_raw in ("TIMED", "SCHEDULED"):
                status_badge = f'<span class="sched-status {status_cls}">{m["local_time"]}</span>'
            else:
                status_badge = f'<span class="sched-status {status_cls}">{status_txt}</span>'

            md_badge = f'<span class="sched-matchday">MD {m["matchday"]}</span>' if m["matchday"] else ""

            cur_standings = all_standings.get(m["league"], {})
            # Normalize team names (schedule API returns raw names, standings uses mapped names)
            h_norm = TEAM_NAME_MAP.get(m["home"], m["home"])
            a_norm = TEAM_NAME_MAP.get(m["away"], m["away"])
            # Use standings data only — no per-match API calls (prevents rate limiting)
            probs, pred_meta = predict_match(cur_standings, h_norm, a_norm)
            xg = predict_expected_score(cur_standings, h_norm, a_norm)
            pred_html = ""
            if probs is not None and status_raw in ("TIMED", "SCHEDULED"):
                hw = int(probs[0] * 100)
                dr = int(probs[1] * 100)
                aw = int(probs[2] * 100)

                # Build momentum indicator — shows how much recent form affected the prediction
                shift = pred_meta.get("total_shift", 0) if pred_meta else 0
                shift_pct = abs(int(shift * 100))
                if shift > 0.05:
                    momentum_txt = f"↑ {m['home'].split()[-1]} in form (+{shift_pct}%)"
                    momentum_col = "#4CAF50"
                elif shift < -0.05:
                    momentum_txt = f"↑ {m['away'].split()[-1]} in form (+{shift_pct}%)"
                    momentum_col = "#F44336"
                else:
                    momentum_txt = "Form balanced"
                    momentum_col = "var(--mid)"

                # Confidence label
                conf = pred_meta.get("confidence", 0) if pred_meta else 0
                if conf > 0.20:
                    conf_label = "HIGH"
                    conf_col = "#00C875"
                elif conf > 0.10:
                    conf_label = "MEDIUM"
                    conf_col = "#FFB800"
                else:
                    conf_label = "LOW"
                    conf_col = "#F2827F"

                # Expected score line
                xg_html = ""
                if xg:
                    mls = xg["most_likely_score"]
                    xg_html = (
                        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:.5rem;margin-top:.3rem;padding-top:.3rem;border-top:1px dashed var(--beige);">'
                        f'<span style="font-size:.58rem;color:var(--mid);letter-spacing:.06em;text-transform:uppercase;font-weight:800;">Expected score</span>'
                        f'<span style="font-size:.85rem;font-weight:900;color:var(--dark);letter-spacing:.05em;">{mls[0]} – {mls[1]}</span>'
                        f'<span style="font-size:.52rem;color:var(--mid);font-weight:700;">xG {xg["xg_home"]} · {xg["xg_away"]}</span>'
                        f'</div>'
                    )

                pred_html = (
                    f'<div class="sched-pred">'
                    f'<div class="sched-pred-bar">'
                    f'<div style="width:{hw}%;background:#4CAF50;border-radius:4px 0 0 4px"></div>'
                    f'<div style="width:{dr}%;background:#FFC107"></div>'
                    f'<div style="width:{aw}%;background:#F44336;border-radius:0 4px 4px 0"></div>'
                    f'</div>'
                    f'<div class="sched-pred-labels">'
                    f'<span style="color:#4CAF50">{m["home"].split()[-1]} {hw}%</span>'
                    f'<span style="color:#FFC107">D {dr}%</span>'
                    f'<span style="color:#F44336">{aw}% {m["away"].split()[-1]}</span>'
                    f'</div>'
                    f'{xg_html}'
                    f'<div style="display:flex;align-items:center;justify-content:flex-end;gap:.35rem;margin-top:.35rem;padding-top:.35rem;border-top:1px dashed var(--beige);font-size:.58rem;font-weight:700;">'
                    f'<span style="color:{momentum_col};letter-spacing:.02em;">{momentum_txt}</span>'
                    f'<span style="background:{conf_col};color:white;padding:.08rem .35rem;border-radius:3px;font-size:.5rem;letter-spacing:.08em;">{conf_label}</span>'
                    f'</div>'
                    f'</div>'
                )

            html += (
                f'<div class="sched-match" style="border-left-color:{m["color"]}">'
                f'<div class="sched-league-dot" style="background:{m["color"]}"></div>'
                f'<div style="flex:1">'
                f'<div class="sched-teams">{m["home"]} <span style="color:var(--mid);font-weight:700">vs</span> {m["away"]}</div>'
                f'{pred_html}'
                f'</div>'
                f'{score_html}{status_badge}{md_badge}'
                f'</div>'
            )
        html += '</div>'

    st.markdown(html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTBALL RULES DATA
# ══════════════════════════════════════════════════════════════════════════════
FOOTBALL_RULES = [
    ("The Objective", "⚽", "Score more goals than the opponent before time runs out. A goal is scored when the ball fully crosses the opponent's goal line between the posts and under the crossbar. The team with the most goals wins. If both teams score the same number, the match is a draw — unless a winner must be decided (knockout round), in which case extra time or a penalty shootout follows."),
    ("The Duration", "⏱️", "A match lasts 90 minutes, split into two halves of 45 minutes each, with a 15-minute break at half-time. The referee adds extra minutes at the end of each half (called added time or stoppage time) to compensate for injuries, substitutions, or time-wasting. These extra minutes are displayed on a board by the 4th official on the touchline."),
    ("The Teams", "👥", "Each team fields 11 players on the pitch at any one time, including one goalkeeper. A team needs at least 7 players to continue a match — if a team drops below 7 (due to red cards or injuries with no substitutes left), the game is abandoned. Both teams must wear different coloured kits so they are easy to tell apart."),
    ("The Goalkeeper", "🧤", "The goalkeeper is the only player allowed to handle the ball with their hands — but only inside their own penalty area. Outside the area, they must play like any other outfield player. They wear a different colour shirt to distinguish themselves. The keeper cannot pick up the ball with their hands if a teammate deliberately passes it back to them with their feet (back-pass rule)."),
    ("Handball", "✋", "A handball is called when the ball touches a player's hand or arm in an unnatural position — meaning the arm is making the body bigger or in a position that was not expected. Accidental handballs are not always penalised. However, if a player scores directly after an accidental handball by a teammate, the goal is disallowed. Deliberate handball is always a foul. Goalkeepers can handle the ball freely inside their penalty area."),
    ("Offside", "🚩", "A player is offside if, at the moment the ball is played to them, any part of their body that can score a goal (head, torso, legs) is closer to the opponent's goal line than both the ball AND the second-to-last defender (usually the last outfield player). Being offside is not a foul by itself — it only becomes an offside infringement if the player is actively involved in play (receives the ball, influences an opponent, or gains an advantage). You cannot be offside from a throw-in, corner, or goal kick."),
    ("The Foul", "🦵", "A foul is an unfair act against an opponent, judged by the referee. Common fouls include: kicking, tripping, pushing, holding, or charging an opponent in a careless, reckless, or excessively forceful way. Fouls result in a free kick (or penalty if committed inside the penalty area) for the opposing team. The severity of the foul determines whether a yellow or red card is shown."),
    ("The Free Kick", "🎯", "Awarded to a team after a foul or infringement by the opponent. There are two types: a direct free kick (can be shot directly into goal) and an indirect free kick (must touch another player before a goal can be scored). The opposing players must stand at least 9.15 metres (10 yards) from the ball. The kick must be taken from where the foul occurred."),
    ("The Penalty Area", "📐", "The large rectangle in front of each goal (18-yard box). Inside this area, the goalkeeper can handle the ball. Any foul committed by a defending player inside this box results in a penalty kick for the attacking team. The smaller box inside it (6-yard box) marks the goal area, from where goal kicks are taken."),
    ("The Penalty", "💥", "Awarded when a defending player commits a direct free kick foul inside their own penalty area. The ball is placed on the penalty spot (12 yards from goal). Only the penalty taker and the opposing goalkeeper are allowed in the area at the moment of the kick. All other players must be outside the penalty area and the penalty arc. The goalkeeper must stay on their goal line until the ball is kicked."),
    ("The Yellow Card", "🟨", "A formal warning shown by the referee, also called a caution. A player receives a yellow card for: persistent fouling, unsporting behaviour (diving, time-wasting, pulling a shirt), showing dissent towards the referee, entering or leaving the pitch without permission, or deliberately handling the ball. If a player receives two yellow cards in the same match, they are immediately shown a red card and sent off."),
    ("The Red Card", "🟥", "Shown for serious offences that result in immediate dismissal from the match. A player receives a direct red card (no prior yellow needed) for: violent conduct (punching, headbutting, biting), serious foul play (a tackle that endangers the opponent's safety), spitting, using offensive or abusive language or gestures, or denying an obvious goal-scoring opportunity by deliberate handball or a foul (DOGSO). After a red card, the team plays with 10 players and cannot replace the sent-off player."),
    ("Expulsion", "🚫", "When a player receives a red card — either directly or after two yellow cards — they must immediately leave the pitch and the technical area. They cannot be replaced by a substitute, so their team plays with one fewer player for the rest of the match. The player is also automatically suspended for at least one subsequent match, sometimes more depending on the seriousness of the offence."),
    ("The Corner", "🚩", "Awarded to the attacking team when the ball goes out of play over the goal line and was last touched by a defending player. The ball is placed in the corner arc (a small quarter-circle in the corner of the pitch) and kicked back into play. A goal can be scored directly from a corner. Defending players must stand at least 9.15 metres from the corner arc until the ball is in play."),
    ("The Throw-In", "🤾", "Awarded when the ball goes out of play over the touchline (the long sides of the pitch). The throw-in is taken by the team that did not touch the ball last. The player must throw the ball with both hands, from behind and over their head, and both feet must be on or behind the touchline at the moment of release. A goal cannot be scored directly from a throw-in."),
    ("The Goal", "🥅", "A goal is scored when the entire ball crosses the goal line between the goalposts and under the crossbar, provided no infringement (offside, foul, handball) occurred in the build-up. The goal counts the moment the ball fully crosses the line — even if it bounces back out. Goal-line technology or VAR may be used to confirm whether the ball crossed the line."),
    ("The Ball", "⚪", "A standard football is spherical, made of leather or a similar material, with a circumference of 68–70 cm and a weight of 410–450 grams at the start of the match. If the ball bursts or deflates during play, the game is stopped and restarted with a new ball. The restart depends on where the ball was when it stopped — usually a dropped ball from where it became defective."),
    ("The Substitute", "🔄", "Teams can make up to 5 substitutions per match (in most competitions), with a sixth allowed in extra time. Substitutions can only be made during a stoppage in play and must be confirmed with the 4th official. Once a player is substituted off, they cannot return to the match. A player who receives a red card cannot be replaced by a substitute — the team simply plays with fewer players."),
    ("Extra Time", "⏰", "If a knockout match is level after 90 minutes, the game goes into extra time: two additional periods of 15 minutes each (30 minutes total). If the score is still level after extra time, the match is decided by a penalty shootout. Unlike regular time, a goal scored in extra time does not end the game immediately — both periods must be played."),
    ("Penalty Shootout", "🥊", "Used to decide a knockout match that is still level after extra time. Each team takes turns shooting 5 penalties alternately. If still level after 5 each, it goes to sudden death (one penalty at a time, the first team to score while the other misses wins). Only players on the pitch at the end of extra time are eligible to take penalties (except the expelled players)."),
    ("The Referee", "👨‍⚖️", "The referee is the authority on the pitch. They enforce the rules, start and stop play, award free kicks and penalties, show yellow and red cards, and add stoppage time. The referee's decision is final on the pitch. They can change a decision if they realise it was wrong, as long as play has not resumed. The referee is assisted by two assistant referees and, in top competitions, a VAR team."),
    ("The Assistant Referee", "🚩", "Two assistant referees (ARs) patrol the touchlines during a match, one on each side. Their main jobs are to signal when the ball goes out of play (and which team gets the throw-in or corner), to flag for offside, and to assist the referee with decisions near their side of the pitch. They communicate with the referee via earpiece. They can only recommend decisions — the final call always belongs to the referee."),
    ("VAR", "📺", "Video Assistant Referee — a technology system used in top competitions to review four key match-changing decisions: goals (and the build-up to them), penalty decisions, direct red cards, and cases of mistaken identity. A VAR team watches multiple camera angles in a review centre. They can recommend the referee to review footage on a pitchside monitor. VAR only intervenes for a 'clear and obvious error' — not for every debatable decision."),
    ("Assisted Offside (OGSO)", "📏", "In competitions using VAR, offside decisions are confirmed using a 'semi-automated offside technology' (SAOT) system that tracks players' body positions using camera data and draws precise lines to determine if any part of the attacking player's body was ahead of the last defender. This removes the need for a subjective judgment call by the assistant referee and has made offside decisions much more accurate — though sometimes controversial for millimetre-thin calls."),
    ("Kick-Off", "🔔", "Used to start each half of the match, to restart after a goal is scored, and to begin extra time periods. The ball is placed on the centre spot. Both teams must be in their own half. The team kicking off can kick the ball in any direction. The opposing team must be outside the centre circle (radius 9.15 m) until the ball is in play. A goal can be scored directly from a kick-off."),
    ("The Back Pass", "↩️", "When a player deliberately passes the ball back to their own goalkeeper using their feet. The goalkeeper is NOT allowed to pick up the ball with their hands in this situation — they must play it with their feet. This rule was introduced in 1992 to prevent time-wasting. However, if the ball is headed or controlled with another body part (not the feet) and played back, the goalkeeper CAN pick it up."),
    ("The Wall", "🧱", "When a free kick is awarded near the penalty area, the defending team can form a 'wall' of players to block the direct shot on goal. The wall must stand at least 9.15 metres from the ball. Only the attacking team's players are allowed inside this 9.15-metre zone — a recent rule change prevents defending players from standing in the wall on the same line as attackers before the kick is taken."),
    ("Simulation (Diving)", "🎭", "When a player intentionally falls to the ground or exaggerates contact to deceive the referee into awarding a free kick or penalty. This is considered unsporting behaviour and is punishable by a yellow card. VAR has made it easier to identify simulation after the fact. Despite this, diving remains a controversial topic in football, as the line between embellishment and genuine reaction to contact can be very thin."),
    ("Added Time", "➕", "Also known as stoppage time or injury time. At the end of each half, the referee adds extra minutes to compensate for time lost during the half — due to injuries, substitutions, goal celebrations, VAR checks, time-wasting, and other stoppages. The 4th official displays the minimum added time on a board. Since 2023, FIFA encouraged referees to add more accurate amounts of time, sometimes exceeding 10 minutes per half in high-stoppage games."),
]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE RULES
# ══════════════════════════════════════════════════════════════════════════════
def page_regles():
    st.markdown('<div class="sec-label">Football Basics</div><div class="sec-title">Rules of the Game</div>', unsafe_allow_html=True)
    st.markdown("""<style>
.rule-card{background:var(--white);border-radius:16px;padding:0;margin-bottom:.7rem;box-shadow:0 2px 10px rgba(26,26,46,.06);overflow:hidden;}
.rule-card summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:.75rem;padding:.85rem 1.1rem;user-select:none;}
.rule-card summary::-webkit-details-marker{display:none;}
.rule-icon{font-size:1.2rem;flex-shrink:0;width:32px;text-align:center;}
.rule-title{font-size:.9rem;font-weight:800;color:var(--dark);flex:1;}
.rule-arrow{font-size:.75rem;font-weight:900;color:var(--mid);transition:transform .2s;flex-shrink:0;}
details.rule-card[open] .rule-arrow{transform:rotate(90deg);}
details.rule-card summary::after{content:none;}
.rule-body{padding:.1rem 1.1rem 1rem 1.1rem;font-size:.88rem;line-height:1.75;color:var(--mid);font-weight:600;border-top:1px solid var(--beige);}

</style>""", unsafe_allow_html=True)

    for title, icon, description in FOOTBALL_RULES:
        st.markdown(
            f'<details class="rule-card">'
            f'<summary>'
            f'<span class="rule-icon">{icon}</span>'
            f'<span class="rule-title">{title}</span>'
            f'<span class="rule-arrow">›</span>'
            f'</summary>'
            f'<div class="rule-body">{description}</div>'
            f'</details>',
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.page == "definition":
    page_definition()
else:
    render_header()
    render_nav()
    if st.session_state.page == "classement":
        page_classement()
    elif st.session_state.page == "schedule":
        page_schedule()
    elif st.session_state.page == "regles":
        page_regles()
    elif st.session_state.page == "glossaire":
        page_glossaire()
    else:
        page_main()
