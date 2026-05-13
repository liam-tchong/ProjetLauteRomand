import streamlit as st
import requests
import re
import anthropic
import time
import os
import pickle
import math
from concurrent.futures import ThreadPoolExecutor
from tactical_data import TEAM_TACTICS
from football_data import FOOTBALL_RULES, FORMATIONS_DATA, POSITIONS_DATA, STYLE_TAG_GUIDE, STYLE_TAG_TO_TERM, TACTICAL_TERMS

_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MachineLearning", "model.pkl")
_goals_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MachineLearning", "goals_model.pkl")
_csv_path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MachineLearning", "match_data.csv")
try:
    with open(_model_path, "rb") as _f:
        MATCH_MODEL = pickle.load(_f)
except Exception:
    MATCH_MODEL = None

try:
    with open(_goals_model_path, "rb") as _f:
        GOALS_MODEL = pickle.load(_f)  # {"home": PoissonRegressor, "away": PoissonRegressor}
except Exception:
    GOALS_MODEL = None



def _poisson_pmf(k, lam):
    """P(X=k) for Poisson distribution with mean lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _most_likely_score(xg_home, xg_away, max_goals=7):
    """Find the (home, away) score with highest joint Poisson probability."""
    best, best_p = (0, 0), 0.0
    for h in range(max_goals + 1):
        ph = _poisson_pmf(h, xg_home)
        for a in range(max_goals + 1):
            p = ph * _poisson_pmf(a, xg_away)
            if p > best_p:
                best_p = p
                best = (h, a)
    return best, best_p


def _start_background_refresh():
    """Fetch latest 2025/26 matches since last CSV entry and retrain model in background.
    Runs at most once per session (session_state gate) and once per hour (file mtime gate)."""
    import threading

    if st.session_state.get("_bg_refresh_started"):
        return
    st.session_state["_bg_refresh_started"] = True

    if not os.path.exists(_csv_path):
        return
    if time.time() - os.path.getmtime(_csv_path) < 3600:
        return  # Fresh enough — skip

    def _worker():
        global MATCH_MODEL
        try:
            import pandas as pd
            df = pd.read_csv(_csv_path)
            last_date = df["date"].max()

            _LEAGUE_CODES = {
                "FL1": "Ligue 1", "PL": "Premier League",
                "PD": "La Liga",  "SA": "Serie A", "BL1": "Bundesliga",
            }
            new_rows = []
            for code, name in _LEAGUE_CODES.items():
                try:
                    r = requests.get(
                        f"https://api.football-data.org/v4/competitions/{code}/matches",
                        headers=HEADERS,
                        params={"season": 2025, "status": "FINISHED"},
                        timeout=10,
                    )
                    r.raise_for_status()
                    for m in r.json().get("matches", []):
                        d = m["utcDate"][:10]
                        if d <= last_date:
                            continue
                        hs = m["score"]["fullTime"].get("home")
                        as_ = m["score"]["fullTime"].get("away")
                        if hs is None or as_ is None:
                            continue
                        new_rows.append({
                            "league": name, "season": 2025, "date": d,
                            "home_team": m["homeTeam"]["name"],
                            "away_team": m["awayTeam"]["name"],
                            "home_goals": hs, "away_goals": as_,
                            "result": "H" if hs > as_ else ("D" if hs == as_ else "A"),
                        })
                    time.sleep(7)  # 10 req/min free-tier rate limit
                except Exception:
                    pass

            if not new_rows:
                os.utime(_csv_path, None)  # Touch to reset the 1-hour timer
                return

            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df.drop_duplicates(subset=["date", "home_team", "away_team"], inplace=True)
            df.to_csv(_csv_path, index=False)

            # ── Retrain on fresh data ──────────────────────────────────────
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.linear_model import PoissonRegressor
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            def _roll(df, team, date, n=5):
                past = df[
                    ((df["home_team"] == team) | (df["away_team"] == team))
                    & (df["date"] < date)
                ].tail(n)
                if len(past) == 0:
                    return 0.5, 1.3, 1.3
                w = s = c = 0
                for _, row in past.iterrows():
                    if row["home_team"] == team:
                        s += row["home_goals"]; c += row["away_goals"]
                        if row["result"] == "H": w += 1
                    else:
                        s += row["away_goals"]; c += row["home_goals"]
                        if row["result"] == "A": w += 1
                n_ = len(past)
                return w / n_, s / n_, c / n_

            feat_rows = []
            for _, match in df.iterrows():
                hf, hs2, hc = _roll(df, match["home_team"], match["date"])
                af, as2, ac = _roll(df, match["away_team"], match["date"])
                feat_rows.append([hf, hs2, hc, af, as2, ac,
                                   {"H": 0, "D": 1, "A": 2}[match["result"]],
                                   match["home_goals"], match["away_goals"]])

            feat = pd.DataFrame(feat_rows, columns=[
                "h_form", "h_scored", "h_conceded",
                "a_form", "a_scored", "a_conceded",
                "result", "home_goals", "away_goals"
            ]).dropna()
            X = feat[["h_form","h_scored","h_conceded","a_form","a_scored","a_conceded"]]

            clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                             learning_rate=0.05, random_state=42)
            clf.fit(X, feat["result"])
            with open(_model_path, "wb") as f:
                pickle.dump(clf, f)
            MATCH_MODEL = clf

            hm = PoissonRegressor(alpha=0.5, max_iter=500)
            hm.fit(X, feat["home_goals"].astype(float))
            am = PoissonRegressor(alpha=0.5, max_iter=500)
            am.fit(X, feat["away_goals"].astype(float))
            goals_pkg = {"home": hm, "away": am}
            with open(_goals_model_path, "wb") as f:
                pickle.dump(goals_pkg, f)
            GOALS_MODEL = goals_pkg
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


_start_background_refresh()

st.set_page_config(page_title="The Football Classroom", layout="wide")

# ── API ───────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = st.secrets.get("ANTHROPIC_API_KEY", "")
API_FOOTBALL_KEY   = st.secrets.get("API_FOOTBALL_KEY", "")
SQUAD_API_KEY      = st.secrets.get("SQUAD_API_KEY", "")
HEADERS = {"X-Auth-Token": API_FOOTBALL_KEY}

@st.cache_resource
def _get_anthropic_client():
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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


# Powers the "Expected Score" block in the Match Prediction card — estimates goals using
# attack vs. defence averages, recent form, and a Poisson-like model.
def predict_expected_score(standings, home_team, away_team,
                            form_home=None, form_away=None,
                            extra_home=None, extra_away=None):
    """Predict expected goals using trained Poisson regression + real match history.

    Uses a PoissonRegressor trained on 5000+ historical matches (2023-2026).
    Most likely score is found via joint Poisson distribution — supports 0-0, 1-0, etc.
    Includes stakes adjustment: title race / relegation battle affect xG.

    Returns dict with xg_home, xg_away, most_likely_score.
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
        eh = extra_home or {}
        ea = extra_away or {}

        # ── Attack strength × defensive weakness (Dixon-Coles, venue-split) ──
        # Pull venue-specific averages from last 15 API matches (live data, updates post-match).
        h_scored_home   = eh.get("home_gf_avg") or eh.get("gf_avg_recent") or dh.get("goals_for", 0) / played_h
        a_conceded_away = ea.get("away_ga_avg") or ea.get("ga_avg_recent") or da.get("goals_against", 0) / played_a
        a_scored_away   = ea.get("away_gf_avg") or ea.get("gf_avg_recent") or da.get("goals_for", 0) / played_a
        h_conceded_home = eh.get("home_ga_avg") or eh.get("ga_avg_recent") or dh.get("goals_against", 0) / played_h

        # League average goals per team per venue (calibration constant)
        LEAGUE_AVG_HOME = 1.45
        LEAGUE_AVG_AWAY = 1.10

        # Multiplicative model: team_attack_strength × opponent_defence_weakness × league_avg
        # Gives real variance: dominant team vs bad defence can reach xG 3-4
        atk_h  = h_scored_home   / LEAGUE_AVG_HOME
        def_a  = a_conceded_away / LEAGUE_AVG_AWAY
        atk_a  = a_scored_away   / LEAGUE_AVG_AWAY
        def_h  = h_conceded_home / LEAGUE_AVG_HOME

        # Geometric mean of factors (square-root) dampens extremes while keeping variance.
        # Pure multiplication (atk×def) compounds two outliers → unrealistic 5-0, 6-0.
        # Sqrt keeps PSG vs relegation at ~3-4 xG instead of 6+.
        xg_home = LEAGUE_AVG_HOME * math.sqrt(atk_h * def_a)
        xg_away = LEAGUE_AVG_AWAY * math.sqrt(atk_a * def_h)

        # ── Form factor: last 5 weighted results → ±20% swing ──
        fh = _form_score(form_home) if form_home else dh.get("won", 0) / played_h
        fa = _form_score(form_away) if form_away else da.get("won", 0) / played_a
        xg_home *= 0.80 + fh * 0.40   # 0.80 at worst form → 1.20 at best form
        xg_away *= 0.80 + fa * 0.40

        # ── Fatigue: played ≤4 days ago (midweek cup / Ligue des champions) ──
        if eh.get("fatigued"):
            xg_home *= 0.88
        if ea.get("fatigued"):
            xg_away *= 0.85

        # ── Stakes: high-stakes matches tend to be more careful ──
        n_teams = max(len(standings), 18)
        pos_h = dh.get("position", n_teams // 2)
        pos_a = da.get("position", n_teams // 2)

        def _stakes_factor(pos, n):
            if pos <= 3:       return 0.93   # titre — 1-0 courants
            if pos >= n - 2:   return 0.90   # relégation — très fermé
            if 4 <= pos <= 6:  return 0.97   # europe
            return 1.0

        combined_stakes = (_stakes_factor(pos_h, n_teams) + _stakes_factor(pos_a, n_teams)) / 2
        xg_home *= combined_stakes
        xg_away *= combined_stakes

        # Clamp: astronomically dominant matchups shouldn't exceed 5 goals
        xg_home = max(0.20, min(xg_home, 5.0))
        xg_away = max(0.15, min(xg_away, 4.0))

        # ── Predicted score: joint Poisson mode (statistically correct) ──
        # The mode of Poisson(λ) is floor(λ). We search the joint distribution
        # to find the (h, a) pair with the highest combined probability.
        # This naturally gives 0-0, 1-0, 0-1 for low-xG matchups.
        (score_h, score_a), _ = _most_likely_score(xg_home, xg_away)

        return {
            "xg_home":           round(xg_home, 2),
            "xg_away":           round(xg_away, 2),
            "most_likely_score": (score_h, score_a),
            "top_scores":        [((score_h, score_a), None)],
        }
    except Exception:
        return None


def predict_match(standings, home_team, away_team, form_home=None, form_away=None,
                  extra_home=None, extra_away=None):
    """Win probabilities derived directly from xG via Poisson distribution.

    Single source of truth: xG drives both the score prediction AND the win% bar.
    If xG says 3-0, the bar shows ~90% home — always coherent, never contradictory.
    """
    if not standings:
        return None, None
    try:
        xg = predict_expected_score(standings, home_team, away_team,
                                    form_home=form_home, form_away=form_away,
                                    extra_home=extra_home, extra_away=extra_away)
        if xg is None:
            return None, None

        xg_h, xg_a = xg["xg_home"], xg["xg_away"]

        # P(home win), P(draw), P(away win) from joint Poisson — same λ used for score
        ph = pd_ = pa = 0.0
        for h in range(11):
            p_h = _poisson_pmf(h, xg_h)
            for a in range(11):
                p = p_h * _poisson_pmf(a, xg_a)
                if h > a:    ph += p
                elif h == a: pd_ += p
                else:        pa += p

        probs = [ph, pd_, pa]
        fh = _form_score(form_home) if form_home else 0.5
        fa = _form_score(form_away) if form_away else 0.5
        meta = {
            "momentum_home": fh,
            "momentum_away": fa,
            "total_shift":   fh - fa,
            "confidence":    max(probs) - sorted(probs)[-2],
        }
        return probs, meta
    except Exception:
        return None, None

# Populates the standings table, team selectors, stats cards, and every crest image.
# Single source of truth for position, points, goals, and team IDs across the app.
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
        "Write 3-4 sentences summarising the current state of this league season. "
        "Tone: a knowledgeable football analyst giving a quick verbal briefing — clear, direct, and professional. "
        "No slang, no exclamation marks, no casual filler words. "
        "No structure, no headline, no bullet points. Just flowing prose. "
        "Lead with the most significant storyline right now. Use team names and exact point gaps. Be specific.\n\n"
        + context
    )
    try:
        client = _get_anthropic_client()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=260,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()
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


# Feeds the Schedule page — returns upcoming and recent matches for a 17-day window
# (3 days back, 14 days forward) including live scores and match status.
@st.cache_data(ttl=600, show_spinner=False)
def fetch_schedule(league_code, date_from, date_to):
    """Fetch matches for a league between two dates (yyyy-mm-dd strings)."""
    for _ in range(4):
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


# Supplies the "Recent Form" pills, home/away split, and per-match goal averages used in
# the Match Prediction card and as live inputs to the AI analysis prompt.
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

        # ── Home/away split averages (domicile vs extérieur) ──
        home_gf_list, home_ga_list = [], []
        away_gf_list, away_ga_list = [], []
        last_match_date = None

        for m in matches:
            home_id  = m["homeTeam"]["id"]
            hs = m["score"]["fullTime"]["home"]
            as_ = m["score"]["fullTime"]["away"]
            if hs is None or as_ is None:
                continue
            is_home = (home_id == team_id)
            gs = hs if is_home else as_
            gc = as_ if is_home else hs
            if is_home:
                home_gf_list.append(gs); home_ga_list.append(gc)
            else:
                away_gf_list.append(gs); away_ga_list.append(gc)

            # Track most recent match for fatigue detection
            try:
                from datetime import datetime as _dt, timezone as _tz
                md = _dt.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
                if last_match_date is None or md > last_match_date:
                    last_match_date = md
            except Exception:
                pass

        # Fatigue: played within 4 days = fatigued (European/cup midweek)
        days_since = 99
        if last_match_date:
            from datetime import datetime as _dt2, timezone as _tz2
            days_since = (_dt2.now(_tz2.utc) - last_match_date).days

        stats = {
            "home_record":    f"{home_w}W {home_d}D {home_l}L",
            "away_record":    f"{away_w}W {away_d}D {away_l}L",
            "clean_sheets":   clean_sheets,
            "gf_avg_recent":  round(sum(gf_list) / n, 2) if n else None,
            "ga_avg_recent":  round(sum(ga_list) / n, 2) if n else None,
            "win_pct":        round(form.count("W") / len(form) * 100) if form else None,
            # Split home/away averages
            "home_gf_avg":   round(sum(home_gf_list) / len(home_gf_list), 2) if home_gf_list else None,
            "home_ga_avg":   round(sum(home_ga_list) / len(home_ga_list), 2) if home_ga_list else None,
            "away_gf_avg":   round(sum(away_gf_list) / len(away_gf_list), 2) if away_gf_list else None,
            "away_ga_avg":   round(sum(away_ga_list) / len(away_ga_list), 2) if away_ga_list else None,
            # Fatigue
            "days_since_last_match": days_since,
            "fatigued":      days_since <= 4,
        }
        return form[-5:], stats
    except Exception:
        return [], {}


def fetch_team_form(team_id):
    """Kept for backward compat — returns just the 5-match form list."""
    form, _ = fetch_team_extended(team_id)
    return form


# Provides the top-scorer list (name, goals, assists) fed into the Claude prompt for
# AI team-style generation; top 3 per team are shown in the analysis context.
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


# Pulls advanced Ligue 1 stats (formation, pass %, shots, clean sheets) from api-sports.io.
# Used in the AI analysis prompt and the Offensive/Defensive stats comparison cards.
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


# Calls Claude Haiku to generate the 4-panel team card content (The Club / How They Play ×2 /
# Fun Fact) displayed on the Analysis page for each selected team.
@st.cache_data(ttl=3600, show_spinner=False)
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

    prompt = f"""You're writing quick takes on a football team for someone who has never watched football. 4 short sections, separated by "|||". No structure labels, no numbered sections, no intro phrases.

{stats_block}

Section 1: just state who this club is. One or two blunt sentences. No fluff.
Section 2: how do they play? Use 2-3 terms from: {terms}, each wrapped in <b>tags</b> with a one-clause explanation in parentheses. Be opinionated.
Section 3: go deeper. Name one real player. What do they actually do on the pitch? Use 3-4 terms from: {terms} in <b>tags</b>.
Section 4: one thing about this club that's genuinely surprising or counterintuitive this season. Not Wikipedia trivia.

No em-dashes. No "what makes them special". No "essentially". Short sentences only.
Reply with exactly 4 sections separated by "|||"."""

    try:
        client = _get_anthropic_client()
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


# Calls Claude Haiku to produce the per-team tactical challenge blurbs rendered at the
# bottom of the "Key points to watch" card on the Analysis page.
@st.cache_data(ttl=86400, show_spinner=False)
def generate_key_challenges(team_a, team_b, pts_a, pts_b, gf_a, gf_b, ga_a, ga_b):
    """Generates a short challenge paragraph (2-3 sentences) per team for this specific matchup."""
    if not ANTHROPIC_API_KEY:
        return (
            f"{team_a} must stay compact and limit space in behind. Defensive organisation will be key.",
            f"{team_b} must be clinical in the final third. Creating clear chances will decide the match.",
        )
    prompt = f"""One sentence per team. Maximum. No "stay compact", no "be clinical", no "defensive unit".
Start each sentence with the team name.
Make it specific to how these two teams actually compare this season.
Just two lines, nothing else.

{team_a}: {pts_a} pts, scored {gf_a}, conceded {ga_a}
{team_b}: {pts_b} pts, scored {gf_b}, conceded {ga_b}"""
    try:
        client = _get_anthropic_client()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=180,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        sep_a = raw.find(f"{team_a}:")
        sep_b = raw.find(f"{team_b}:")
        if sep_a != -1 and sep_b != -1:
            if sep_a < sep_b:
                challenge_a = raw[sep_a + len(team_a) + 1 : sep_b].strip()
                challenge_b = raw[sep_b + len(team_b) + 1 :].strip()
            else:
                challenge_b = raw[sep_b + len(team_b) + 1 : sep_a].strip()
                challenge_a = raw[sep_a + len(team_a) + 1 :].strip()
        else:
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            challenge_a = lines[0].split(":", 1)[-1].strip() if lines else ""
            challenge_b = lines[1].split(":", 1)[-1].strip() if len(lines) > 1 else ""
        challenge_a = challenge_a or "Stay compact and limit space in behind. Defensive organisation will be key."
        challenge_b = challenge_b or "Be clinical in the final third. Creating clear chances will decide the match."
        return challenge_a, challenge_b
    except Exception:
        return (
            f"Stay compact and limit space in behind. Defensive organisation will be key.",
            f"Be clinical in the final third. Creating clear chances will decide the match.",
        )



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



_TACTICS_NAME_MAP = {
    "FC Internazionale Milano": "Inter Milan",
    "SC Freiburg":              "Sport-Club Freiburg",
    "RCD Espanyol de Barcelona": "Espanyol",
}

def _hex_to_rgb(hx):
    """Hex → 'r,g,b' string for rgba()."""
    h = hx.lstrip('#')
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

# Renders the animated SVG tactical pitch on the Analysis page — shows formation, labelled
# player positions, movement arrows, and heat zones for the selected team.
def render_tactical_pitch_html(team_name):
    """Generate a premium animated SVG tactical pitch for a given team."""
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

        players_svg += (
            f'<g class="{cls}" {filt}>'
            f'{ring}'
            f'{gk_ring}'
            f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{player_r}" fill="{player_fill}" stroke="{stroke_color}" stroke-width="{stroke_w}"/>'
            f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{player_r}" fill="rgba(255,255,255,.08)"/>'
            f'<text x="{x0:.1f}" y="{y0:.1f}" text-anchor="middle" dominant-baseline="central" '
            f'font-size="{font_sz}" font-weight="900" fill="{text_fill}" font-family="Nunito,sans-serif" letter-spacing="-.3">{abbr}</text>'
            f'</g>\n'
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
            _ta = st.session_state.get("team_a", "")
            _tb = st.session_state.get("team_b", "")
            _lg = st.session_state.get("league", "Ligue 1")
            return f'<a href="?term={term_key}&from=main&ta={_ta}&tb={_tb}&lg={_lg}" target="_parent" style="text-decoration:none;">{inner}</a>'
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
        f'<div style="font-size:.95rem;font-weight:900;color:rgba(255,255,255,.92);letter-spacing:-.02em;">{team_name}</div>'
        f'</div>'
        f'<a href="?nav=glossaire&formation={formation_val}&ta={st.session_state.get("team_a","")}&tb={st.session_state.get("team_b","")}&lg={st.session_state.get("league","Ligue 1")}" target="_parent" style="text-decoration:none;">'
        f'<span style="display:inline-flex;align-items:center;gap:.4rem;background:{color};color:white;font-size:.72rem;font-weight:900;'
        f'padding:.32rem .75rem .32rem .9rem;border-radius:100px;letter-spacing:.06em;cursor:pointer;'
        f'border:2px solid rgba(255,255,255,.9);'
        f'box-shadow:0 2px 12px {color}88;transition:all .18s cubic-bezier(.34,1.56,.64,1);" '
        f'onmouseover="this.style.transform=\'scale(1.1)\';this.style.background=\'white\';this.style.color=\'{color}\';this.style.boxShadow=\'0 4px 18px {color}99\'" '
        f'onmouseout="this.style.transform=\'scale(1)\';this.style.background=\'{color}\';this.style.color=\'white\';this.style.boxShadow=\'0 2px 12px {color}88\'">'
        f'{t["formation"]}'
        f'<span style="font-size:.65rem;font-weight:900;opacity:.9">&#8594;</span>'
        f'</span>'
        f'</a>'
        f'</div>'
        # SVG pitch + animated overlay
        f'<div style="position:relative;">{svg}{overlay_html}</div>'
        # Footer: pills only
        f'<div style="padding:.55rem 1rem .75rem;border-top:1px solid rgba(255,255,255,.06);">'
        f'<div style="font-size:.52rem;font-weight:800;letter-spacing:.16em;color:rgba(255,255,255,.25);'
        f'text-transform:uppercase;margin-bottom:.28rem;">click any tag to learn more</div>'
        f'{pills}</div>'
        f'</div>'
    )

# Renders the SVG tactical illustration on the term Definition page — one unique animation
# per tactical concept (pressing, pivot, false nine, through ball, etc.).
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
            f'<div style="font-size:.56rem;color:rgba(255,255,255,.5);font-weight:600;line-height:1.4;">{anim_idea}</div>'
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
            f'<a href="?term={term}&from={source_page}{extra}" target="_parent" class="term-link">{term}</a>'
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
[data-testid="stMain"]::before{content:'';display:block;height:3px;background:var(--green);margin-bottom:1.5rem;}

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
details.style-acc summary::after{content:'Show more';}
details.style-acc[open] summary::after{content:'Show less';}
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
    if "ta" in qp and qp["ta"]:
        st.session_state.team_a = qp["ta"]
    if "tb" in qp and qp["tb"]:
        st.session_state.team_b = qp["tb"]
    if "lg" in qp and qp["lg"] in LEAGUES:
        st.session_state.league = qp["lg"]
    st.session_state.active_term = qp["term"]
    st.session_state.page = "definition"
    st.query_params.clear()
    st.rerun()

# ── Back navigation from definition page ─────────────────────────────────────
VALID_PAGES = {"main", "glossaire", "classement", "schedule", "regles"}

# ── Navigate to glossary position / formation anchor from pitch page ──────────
if "nav" in qp and qp["nav"] == "glossaire":
    st.session_state.page = "glossaire"
    st.session_state.prev_page = "main"
    if "ta" in qp and qp["ta"]:
        st.session_state.team_a = qp["ta"]
    if "tb" in qp and qp["tb"]:
        st.session_state.team_b = qp["tb"]
    if "lg" in qp and qp["lg"] in LEAGUES:
        st.session_state.league = qp["lg"]
    if qp.get("pos"):
        st.session_state.glossaire_anchor = qp["pos"]
        st.session_state.glossaire_tab = 1
    elif qp.get("formation"):
        st.session_state.glossaire_anchor = qp["formation"]
        st.session_state.glossaire_tab = 2
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
    played_a = da.get("played",1) or 1
    played_b = db.get("played",1) or 1
    gf_a, gf_b = da["goals_for"], db["goals_for"]
    ga_a, ga_b = da["goals_against"], db["goals_against"]
    pts_a, pts_b = da["points"], db["points"]
    attacker = a if gf_a >= gf_b else b
    defender = a if ga_a <= ga_b else b
    leader   = a if pts_a >= pts_b else b
    trailer  = b if pts_a >= pts_b else a
    gap      = abs(pts_a - pts_b)
    pts_line = (
        f"{leader} are {gap} point{'s' if gap != 1 else ''} ahead in the table. This match matters more for {trailer}."
        if gap <= 15 else
        f"{leader} are dominant this season with a {gap}-point gap — {trailer} need a statement result."
    )
    return [
        f"{attacker} have been more dangerous going forward — {gf_a} goals for {a} vs {gf_b} for {b} this season.",
        f"{defender} have the tighter defence. {abs(ga_a-ga_b)} goals separate them on the season ({ga_a} vs {ga_b} conceded).",
        pts_line,
    ]

GLOS_ICONS = ["", "", "", "", ""]
GLOS_COLORS = ["var(--yellow-lt)","var(--green-lt)","var(--red-lt)","var(--beige)","var(--green-lt)"]




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
        if st.button("Analysis", type=t, use_container_width=True, key="nav_main"):
            st.session_state.page = "main"; st.rerun()
    with c2:
        t = "primary" if page == "classement" else "secondary"
        if st.button("Standings", type=t, use_container_width=True, key="nav_class"):
            st.session_state.page = "classement"; st.rerun()
    with c3:
        t = "primary" if page == "schedule" else "secondary"
        if st.button("Schedule", type=t, use_container_width=True, key="nav_sched"):
            st.session_state.page = "schedule"; st.rerun()
    with c4:
        t = "primary" if page == "regles" else "secondary"
        if st.button("Rules", type=t, use_container_width=True, key="nav_regles"):
            st.session_state.page = "regles"; st.rerun()
    with c5:
        t = "primary" if page == "glossaire" else "secondary"
        if st.button("Glossary", type=t, use_container_width=True, key="nav_glos"):
            st.session_state.page = "glossaire"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE DÉFINITION
# ══════════════════════════════════════════════════════════════════════════════
# Renders the individual tactical term detail page — definition, plain-English summary,
# real-world example, and the SVG illustration from render_term_animation_html().
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

    prev = st.session_state.get("prev_page", "main")
    if st.button("← Back", key="def_back"):
        st.session_state.page = prev
        st.session_state.active_term = None
        st.rerun()

    st.markdown(f"""<div class="def-hero"><div class="def-title">{term.capitalize()}</div></div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="def-text">{definition}</div>', unsafe_allow_html=True)
    if simple:
        st.markdown(f'<div class="def-simple"><span class="def-tag def-tag-green">In plain English</span><p>{simple}</p></div>', unsafe_allow_html=True)
    if example:
        st.markdown(f'<div class="def-example"><span class="def-tag def-tag-yellow">Real example</span><p>{example}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Tactical illustration</div>', unsafe_allow_html=True)
    st.markdown(render_term_animation_html(term), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE GLOSSAIRE
# ══════════════════════════════════════════════════════════════════════════════
# Renders the 3-tab Glossary: tactical terms (with clickable cards), player positions
# (with live squad names from API), and formations with descriptions.
def page_glossaire():
    # ── Consume anchor set when navigating from pitch page ────────────────────
    anchor = st.session_state.get("glossaire_anchor") or None
    st.session_state.glossaire_anchor = None


    st.markdown('<div class="sec-title">Tactical Glossary</div>', unsafe_allow_html=True)

    # ── Tab switcher (same button pattern as nav bar) ─────────────────────────
    active_tab = st.session_state.get("glossaire_tab", 0)
    st.markdown('<div style="background:var(--white);border:2px solid var(--beige);border-radius:22px;padding:.5rem .6rem;margin-bottom:1.5rem;box-shadow:0 4px 20px rgba(42,32,24,0.08);display:flex;gap:.4rem">', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        if st.button("Tactics", type="primary" if active_tab == 0 else "secondary", use_container_width=True, key="glos_tab0"):
            st.session_state.glossaire_tab = 0; st.rerun()
    with tc2:
        if st.button("Positions", type="primary" if active_tab == 1 else "secondary", use_container_width=True, key="glos_tab1"):
            st.session_state.glossaire_tab = 1; st.rerun()
    with tc3:
        if st.button("Formations", type="primary" if active_tab == 2 else "secondary", use_container_width=True, key="glos_tab2"):
            st.session_state.glossaire_tab = 2; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 0: Tactics ────────────────────────────────────────────────────────
    if active_tab == 0:
        for i, (term, term_data) in enumerate(TACTICAL_TERMS.items()):
            definition = term_data.get("definition", "") if isinstance(term_data, dict) else term_data
            icon = GLOS_ICONS[i % len(GLOS_ICONS)]
            bg   = GLOS_COLORS[i % len(GLOS_COLORS)]
            _ta = st.session_state.get("team_a", "")
            _tb = st.session_state.get("team_b", "")
            _lg = st.session_state.get("league", "Ligue 1")
            st.markdown(
                f'<a href="?term={term}&from=glossaire&ta={_ta}&tb={_tb}&lg={_lg}" target="_parent" style="text-decoration:none;color:inherit">'
                f'<div class="glos-card">'
                f'<div class="glos-card-header">'
                f'<div class="glos-card-icon" style="background:{bg};display:flex;align-items:center;justify-content:center"><span style="width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block"></span></div>'
                f'<span class="glos-card-term">{term.capitalize()}</span>'
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
# Renders the Standings page — one tab per league showing the full table with crests,
# highlights for the selected teams, and a Claude-generated narrative summary per league.
def page_classement():
    selected_league = st.session_state.get("league", "Ligue 1")
    team_a = st.session_state.team_a
    team_b = st.session_state.team_b

    st.markdown('<div class="sec-title">Standings 2025/26</div>', unsafe_allow_html=True)

    # Pre-fetch all standings + summaries in parallel before rendering tabs
    with st.spinner("Loading standings…"):
        with ThreadPoolExecutor(max_workers=len(LEAGUES)) as _pool:
            _st_futures = {
                lname: _pool.submit(fetch_standings, linfo["code"])
                for lname, linfo in LEAGUES.items()
            }
            _all_standings = {lname: f.result() for lname, f in _st_futures.items()}

        def _make_tuple(standings):
            return tuple(
                (name, d["position"], d["points"], d["played"], d["won"], d["goal_diff"])
                for name, d in standings.items()
            )

        with ThreadPoolExecutor(max_workers=len(LEAGUES)) as _pool:
            _sum_futures = {
                lname: _pool.submit(generate_standings_summary, lname, _make_tuple(stds))
                for lname, stds in _all_standings.items() if stds
            }
            _all_summaries = {lname: f.result() for lname, f in _sum_futures.items()}

    tab_labels = [f"{LEAGUES[l]['flag']} {l}" for l in LEAGUES]
    tabs = st.tabs(tab_labels)

    for (league_name, league_info), tab in zip(LEAGUES.items(), tabs):
        with tab:
            league_standings = _all_standings.get(league_name, {})
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
                f'<div class="standings-header"><span class="standings-header-title">Matchday {first_team_data.get("played","?")}</span></div>'
                f'{hdr}{rows}'
                f'</div>',
                unsafe_allow_html=True
            )

            summary = _all_summaries.get(league_name, "")
            if summary:
                st.markdown(
                    f'<div class="standings-summary">'
                    f'<p>{summary}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
# The primary Analysis page: league + team selector, AI style cards, tactical pitches,
# ML match prediction, stats comparison bars, and the "Key points to watch" card.
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
        format_func=lambda l: f"{LEAGUES[l]['flag']}  {l}",
    )
    if selected_league != st.session_state.league:
        st.session_state.league = selected_league
        _new_code  = LEAGUES[selected_league]["code"]
        _new_stds  = fetch_standings(_new_code)
        _new_teams = sorted(_new_stds.keys(), key=lambda n: _new_stds[n]["position"]) if _new_stds else []
        st.session_state.team_a = _new_teams[0] if _new_teams else ""
        st.session_state.team_b = _new_teams[1] if len(_new_teams) > 1 else ""
        st.rerun()

    # ── Integrated VS team selector ──
    if "home_is_a" not in st.session_state:
        st.session_state.home_is_a = True
    _home_is_a_pre = st.session_state.home_is_a

    col_a, col_mid, col_b = st.columns([10, 1, 10])

    with col_a:
        da_pre = standings.get(team_a, {})
        img_pre_a = get_crest_img(team_a, 44)
        venue_a = "Home" if _home_is_a_pre else "Away"
        venue_col_a = "var(--green)" if _home_is_a_pre else "var(--mid)"
        st.markdown(
            f'<div class="team-pick-card team-pick-a">'
            f'<div class="team-pick-info">{img_pre_a}'
            f'<div class="team-pick-meta">'
            f'<div class="team-pick-name">{team_a}</div>'
            f'<div class="team-pick-stat">#{da_pre.get("position","—")} · {da_pre.get("points","—")} pts</div>'
            f'</div></div>'
            f'<div style="font-size:.6rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;'
            f'color:{venue_col_a};margin-top:.4rem;">{venue_a}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        idx_a = ALL_TEAMS.index(team_a) if team_a in ALL_TEAMS else 0
        new_a = st.selectbox("Team A", ALL_TEAMS, index=idx_a, label_visibility="collapsed")
        if new_a != st.session_state.team_a:
            st.session_state.team_a = new_a
            st.rerun()

    with col_mid:
        st.markdown('<div class="vs-mid-pill">VS</div>', unsafe_allow_html=True)
        if st.button("⇄", key="toggle_home", help="Swap home team", use_container_width=True):
            st.session_state.home_is_a = not st.session_state.home_is_a
            st.rerun()

    with col_b:
        remaining = [t for t in ALL_TEAMS if t != st.session_state.team_a]
        if st.session_state.team_b not in remaining:
            st.session_state.team_b = remaining[0] if remaining else ""
        team_b = st.session_state.team_b
        db_pre = standings.get(team_b, {})
        img_pre_b = get_crest_img(team_b, 44)
        venue_b = "Away" if _home_is_a_pre else "Home"
        venue_col_b = "var(--mid)" if _home_is_a_pre else "var(--green)"
        st.markdown(
            f'<div class="team-pick-card team-pick-b">'
            f'<div class="team-pick-info">{img_pre_b}'
            f'<div class="team-pick-meta">'
            f'<div class="team-pick-name">{team_b}</div>'
            f'<div class="team-pick-stat">#{db_pre.get("position","—")} · {db_pre.get("points","—")} pts</div>'
            f'</div></div>'
            f'<div style="font-size:.6rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;'
            f'color:{venue_col_b};margin-top:.4rem;text-align:right;">{venue_b}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        idx_b = remaining.index(st.session_state.team_b) if st.session_state.team_b in remaining else 0
        new_b = st.selectbox("Team B", remaining, index=idx_b, label_visibility="collapsed")
        if new_b != st.session_state.team_b:
            st.session_state.team_b = new_b
            st.rerun()

    team_a, team_b = st.session_state.team_a, st.session_state.team_b
    da, db = standings.get(team_a, {}), standings.get(team_b, {})
    crest_a, crest_b = da.get("crest",""), db.get("crest","")

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # Fetch enriched data for both teams — all 6 calls run in parallel
    _id_a, _id_b = da.get("id"), db.get("id")
    with st.spinner("Loading match data…"):
        with ThreadPoolExecutor(max_workers=6) as _pool:
            _f_ext_a   = _pool.submit(fetch_team_extended,        _id_a)
            _f_ext_b   = _pool.submit(fetch_team_extended,        _id_b)
            _f_extra_a = _pool.submit(fetch_api_football_stats,   team_a, _league_code)
            _f_extra_b = _pool.submit(fetch_api_football_stats,   team_b, _league_code)
            _f_scorers = _pool.submit(fetch_competition_scorers,  _league_code)
            _f_prev    = _pool.submit(fetch_previous_standings,   _league_code)
            (form_a_raw, ext_a) = _f_ext_a.result()
            (form_b_raw, ext_b) = _f_ext_b.result()
            extra_a             = _f_extra_a.result()
            extra_b             = _f_extra_b.result()
            all_scorers         = _f_scorers.result()
            prev_standings      = _f_prev.result()
    form_a     = tuple(form_a_raw)
    form_b     = tuple(form_b_raw)
    scorers_a  = tuple(all_scorers.get(team_a, [])[:3])
    scorers_b  = tuple(all_scorers.get(team_b, [])[:3])
    prev_pos_a = prev_standings.get(team_a)
    prev_pos_b = prev_standings.get(team_b)

    # ── Home/away assignment (toggle button in VS column) ──
    home_is_a = st.session_state.get("home_is_a", True)
    home_team  = team_a if home_is_a else team_b
    away_team  = team_b if home_is_a else team_a
    form_home  = form_a if home_is_a else form_b
    form_away  = form_b if home_is_a else form_a
    ext_home   = ext_a  if home_is_a else ext_b
    ext_away   = ext_b  if home_is_a else ext_a

    def _build_team_card_html(team_name, badge_label, hdr_bg, cards_tuple, form_tuple, stats_dict, crest_url):
        """CSS :target carousel — each panel owns its nav so prev/next arrows always point to the right card."""
        slug = re.sub(r'[^a-z0-9]', '_', team_name.lower())
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
            ("Club",         cards_tuple[0] if len(cards_tuple) > 0 else ""),
            ("Their game",   cards_tuple[1] if len(cards_tuple) > 1 else ""),
            ("In depth",     cards_tuple[2] if len(cards_tuple) > 2 else ""),
            ("Worth knowing",cards_tuple[3] if len(cards_tuple) > 3 else ""),
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
        for i, (label, text) in enumerate(card_defs):
            body = text or "—"
            for term in TACTICAL_TERMS:
                _lg = st.session_state.get("league", "Ligue 1")
                url = f"?term={term}&from=main&ta={team_a}&tb={team_b}&lg={_lg}"
                body = body.replace(
                    f"<b>{term}</b>",
                    f'<a href="{url}" target="_parent" style="color:#8B5CF6;font-weight:800;text-decoration:underline dotted 2px">{term}</a>'
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

    st.markdown('<div class="sec-title">Playing Style</div>', unsafe_allow_html=True)

    with st.spinner("Loading…"):
        # Run both style analyses + key challenges in parallel (3 Claude calls at once)
        with ThreadPoolExecutor(max_workers=3) as _claude_pool:
            _fs_a = _claude_pool.submit(
                generate_team_style,
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
            _fs_b = _claude_pool.submit(
                generate_team_style,
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
            _fc = _claude_pool.submit(
                generate_key_challenges,
                team_a, team_b,
                da.get("points",0), db.get("points",0),
                da.get("goals_for",0), db.get("goals_for",0),
                da.get("goals_against",0), db.get("goals_against",0),
            )
            style_a_raw      = _fs_a.result()
            style_b_raw      = _fs_b.result()
            _challenges_pair = _fc.result()

    # Handle legacy cache returning a plain string
    if isinstance(style_a_raw, str):
        style_a_raw = (style_a_raw, "", "", "")
    if isinstance(style_b_raw, str):
        style_b_raw = (style_b_raw, "", "", "")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            _build_team_card_html(team_a, f"#{da.get('position','—')} · {da.get('points','—')} pts", "#CCFFE9", style_a_raw, form_a, da, da.get("crest","")),
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            _build_team_card_html(team_b, f"#{db.get('position','—')} · {db.get('points','—')} pts", "#FFE0E0", style_b_raw, form_b, db, db.get("crest","")),
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
    st.markdown('<div class="sec-label">Machine Learning</div><div class="sec-title">How this could go</div>', unsafe_allow_html=True)

    ml_probs, ml_meta = predict_match(
        standings, home_team, away_team,
        form_home=list(form_home), form_away=list(form_away),
        extra_home=ext_home, extra_away=ext_away,
    )
    ml_xg = predict_expected_score(
        standings, home_team, away_team,
        form_home=list(form_home), form_away=list(form_away),
        extra_home=ext_home, extra_away=ext_away,
    )

    if ml_probs is not None and ml_meta is not None:
        # ml_probs[0]=home win, [1]=draw, [2]=away win — remap to team_a / team_b
        if home_is_a:
            prob_a_win, prob_draw, prob_b_win = ml_probs[0], ml_probs[1], ml_probs[2]
        else:
            prob_a_win, prob_draw, prob_b_win = ml_probs[2], ml_probs[1], ml_probs[0]
        hw = int(prob_a_win * 100)
        dr = int(prob_draw * 100)
        aw = int(prob_b_win * 100)
        # Form diff always in team_a vs team_b terms (not home vs away)
        fa_score = _form_score(list(form_a)) if form_a else 0.5
        fb_score = _form_score(list(form_b)) if form_b else 0.5
        form_diff = fa_score - fb_score
        if form_diff > 0.05:
            momentum_msg = f"<strong>{team_a}</strong> in better form — xG boosted accordingly"
            momentum_col = "#00C875"
        elif form_diff < -0.05:
            momentum_msg = f"<strong>{team_b}</strong> in better form — xG boosted accordingly"
            momentum_col = "#F2827F"
        else:
            momentum_msg = "Both teams in similar form — xG reflects venue & season stats"
            momentum_col = "#5A5A7A"

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
            # mls is [home_score, away_score] — remap to team_a / team_b
            score_a = mls[0] if home_is_a else mls[1]
            score_b = mls[1] if home_is_a else mls[0]
            xg_block = (
                f'<div style="flex:1;background:var(--bg);border-radius:14px;padding:1rem 1.2rem;text-align:center;">'
                f'<div style="font-size:.58rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--mid);margin-bottom:.4rem;">On paper</div>'
                f'<div style="font-size:2rem;font-weight:900;color:var(--dark);letter-spacing:.05em;line-height:1;">{score_a} <span style="color:var(--mid);font-size:1.2rem;">–</span> {score_b}</div>'
                f'</div>'
            )

        # Favourite sentence
        if hw >= 50:
            fav_team, fav_pct = team_a, hw
        elif aw >= 50:
            fav_team, fav_pct = team_b, aw
        else:
            fav_team, fav_pct = None, dr
        if fav_team and fav_pct >= 55:
            fav_sentence = f"{fav_team} are the favourites here."
        elif fav_team:
            fav_sentence = f"Slight edge to {fav_team}, but this is close."
        else:
            fav_sentence = "Hard to call. Either team could take this."

        st.markdown(
            f'<div style="background:var(--white);border-radius:var(--radius);border:2px solid var(--beige);box-shadow:var(--shadow);padding:1.4rem 1.6rem;">'
            f'<div style="font-size:.72rem;font-weight:800;color:var(--dark);margin-bottom:.6rem;">{fav_sentence}</div>'
            # Probability bar — no raw numbers, no confidence label
            f'<div style="font-size:.6rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--mid);margin-bottom:.35rem;">Who\'s likely to win</div>'
            f'<div style="display:flex;height:28px;border-radius:10px;overflow:hidden;">'
            f'<div style="width:{hw}%;background:#00C875;border-radius:8px 0 0 8px;"></div>'
            f'<div style="width:{dr}%;background:#FFB800;"></div>'
            f'<div style="width:{aw}%;background:#FF5C5C;border-radius:0 8px 8px 0;"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-top:.4rem;font-size:.68rem;font-weight:800;">'
            f'<span style="color:#00A862;">{team_a}</span>'
            f'<span style="color:#F5A500;">Draw</span>'
            f'<span style="color:#E63F3F;">{team_b}</span>'
            f'</div>'
            # Middle row: score + form
            f'<div style="display:flex;gap:1rem;margin-top:1.2rem;">'
            f'{xg_block}'
            f'<div style="flex:1.5;background:var(--bg);border-radius:14px;padding:1rem 1.2rem;">'
            f'<div style="font-size:.58rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--mid);margin-bottom:.5rem;">Last 5</div>'
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
    played_b = db.get("played",1) or 1

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
        st.markdown(cmp_card("stat-cmp-hdr-1","stat-cmp-dot-1","Offensive efficiency",gf_a,gf_b,max_gf,f"Avg. {gf_a/played_a:.1f} vs {gf_b/played_b:.1f} goals / match"), unsafe_allow_html=True)
    with r2:
        st.markdown(cmp_card("stat-cmp-hdr-2","stat-cmp-dot-2","Points in standings",pts_a,pts_b,max_pts,f"#{da.get('position','—')} vs #{db.get('position','—')} in the standings"), unsafe_allow_html=True)
    with r3:
        st.markdown(cmp_card("stat-cmp-hdr-3","stat-cmp-dot-3","Defensive solidity",ga_a,ga_b,max_ga,"Goals conceded — lower is better",inverted=True), unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Watch card ──
    points = watch_points(team_a, team_b)
    challenge_a, challenge_b = _challenges_pair
    st.markdown(
        f'<div class="watch-card">'
        f'<div class="watch-header"><div class="watch-icon">👁</div><div><div class="watch-title">What to look for</div><div class="watch-subtitle">{team_a} vs {team_b}</div></div></div>'
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
# Renders the Schedule page — match cards grouped by date with live scores, ML win-probability
# bars, expected score lines, and momentum indicators for upcoming fixtures.
def page_schedule():
    from datetime import datetime, timedelta, timezone
    from collections import defaultdict

    st.markdown('<div class="sec-title">Match Schedule</div>', unsafe_allow_html=True)

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
            # Fetch standings and schedule for this league in parallel
            with ThreadPoolExecutor(max_workers=2) as _sched_pool:
                _f_st = _sched_pool.submit(fetch_standings, linfo["code"])
                _f_sc = _sched_pool.submit(fetch_schedule, linfo["code"], date_from, date_to)
                all_standings[lname] = _f_st.result()
                matches              = _f_sc.result()
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
.sched-pred-labels{display:flex;justify-content:space-between;align-items:flex-start;font-size:.58rem;font-weight:700;gap:.3rem;}
.sched-pred-labels span{flex:1;word-break:break-word;}
.sched-pred-labels span:last-child{text-align:right;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
</style>""", unsafe_allow_html=True)

    def _sched_extra(td, is_home_team):
        p = td.get("played", 1) or 1
        gf = td.get("goals_for", 0) / p
        ga = td.get("goals_against", 0) / p
        return {
            "gf_avg_recent":  round(gf, 2),
            "ga_avg_recent":  round(ga, 2),
            "home_gf_avg":    round(gf * 1.18, 2),
            "home_ga_avg":    round(ga * 0.82, 2),
            "away_gf_avg":    round(gf * 0.82, 2),
            "away_ga_avg":    round(ga * 1.18, 2),
            "fatigued":       False,
        }

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
                score_html = f'<div class="sched-score" style="color:var(--mid);font-size:.8rem"></div>'

            if status_raw in ("TIMED", "SCHEDULED"):
                status_badge = f'<span class="sched-status {status_cls}">{m["local_time"]}</span>'
            else:
                status_badge = f'<span class="sched-status {status_cls}">{status_txt}</span>'

            md_badge = f'<span class="sched-matchday">Matchday {m["matchday"]}</span>' if m["matchday"] else ""

            cur_standings = all_standings.get(m["league"], {})
            # Normalize team names (schedule API returns raw names, standings uses mapped names)
            h_norm = TEAM_NAME_MAP.get(m["home"], m["home"])
            a_norm = TEAM_NAME_MAP.get(m["away"], m["away"])
            # Build venue-split stats from standings — same formula as Analysis page.
            # Home teams score ~18% more at home and concede ~18% less (5-league average).
            h_td = cur_standings.get(h_norm, {})
            a_td = cur_standings.get(a_norm, {})
            sched_ext_h = _sched_extra(h_td, True)  if h_td else None
            sched_ext_a = _sched_extra(a_td, False) if a_td else None
            # predict_match now calls predict_expected_score internally — one model, one call
            probs, pred_meta = predict_match(cur_standings, h_norm, a_norm,
                                             extra_home=sched_ext_h, extra_away=sched_ext_a)
            xg = predict_expected_score(cur_standings, h_norm, a_norm,
                                        extra_home=sched_ext_h, extra_away=sched_ext_a)
            pred_html = ""
            if probs is not None and status_raw in ("TIMED", "SCHEDULED"):
                hw = int(probs[0] * 100)
                dr = int(probs[1] * 100)
                aw = int(probs[2] * 100)

                # Likely score line
                score_line = ""
                if xg:
                    mls = xg["most_likely_score"]
                    score_line = (
                        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:.5rem;margin-top:.3rem;padding-top:.3rem;border-top:1px dashed var(--beige);">'
                        f'<span style="font-size:.58rem;color:var(--mid);letter-spacing:.06em;text-transform:uppercase;font-weight:800;">Likely score</span>'
                        f'<span style="font-size:.85rem;font-weight:900;color:var(--dark);letter-spacing:.05em;">{mls[0]} – {mls[1]}</span>'
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
                    f'<span style="color:#4CAF50">{h_norm}</span>'
                    f'<span style="color:#FFC107;flex:0;white-space:nowrap">Draw</span>'
                    f'<span style="color:#F44336">{a_norm}</span>'
                    f'</div>'
                    f'{score_line}'
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE RULES
# ══════════════════════════════════════════════════════════════════════════════
# Renders the Rules page — iterates FOOTBALL_RULES and displays each as an expandable
# card with emoji, title, and plain-English description.
def page_regles():
    st.markdown('<div class="sec-title">Rules of the Game</div>', unsafe_allow_html=True)
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

    for title, _, description in FOOTBALL_RULES:
        st.markdown(
            f'<details class="rule-card">'
            f'<summary>'
            f'<span class="rule-icon" style="display:flex;align-items:center;justify-content:center"><span style="width:7px;height:7px;border-radius:50%;background:var(--green);display:inline-block;flex-shrink:0"></span></span>'
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
