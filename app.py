import streamlit as st
import requests
import re
import anthropic

st.set_page_config(page_title="The Football Classroom", layout="wide")

# ── API ───────────────────────────────────────────────────────────────────────
API_KEY = "911605e549af4b759c5d7d2ffa977742"
HEADERS = {"X-Auth-Token": API_KEY}

ANTHROPIC_API_KEY  = st.secrets.get("ANTHROPIC_API_KEY", "")
API_FOOTBALL_KEY   = st.secrets.get("API_FOOTBALL_KEY", "")

LEAGUES = {
    "Ligue 1":        {"code": "FL1",  "flag": "🇫🇷", "country": "France",  "color": "#1A56C4", "color_lt": "#E8F0FB"},
    "La Liga":        {"code": "PD",   "flag": "🇪🇸", "country": "Spain",   "color": "#F5C800", "color_lt": "#FFFBE6"},
    "Serie A":        {"code": "SA",   "flag": "🇮🇹", "country": "Italy",   "color": "#1E9E4A", "color_lt": "#E6F7EC"},
    "Premier League": {"code": "PL",   "flag": "🇬🇧", "country": "England", "color": "#E02020", "color_lt": "#FDEAED"},
    "Bundesliga":     {"code": "BL1",  "flag": "🇩🇪", "country": "Germany", "color": "#7B4A1E", "color_lt": "#F5EDE6"},
}

# Ligue 1 team IDs on API-Football (only used for advanced stats)
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

# Name normalization for Ligue 1 (football-data.org names → display names)
FL1_NAME_MAP = {
    "Paris Saint-Germain FC": "Paris Saint-Germain",
    "Racing Club de Lens":    "RC Lens",
    "Olympique de Marseille": "Olympique de Marseille",
    "Olympique Lyonnais":     "Olympique Lyonnais",
    "Lille OSC":              "LOSC Lille",
    "AS Monaco FC":           "AS Monaco",
    "Stade Rennais FC 1901":  "Stade Rennais",
    "RC Strasbourg Alsace":   "RC Strasbourg",
    "Toulouse FC":            "Toulouse FC",
    "FC Lorient":             "FC Lorient",
    "Stade Brestois 29":      "Stade Brestois",
    "Angers SCO":             "Angers SCO",
    "Paris FC":               "Paris FC",
    "Le Havre AC":            "Le Havre AC",
    "OGC Nice":               "OGC Nice",
    "AJ Auxerre":             "AJ Auxerre",
    "FC Nantes":              "FC Nantes",
    "FC Metz":                "FC Metz",
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_standings(league_code):
    name_map = FL1_NAME_MAP if league_code == "FL1" else {}
    try:
        r = requests.get(
            f"https://api.football-data.org/v4/competitions/{league_code}/standings",
            headers=HEADERS, timeout=10
        )
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
        return {}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_previous_standings(league_code):
    name_map = FL1_NAME_MAP if league_code == "FL1" else {}
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
    try:
        r = requests.get(
            f"https://api.football-data.org/v4/competitions/{league_code}/matches",
            headers=HEADERS,
            params={"dateFrom": date_from, "dateTo": date_to},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("matches", [])
    except Exception:
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
    name_map = FL1_NAME_MAP if league_code == "FL1" else {}
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

    prompt = f"""You are a passionate football commentator writing for fans of all levels — from total beginners to die-hard supporters.
Using the stats below AND your own knowledge of this club, write an analysis in EXACTLY 3 paragraphs in English.

{stats_block}

CRITICAL: Your response must contain EXACTLY 3 paragraphs separated by a blank line. Not 2, not 4. Exactly 3.

Paragraph 1 — BIG PICTURE (2 to 3 sentences MAX): Playing style, recent form, last season comparison, key players or trophies. Be concise.

Paragraph 2 — HOW THEY PLAY (2 to 3 sentences MAX): Formation, strengths, how they press/build up, one or two stats translated into behaviour. Simple and accessible.

Paragraph 3 — WHAT'S HAPPENING NOW (1 to 2 sentences MAX): NOT technical. One current news item — injured player, player returning from injury, transfer, record, or fun fact.

Rules:
- Write in English, vivid and accessible — no unexplained jargon
- Name real players when relevant
- You MUST use AT LEAST 5 terms from this glossary: {terms}
- Every time you use one of those terms, wrap it EXACTLY like this: <b>term</b>
- Separate the 3 paragraphs with a blank line (\\n\\n)
- No titles, no bullet points, no numbering
- If unsure of a specific fact, stay vague rather than inventing

Reply with the 3 paragraphs only, nothing else."""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        # If response was cut mid-sentence, trim to last complete sentence
        if raw and raw[-1] not in ".!?":
            for sep in (".", "!", "?"):
                idx = raw.rfind(sep)
                if idx != -1:
                    raw = raw[:idx+1]
                    break
        return raw
    except Exception:
        return TEAM_STYLES.get(team_name, DEFAULT_STYLE)


@st.cache_data(ttl=3600, show_spinner=False)
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
        "animation_idea":    "The striker drops into midfield, leaving space that midfielders run into behind.",
    },
    "build-up play": {
        "definition":        "The phase of play when a team in possession tries to move the ball forward from defence into attack against an organized opposition.",
        "simple_explanation":"The team passes the ball from the back to the front in an organized, controlled way.",
        "example":           "A goalkeeper passes to a centre-back, who plays to the midfielder, who finds the striker.",
        "animation_idea":    "Arrows showing ball movement from defence through midfield into the final third.",
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
        "animation_idea":    "Players swarming as a group toward the ball the instant possession is lost.",
    },
    "transition": {
        "definition":        "The moment a team switches from attack to defence (negative transition) or from defence to attack (positive transition).",
        "simple_explanation":"The moment your team changes from defending to attacking — or the other way around.",
        "example":           "A team is caught in a bad transition when they lose the ball with players pushed too far forward.",
        "animation_idea":    "Half the arrows reversing direction simultaneously as the ball changes hands.",
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
        "animation_idea":    "Two vertical channels highlighted between the centre circle and the touchlines.",
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
        "animation_idea":    "Rapid short passes between many players forming fluid triangles across the pitch.",
    },
    "total football": {
        "definition":        "A tactical theory in which any outfield player can take over the role of any other player, requiring universal positional flexibility.",
        "simple_explanation":"Every player is comfortable in any position — if someone moves, a teammate fills the gap.",
        "example":           "The Netherlands team of the 1970s rotated positions fluidly under coach Rinus Michels.",
        "animation_idea":    "Players swapping positions with arrows showing continuous positional rotations.",
    },
    "positional play": {
        "definition":        "A tactical system focused on controlling the game through intelligent positioning, occupying key spaces to dominate the pitch.",
        "simple_explanation":"Players take up smart positions to control space and make the team impossible to press.",
        "example":           "Manchester City under Guardiola use positional play to maintain passing structures at all times.",
        "animation_idea":    "Players in structured positions forming triangles and diamonds across the full pitch.",
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
        "animation_idea":    "Two players exchanging the ball as a third player runs into open space to receive.",
    },
    "line-breaking pass": {
        "definition":        "A pass that bypasses one or more lines of opposing players in a single movement, skipping midfield or the defensive line.",
        "simple_explanation":"A pass that jumps over a whole line of opponents in one go, instantly unlocking the defence.",
        "example":           "A centre-back plays a line-breaking pass into the feet of a forward between the midfield and defensive lines.",
        "animation_idea":    "Ball traveling through a defensive line with a dotted arrow bypassing all players in one move.",
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
    "1. FSV Mainz 05": {
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
    "1. FC Union Berlin": {
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

def render_tactical_pitch_html(team_name):
    """Generate a premium animated SVG tactical pitch for a given team."""
    import math
    PAD, PW, PH = 14, 252, 360
    SW, SH = PW + 2 * PAD, PH + 2 * PAD

    def sx(p): return PAD + p / 100 * PW
    def sy(p): return PAD + p / 100 * PH

    t = TEAM_TACTICS.get(team_name)
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
        players_svg += (
            f'<g class="{cls}" {filt}>'
            f'{ring}'
            f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="11.5" fill="{color}" stroke="rgba(255,255,255,.9)" stroke-width="1.8"/>'
            f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="11.5" fill="rgba(255,255,255,.08)"/>'
            f'<text x="{x0:.1f}" y="{y0:.1f}" text-anchor="middle" dominant-baseline="central" '
            f'font-size="6.2" font-weight="900" fill="white" font-family="Nunito,sans-serif" letter-spacing="-.3">{abbr}</text>'
            f'</g>\n'
        )

    # ── Style pills ──
    pills = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .72rem;'
        f'border-radius:100px;background:rgba(255,255,255,.08);color:rgba(255,255,255,.75);'
        f'font-size:.6rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;'
        f'border:1px solid rgba(255,255,255,.12);margin:.15rem .15rem 0 0">'
        f'<span style="width:5px;height:5px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0"></span>'
        f'{s}</span>'
        for s in t.get("style_tags", [])
    )

    css_block = "<style>" + "".join(css_lines) + "</style>"

    svg = (
        f'<svg viewBox="0 0 {SW} {SH}" xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;width:100%;background:#1e5c1e;">'
        f'{defs}{pitch_bg}{m}{zones_svg}{formation_lines_svg}{arrows_svg}{players_svg}'
        f'</svg>'
    )

    return (
        f'{css_block}'
        f'<div style="background:#0F1C0F;border-radius:20px;overflow:hidden;'
        f'box-shadow:0 8px 32px rgba(0,0,0,.45),0 0 0 1px rgba(255,255,255,.06);">'
        # Header
        f'<div style="padding:.9rem 1.1rem .6rem;display:flex;align-items:center;'
        f'justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.07);">'
        f'<div>'
        f'<div style="font-size:.64rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase;'
        f'color:{color};margin-bottom:.18rem;">Team Analysis</div>'
        f'<div style="font-size:.95rem;font-weight:900;color:rgba(255,255,255,.92);letter-spacing:-.02em;">{team_name}</div>'
        f'</div>'
        f'<span style="background:{color};color:white;font-size:.72rem;font-weight:900;'
        f'padding:.3rem .9rem;border-radius:100px;letter-spacing:.06em;'
        f'box-shadow:0 2px 10px {color}66;">{t["formation"]}</span>'
        f'</div>'
        # SVG pitch
        f'<div style="position:relative;">{svg}</div>'
        # Footer pills
        f'<div style="padding:.6rem 1rem .8rem;border-top:1px solid rgba(255,255,255,.06);">'
        f'<div style="font-size:.56rem;font-weight:800;letter-spacing:.16em;color:rgba(255,255,255,.3);'
        f'text-transform:uppercase;margin-bottom:.3rem;">Playing style</div>'
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
        css = (
            f"@keyframes bu1_{slug}{{0%{{stroke-dashoffset:80;opacity:0}}20%,38%{{stroke-dashoffset:0;opacity:.9}}55%,100%{{stroke-dashoffset:80;opacity:0}}}}"
            f"@keyframes bu2_{slug}{{0%,20%{{stroke-dashoffset:80;opacity:0}}38%,56%{{stroke-dashoffset:0;opacity:.9}}72%,100%{{stroke-dashoffset:80;opacity:0}}}}"
            f"@keyframes bu3_{slug}{{0%,38%{{stroke-dashoffset:80;opacity:0}}56%,74%{{stroke-dashoffset:0;opacity:.9}}90%,100%{{stroke-dashoffset:80;opacity:0}}}}"
            f".bua1_{slug}{{stroke-dasharray:80;animation:bu1_{slug} 3.2s ease-in-out infinite}}"
            f".bua2_{slug}{{stroke-dasharray:80;animation:bu2_{slug} 3.2s ease-in-out infinite}}"
            f".bua3_{slug}{{stroke-dasharray:80;animation:bu3_{slug} 3.2s ease-in-out infinite}}"
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

    # ── Style pills (same structure as tactical pitch) ──
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
        # Header — identical layout to tactical pitch
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
        # SVG
        f'<div style="position:relative;">{svg}</div>'
        # Footer — identical layout to tactical pitch
        f'<div style="padding:.6rem 1rem .8rem;border-top:1px solid rgba(255,255,255,.06);">'
        f'<div style="font-size:.56rem;font-weight:800;letter-spacing:.16em;color:rgba(255,255,255,.3);'
        f'text-transform:uppercase;margin-bottom:.3rem;">Key attributes</div>'
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

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("page","main"), ("prev_page","main"), ("active_term",None),
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
    st.markdown('<div class="sec-label">Vocabulary</div><div class="sec-title">Tactical Glossary</div>', unsafe_allow_html=True)
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
            unsafe_allow_html=True
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

    # ── Styles de jeu ──
    st.markdown('<div class="sec-label">AI Analysis</div><div class="sec-title">Playing Style</div>', unsafe_allow_html=True)

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

    def _render_form(form):
        if not form:
            return ""
        colors = {"W": "var(--green)", "D": "var(--yellow-dk)", "L": "var(--red)"}
        pills = "".join(
            f'<span style="display:inline-block;padding:.15rem .5rem;border-radius:6px;'
            f'background:{"var(--green-lt)" if r=="W" else ("var(--yellow-lt)" if r=="D" else "var(--red-lt)")};'
            f'color:{colors[r]};font-size:.65rem;font-weight:900;margin-right:.25rem">{r}</span>'
            for r in form
        )
        return f'<div style="margin-bottom:.6rem;font-size:.62rem;font-weight:800;color:var(--mid);letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem">Recent form &nbsp;{pills}</div>'

    def _fmt_style(raw):
        """Show 1st paragraph by default, rest collapsible behind a + button."""
        html = raw.replace("\n\n", "<br><br>").replace("\n", " ")
        html = linkify_terms(html, source_page="main", ta=team_a, tb=team_b)
        parts = html.split("<br><br>", 1)
        summary = f'<div class="style-summary">{parts[0]}</div>'
        if len(parts) > 1:
            detail = f'<div class="style-details">{parts[1]}</div>'
            return f'{summary}<details class="style-acc"><summary></summary>{detail}</details>'
        return summary

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

    style_a = _fmt_style(style_a_raw)
    style_b = _fmt_style(style_b_raw)

    c1, c2 = st.columns(2)
    img28_a = get_crest_img(team_a, 28)
    img28_b = get_crest_img(team_b, 28)

    with c1:
        st.markdown(
            f'<div class="team-card card-a">'
            f'<div class="team-card-header">{img28_a} {team_a}<span class="badge">Team A</span></div>'
            f'<div class="team-card-body">{_render_form(form_a)}{style_a}</div>'
            f'<div class="team-stats-row">'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("points","—")}</div><div class="team-stat-box-lbl">Pts</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("won","—")}</div><div class="team-stat-box-lbl">W</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("draw","—")}</div><div class="team-stat-box-lbl">D</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("lost","—")}</div><div class="team-stat-box-lbl">L</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("goals_for","—")}</div><div class="team-stat-box-lbl">Goals</div></div>'
            f'</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="team-card card-b">'
            f'<div class="team-card-header">{img28_b} {team_b}<span class="badge">Team B</span></div>'
            f'<div class="team-card-body">{_render_form(form_b)}{style_b}</div>'
            f'<div class="team-stats-row">'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("points","—")}</div><div class="team-stat-box-lbl">Pts</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("won","—")}</div><div class="team-stat-box-lbl">W</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("draw","—")}</div><div class="team-stat-box-lbl">D</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("lost","—")}</div><div class="team-stat-box-lbl">L</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("goals_for","—")}</div><div class="team-stat-box-lbl">Goals</div></div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Terrain ──
    st.markdown('<div class="sec-label">Tactics</div><div class="sec-title">Tactical pitch</div>', unsafe_allow_html=True)
    pitch_col_a, pitch_col_b = st.columns(2)
    with pitch_col_a:
        st.markdown(render_tactical_pitch_html(team_a), unsafe_allow_html=True)
    with pitch_col_b:
        st.markdown(render_tactical_pitch_html(team_b), unsafe_allow_html=True)

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

    # ── League filter (toggle buttons) ──
    if "sched_leagues" not in st.session_state:
        st.session_state.sched_leagues = set(LEAGUES.keys())

    btn_cols = st.columns(len(LEAGUES))
    for col, (lname, linfo) in zip(btn_cols, LEAGUES.items()):
        with col:
            active = lname in st.session_state.sched_leagues
            if st.button(f"{linfo['flag']} {lname}", key=f"sched_btn_{lname}",
                         type="primary" if active else "secondary",
                         use_container_width=True):
                if active:
                    st.session_state.sched_leagues.discard(lname)
                else:
                    st.session_state.sched_leagues.add(lname)
                st.rerun()

    selected = st.session_state.sched_leagues
    if not selected:
        st.markdown('<p style="color:var(--mid);text-align:center;padding:2rem 0">Select at least one league above.</p>', unsafe_allow_html=True)
        return

    now       = datetime.now(timezone.utc)
    date_from = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    date_to   = (now + timedelta(days=14)).strftime("%Y-%m-%d")

    all_matches = []
    with st.spinner("Loading schedule…"):
        for lname in selected:
            linfo = LEAGUES[lname]
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

            html += (
                f'<div class="sched-match" style="border-left-color:{m["color"]}">'
                f'<div class="sched-league-dot" style="background:{m["color"]}"></div>'
                f'<div class="sched-teams">{m["home"]} <span style="color:var(--mid);font-weight:700">vs</span> {m["away"]}</div>'
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
