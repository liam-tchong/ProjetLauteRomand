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
    "Ligue 1":        {"code": "FL1",  "flag": "🇫🇷", "country": "France"},
    "La Liga":        {"code": "PD",   "flag": "🇪🇸", "country": "Spain"},
    "Serie A":        {"code": "SA",   "flag": "🇮🇹", "country": "Italy"},
    "Premier League": {"code": "PL",   "flag": "🇬🇧", "country": "England"},
    "Bundesliga":     {"code": "BL1",  "flag": "🇩🇪", "country": "Germany"},
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_team_form(team_id):
    """Last 5 played matches for a football-data.org team_id."""
    if not team_id:
        return []
    try:
        r = requests.get(
            f"https://api.football-data.org/v4/teams/{team_id}/matches",
            headers=HEADERS,
            params={"status": "FINISHED", "limit": 5},
            timeout=10
        )
        r.raise_for_status()
        matches = r.json().get("matches", [])
        form = []
        for m in matches:
            home_id    = m["homeTeam"]["id"]
            home_score = m["score"]["fullTime"]["home"]
            away_score = m["score"]["fullTime"]["away"]
            is_home    = (home_id == team_id)
            gs = home_score if is_home else away_score
            gc = away_score if is_home else home_score
            if gs is None or gc is None:
                continue
            if gs > gc:   form.append("W")
            elif gs < gc: form.append("L")
            else:          form.append("D")
        return form[-5:]
    except Exception:
        return []


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
                        extra_clean_sheets, extra_failed_to_score):
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
    """Generates one key challenge sentence per team for this specific matchup."""
    if not ANTHROPIC_API_KEY:
        return (
            f"{team_a} must stay compact and limit space in behind.",
            f"{team_b} must be clinical in the final third.",
        )
    prompt = f"""You are a concise football analyst. Given these two teams and their season stats, write exactly 2 lines — one per team — describing each team's single biggest tactical challenge in this specific matchup.

{team_a}: {pts_a} pts, {gf_a} goals scored, {ga_a} goals conceded this season.
{team_b}: {pts_b} pts, {gf_b} goals scored, {ga_b} goals conceded this season.

Format (reply with exactly these 2 lines, nothing else):
{team_a}: <one sentence, max 12 words>
{team_b}: <one sentence, max 12 words>"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}]
        )
        lines = msg.content[0].text.strip().split("\n")
        def _extract(line, team):
            if ":" in line:
                return line.split(":", 1)[1].strip()
            return line.strip()
        challenge_a = _extract(lines[0], team_a) if len(lines) > 0 else "Stay compact and limit space in behind."
        challenge_b = _extract(lines[1], team_b) if len(lines) > 1 else "Be clinical in the final third."
        return challenge_a, challenge_b
    except Exception:
        return (
            f"Stay compact and limit space in behind.",
            f"Be clinical in the final third.",
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

/* VS Banner */
.vs-banner{display:flex;align-items:stretch;background:var(--white);border-radius:var(--radius);border:2px solid var(--beige);overflow:hidden;box-shadow:var(--shadow);margin-top:.6rem;}
.vs-team{flex:1;padding:1.2rem 1.8rem;display:flex;flex-direction:column;gap:.3rem;}
.vs-team-crest{display:flex;align-items:center;gap:.75rem;}
.vs-team-label{font-size:.62rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;}
.vs-team-label-a{color:var(--green-dk);}
.vs-team-label-b{color:var(--red-dk);}
.vs-team-name{font-size:1.05rem;font-weight:900;color:var(--dark);letter-spacing:-.02em;line-height:1.2;}
.vs-team-stat{font-size:.72rem;font-weight:700;color:var(--mid);}
.vs-sep{background:var(--dark);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 1.4rem;gap:.15rem;}
.vs-sep-text{font-size:.9rem;font-weight:900;color:var(--white);letter-spacing:.06em;}
.vs-sep-dot{width:5px;height:5px;border-radius:50%;background:rgba(255,255,255,.2);}

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
details.style-acc summary{list-style:none;cursor:pointer;display:inline-flex;align-items:center;gap:.4rem;font-size:.72rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--dark);background:var(--beige);border-radius:100px;padding:.25rem .75rem;margin-top:.5rem;user-select:none;}
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

/* Glossary accordion */
details.glos-acc{margin-bottom:1rem;}
details.glos-acc summary{list-style:none;cursor:pointer;display:block;}
details.glos-acc summary::-webkit-details-marker{display:none;}
details.glos-acc summary .glos-card{margin-bottom:0;border-radius:var(--radius);transition:border-radius .15s;}
details.glos-acc[open] summary .glos-card{border-radius:var(--radius) var(--radius) 0 0!important;}
.glos-acc-body{background:var(--white);border:2px solid var(--beige);border-top:none;border-radius:0 0 var(--radius) var(--radius);padding:1.2rem 1.4rem 1.4rem;box-shadow:var(--shadow);}
.pill-tac::after{content:'Tactical →';}
details.glos-acc[open] .pill-tac::after{content:'Close ↑';}
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
    c1, c2, c3 = st.columns(3)
    with c1:
        t = "primary" if page == "main" else "secondary"
        if st.button("⚽  Analysis", type=t, use_container_width=True, key="nav_main"):
            st.session_state.page = "main"; st.rerun()
    with c2:
        t = "primary" if page == "classement" else "secondary"
        if st.button("📊  Standings", type=t, use_container_width=True, key="nav_class"):
            st.session_state.page = "classement"; st.rerun()
    with c3:
        t = "primary" if page == "glossaire" else "secondary"
        if st.button("📖  Glossary", type=t, use_container_width=True, key="nav_glos"):
            st.session_state.page = "glossaire"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PITCH ANIMATIONS
# ══════════════════════════════════════════════════════════════════════════════
def build_pitch_animation(term):
    import math
    PAD, PW, PH = 14, 252, 360
    SW, SH = PW + 2*PAD, PH + 2*PAD
    def sx(p): return PAD + p/100*PW
    def sy(p): return PAD + p/100*PH

    GLOS_SCENARIOS = {
        "pressing": {
            "label": "Pressing",
            "subtitle": "Ball Recovery",
            "tags": ["High Press", "Intensity", "Collective"],
            "zones": [(50, 28, 28, 18, 0.30)],
            # groups: (color, [(x%, y%, label)], {player_idx: (to_x%, to_y%)})
            "groups": [
                ("#E05555", [(50, 22, "O")],  {}),
                ("#F5D06E", [(28, 42, "P"), (50, 46, "P"), (72, 42, "P")],
                            {0: (44, 27), 1: (50, 27), 2: (56, 27)}),
            ],
        },
        "pivot": {
            "label": "Pivot",
            "subtitle": "Hold-Up Play",
            "tags": ["Target Man", "Link Play", "Hold-Up"],
            "zones": [(50, 42, 24, 14, 0.22)],
            "groups": [
                ("#4CAF85", [(50, 60, "8")],  {0: (50, 44)}),
                ("#F5D06E", [(50, 44, "9")],  {}),
                ("#4CAF85", [(28, 28, "10"), (72, 28, "7")], {0: (28, 20), 1: (72, 20)}),
            ],
        },
        "false nine": {
            "label": "False Nine",
            "subtitle": "Space Creation",
            "tags": ["False 9", "Space Creation", "Confusion"],
            "zones": [(50, 24, 26, 14, 0.28)],
            "groups": [
                ("#E05555", [(50, 32, "CB")], {0: (50, 46)}),
                ("#F5D06E", [(50, 20, "9")],  {0: (50, 44)}),
                ("#4CAF85", [(36, 52, "8")],  {0: (44, 26)}),
            ],
        },
        "build-up play": {
            "label": "Build-Up Play",
            "subtitle": "Ball Progression",
            "tags": ["Structure", "Patience", "Circulation"],
            "zones": [(50, 78, 30, 10, 0.16), (50, 58, 26, 10, 0.12)],
            "groups": [
                ("#4CAF85", [
                    (50, 88, "GK"),
                    (30, 76, "CB"), (70, 76, "CB"),
                    (36, 60, "DM"), (64, 60, "DM"),
                    (50, 42, "CAM"),
                ], {0: (30, 76), 1: (36, 60), 3: (50, 42)}),
            ],
        },
        "through ball": {
            "label": "Through Ball",
            "subtitle": "Line Breaker",
            "tags": ["Penetration", "Timing", "Space"],
            "zones": [(62, 18, 18, 12, 0.30)],
            "groups": [
                ("#E05555", [(25,34,"CB"),(42,34,"CB"),(58,34,"CB"),(75,34,"CB")], {}),
                ("#4CAF85", [(35, 50, "8")], {0: (62, 18)}),
                ("#4CAF85", [(62, 30, "9")], {0: (62, 14)}),
            ],
        },
        "switch of play": {
            "label": "Switch of Play",
            "subtitle": "Width Exploitation",
            "tags": ["Width", "Diagonal", "Space"],
            "zones": [(82, 36, 14, 20, 0.25)],
            "groups": [
                ("#4CAF85", [
                    (18, 36, "LB"), (32, 46, "CM"), (50, 44, "CM"),
                    (82, 36, "RB"),
                ], {2: (82, 36)}),
                ("#E05555", [(40,34,"CB"),(60,34,"CB"),(40,48,"DM"),(60,48,"DM")], {}),
            ],
        },
        "overlap": {
            "label": "Overlap",
            "subtitle": "Attacking Run",
            "tags": ["Full-Back", "Width", "Overload"],
            "zones": [(88, 22, 12, 20, 0.28)],
            "groups": [
                ("#4CAF85", [(78, 32, "RW")], {}),
                ("#F5D06E", [(84, 54, "RB")], {0: (92, 20)}),
                ("#E05555", [(72, 38, "LB")], {}),
            ],
        },
        "underlap": {
            "label": "Underlap",
            "subtitle": "Inside Run",
            "tags": ["Half-Space", "Cut Inside", "Diagonal"],
            "zones": [(64, 26, 16, 14, 0.26)],
            "groups": [
                ("#4CAF85", [(80, 30, "RW")], {}),
                ("#F5D06E", [(68, 50, "RM")], {0: (62, 24)}),
                ("#E05555", [(76, 38, "LB"),(62, 38, "CB")], {}),
            ],
        },
        "cross": {
            "label": "Cross",
            "subtitle": "Wide Delivery",
            "tags": ["Crossing", "Width", "Aerial"],
            "zones": [(40, 14, 20, 12, 0.28)],
            "groups": [
                ("#F5D06E", [(84, 28, "RB")], {0: (84, 20)}),
                ("#4CAF85", [(36, 18, "9"), (52, 22, "10")], {0: (36, 12), 1: (52, 14)}),
                ("#E05555", [(40, 30, "CB"),(56, 30, "CB")], {}),
            ],
        },
        "final third": {
            "label": "Final Third",
            "subtitle": "Danger Zone",
            "tags": ["Attacking Zone", "Chance Creation", "Pressure"],
            "zones": [(50, 20, 46, 22, 0.22)],
            "groups": [
                ("#F5D06E", [(30, 28, "LW"), (50, 18, "9"), (70, 28, "RW")], {1: (50, 12)}),
                ("#4CAF85", [(36, 44, "LCM"), (64, 44, "RCM")], {0: (36, 30), 1: (64, 30)}),
                ("#E05555", [(28,32,"LB"),(44,34,"CB"),(56,34,"CB"),(72,32,"RB")], {}),
            ],
        },
        "counter-attack": {
            "label": "Counter-Attack",
            "subtitle": "Fast Transition",
            "tags": ["Speed", "Transition", "Vertical"],
            "zones": [(50, 20, 36, 16, 0.26)],
            "groups": [
                ("#4CAF85", [
                    (26, 62, "LW"), (50, 66, "ST"), (74, 62, "RW"),
                ], {0: (18, 24), 1: (50, 18), 2: (82, 24)}),
                ("#E05555", [(38,36,"CB"),(62,36,"CB"),(44,46,"DM"),(56,46,"DM")], {}),
            ],
        },
        "high press": {
            "label": "High Press",
            "subtitle": "Aggressive Pressure",
            "tags": ["Gegenpressing", "High Line", "Intensity"],
            "zones": [(50, 20, 36, 18, 0.28)],
            "groups": [
                ("#E05555", [(30,18,"CB"),(70,18,"CB"),(50,28,"DM")], {}),
                ("#F5D06E", [
                    (22, 32, "LW"), (50, 34, "ST"), (78, 32, "RW"),
                    (38, 42, "LCM"), (62, 42, "RCM"),
                ], {0: (26, 22), 1: (50, 24), 2: (74, 22)}),
            ],
        },
        "low block": {
            "label": "Low Block",
            "subtitle": "Deep Defense",
            "tags": ["Defensive", "Compact", "Resilience"],
            "zones": [(50, 74, 46, 16, 0.20), (50, 86, 36, 10, 0.18)],
            "groups": [
                ("#4CAF85", [
                    (18,66,"LM"),(36,66,"CM"),(64,66,"CM"),(82,66,"RM"),
                    (22,76,"LB"),(38,78,"CB"),(62,78,"CB"),(78,76,"RB"),
                    (50, 88, "GK"),
                ], {}),
                ("#E05555", [(30,52,"LW"),(50,48,"ST"),(70,52,"RW")], {0:(30,60),1:(50,58),2:(70,60)}),
            ],
        },
        "man marking": {
            "label": "Man Marking",
            "subtitle": "Individual Defense",
            "tags": ["Tracking", "1v1", "Discipline"],
            "zones": [],
            "groups": [
                ("#E05555", [(22,30,"LW"),(50,22,"ST"),(78,30,"RW"),(38,44,"CM")], {}),
                ("#4CAF85", [(22,40,"LB"),(50,32,"CB"),(78,40,"RB"),(38,54,"DM")],
                            {0:(22,30), 1:(50,22), 2:(78,30), 3:(38,44)}),
            ],
        },
        "zonal marking": {
            "label": "Zonal Marking",
            "subtitle": "Area Defense",
            "tags": ["Structure", "Zones", "Collective"],
            "zones": [(22,42,16,22,0.18),(50,38,16,22,0.18),(78,42,16,22,0.18),(36,62,16,18,0.14),(64,62,16,18,0.14)],
            "groups": [
                ("#4CAF85", [(22,42,"LB"),(50,38,"CB"),(78,42,"RB"),(36,62,"LM"),(64,62,"RM")], {}),
                ("#E05555", [(28,34,"LW"),(50,26,"ST"),(72,34,"RW")], {0:(22,42),1:(50,38),2:(78,42)}),
            ],
        },
        "tackle": {
            "label": "Tackle",
            "subtitle": "Ball Challenge",
            "tags": ["Duel", "Physicality", "Defense"],
            "zones": [(50, 44, 18, 12, 0.20)],
            "groups": [
                ("#E05555", [(50, 38, "O")], {0: (50, 46)}),
                ("#4CAF85", [(50, 56, "D")], {0: (50, 44)}),
            ],
        },
        "interception": {
            "label": "Interception",
            "subtitle": "Pass Reading",
            "tags": ["Anticipation", "Reading", "Positioning"],
            "zones": [(52, 46, 14, 10, 0.22)],
            "groups": [
                ("#E05555", [(24, 50, "O1"), (80, 42, "O2")], {0: (52, 46)}),
                ("#4CAF85", [(52, 52, "D")], {0: (52, 46)}),
            ],
        },
        "counter-pressing": {
            "label": "Counter-Pressing",
            "subtitle": "Immediate Recovery",
            "tags": ["Gegenpressing", "Urgency", "Collective"],
            "zones": [(50, 46, 24, 14, 0.28)],
            "groups": [
                ("#E05555", [(50, 46, "O")], {}),
                ("#F5D06E", [
                    (30, 56, "P"), (50, 60, "P"), (70, 56, "P"), (40, 50, "P"),
                ], {0:(40,48), 1:(50,48), 2:(60,48), 3:(46,46)}),
            ],
        },
        "transition": {
            "label": "Transition",
            "subtitle": "Phase Change",
            "tags": ["Speed", "Switching", "Vertical"],
            "zones": [(50, 32, 36, 14, 0.20)],
            "groups": [
                ("#4CAF85", [
                    (22, 62, "LW"), (50, 68, "ST"), (78, 62, "RW"),
                    (38, 74, "CM"), (62, 74, "CM"),
                ], {0:(18,28), 1:(50,22), 2:(82,28), 3:(36,38), 4:(64,38)}),
            ],
        },
        "formation": {
            "label": "Formation",
            "subtitle": "Team Structure",
            "tags": ["4-3-3", "Organisation", "Spacing"],
            "zones": [],
            "groups": [
                ("#4CAF85", [
                    (50, 88, "GK"),
                    (18,72,"LB"),(38,74,"CB"),(62,74,"CB"),(82,72,"RB"),
                    (28,54,"LCM"),(50,50,"CM"),(72,54,"RCM"),
                    (16,28,"LW"),(50,22,"ST"),(84,28,"RW"),
                ], {}),
            ],
        },
        "shape": {
            "label": "Shape",
            "subtitle": "Team Compactness",
            "tags": ["Organisation", "Compact", "Unit"],
            "zones": [(62, 54, 36, 26, 0.16)],
            "groups": [
                ("#4CAF85", [
                    (40,46,"LM"),(56,46,"CM"),(72,46,"RM"),
                    (36,58,"LB"),(52,60,"CB"),(68,60,"CB"),(84,58,"RB"),
                ], {0:(56,46), 1:(72,46), 2:(88,46), 3:(52,60), 4:(68,60), 5:(84,60), 6:(100,58)}),
            ],
        },
        "width": {
            "label": "Width",
            "subtitle": "Pitch Stretching",
            "tags": ["Wide Play", "Space", "Stretch"],
            "zones": [(14, 38, 12, 28, 0.22), (86, 38, 12, 28, 0.22)],
            "groups": [
                ("#F5D06E", [(12, 38, "LW"), (88, 38, "RW")], {}),
                ("#4CAF85", [(50, 42, "CM"), (34, 56, "LB"), (66, 56, "RB")], {0:(50,38)}),
                ("#E05555", [(36,42,"CB"),(50,38,"CB"),(64,42,"CB")], {}),
            ],
        },
        "depth": {
            "label": "Depth",
            "subtitle": "Vertical Spacing",
            "tags": ["Staggering", "Options", "Layers"],
            "zones": [(50, 42, 20, 30, 0.16)],
            "groups": [
                ("#4CAF85", [
                    (50, 24, "ST"),
                    (38, 40, "LCM"), (62, 40, "RCM"),
                    (50, 58, "DM"),
                    (34, 72, "LB"), (66, 72, "RB"),
                ], {3:(50,48), 1:(38,32), 2:(62,32)}),
            ],
        },
        "half-space": {
            "label": "Half-Space",
            "subtitle": "Channel Exploitation",
            "tags": ["Half-Space", "Diagonal", "Danger Zone"],
            "zones": [(30, 36, 12, 32, 0.26), (70, 36, 12, 32, 0.26)],
            "groups": [
                ("#F5D06E", [(30, 50, "10"), (70, 50, "8")], {0:(30,28), 1:(70,28)}),
                ("#4CAF85", [(12,36,"LW"),(50,30,"ST"),(88,36,"RW")], {}),
                ("#E05555", [(22,34,"LB"),(40,32,"CB"),(60,32,"CB"),(78,34,"RB")], {}),
            ],
        },
        "lines": {
            "label": "Lines",
            "subtitle": "Defensive Structure",
            "tags": ["Block", "Compactness", "Defense"],
            "zones": [(50, 38, 46, 8, 0.16), (50, 52, 46, 8, 0.16), (50, 66, 46, 8, 0.16)],
            "groups": [
                ("#E05555", [(28,32,"LW"),(50,26,"ST"),(72,32,"RW")], {}),
                ("#4CAF85", [
                    (22,38,"LM"),(40,38,"CM"),(60,38,"CM"),(78,38,"RM"),
                    (24,52,"LB"),(40,52,"CB"),(60,52,"CB"),(76,52,"RB"),
                ], {}),
                ("#F5D06E", [(50, 60, "DM")], {0:(50,40)}),
            ],
        },
        "tiki-taka": {
            "label": "Tiki-Taka",
            "subtitle": "Short Passing",
            "tags": ["Possession", "Triangles", "Control"],
            "zones": [(50, 42, 36, 28, 0.14)],
            "groups": [
                ("#4CAF85", [
                    (36,30,"10"),(50,22,"9"),(64,30,"8"),
                    (30,46,"LM"),(50,50,"CM"),(70,46,"RM"),
                ], {0:(50,22), 1:(64,30), 2:(50,50), 3:(36,30), 4:(30,46), 5:(64,30)}),
            ],
        },
        "total football": {
            "label": "Total Football",
            "subtitle": "Universal Roles",
            "tags": ["Rotations", "Flexibility", "Ajax Style"],
            "zones": [(50, 44, 40, 30, 0.14)],
            "groups": [
                ("#F5D06E", [
                    (22,30,"LW"),(50,22,"9"),(78,30,"RW"),
                    (30,48,"LCM"),(50,50,"CM"),(70,48,"RCM"),
                    (18,62,"LB"),(50,66,"CB"),(82,62,"RB"),
                ], {0:(30,48), 2:(70,48), 3:(22,30), 5:(78,30), 6:(30,48), 8:(70,48)}),
            ],
        },
        "positional play": {
            "label": "Positional Play",
            "subtitle": "Space Control",
            "tags": ["Triangles", "Diamonds", "Pep Style"],
            "zones": [(50, 38, 40, 32, 0.16)],
            "groups": [
                ("#4CAF85", [
                    (50,20,"9"),
                    (28,30,"LW"),(72,30,"RW"),
                    (36,44,"LCM"),(64,44,"RCM"),
                    (50,50,"DM"),
                    (22,56,"LB"),(78,56,"RB"),
                ], {3:(36,36), 4:(64,36), 5:(50,40)}),
            ],
        },
        "overload": {
            "label": "Overload",
            "subtitle": "Numerical Superiority",
            "tags": ["3v2", "Overload", "Combination"],
            "zones": [(22, 32, 18, 22, 0.28)],
            "groups": [
                ("#E05555", [(30, 36, "D1"), (18, 44, "D2")], {}),
                ("#F5D06E", [(14, 28, "LW"), (22, 22, "ST"), (32, 28, "10")],
                            {0:(14,36), 1:(22,32), 2:(30,28)}),
            ],
        },
        "third man run": {
            "label": "Third Man Run",
            "subtitle": "Combination Play",
            "tags": ["One-Two", "Run", "Timing"],
            "zones": [(62, 26, 16, 14, 0.26)],
            "groups": [
                ("#4CAF85", [(38, 52, "A")], {0: (50, 40)}),
                ("#F5D06E", [(50, 40, "B")], {0: (62, 28)}),
                ("#4CAF85", [(64, 50, "C")], {0: (64, 22)}),
            ],
        },
        "line-breaking pass": {
            "label": "Line-Breaking Pass",
            "subtitle": "Bypassing Defense",
            "tags": ["Penetration", "Vision", "Key Pass"],
            "zones": [(50, 22, 28, 14, 0.28)],
            "groups": [
                ("#E05555", [
                    (22,36,"D"),(40,36,"D"),(60,36,"D"),(78,36,"D"),
                    (30,50,"D"),(50,50,"D"),(70,50,"D"),
                ], {}),
                ("#4CAF85", [(50, 62, "8")], {0: (50, 22)}),
                ("#F5D06E", [(50, 28, "9")], {0: (50, 16)}),
            ],
        },
    }

    t = GLOS_SCENARIOS.get(term)
    if not t:
        return ('<div style="background:#0F1C0F;border-radius:20px;height:260px;'
                'display:flex;align-items:center;justify-content:center;">'
                '<span style="color:rgba(255,255,255,.4);font-size:.78rem;font-weight:700;'
                'letter-spacing:.12em;text-transform:uppercase;">Visualization — coming soon</span></div>')

    slug = re.sub(r'[^a-z0-9]', '_', term)
    cx, cy = PAD + PW//2, PAD + PH//2
    stripe_h = 40

    # ── Defs ──
    defs = (
        f'<defs>'
        f'<pattern id="g_{slug}" x="0" y="0" width="{PW}" height="{stripe_h}" patternUnits="userSpaceOnUse">'
        f'<rect x="0" y="0" width="{PW}" height="{stripe_h//2}" fill="rgba(0,0,0,0.045)"/>'
        f'</pattern>'
        f'<radialGradient id="vig_{slug}" cx="50%" cy="50%" r="70%">'
        f'<stop offset="0%" stop-color="transparent"/>'
        f'<stop offset="100%" stop-color="rgba(0,0,0,0.25)"/>'
        f'</radialGradient>'
        f'<filter id="pshadow_{slug}" x="-30%" y="-30%" width="160%" height="160%">'
        f'<feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="rgba(0,0,0,0.5)"/>'
        f'</filter>'
        f'<filter id="pglow_{slug}" x="-40%" y="-40%" width="180%" height="180%">'
        f'<feGaussianBlur stdDeviation="3" result="blur"/>'
        f'<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f'</filter>'
        f'<marker id="arr_{slug}" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 7 3.5, 0 7" fill="rgba(255,255,255,0.9)"/>'
        f'</marker>'
    )
    for gi, (color, _, _) in enumerate(t["groups"]):
        for zi, (zcx, zcy, zrx, zry, op) in enumerate(t.get("zones", [])):
            defs += (
                f'<radialGradient id="hz_{slug}_{gi}_{zi}" cx="50%" cy="50%" r="50%">'
                f'<stop offset="0%" stop-color="{color}" stop-opacity="{min(op*2.2, 0.55):.2f}"/>'
                f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
                f'</radialGradient>'
            )
    defs += '</defs>'

    # ── Background ──
    pitch_bg = (
        f'<rect x="0" y="0" width="{SW}" height="{SH}" fill="url(#g_{slug})"/>'
        f'<rect x="0" y="0" width="{SW}" height="{SH}" fill="url(#vig_{slug})"/>'
    )

    # ── Markings ──
    markings = (
        f'<rect x="{PAD}" y="{PAD}" width="{PW}" height="{PH}" fill="none" stroke="rgba(255,255,255,.55)" stroke-width="1.5"/>'
        f'<line x1="{PAD}" y1="{cy}" x2="{PAD+PW}" y2="{cy}" stroke="rgba(255,255,255,.45)" stroke-width="1"/>'
        f'<circle cx="{cx}" cy="{cy}" r="32" fill="none" stroke="rgba(255,255,255,.4)" stroke-width="1"/>'
        f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="rgba(255,255,255,.6)"/>'
        f'<rect x="{PAD+58}" y="{PAD}" width="136" height="66" fill="rgba(255,255,255,.03)" stroke="rgba(255,255,255,.38)" stroke-width="1"/>'
        f'<rect x="{PAD+96}" y="{PAD}" width="60" height="23" fill="rgba(255,255,255,.02)" stroke="rgba(255,255,255,.28)" stroke-width="1"/>'
        f'<circle cx="{cx}" cy="{PAD+52}" r="2" fill="rgba(255,255,255,.45)"/>'
        f'<path d="M {PAD+80} {PAD+66} A 32 32 0 0 0 {PAD+172} {PAD+66}" fill="none" stroke="rgba(255,255,255,.3)" stroke-width="1"/>'
        f'<rect x="{PAD+58}" y="{PAD+PH-66}" width="136" height="66" fill="rgba(255,255,255,.03)" stroke="rgba(255,255,255,.38)" stroke-width="1"/>'
        f'<rect x="{PAD+96}" y="{PAD+PH-23}" width="60" height="23" fill="rgba(255,255,255,.02)" stroke="rgba(255,255,255,.28)" stroke-width="1"/>'
        f'<circle cx="{cx}" cy="{PAD+PH-52}" r="2" fill="rgba(255,255,255,.45)"/>'
        f'<path d="M {PAD+80} {PAD+PH-66} A 32 32 0 0 1 {PAD+172} {PAD+PH-66}" fill="none" stroke="rgba(255,255,255,.3)" stroke-width="1"/>'
        f'<rect x="{PAD+102}" y="{PAD-8}" width="48" height="8" fill="rgba(255,255,255,.12)" stroke="rgba(255,255,255,.45)" stroke-width="1"/>'
        f'<rect x="{PAD+102}" y="{PAD+PH}" width="48" height="8" fill="rgba(255,255,255,.12)" stroke="rgba(255,255,255,.45)" stroke-width="1"/>'
        f'<path d="M {PAD} {PAD+9} A 9 9 0 0 1 {PAD+9} {PAD}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
        f'<path d="M {PAD+PW-9} {PAD} A 9 9 0 0 1 {PAD+PW} {PAD+9}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
        f'<path d="M {PAD} {PAD+PH-9} A 9 9 0 0 1 {PAD+9} {PAD+PH}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
        f'<path d="M {PAD+PW-9} {PAD+PH} A 9 9 0 0 1 {PAD+PW} {PAD+PH-9}" fill="none" stroke="rgba(255,255,255,.32)" stroke-width="1"/>'
    )

    # ── Heat zones ──
    zones_svg = ""
    for gi, (color, _, _) in enumerate(t["groups"]):
        for zi, (zcx, zcy, zrx, zry, op) in enumerate(t.get("zones", [])):
            zones_svg += (
                f'<ellipse cx="{sx(zcx):.1f}" cy="{sy(zcy):.1f}" '
                f'rx="{zrx/100*PW:.1f}" ry="{zry/100*PH:.1f}" '
                f'fill="url(#hz_{slug}_{gi}_{zi})" class="hz_{slug}"/>\n'
            )

    # ── Players + arrows ──
    css_lines = [
        f"@keyframes hz_p_{slug}{{0%,100%{{opacity:.75}}50%{{opacity:1}}}}",
        f".hz_{slug}{{animation:hz_p_{slug} 4.5s ease-in-out infinite;}}",
    ]
    players_svg = ""
    arrows_svg  = ""
    global_idx  = 0
    global_seq  = 0

    for gi, (color, players, moves_dict) in enumerate(t["groups"]):
        for pi, (ppx, ppy, abbr) in enumerate(players):
            x0, y0 = sx(ppx), sy(ppy)
            cls    = f"pl_{slug}_{gi}_{pi}"
            delay  = f"{global_idx * 0.22:.2f}s"
            is_mover = pi in moves_dict
            filt = f'filter="url(#pglow_{slug})"' if is_mover else f'filter="url(#pshadow_{slug})"'

            if is_mover:
                tx, ty   = moves_dict[pi]
                ddx, ddy = sx(tx) - x0, sy(ty) - y0
                an = f"pm_{slug}_{gi}_{pi}"
                css_lines.append(
                    f"@keyframes {an}{{"
                    f"0%,18%{{transform:translate(0px,0px)}}"
                    f"38%,62%{{transform:translate({ddx:.1f}px,{ddy:.1f}px)}}"
                    f"82%,100%{{transform:translate(0px,0px)}}}}"
                    f".{cls}{{animation:{an} 9s ease-in-out infinite;animation-delay:{delay};}}"
                )
                L      = math.hypot(ddx, ddy) or 1
                perp_x, perp_y = -ddy/L, ddx/L
                offset = min(L * 0.22, 18)
                cpx    = (x0 + sx(tx))/2 + perp_x * offset
                cpy    = (y0 + sy(ty))/2 + perp_y * offset
                path_d = f"M {x0:.1f} {y0:.1f} Q {cpx:.1f} {cpy:.1f} {sx(tx):.1f} {sy(ty):.1f}"
                path_len = int(L * 1.18) + 10
                d_s    = global_seq * 1.8
                arr_an  = f"aw_{slug}_{gi}_{pi}"
                arr_cls = f"ac_{slug}_{gi}_{pi}"
                css_lines.append(
                    f"@keyframes {arr_an}{{"
                    f"0%,{int(d_s/9*100)}%{{stroke-dashoffset:{path_len};opacity:0}}"
                    f"{int((d_s+0.5)/9*100)}%,{int((d_s+2.2)/9*100)}%{{stroke-dashoffset:0;opacity:.92}}"
                    f"{int((d_s+2.8)/9*100)}%,100%{{stroke-dashoffset:-{path_len};opacity:0}}}}"
                    f".{arr_cls}{{stroke-dasharray:{path_len};stroke-dashoffset:{path_len};"
                    f"animation:{arr_an} 9s ease-in-out infinite;}}"
                )
                arrows_svg += (
                    f'<path class="{arr_cls}" d="{path_d}" fill="none" stroke="{color}" '
                    f'stroke-width="2.2" marker-end="url(#arr_{slug})"/>\n'
                )
                global_seq += 1
                ring = f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="15" fill="none" stroke="{color}" stroke-width="1" opacity="0.4" stroke-dasharray="3 3"/>'
            else:
                an = f"ps_{slug}_{gi}_{pi}"
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
            global_idx += 1

    pills = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .72rem;'
        f'border-radius:100px;background:rgba(255,255,255,.08);color:rgba(255,255,255,.75);'
        f'font-size:.6rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;'
        f'border:1px solid rgba(255,255,255,.12);margin:.15rem .15rem 0 0">'
        f'<span style="width:5px;height:5px;border-radius:50%;background:#F5D06E;display:inline-block;flex-shrink:0"></span>'
        f'{s}</span>'
        for s in t.get("tags", [])
    )

    css_block = "<style>" + "".join(css_lines) + "</style>"
    svg = (
        f'<svg viewBox="0 0 {SW} {SH}" xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;width:100%;background:#1e5c1e;">'
        f'{defs}{pitch_bg}{markings}{zones_svg}{arrows_svg}{players_svg}'
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
        f'color:#F5D06E;margin-bottom:.18rem;">Tactical Animation</div>'
        f'<div style="font-size:.95rem;font-weight:900;color:rgba(255,255,255,.92);letter-spacing:-.02em;">{t["label"]}</div>'
        f'</div>'
        f'<span style="background:#F5D06E;color:#1A1A2E;font-size:.72rem;font-weight:900;'
        f'padding:.3rem .9rem;border-radius:100px;letter-spacing:.06em;">{t["subtitle"]}</span>'
        f'</div>'
        f'<div style="position:relative;">{svg}</div>'
        f'<div style="padding:.6rem 1rem .8rem;border-top:1px solid rgba(255,255,255,.06);">'
        f'<div style="font-size:.56rem;font-weight:800;letter-spacing:.16em;color:rgba(255,255,255,.3);'
        f'text-transform:uppercase;margin-bottom:.3rem;">Key concepts</div>'
        f'{pills}</div>'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE DÉFINITION
# ══════════════════════════════════════════════════════════════════════════════
def page_definition():
    term = st.session_state.active_term
    term_data = TACTICAL_TERMS.get(term, {})
    definition      = term_data.get("definition",        "Definition coming soon.") if isinstance(term_data, dict) else term_data
    simple          = term_data.get("simple_explanation", "") if isinstance(term_data, dict) else ""
    example         = term_data.get("example",            "") if isinstance(term_data, dict) else ""
    animation       = term_data.get("animation_idea",     "") if isinstance(term_data, dict) else ""

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
    st.markdown(build_pitch_animation(term), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE GLOSSAIRE
# ══════════════════════════════════════════════════════════════════════════════
def page_glossaire():
    st.markdown('<div class="sec-label">Vocabulary</div><div class="sec-title">Tactical Glossary</div>', unsafe_allow_html=True)
    for i, (term, term_data) in enumerate(TACTICAL_TERMS.items()):
        definition = term_data.get("definition", "") if isinstance(term_data, dict) else term_data
        simple     = term_data.get("simple_explanation", "") if isinstance(term_data, dict) else ""
        example    = term_data.get("example", "") if isinstance(term_data, dict) else ""
        icon = GLOS_ICONS[i % len(GLOS_ICONS)]
        bg   = GLOS_COLORS[i % len(GLOS_COLORS)]

        simple_html  = f'<div class="def-simple"><span class="def-tag def-tag-green">💡 In simple terms</span><p>{simple}</p></div>' if simple else ""
        example_html = f'<div class="def-example"><span class="def-tag def-tag-yellow">⚽ Example</span><p>{example}</p></div>' if example else ""
        anim_html    = build_pitch_animation(term)

        st.markdown(
            f'<details class="glos-acc">'
            f'<summary>'
            f'<div class="glos-card">'
            f'<div class="glos-card-header">'
            f'<div class="glos-card-icon" style="background:{bg}">{icon}</div>'
            f'<span class="glos-card-term">{term.capitalize()}</span>'
            f'<span class="pill pill-yellow pill-tac" style="margin-left:auto"></span>'
            f'</div>'
            f'<div class="glos-card-body">{definition}</div>'
            f'</div>'
            f'</summary>'
            f'<div class="glos-acc-body">'
            f'{simple_html}{example_html}{anim_html}'
            f'</div>'
            f'</details>',
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

    # ── Team selectors ──
    st.markdown('<div class="match-card"><span class="match-card-label">Choose the match to analyse</span><span class="match-vs-badge">VS</span></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        idx_a = ALL_TEAMS.index(team_a) if team_a in ALL_TEAMS else 0
        st.session_state.team_a = st.selectbox("Team A", ALL_TEAMS, index=idx_a, key="sel_a")
    with col_b:
        remaining = [t for t in ALL_TEAMS if t != st.session_state.team_a]
        if st.session_state.team_b not in remaining:
            st.session_state.team_b = remaining[0] if remaining else ""
        st.session_state.team_b = st.selectbox("Team B", remaining, index=remaining.index(st.session_state.team_b) if st.session_state.team_b in remaining else 0, key="sel_b")

    team_a, team_b = st.session_state.team_a, st.session_state.team_b
    da, db = standings.get(team_a, {}), standings.get(team_b, {})
    crest_a, crest_b = da.get("crest",""), db.get("crest","")
    img_a = get_crest_img(team_a, 36)
    img_b = get_crest_img(team_b, 36)

    # ── VS Banner ──
    st.markdown(
        f'<div class="vs-banner">'
        f'<div class="vs-team"><span class="vs-team-label vs-team-label-a">Team A</span>'
        f'<div class="vs-team-crest">{img_a}<span class="vs-team-name">{team_a}</span></div>'
        f'<span class="vs-team-stat">#{da.get("position","—")} · {da.get("points","—")} pts</span></div>'
        f'<div class="vs-sep"><span class="vs-sep-dot"></span><span class="vs-sep-text">VS</span><span class="vs-sep-dot"></span></div>'
        f'<div class="vs-team"><span class="vs-team-label vs-team-label-b">Team B</span>'
        f'<div class="vs-team-crest">{img_b}<span class="vs-team-name">{team_b}</span></div>'
        f'<span class="vs-team-stat">#{db.get("position","—")} · {db.get("points","—")} pts</span></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Styles de jeu ──
    st.markdown('<div class="sec-label">AI Analysis</div><div class="sec-title">Playing Style</div>', unsafe_allow_html=True)

    # Fetch enriched data for both teams
    form_a        = tuple(fetch_team_form(da.get("id")))
    form_b        = tuple(fetch_team_form(db.get("id")))
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
        """Split into summary (1st paragraph) + collapsible detail (rest)."""
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


# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.page == "definition":
    page_definition()
else:
    render_header()
    render_nav()
    if st.session_state.page == "classement":
        page_classement()
    elif st.session_state.page == "glossaire":
        page_glossaire()
    else:
        page_main()
