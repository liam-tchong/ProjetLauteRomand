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

# Ligue 1 team IDs on API-Football (league 61, season 2025)
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

API_TO_DISPLAY = {
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
def fetch_standings():
    try:
        r = requests.get(
            "https://api.football-data.org/v4/competitions/FL1/standings",
            headers=HEADERS, timeout=10
        )
        r.raise_for_status()
        table = r.json()["standings"][0]["table"]
        result = {}
        for row in table:
            t = row["team"]
            name = API_TO_DISPLAY.get(t["name"], t["name"])
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
def fetch_previous_standings():
    """Final Ligue 1 standings for the 2024 season (previous season)."""
    try:
        r = requests.get(
            "https://api.football-data.org/v4/competitions/FL1/standings",
            headers=HEADERS,
            params={"season": 2024},
            timeout=10
        )
        r.raise_for_status()
        table = r.json()["standings"][0]["table"]
        return {
            API_TO_DISPLAY.get(row["team"]["name"], row["team"]["name"]): row["position"]
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
def fetch_competition_scorers():
    """Ligue 1 top scorers — returns a dict {team_display_name: [(player, goals), ...]}."""
    try:
        r = requests.get(
            "https://api.football-data.org/v4/competitions/FL1/scorers",
            headers=HEADERS,
            params={"limit": 50},
            timeout=10
        )
        r.raise_for_status()
        scorers_by_team = {}
        for s in r.json().get("scorers", []):
            raw_team = s["team"]["name"]
            team = API_TO_DISPLAY.get(raw_team, raw_team)
            player = s["player"]["name"]
            goals  = s.get("goals", 0)
            assists = s.get("assists") or 0
            scorers_by_team.setdefault(team, []).append((player, goals, assists))
        return scorers_by_team
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_api_football_stats(team_name):
    """Advanced stats from API-Football: formation, passes, shots, clean sheets."""
    if not API_FOOTBALL_KEY:
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

    prev_str = f"{prev_position}" if prev_position else "N/A"
    pos_delta = ""
    if prev_position and position:
        diff = prev_position - position
        if diff > 0:   pos_delta = f" (↑ +{diff} vs last season)"
        elif diff < 0: pos_delta = f" (↓ {diff} vs last season)"
        else:           pos_delta = " (= same position as last year)"

    stats_block = f"""Team: {team_name}
Current ranking: {position}th place{pos_delta} — {pts} points in {played} matches
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
        stats_block += f"\nShots per match: {extra_shots_pg} ({extra_shots_on_pg} on target)"
    if extra_clean_sheets is not None:
        stats_block += f"\nClean sheets: {extra_clean_sheets}"
    if extra_failed_to_score is not None:
        stats_block += f"\nMatches without scoring: {extra_failed_to_score}"
    if extra_top_slot:
        stats_block += f"\nTime slot where the team scores most: {extra_top_slot} min"
    if extra_gf_avg:
        stats_block += f"\nAverage goals for / against per match: {extra_gf_avg} / {extra_ga_avg}"

    terms = ", ".join(TACTICAL_TERMS.keys())

    prompt = f"""You are a passionate football analyst explaining the game to fans of all levels — from beginners to seasoned supporters.
Using the statistics below and your knowledge of this club, write a structured analysis in exactly 3 paragraphs, in English.

{stats_block}

Required structure:
§1 — OVERALL STRATEGY (2 sentences max): formation, defensive and offensive philosophy, tactical identity this season.
§2 — TECHNICAL DETAILS (3 sentences max): key players, how they press/defend/attack, one or two stats.
§3 — FUN STRATEGIC FACT (1-2 sentences max): something memorable — a historical style, legendary coach, or tactical reputation.

Rules:
- Simple, vivid language — no unexplained jargon
- STRICT limit: 3 paragraphs, 7 sentences total across the whole analysis
- You MUST use at least 6 terms from this list: {terms}
- Every time you use one of those terms, wrap it exactly like this: <b>term</b> (e.g. <b>pressing</b>, <b>high press</b>, <b>counter-attack</b>)
- Separate the 3 paragraphs with a blank line (\\n\\n)
- No titles, no bullet points, no numbering
- If unsure about a specific fact, stay vague rather than inventing

Reply with the 3 paragraphs only, nothing else."""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=420,
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
    prompt = f"""You are a concise football analyst. Given these two Ligue 1 teams and their season stats, write exactly 2 lines — one per team — describing each team's single biggest tactical challenge in this specific matchup.

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
  --bg:#FAF6EE; --beige:#EDE4D0;
  --green:#7CC99A; --green-lt:#D4F0E0; --green-dk:#2E7D52;
  --red:#F2827F; --red-lt:#FCE0DF; --red-dk:#C0302E;
  --yellow:#F5D06E; --yellow-lt:#FEF3CF; --yellow-dk:#8A6800;
  --dark:#2A2018; --mid:#6B5A45; --white:#FFFFFF;
  --radius:22px; --shadow:0 4px 20px rgba(42,32,24,0.08); --shadow-lg:0 8px 32px rgba(42,32,24,0.13);
}
#MainMenu,header,footer{visibility:hidden;}
.block-container{padding-top:0!important;padding-bottom:3rem;max-width:1200px;margin:0 auto;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background-color:var(--bg)!important;font-family:'Nunito',sans-serif!important;}
[data-testid="stVerticalBlock"]{background:transparent;}
[data-testid="stMain"]::before{content:'';display:block;height:4px;background:linear-gradient(90deg,var(--green) 0%,var(--yellow) 50%,var(--red) 100%);border-radius:0 0 8px 8px;margin-bottom:1.5rem;}

/* Live dot */
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.75);}}
.live-badge{display:inline-flex;align-items:center;gap:.45rem;background:rgba(124,201,154,.12);border:1.5px solid rgba(124,201,154,.35);border-radius:100px;padding:.3rem .85rem;font-size:.68rem;font-weight:800;color:var(--green);letter-spacing:.1em;text-transform:uppercase;}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);flex-shrink:0;animation:pulse-dot 1.8s ease-in-out infinite;}

/* Header */
.app-header{background:var(--dark);background-image:radial-gradient(ellipse at 80% 50%,rgba(124,201,154,.07) 0%,transparent 60%);border-radius:var(--radius);padding:1.8rem 2.5rem;margin-bottom:.8rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;}
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

# ── Load data ──────────────────────────────────────────────────────────────────
standings = fetch_standings()
ALL_TEAMS = sorted(standings.keys(), key=lambda n: standings[n]["position"]) if standings else list(TEAM_STYLES.keys())

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
    st.markdown("""
<div class="app-header">
<div><div class="app-title">The Football <span>Classroom</span></div><div class="app-sub">Tactical analysis · Ligue 1</div></div>
<div class="app-header-right">
<div class="live-badge"><span class="live-dot"></span>Live</div>
<div class="app-badges"><span class="app-badge">Ligue 1</span><span class="app-badge">2025/26</span><span class="app-badge">18 teams</span></div>
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
    terrain_label = animation if animation else "Visualization — coming soon"
    st.markdown(f"""<div class="terrain-wrap"><div class="terrain-border"></div><div class="terrain-center"></div><div class="terrain-center-dot"></div><div class="terrain-box-top"></div><div class="terrain-box-bot"></div><div class="terrain-small-top"></div><div class="terrain-small-bot"></div><span class="terrain-label">{terrain_label}</span></div>""", unsafe_allow_html=True)
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
    team_a = st.session_state.team_a
    team_b = st.session_state.team_b
    da = standings.get(team_a, {})

    st.markdown('<div class="sec-label">Ligue 1</div><div class="sec-title">Standings 2025/26</div>', unsafe_allow_html=True)

    # Build rows — NO indentation to avoid markdown code-block interpretation
    hdr = '<div class="standings-hdr-row"><span class="standings-pos">#</span><span style="width:20px"></span><span style="flex:1">Team</span><span class="standings-stat">P</span><span class="standings-stat">W</span><span class="standings-stat">D</span><span class="standings-stat">L</span><span class="standings-gd">GD</span><span class="standings-pts">Pts</span></div>'

    rows = ""
    for name in ALL_TEAMS:
        d = standings[name]
        gd = d["goal_diff"]
        gd_str = f"{gd:+}" if gd != 0 else "0"
        cls = "highlighted-a" if name == team_a else ("highlighted-b" if name == team_b else "")
        img = f'<img class="standings-crest" src="{d["crest"]}">' if d.get("crest") else '<span style="width:20px"></span>'
        rows += f'<div class="standings-row {cls}"><span class="standings-pos">{d["position"]}</span>{img}<span class="standings-name">{name}</span><span class="standings-stat">{d["played"]}</span><span class="standings-stat">{d["won"]}</span><span class="standings-stat">{d["draw"]}</span><span class="standings-stat">{d["lost"]}</span><span class="standings-gd {gd_class(gd)}">{gd_str}</span><span class="standings-pts">{d["points"]}</span></div>'

    st.markdown(
        f'<div class="standings-card">'
        f'<div class="standings-header"><span class="standings-header-title">Full standings — Matchday {da.get("played","?")}</span></div>'
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

    # ── Sélecteurs ──
    st.markdown('<div class="match-card"><span class="match-card-label">Choose the match to analyse</span><span class="match-vs-badge">VS</span></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        idx_a = ALL_TEAMS.index(team_a) if team_a in ALL_TEAMS else 0
        st.session_state.team_a = st.selectbox("Team A", ALL_TEAMS, index=idx_a, key="sel_a")
    with col_b:
        remaining = [t for t in ALL_TEAMS if t != st.session_state.team_a]
        if st.session_state.team_b not in remaining:
            st.session_state.team_b = remaining[0]
        st.session_state.team_b = st.selectbox("Team B", remaining, index=remaining.index(st.session_state.team_b), key="sel_b")

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
    extra_a       = fetch_api_football_stats(team_a)
    extra_b       = fetch_api_football_stats(team_b)
    all_scorers   = fetch_competition_scorers()
    scorers_a     = tuple(all_scorers.get(team_a, [])[:3])
    scorers_b     = tuple(all_scorers.get(team_b, [])[:3])
    prev_standings = fetch_previous_standings()
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
        """Converts line breaks to <br> for HTML display, then linkifies glossary terms."""
        html = raw.replace("\n\n", "<br><br>").replace("\n", " ")
        return linkify_terms(html, source_page="main", ta=team_a, tb=team_b)

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
    st.markdown('<div class="terrain-wrap"><div class="terrain-border"></div><div class="terrain-center"></div><div class="terrain-center-dot"></div><div class="terrain-box-top"></div><div class="terrain-box-bot"></div><div class="terrain-small-top"></div><div class="terrain-small-bot"></div><span class="terrain-label">Tactical pitch — coming soon</span></div>', unsafe_allow_html=True)

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
