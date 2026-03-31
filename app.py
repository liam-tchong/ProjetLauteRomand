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

# IDs équipes Ligue 1 sur API-Football (league 61, saison 2025)
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
    """Classement final Ligue 1 saison 2024 (saison précédente)."""
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
    """Derniers 5 matchs joués pour un team_id football-data.org."""
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
    """Top buteurs de Ligue 1 — retourne un dict {team_display_name: [(joueur, buts), ...]}."""
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
    """Stats avancées depuis API-Football : formation, passes, tirs, clean sheets."""
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

        # Tranche horaire où l'équipe marque le plus
        minutes  = goals.get("for", {}).get("minute", {})
        top_slot = max(minutes, key=lambda k: (minutes[k].get("total") or 0)) if minutes else None

        # Passes & tirs
        passes_pct     = passes.get("percentage")
        shots_total    = shots.get("total",  {}).get("total")
        shots_on       = shots.get("on",     {}).get("total")
        played_total   = fixtures.get("played", {}).get("total") or 1
        shots_pg       = round(shots_total / played_total, 1) if shots_total else None
        shots_on_pg    = round(shots_on    / played_total, 1) if shots_on    else None

        # Clean sheets & matchs sans marquer
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
    """Génère une analyse tactique en 3 paragraphes via Claude."""
    if not ANTHROPIC_API_KEY:
        return TEAM_STYLES.get(team_name, DEFAULT_STYLE)

    form_str = " → ".join(form_tuple) if form_tuple else "N/A"
    avg_gf   = round(goals_for      / max(played, 1), 2)
    avg_ga   = round(goals_against  / max(played, 1), 2)

    prev_str = f"{prev_position}e" if prev_position else "N/A"
    pos_delta = ""
    if prev_position and position:
        diff = prev_position - position
        if diff > 0:   pos_delta = f" (↑ +{diff} vs saison dernière)"
        elif diff < 0: pos_delta = f" (↓ {diff} vs saison dernière)"
        else:           pos_delta = " (= même position que l'an dernier)"

    stats_block = f"""Équipe : {team_name}
Classement actuel : {position}e place{pos_delta} — {pts} points en {played} matchs
Classement saison précédente (2024/25) : {prev_str}
Bilan : {won}V / {draw}N / {lost}D
Buts marqués : {goals_for} ({avg_gf}/match) | Buts encaissés : {goals_against} ({avg_ga}/match)
Différence de buts : {goal_diff:+}
Forme récente (5 derniers matchs) : {form_str}"""

    if key_scorers_tuple:
        scorers_str = ", ".join(f"{n} ({g} buts{', '+str(a)+' passes dét.' if a else ''})"
                                for n, g, a in key_scorers_tuple)
        stats_block += f"\nJoueurs clés (buteurs) : {scorers_str}"
    if extra_formation:
        stats_block += f"\nFormation principale : {extra_formation}"
    if extra_wins_home is not None:
        stats_block += f"\nVictoires domicile / extérieur : {extra_wins_home} / {extra_wins_away}"
    if extra_passes_pct:
        stats_block += f"\nPrécision des passes : {extra_passes_pct}"
    if extra_shots_pg:
        stats_block += f"\nTirs par match : {extra_shots_pg} (dont {extra_shots_on_pg} cadrés)"
    if extra_clean_sheets is not None:
        stats_block += f"\nClean sheets (matchs sans encaisser) : {extra_clean_sheets}"
    if extra_failed_to_score is not None:
        stats_block += f"\nMatchs sans marquer : {extra_failed_to_score}"
    if extra_top_slot:
        stats_block += f"\nTranche de jeu où l'équipe marque le plus : {extra_top_slot} min"
    if extra_gf_avg:
        stats_block += f"\nMoyenne buts pour / contre par match : {extra_gf_avg} / {extra_ga_avg}"

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
            max_tokens=320,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()
    except Exception:
        return TEAM_STYLES.get(team_name, DEFAULT_STYLE)


# ── Data ─────────────────────────────────────────────────────────────────────
TACTICAL_TERMS = {
    "pressing": {
        "definition":        "Technique collective consistant à exercer une pression immédiate sur le porteur du ballon dès la perte de balle, afin de récupérer rapidement la possession.",
        "simple_explanation":"Au lieu d'attendre, les joueurs courent vers l'adversaire pour lui forcer des erreurs.",
        "example":           "Liverpool presse si haut que les gardiens adverses font régulièrement des erreurs sous pression.",
        "animation_idea":    "Plusieurs joueurs convergeant rapidement vers le porteur du ballon.",
    },
    "pivot": {
        "definition":        "Joueur central servant de point d'appui dans le jeu offensif, capable de recevoir dos au but, protéger le ballon et redistribuer.",
        "simple_explanation":"Un joueur costaud qui reçoit le ballon dos au but et le redistribue à ses coéquipiers.",
        "example":           "Ibrahimovic était un pivot parfait : il contrôlait, protégeait et relançait le jeu.",
        "animation_idea":    "Joueur dos au but recevant le ballon, puis le déviant vers des coéquipiers qui arrivent.",
    },
    "faux neuf": {
        "definition":        "Attaquant axial qui décroche vers le milieu plutôt que de rester en pointe, créant de la confusion dans la défense adverse.",
        "simple_explanation":"Un attaquant qui s'éloigne du but pour perturber les défenseurs qui ne savent plus s'ils doivent le suivre.",
        "example":           "Messi au Barça jouait faux neuf sous Guardiola, laissant les défenseurs centraux désorientés.",
        "animation_idea":    "L'attaquant décroche vers le milieu, laissant un espace dans lequel les milieux s'engouffrent.",
    },
    "contre-attaque": {
        "definition":        "Transition offensive rapide dès la récupération du ballon, exploitant le déséquilibre défensif adverse avant réorganisation.",
        "simple_explanation":"Dès que tu récupères le ballon, attaque vite avant que les adversaires ne se replacent.",
        "example":           "Le PSG récupère le ballon au milieu et trois joueurs sprintent pour marquer en quelques secondes.",
        "animation_idea":    "Flèches montrant un mouvement rapide de joueurs depuis la défense vers le but adverse.",
    },
    "bloc bas": {
        "definition":        "Organisation défensive dans laquelle l'équipe se replie profondément dans sa moitié de terrain pour réduire les espaces.",
        "simple_explanation":"Toute l'équipe se place près de son propre but pour ne laisser aucun espace à l'adversaire.",
        "example":           "L'Atletico Madrid défend avec 8 ou 9 joueurs derrière le ballon dans un bloc bas organisé.",
        "animation_idea":    "Tous les joueurs de champ serrés dans la moitié défensive formant un mur compact.",
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
    "Paris Saint-Germain": "Le PSG pratique un <b>high press</b> intense pour récupérer le ballon dans la moitié adverse, soutenu par un <b>counter-pressing</b> immédiat à chaque perte. En phase de <b>build-up play</b>, le jeu repose sur le <b>positional play</b> avec des milieux qui s'infiltrent dans le <b>half-space</b> pour créer des décalages. Un <b>faux neuf</b> décroche entre les <b>lines</b> pour déclencher des <b>line-breaking pass</b>es vers les attaquants lancés en profondeur. Les ailiers exploitent la <b>width</b> du terrain pour étirer la défense, et en <b>transition</b> offensive, la vitesse collective est l'arme principale.",

    "Olympique de Marseille": "L'OM impose un <b>pressing</b> collectif agressif et un <b>counter-pressing</b> immédiat dès la perte de balle. La <b>shape</b> défensive alterne entre <b>man marking</b> ciblé sur les relanceurs et <b>zonal marking</b> sur les couloirs latéraux. La <b>transition</b> offensive est l'arme principale : le ballon remonte vite vers les attaquants en <b>contre-attaque</b>. Un <b>switch of play</b> rapide libère un ailier côté faible pour un <b>cross</b> dans le <b>final third</b>.",

    "AS Monaco": "Monaco construit avec un <b>build-up play</b> patient depuis l'arrière, exploitant le <b>depth</b> du terrain pour avancer par étapes. Les latéraux réalisent des <b>overlap</b>s constants pour créer des surnombres sur les ailes, tandis que les milieux proposent des <b>underlap</b>s dans le <b>half-space</b>. Des <b>third man run</b>s bien coordonnés désorganisent la défense après des échanges rapides, et un <b>through ball</b> précis brise les <b>lines</b> pour libérer les attaquants. En cas de perte, un <b>counter-pressing</b> immédiat tente de récupérer la possession avant la réorganisation adverse.",

    "LOSC Lille": "Lille est reconnu pour son <b>high press</b> collectif et son <b>counter-pressing</b> immédiat — chaque perte déclenche une réaction groupée. La <b>formation</b> en 4-4-2 impose une <b>shape</b> très serrée qui compresse les espaces entre les <b>lines</b> adverses. Des <b>line-breaking pass</b>es rapides libèrent les attaquants en <b>transition</b> après récupération haute. Un <b>overload</b> côté balle crée le surnombre, avant un <b>switch of play</b> vers le côté libéré pour atteindre le <b>final third</b>.",

    "Olympique Lyonnais": "L'OL adopte un style proche du <b>tiki-taka</b>, avec un <b>positional play</b> rigoureux et un <b>build-up play</b> soigné depuis les défenseurs. Les milieux se glissent dans les <b>half-space</b>s pour recevoir entre les <b>lines</b>, exploitant à la fois le <b>depth</b> et la <b>width</b> du terrain. Un <b>overload</b> côté balle est suivi d'un <b>switch of play</b> rapide pour exploiter l'espace libéré. En phase défensive, un <b>zonal marking</b> bien organisé et un <b>low block</b> discipliné protègent les espaces, avec une <b>formation</b> 4-3-3 adaptable.",

    "RC Lens": "Lens se distingue par un <b>pressing</b> collectif intense et une <b>shape</b> très organisée en <b>formation</b> 3-4-3. Les pistons offrent une grande <b>width</b> et combinent <b>overlap</b>s et <b>cross</b>es vers le <b>pivot</b> dans la surface. En récupération, l'équipe déclenche immédiatement une <b>counter-attack</b> verticale. Des <b>tackle</b>s bien placés et des <b>interception</b>s dans l'entrejeu alimentent la <b>transition</b> offensive, et des <b>third man run</b>s permettent de trouver le joueur libre dans les espaces.",

    "OGC Nice": "Nice pratique un <b>positional play</b> structuré avec un <b>build-up play</b> méthodique depuis les défenseurs centraux. La <b>shape</b> défensive repose sur un <b>zonal marking</b> strict et un <b>bloc bas</b> organisé quand l'adversaire possède. Des <b>line-breaking pass</b>es cherchent à pénétrer dans le <b>final third</b> en exploitant le <b>depth</b> et les <b>half-space</b>s. Un <b>switch of play</b> rapide après une <b>interception</b> libère les ailiers côté faible pour un <b>cross</b> dans la surface.",

    "Stade Rennais": "Rennes adopte une philosophie proche du <b>total football</b>, avec des rotations permanentes entre les lignes. Le <b>positional play</b> impose une grande <b>depth</b> et de la <b>width</b> pour occuper tout le terrain simultanément. Des <b>third man run</b>s fréquents désorganisent les défenses adverses, et des <b>overlap</b>s des latéraux créent les surnombres sur les côtés. En <b>transition</b>, le <b>counter-pressing</b> empêche l'adversaire de relancer proprement, avant de chercher un <b>through ball</b> vers un attaquant lancé dans le <b>half-space</b>.",

    "RC Strasbourg": "Strasbourg privilégie un jeu direct avec un <b>pressing</b> haut pour forcer les erreurs adverses. Les ailiers cherchent constamment la <b>width</b> pour effectuer des <b>cross</b>es vers le <b>pivot</b> ou les milieux arrivant en retard. En phase défensive, une <b>shape</b> compacte avec un <b>low block</b> bien organisé absorbe la pression. Des <b>tackle</b>s et des <b>interception</b>s dans l'entrejeu alimentent des <b>counter-attack</b>s directes, profitant des espaces laissés en <b>transition</b>. Un <b>switch of play</b> rapide permet de changer d'axe d'attaque.",

    "Toulouse FC": "Toulouse construit patiemment avec un <b>build-up play</b> propre et un <b>positional play</b> rigoureux depuis l'arrière. Les milieux se glissent dans les <b>half-space</b>s pour recevoir entre les <b>lines</b> adverses et proposer des <b>through ball</b>s aux attaquants. La <b>formation</b> est flexible : en défense, un <b>zonal marking</b> strict couvre les couloirs et les zones centrales. Des <b>line-breaking pass</b>es déclenchent les courses en profondeur vers le <b>final third</b>, et en <b>transition</b> négative, l'équipe se replace rapidement.",

    "Stade Brestois": "Brest repose sur un <b>bloc bas</b> très compact qui limite l'espace dans son propre <b>final third</b> en resserrant les <b>lines</b>. La <b>shape</b> défensive combine <b>man marking</b> strict sur les porteurs et <b>zonal marking</b> sur les zones dangereuses. Des <b>tackle</b>s et <b>interception</b>s fréquents alimentent des <b>counter-attack</b>s rapides sur les flancs. En <b>transition</b> offensive, le <b>pivot</b> sert de relais pour distribuer en <b>width</b> et exploiter les espaces. Un <b>pressing</b> ponctuel peut surprendre l'adversaire dans ses relances.",

    "FC Nantes": "Nantes mise sur un jeu de <b>cross</b>es depuis les ailes, avec des latéraux pratiquant des <b>overlap</b>s réguliers pour centrer dans la surface. Le <b>pivot</b> est au cœur du dispositif offensif dans le <b>final third</b>, servant de point d'appui pour les milieux arrivant en retard. En phase défensive, un <b>pressing</b> collectif couvre tout le terrain avec un <b>man marking</b> ciblant les joueurs clés adverses. Un <b>switch of play</b> rapide libère le côté faible, et des <b>tackle</b>s robustes alimentent les <b>transition</b>s offensives.",

    "Angers SCO": "Angers adopte un <b>low block</b> défensif qui resserre les espaces entre les <b>lines</b> et prive l'adversaire d'espace dans le <b>final third</b>. La <b>shape</b> compacte s'appuie sur un <b>zonal marking</b> organisé sur l'ensemble du terrain. Des <b>interception</b>s bien placées et des <b>tackle</b>s propres alimentent des <b>counter-attack</b>s en <b>transition</b>. Le <b>pressing</b> est ciblé et non systématique, utilisé uniquement lorsque l'adversaire est en difficulté. En possession, un <b>build-up play</b> prudent maintient la balle et évite les risques.",

    "Le Havre AC": "Le Havre repose sur un <b>bloc bas</b> rigoureux pour protéger son but et limiter les espaces en profondeur. La <b>shape</b> défensive combine <b>man marking</b> sur les attaquants et <b>zonal marking</b> sur les couloirs, en maintenant une grande <b>depth</b> défensive pour réduire l'espace derrière les <b>lines</b>. En <b>transition</b> offensive, les joueurs cherchent rapidement une <b>counter-attack</b> via un jeu direct vers l'avant. Un <b>pressing</b> collectif ponctuel peut être déclenché pour récupérer dans l'entrejeu.",

    "AJ Auxerre": "Auxerre pratique un <b>high press</b> dynamique pour reprendre le ballon haut sur le terrain, suivi d'un <b>counter-pressing</b> immédiat. En possession, l'équipe crée des <b>overload</b>s sur un côté avant d'utiliser un <b>switch of play</b> pour exploiter l'espace libéré. Des <b>line-breaking pass</b>es et des <b>through ball</b>s permettent d'atteindre rapidement le <b>final third</b> adverse. Des <b>third man run</b>s bien coordonnés brisent les <b>lines</b> défensives adverses, et un <b>build-up play</b> propre depuis l'arrière permet de repartir sereinement.",

    "FC Metz": "Metz s'appuie sur un <b>low block</b> très compact pour absorber la pression et protéger les espaces dans son <b>final third</b>. La <b>shape</b> défensive repose sur un <b>man marking</b> strict pour neutraliser les joueurs clés adverses. Des <b>tackle</b>s bien placés et des <b>interception</b>s alimentent des <b>counter-attack</b>s directes en <b>transition</b>. Le <b>pressing</b> est utilisé de manière ciblée dans les zones de récupération prioritaires. En phase offensive, un jeu de <b>cross</b>es depuis les ailes cherche à exploiter le <b>pivot</b> dans la surface.",

    "Paris FC": "Le Paris FC construit son jeu sur un <b>build-up play</b> soigné, avec des milieux se positionnant dans les <b>half-space</b>s pour progresser entre les <b>lines</b> adverses. La philosophie de <b>positional play</b> s'appuie sur une grande <b>width</b> et un <b>depth</b> bien maîtrisé pour occuper tout l'espace. Des <b>line-breaking pass</b>es atteignent les attaquants dans le <b>final third</b>, suivies de <b>through ball</b>s vers les coureurs en profondeur. En défense, une <b>shape</b> compacte et un <b>pressing</b> bien calibré maintiennent l'organisation, et la <b>transition</b> est rapide dans les deux sens.",

    "FC Lorient": "Lorient s'appuie sur un <b>pressing</b> collectif pour récupérer le ballon dans l'entrejeu, avec une <b>shape</b> défensive très organisée. La <b>counter-attack</b> est la principale arme offensive : après une <b>interception</b> ou un <b>tackle</b>, le ballon remonte vite vers les attaquants. Un <b>switch of play</b> rapide exploite les espaces laissés par l'adversaire en <b>transition</b>. Des <b>overlap</b>s des latéraux suivis de <b>cross</b>es dans la surface sont la combinaison offensive privilégiée. En défense profonde, un <b>bloc bas</b> bien en place protège les espaces dans le <b>final third</b>.",
}
DEFAULT_STYLE = "Style de jeu à documenter."
WATCH_COLORS = ["#7CC99A", "#F5D06E", "#F2827F"]

def linkify_terms(text):
    """Replace <b>term</b> with a colored clickable link."""
    for term in TACTICAL_TERMS:
        text = text.replace(
            f'<b>{term}</b>',
            f'<a href="?term={term}" class="term-link">{term}</a>'
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
.stat-cmp-lbl{font-size:.72rem;font-weight:800;color:var(--dark);min-width:38px;letter-spacing:-.01em;}
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
    st.session_state.active_term = qp["term"]
    st.session_state.page = "definition"
    st.query_params.clear()
    st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("page","main"), ("active_term",None),
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
        return ["Données non disponibles."]*3
    played = da.get("played",1) or 1
    gf_a, gf_b = da["goals_for"], db["goals_for"]
    ga_a, ga_b = da["goals_against"], db["goals_against"]
    pts_a, pts_b = da["points"], db["points"]
    return [
        f"<b>{a if gf_a>=gf_b else b}</b> mène offensivement : {gf_a} buts pour {a} vs {gf_b} pour {b} ({gf_a/played:.1f} vs {gf_b/played:.1f} par match).",
        f"Solidité défensive : <b>{a if ga_a<=ga_b else b}</b> encaisse moins ({ga_a} vs {ga_b} buts encaissés). Écart de {abs(ga_a-ga_b)} buts.",
        f"Au classement, <b>{a if pts_a>=pts_b else b}</b> devance ({pts_a} pts vs {pts_b} pts). Diff. de buts : {da['goal_diff']:+} vs {db['goal_diff']:+}.",
    ]

GLOS_ICONS = ["⚡","🎯","🔄","💨","🛡️"]
GLOS_COLORS = ["var(--yellow-lt)","var(--green-lt)","var(--red-lt)","var(--beige)","var(--green-lt)"]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER (shown on all pages except definition)
# ══════════════════════════════════════════════════════════════════════════════
def render_header():
    st.markdown("""
<div class="app-header">
<div><div class="app-title">The Football <span>Classroom</span></div><div class="app-sub">Analyse tactique · Ligue 1</div></div>
<div class="app-header-right">
<div class="live-badge"><span class="live-dot"></span>En direct</div>
<div class="app-badges"><span class="app-badge">Ligue 1</span><span class="app-badge">2025/26</span><span class="app-badge">18 équipes</span></div>
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
        if st.button("⚽  Analyse", type=t, use_container_width=True, key="nav_main"):
            st.session_state.page = "main"; st.rerun()
    with c2:
        t = "primary" if page == "classement" else "secondary"
        if st.button("📊  Classement", type=t, use_container_width=True, key="nav_class"):
            st.session_state.page = "classement"; st.rerun()
    with c3:
        t = "primary" if page == "glossaire" else "secondary"
        if st.button("📖  Glossaire", type=t, use_container_width=True, key="nav_glos"):
            st.session_state.page = "glossaire"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE DÉFINITION
# ══════════════════════════════════════════════════════════════════════════════
def page_definition():
    term = st.session_state.active_term
    term_data = TACTICAL_TERMS.get(term, {})
    definition      = term_data.get("definition",        "Définition à venir.") if isinstance(term_data, dict) else term_data
    simple          = term_data.get("simple_explanation", "") if isinstance(term_data, dict) else ""
    example         = term_data.get("example",            "") if isinstance(term_data, dict) else ""
    animation       = term_data.get("animation_idea",     "") if isinstance(term_data, dict) else ""

    if st.button("← Retour", type="primary", key="back"):
        st.session_state.page = "main"
        st.session_state.active_term = None
        st.rerun()

    st.markdown(f"""<div class="def-hero"><span class="pill pill-yellow">Terme tactique</span><div class="def-title">{term.capitalize()}</div><div class="def-category">Glossaire · Ligue 1</div></div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="def-text">{definition}</div>', unsafe_allow_html=True)
    if simple:
        st.markdown(f'<div class="def-simple"><span class="def-tag def-tag-green">💡 In simple terms</span><p>{simple}</p></div>', unsafe_allow_html=True)
    if example:
        st.markdown(f'<div class="def-example"><span class="def-tag def-tag-yellow">⚽ Example</span><p>{example}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Visualisation</div><div class="sec-title">Illustration tactique</div>', unsafe_allow_html=True)
    terrain_label = animation if animation else "Visualisation — à venir"
    st.markdown(f"""<div class="terrain-wrap"><div class="terrain-border"></div><div class="terrain-center"></div><div class="terrain-center-dot"></div><div class="terrain-box-top"></div><div class="terrain-box-bot"></div><div class="terrain-small-top"></div><div class="terrain-small-bot"></div><span class="terrain-label">{terrain_label}</span></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE GLOSSAIRE
# ══════════════════════════════════════════════════════════════════════════════
def page_glossaire():
    st.markdown('<div class="sec-label">Vocabulaire</div><div class="sec-title">Glossaire tactique</div>', unsafe_allow_html=True)
    for i, (term, term_data) in enumerate(TACTICAL_TERMS.items()):
        definition = term_data.get("definition", "") if isinstance(term_data, dict) else term_data
        icon = GLOS_ICONS[i % len(GLOS_ICONS)]
        bg   = GLOS_COLORS[i % len(GLOS_COLORS)]
        st.markdown(
            f'<div class="glos-card">'
            f'<div class="glos-card-header">'
            f'<div class="glos-card-icon" style="background:{bg}">{icon}</div>'
            f'<span class="glos-card-term">{term.capitalize()}</span>'
            f'<span class="pill pill-yellow" style="margin-left:auto">Tactique</span>'
            f'</div>'
            f'<div class="glos-card-body">{definition}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CLASSEMENT
# ══════════════════════════════════════════════════════════════════════════════
def page_classement():
    team_a = st.session_state.team_a
    team_b = st.session_state.team_b
    da = standings.get(team_a, {})

    st.markdown('<div class="sec-label">Ligue 1</div><div class="sec-title">Classement 2025/26</div>', unsafe_allow_html=True)

    # Build rows — NO indentation to avoid markdown code-block interpretation
    hdr = '<div class="standings-hdr-row"><span class="standings-pos">#</span><span style="width:20px"></span><span style="flex:1">Équipe</span><span class="standings-stat">J</span><span class="standings-stat">V</span><span class="standings-stat">N</span><span class="standings-stat">D</span><span class="standings-gd">DB</span><span class="standings-pts">Pts</span></div>'

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
        f'<div class="standings-header"><span class="standings-header-title">Classement général — Journée {da.get("played","?")}</span></div>'
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
    st.markdown('<div class="match-card"><span class="match-card-label">Choisir le match à analyser</span><span class="match-vs-badge">VS</span></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        idx_a = ALL_TEAMS.index(team_a) if team_a in ALL_TEAMS else 0
        st.session_state.team_a = st.selectbox("Équipe A", ALL_TEAMS, index=idx_a, key="sel_a")
    with col_b:
        remaining = [t for t in ALL_TEAMS if t != st.session_state.team_a]
        if st.session_state.team_b not in remaining:
            st.session_state.team_b = remaining[0]
        st.session_state.team_b = st.selectbox("Équipe B", remaining, index=remaining.index(st.session_state.team_b), key="sel_b")

    team_a, team_b = st.session_state.team_a, st.session_state.team_b
    da, db = standings.get(team_a, {}), standings.get(team_b, {})
    crest_a, crest_b = da.get("crest",""), db.get("crest","")
    img_a = get_crest_img(team_a, 36)
    img_b = get_crest_img(team_b, 36)

    # ── VS Banner ──
    st.markdown(
        f'<div class="vs-banner">'
        f'<div class="vs-team"><span class="vs-team-label vs-team-label-a">Équipe A</span>'
        f'<div class="vs-team-crest">{img_a}<span class="vs-team-name">{team_a}</span></div>'
        f'<span class="vs-team-stat">#{da.get("position","—")} · {da.get("points","—")} pts</span></div>'
        f'<div class="vs-sep"><span class="vs-sep-dot"></span><span class="vs-sep-text">VS</span><span class="vs-sep-dot"></span></div>'
        f'<div class="vs-team"><span class="vs-team-label vs-team-label-b">Équipe B</span>'
        f'<div class="vs-team-crest">{img_b}<span class="vs-team-name">{team_b}</span></div>'
        f'<span class="vs-team-stat">#{db.get("position","—")} · {db.get("points","—")} pts</span></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Styles de jeu ──
    st.markdown('<div class="sec-label">Analyse IA</div><div class="sec-title">Style de jeu</div>', unsafe_allow_html=True)

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
        return f'<div style="margin-bottom:.6rem;font-size:.62rem;font-weight:800;color:var(--mid);letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem">Forme récente &nbsp;{pills}</div>'

    def _fmt_style(raw):
        """Convertit les sauts de ligne en <br> pour l'affichage HTML, puis linkifie les termes."""
        html = raw.replace("\n\n", "<br><br>").replace("\n", " ")
        return linkify_terms(html)

    with st.spinner("Génération de l'analyse IA…"):
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
            f'<div class="team-card-header">{img28_a} {team_a}<span class="badge">Équipe A</span></div>'
            f'<div class="team-card-body">{_render_form(form_a)}{style_a}</div>'
            f'<div class="team-stats-row">'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("points","—")}</div><div class="team-stat-box-lbl">Pts</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("won","—")}</div><div class="team-stat-box-lbl">V</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("draw","—")}</div><div class="team-stat-box-lbl">N</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("lost","—")}</div><div class="team-stat-box-lbl">D</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{da.get("goals_for","—")}</div><div class="team-stat-box-lbl">Buts</div></div>'
            f'</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="team-card card-b">'
            f'<div class="team-card-header">{img28_b} {team_b}<span class="badge">Équipe B</span></div>'
            f'<div class="team-card-body">{_render_form(form_b)}{style_b}</div>'
            f'<div class="team-stats-row">'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("points","—")}</div><div class="team-stat-box-lbl">Pts</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("won","—")}</div><div class="team-stat-box-lbl">V</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("draw","—")}</div><div class="team-stat-box-lbl">N</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("lost","—")}</div><div class="team-stat-box-lbl">D</div></div>'
            f'<div class="team-stat-box"><div class="team-stat-box-num">{db.get("goals_for","—")}</div><div class="team-stat-box-lbl">Buts</div></div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Terrain ──
    st.markdown('<div class="sec-label">Tactique</div><div class="sec-title">Terrain tactique</div>', unsafe_allow_html=True)
    st.markdown('<div class="terrain-wrap"><div class="terrain-border"></div><div class="terrain-center"></div><div class="terrain-center-dot"></div><div class="terrain-box-top"></div><div class="terrain-box-bot"></div><div class="terrain-small-top"></div><div class="terrain-small-bot"></div><span class="terrain-label">Terrain tactique — à venir</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Comparaison stats ──
    st.markdown('<div class="sec-label">Statistiques</div><div class="sec-title">Comparaison directe</div>', unsafe_allow_html=True)

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
        st.markdown(cmp_card("stat-cmp-hdr-1","stat-cmp-dot-1","Efficacité offensive",gf_a,gf_b,max_gf,f"Moy. {gf_a/played_a:.1f} vs {gf_b/played_a:.1f} buts / match"), unsafe_allow_html=True)
    with r2:
        st.markdown(cmp_card("stat-cmp-hdr-2","stat-cmp-dot-2","Points au classement",pts_a,pts_b,max_pts,f"#{da.get('position','—')} vs #{db.get('position','—')} au classement"), unsafe_allow_html=True)
    with r3:
        st.markdown(cmp_card("stat-cmp-hdr-3","stat-cmp-dot-3","Solidité défensive",ga_a,ga_b,max_ga,"Buts encaissés — moins = meilleure défense",inverted=True), unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Watch card ──
    points = watch_points(team_a, team_b)
    st.markdown(
        f'<div class="watch-card">'
        f'<div class="watch-header"><div class="watch-icon">👁</div><div><div class="watch-title">Points clés à surveiller</div><div class="watch-subtitle">{team_a} vs {team_b}</div></div></div>'
        f'<div class="watch-item"><div class="watch-num">01</div><div class="watch-dot" style="background:{WATCH_COLORS[0]}"></div><div class="watch-text">{points[0]}</div></div>'
        f'<div class="watch-item"><div class="watch-num">02</div><div class="watch-dot" style="background:{WATCH_COLORS[1]}"></div><div class="watch-text">{points[1]}</div></div>'
        f'<div class="watch-item"><div class="watch-num">03</div><div class="watch-dot" style="background:{WATCH_COLORS[2]}"></div><div class="watch-text">{points[2]}</div></div>'
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
