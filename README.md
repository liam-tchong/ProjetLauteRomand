# ⚽ The Football Classroom

> A web application for tactical football analysis across Europe's top 5 leagues — powered by Machine Learning, real-time APIs, and AI-generated insights.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Claude AI](https://img.shields.io/badge/Claude-Haiku-purple?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## 📌 Overview

**The Football Classroom** is an interactive Streamlit web app that lets users explore, compare, and predict football matches across 5 major European leagues. It combines live data from external APIs, machine learning models trained on thousands of historical matches, and AI-generated tactical analysis to deliver a rich educational football experience.

---

## 🌍 Supported Leagues

| League | Country | API Code |
|---|---|---|
| 🇫🇷 Ligue 1 | France | `FL1` |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League | England | `PL` |
| 🇪🇸 La Liga | Spain | `PD` |
| 🇮🇹 Serie A | Italy | `SA` |
| 🇩🇪 Bundesliga | Germany | `BL1` |

---

## ✨ Features

### 📊 Live Standings
- Full league table with team crests, points, goal difference, and form
- Position change indicators vs previous season
- AI-generated league summary (Claude Haiku) highlighting the key storylines

### 📅 Match Schedule
- 17-day window (3 days back, 14 days forward)
- Live scores with status badges (LIVE / FIN / Scheduled)
- Matches grouped by date with local timezone conversion

### 🔬 Team vs Team Analysis
The core feature of the app. Select any two teams from any league and get:
- **AI-generated style cards** (4-section tactical breakdown per team via Claude Haiku)
- **Tactical pitch visualization** — animated SVG showing formation, player positions, movement arrows, and heat zones
- **ML match prediction** — win/draw/loss probabilities derived from a joint Poisson distribution
- **Expected score** — most likely scoreline computed via Dixon-Coles inspired xG model
- **Stats comparison** — offensive and defensive metrics side by side
- **Key challenges** — AI-generated matchup-specific insight per team

### 📚 Tactical Glossary
3-tab reference section:
- **Tactics** — clickable cards for every tactical term (High Press, Tiki-Taka, Gegenpressing, etc.) with animated SVG definitions
- **Positions** — real squad composition pulled from api-sports.io, organized by position
- **Formations** — visual explanation of the most common formations

### 📖 Rules of the Game
Expandable cards covering the fundamental rules of football, designed for beginners.

### 🔗 Tactical Term Definitions
Each tactical term links to a dedicated page with:
- Full definition
- SVG animation illustrating the concept
- Clickable related terms

---

## 🧠 Machine Learning

The prediction engine is built on two models trained on **5000+ historical matches** from 2023–2026:

### Models
| Model | Task | Algorithm |
|---|---|---|
| Result classifier | Predict H / D / A | `GradientBoostingClassifier` |
| Goals regressors | Predict home & away goals | `PoissonRegressor` (×2) |

### Features (per team, last 5 matches)
- Win rate (form)
- Average goals scored
- Average goals conceded

### xG Engine
The expected goals model uses a Dixon-Coles inspired multiplicative formula:

```
xG_home = league_avg_home × √(attack_strength_home × defensive_weakness_away)
```

Adjusted for:
- **Recent form** (weighted last-5 results, ±20% swing)
- **Fatigue** (match within 4 days → -12% xG)
- **Stakes** (title race / relegation battle → more cautious play)

Win/draw/loss probabilities are derived from the joint Poisson distribution over the xG values — ensuring the probability bar and score prediction are always coherent.

### Auto-retraining
The model silently re-fetches new match results and retrains in a background thread on app startup (at most once per hour), keeping predictions current without blocking the UI.

---

## 🛠️ Technical Stack

| Layer | Technology |
|---|---|
| Web framework | Streamlit |
| Language | Python 3.10+ |
| ML | scikit-learn (GradientBoosting, Poisson) |
| AI text generation | Anthropic Claude Haiku |
| Football data | football-data.org API v4 |
| Advanced stats | api-sports.io (Ligue 1 only) |
| Visualizations | Custom SVG generated in Python |
| UI components | HTML/CSS injected via `st.markdown` |
| Parallel API calls | `ThreadPoolExecutor` |
| Model persistence | `pickle` |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/liam-tchong/ProjetLauteRomand.git
cd ProjetLauteRomand
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.streamlit/secrets.toml` file at the root of the project:

```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key"
API_FOOTBALL_KEY = "your_api_sports_io_key"
SQUAD_API_KEY = "your_squad_api_key"
```

> The app will still run without these keys — AI features will fall back to static content and Ligue 1 advanced stats will be disabled.

### 4. (Optional) Retrain the ML model

```bash
cd MachineLearning
python collect_data.py   # fetch historical match data
python train_model.py    # train and save models
```

Pre-trained models (`model.pkl`, `goals_model.pkl`) are already included in the repository.

### 5. Run the app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit app (UI, navigation, pages, ML engine)
├── football_data.py        # Football data constants and configurations
├── tactical_data.py        # Tactical formations, player positions, heat zones (all 5 leagues)
├── requirements.txt        # Python dependencies
├── index.html              # Static landing page
├── MachineLearning/
│   ├── collect_data.py     # Script to fetch and store historical match data
│   ├── train_model.py      # Script to train and export ML models
│   ├── match_data.csv      # Historical match dataset (5000+ matches)
│   ├── model.pkl           # Trained GradientBoostingClassifier
│   └── goals_model.pkl     # Trained PoissonRegressors (home + away)
└── .streamlit/
    └── secrets.toml        # API keys (not committed to git)
```

---

## 👥 Team Contributions

### Contribution Matrix

| Section | Abel | Benjamin | Cedric | Liam | Romain |
|---|---|---|---|---|---|
| Idea | Contributor | Contributor | Contributor | Contributor | Contributor |
| Streamlit + GitHub setup | Minor | Minor | Minor | **Main** | Minor |
| Dataset | Contributor | Contributor | Contributor | Contributor | Contributor |
| ML | **Main** | Minor | **Main** | Minor | Minor |
| User Interaction | Contributor | Contributor | Contributor | **Main** | **Main** |
| Visualisation | Contributor | Contributor | Contributor | Contributor | **Main** |
| Video | **Main** | Minor | Minor | Minor | Contributor |
| Matrix | Contributor | Contributor | **Main** | Contributor | Contributor |

### Function Descriptions

| Member | Main Contributions |
|---|---|
| **Abel** | Idea generation, implementation structure, Spotify API exploration, dataset selection, dead code cleanup |
| **Benjamin** | Machine learning deep dive, kNN algorithm, development of the group vector, CSV-to-dataset pipeline, video filming |
| **Cedric** | ML model development, code structuring, hardcoded data extraction (rules, positions, formations, tactical terms), Glossary section (Positions & Formations tabs) |
| **Liam** | Streamlit setup, GitHub setup, navigation system, session state management, all pages UI, visual design |
| **Romain** | Feature development, Claude API integration, AI-generated tactical analysis, SVG visualizations, video filming |

---

## 🔑 API Keys

| API | Purpose | Free tier |
|---|---|---|
| [football-data.org](https://www.football-data.org/) | Standings, matches, scorers | ✅ 10 req/min |
| [api-sports.io](https://api-sports.io/) | Advanced Ligue 1 stats, squad composition | ✅ 100 req/day |
| [Anthropic](https://www.anthropic.com/) | AI text generation (Claude Haiku) | ❌ Paid |

---

## 📄 License

This project was built as part of a school project. All football data belongs to their respective owners.
