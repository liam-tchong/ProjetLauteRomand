# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
streamlit run app.py
```

The static website (`index.html`) can be opened directly in a browser — no build step required.

## Architecture

This repo contains two independent projects sharing a design system:

**`index.html`** — Static marketing website for "FORMA Studio" (architecture firm). Single-file, no dependencies beyond a Google Fonts import. Uses vanilla JS for project grid filtering and a marquee animation.

**`app.py`** — Streamlit app ("LaUte Romand App") for Ligue 1 tactical analysis. All data (teams, tactical terms, team styles) is hardcoded as Python dicts at the top of the file. Navigation between pages is managed via `st.session_state.page` ("main" or "definition"). The UI is entirely custom HTML/CSS injected via `st.markdown(..., unsafe_allow_html=True)` — Streamlit's native widgets are only used for selectboxes and buttons.

## Shared design language

Both files use the same CSS custom properties and visual style (Nunito font, warm beige palette with green/red/yellow accents, rounded cards). Keep this consistent when adding UI.
