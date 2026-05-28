"""
NBA Game Predictor — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# --- Page config must be the very first Streamlit call ---
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.styles import get_css, card, stat_badge, section_header, ACCENT_COLORS, TEAM_COLORS, TEXT_MUTED, CARD_BG, BORDER_COLOR
from ui.charts import (
    win_probability_gauge,
    factor_radar,
    factor_bar_comparison,
    game_log_sparkline,
    shooting_breakdown,
)
from data.fetcher import fetch_all_team_data, get_head_to_head, get_team_id, get_team_info, NBA_HEADERS
from data.predictor import predict_game

# Inject global CSS
st.markdown(get_css(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CURRENT_SEASON = "2025-26"

NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards",
]

TEAM_LOGO_URLS = {
    "Minnesota Timberwolves": "https://cdn.nba.com/logos/nba/1610612750/global/L/logo.svg",
    "San Antonio Spurs":       "https://cdn.nba.com/logos/nba/1610612759/global/L/logo.svg",
}

FACTOR_LABELS = {
    "net_rating":         "Net Rating",
    "off_rating_l10":     "Offense (L10)",
    "def_rating_l10":     "Defense (L10)",
    "win_pct_l10":        "Win % (L10)",
    "true_shooting_l10":  "True Shooting %",
    "rest_advantage":     "Rest Advantage",
    "pace_adjusted_pts":  "Pace-Adj Points",
    "turnover_rate":      "Ball Security",
    "home_court":         "Home Court",
    "rebound_rate":       "Rebounding",
    "clutch_plus_minus":  "Clutch Perf.",
    "assist_to_tov":      "AST/TOV Ratio",
    "bench_contribution": "Bench Depth",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_team_colors(team_name: str) -> dict:
    return TEAM_COLORS.get(team_name, TEAM_COLORS["default"])


def team_card_component(team_id: int, team_name: str, role: str,
                         name_color: str, height: int = 195) -> None:
    """
    Render a team card with logo using st.components.v1.html().
    This bypasses st.markdown HTML sanitisation and renders reliably.
    """
    logo_url = f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
    html = f"""<!DOCTYPE html>
<html><head><style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #161616;
    font-family: 'Segoe UI', sans-serif;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    overflow: hidden;
  }}
  .card {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.2rem 1rem 1rem;
    gap: 8px;
  }}
  .logo {{
    width: 80px;
    height: 80px;
    object-fit: contain;
  }}
  .logo-fallback {{
    font-size: 3rem;
    display: none;
  }}
  .name {{
    color: {name_color};
    font-size: 1.05rem;
    font-weight: 700;
    text-align: center;
    line-height: 1.25;
  }}
  .role {{
    color: #888888;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
</style></head>
<body>
  <div class="card">
    <img class="logo" src="{logo_url}"
         onerror="this.style.display='none'; document.querySelector('.logo-fallback').style.display='block';">
    <span class="logo-fallback">🏀</span>
    <div class="name">{team_name}</div>
    <div class="role">{role}</div>
  </div>
</body></html>"""
    components.html(html, height=height, scrolling=False)


def team_result_card_component(team_id: int, team_name: str, role: str,
                                name_color: str, win_prob: float,
                                season: str, height: int = 230) -> None:
    """
    Render a team result card (with win probability) using st.components.v1.html().
    """
    logo_url = f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
    html = f"""<!DOCTYPE html>
<html><head><style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #161616;
    font-family: 'Segoe UI', sans-serif;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    overflow: hidden;
  }}
  .card {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    gap: 6px;
  }}
  .logo {{
    width: 72px;
    height: 72px;
    object-fit: contain;
  }}
  .logo-fallback {{ font-size: 2.5rem; display: none; }}
  .name {{
    color: {name_color};
    font-size: 1rem;
    font-weight: 700;
    text-align: center;
    line-height: 1.2;
  }}
  .role {{
    color: #888888;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  .prob {{
    color: {name_color};
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    margin-top: 4px;
  }}
  .prob-label {{
    color: #888888;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
</style></head>
<body>
  <div class="card">
    <img class="logo" src="{logo_url}"
         onerror="this.style.display='none'; document.querySelector('.logo-fallback').style.display='block';">
    <span class="logo-fallback">🏀</span>
    <div class="name">{team_name}</div>
    <div class="role">{role} · {season} Playoffs</div>
    <div class="prob">{win_prob * 100:.1f}%</div>
    <div class="prob-label">Win Probability</div>
  </div>
</body></html>"""
    components.html(html, height=height, scrolling=False)


def color_edge(score: float) -> str:
    if score > 65:
        return ACCENT_COLORS["green"]
    elif score > 55:
        return ACCENT_COLORS["blue"]
    elif score < 35:
        return ACCENT_COLORS["red"]
    elif score < 45:
        return ACCENT_COLORS["orange"]
    return TEXT_MUTED


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""<div style="text-align:center; padding: 1rem 0;">
            <span style="font-size:2.5rem;">🏀</span>
            <h2 style="margin:0.3rem 0 0 0; font-size:1.3rem; color:#F0F0F0;">
                NBA Predictor
            </h2>
            <p style="color:{TEXT_MUTED}; font-size:0.75rem; margin:0;">
                Playoff Edition · {CURRENT_SEASON}
            </p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem;'>HOME TEAM</p>", unsafe_allow_html=True)
    team1_name = st.selectbox(
        "Home Team", NBA_TEAMS,
        index=NBA_TEAMS.index("Minnesota Timberwolves"),
        label_visibility="collapsed",
        key="team1",
    )

    st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; margin:0.8rem 0 0.3rem 0;'>AWAY TEAM</p>", unsafe_allow_html=True)
    team2_name = st.selectbox(
        "Away Team", NBA_TEAMS,
        index=NBA_TEAMS.index("San Antonio Spurs"),
        label_visibility="collapsed",
        key="team2",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Prediction", use_container_width=True, type="primary")

    st.divider()
    st.markdown(
        f"""<div style="color:{TEXT_MUTED}; font-size:0.75rem; line-height:1.7;">
            <b style="color:#aaa;">Model factors:</b><br>
            • Net/Off/Def Rating<br>
            • Last 10 game form<br>
            • True Shooting %<br>
            • Rest & schedule<br>
            • Turnover rate<br>
            • Rebounding rate<br>
            • Clutch performance<br>
            • AST/TOV ratio<br>
            • Bench contribution
        </div>""",
        unsafe_allow_html=True,
    )


# Resolve team IDs early — needed for logos on both landing and results pages
_team1_info = get_team_info(team1_name)
_team2_info = get_team_info(team2_name)
_team1_id   = _team1_info["id"]
_team2_id   = _team2_info["id"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
        <h1 style="font-size:2.2rem; font-weight:800; letter-spacing:0.04em; margin:0;">
            <span style="color:{ACCENT_COLORS['green']}">NBA</span>
            <span style="color:#F0F0F0;"> GAME PREDICTOR</span>
        </h1>
        <p style="color:{TEXT_MUTED}; font-size:0.9rem; margin:0.3rem 0 0 0;">
            Statistical analysis &amp; win probability for playoff matchups
        </p>
    </div>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main content — only renders after button press
# ---------------------------------------------------------------------------
if not run_btn:
    # Landing state
    t1c = get_team_colors(team1_name)
    t2c = get_team_colors(team2_name)
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns([1, 0.15, 1])
    with cols[0]:
        team_card_component(_team1_id, team1_name, "Home", t1c["highlight"])
    with cols[1]:
        st.markdown(
            f"<div style='text-align:center; padding-top:3.5rem; font-size:1.8rem; color:{TEXT_MUTED}; font-weight:700;'>VS</div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        team_card_component(_team2_id, team2_name, "Away", t2c["accent"] if t2c["accent"] != "#000000" else "#C4CED4")
    st.markdown(
        f"<div style='text-align:center; margin-top:2rem; color:{TEXT_MUTED};'>"
        f"Select matchup in the sidebar and click <b>Run Prediction</b> to begin.</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
with st.spinner(f"Fetching data for {team1_name} & {team2_name}…"):
    try:
        t1_data = fetch_all_team_data(team1_name, season=CURRENT_SEASON)
        t2_data = fetch_all_team_data(team2_name, season=CURRENT_SEASON)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.info("The ESPN API may be unreachable or the season data may not be available. Try again in a moment.")
        st.stop()

with st.spinner("Running prediction model…"):
    # team1 is the "Home Team" in the UI selectors above.
    result = predict_game(t1_data, t2_data, team_is_home=True)

# Show banner if either team is using sample data
if t1_data.get("_is_sample") or t2_data.get("_is_sample"):
    st.info(
        "**Demo mode** — The NBA stats API is currently unreachable (stats.nba.com blocks "
        "direct Python requests). Showing embedded 2025-26 season data. "
        "All model logic, charts, and UI are fully functional.",
        icon="ℹ️",
    )

t1c = get_team_colors(team1_name)
t2c = get_team_colors(team2_name)
team1_color = t1c["highlight"]
team2_color = t2c["accent"] if t2c["accent"] != "#000000" else "#C4CED4"


# ---------------------------------------------------------------------------
# Matchup header with prediction result
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
header_cols = st.columns([2, 1, 2])
with header_cols[0]:
    team_result_card_component(
        t1_data["team_id"], team1_name, "Home",
        team1_color, result["team_win_prob"], CURRENT_SEASON,
    )
with header_cols[1]:
    st.markdown(
        f"""<div style="text-align:center; padding-top:2.5rem;">
            <div style="color:{TEXT_MUTED}; font-size:1.4rem; font-weight:700; letter-spacing:0.1em;">VS</div>
            <div style="margin-top:1rem; padding:0.5rem 0.8rem;
                        background:{'#1a3a1a' if result['predicted_winner'] == team1_name else '#3a1a1a'};
                        border-radius:8px; border:1px solid {'#2d6b2d' if result['predicted_winner'] == team1_name else '#6b2d2d'};">
                <div style="font-size:0.65rem; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em;">Predicted Winner</div>
                <div style="font-size:0.8rem; color:{ACCENT_COLORS['green'] if result['predicted_winner'] == team1_name else ACCENT_COLORS['red']};
                            font-weight:700; margin-top:0.2rem;">
                    {'🏆 ' + team1_name.split()[-1] if result['predicted_winner'] == team1_name else '🏆 ' + team2_name.split()[-1]}
                </div>
                <div style="font-size:0.7rem; color:{TEXT_MUTED}; margin-top:0.3rem;">
                    Confidence: <b style="color:#aaa">{result['confidence']}</b>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
with header_cols[2]:
    team_result_card_component(
        t2_data["team_id"], team2_name, "Away",
        team2_color, result["opp_win_prob"], CURRENT_SEASON,
    )

# Win probability bar
st.markdown("<br>", unsafe_allow_html=True)
st.plotly_chart(
    win_probability_gauge(team1_name, team2_name,
                          result["team_win_prob"], result["opp_win_prob"],
                          team1_color, team2_color),
    use_container_width=True,
    config={"displayModeBar": False},
)

st.divider()

# ---------------------------------------------------------------------------
# Prediction Reasoning
# ---------------------------------------------------------------------------
rsn = result["reasoning"]
winner_color = team1_color if rsn["winner"] == team1_name else team2_color
loser_color  = team2_color if rsn["winner"] == team1_name else team1_color

st.markdown(section_header("Why We Predict This Outcome", "🔍"), unsafe_allow_html=True)

# Narrative sentence
st.markdown(
    f"""<div style="background:#1a1a1a; border-left:3px solid {winner_color};
                   border-radius:0 8px 8px 0; padding:0.9rem 1.2rem; margin-bottom:1rem;
                   font-size:0.95rem; color:#E0E0E0; line-height:1.6;">
        {rsn['narrative']}
    </div>""",
    unsafe_allow_html=True,
)

reason_col1, reason_col2 = st.columns(2)

with reason_col1:
    st.markdown(
        f"<p style='color:{winner_color}; font-weight:700; font-size:0.85rem; "
        f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>"
        f"✅ {rsn['winner'].split()[-1]} Key Advantages</p>",
        unsafe_allow_html=True,
    )
    if rsn["advantages"]:
        for adv in rsn["advantages"][:6]:
            margin_bar = min(adv["margin"] / 40 * 100, 100)
            st.markdown(
                f"""<div style="background:{CARD_BG}; border:1px solid {BORDER_COLOR};
                               border-radius:8px; padding:0.6rem 0.9rem; margin-bottom:0.5rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="color:#E0E0E0; font-size:0.85rem; font-weight:600;">{adv['label']}</span>
                        <span style="color:{winner_color}; font-size:0.8rem; font-weight:700;">+{adv['margin']:.0f} pts edge</span>
                    </div>
                    <div style="background:#2a2a2a; border-radius:4px; height:5px; margin-bottom:5px;">
                        <div style="background:{winner_color}; width:{margin_bar:.0f}%; height:5px; border-radius:4px;"></div>
                    </div>
                    <span style="color:#888; font-size:0.75rem;">{adv['description']}</span>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.85rem;'>No dominant advantages — this is a very close matchup.</p>", unsafe_allow_html=True)

with reason_col2:
    st.markdown(
        f"<p style='color:#E0E0E0; font-weight:700; font-size:0.85rem; "
        f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>"
        f"📊 Head-to-Head Stats</p>",
        unsafe_allow_html=True,
    )
    for stat in rsn["stats_comparison"]:
        w_val = stat["winner_val"]
        l_val = stat["loser_val"]
        higher_better = stat["higher_better"]
        winner_leads = (w_val > l_val) if higher_better else (w_val < l_val)
        diff = abs(w_val - l_val)

        w_color = winner_color if winner_leads else loser_color
        l_color = loser_color  if winner_leads else winner_color

        st.markdown(
            f"""<div style="display:flex; justify-content:space-between; align-items:center;
                           padding:0.4rem 0.7rem; border-bottom:1px solid {BORDER_COLOR};
                           font-size:0.83rem;">
                <span style="color:{w_color}; font-weight:700; min-width:60px;">
                    {w_val:g}{stat['unit']}
                </span>
                <span style="color:{TEXT_MUTED}; flex:1; text-align:center;">{stat['label']}</span>
                <span style="color:{l_color}; font-weight:700; min-width:60px; text-align:right;">
                    {l_val:g}{stat['unit']}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""<div style="display:flex; justify-content:space-between; padding:0.4rem 0.7rem;
                       font-size:0.72rem; color:{TEXT_MUTED};">
            <span style="color:{winner_color}; font-weight:700;">{rsn['winner'].split()[-1]}</span>
            <span></span>
            <span style="color:{loser_color}; font-weight:700; text-align:right;">{rsn['loser'].split()[-1]}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    if rsn["concerns"]:
        st.markdown(
            f"<p style='color:{TEXT_MUTED}; font-size:0.8rem; margin-top:0.8rem; font-weight:600;'>"
            f"⚠️ {rsn['winner'].split()[-1]} Risk Factors</p>",
            unsafe_allow_html=True,
        )
        for concern in rsn["concerns"][:3]:
            st.markdown(
                f"""<div style="background:#1e1a14; border:1px solid #3a2e1a;
                               border-radius:6px; padding:0.45rem 0.8rem; margin-bottom:0.4rem;">
                    <span style="color:#FFB347; font-size:0.8rem; font-weight:600;">{concern['label']}</span>
                    <span style="color:{TEXT_MUTED}; font-size:0.75rem;"> — {concern['description']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

st.divider()


# ---------------------------------------------------------------------------
# Tabs: Overview | Team Stats | Charts | Head-to-Head
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "  Overview  ", "  Team Stats  ", "  Charts  ", "  Head-to-Head  "
])


# ── Tab 1: Overview ─────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns(2)

    for col, data, name, color in [
        (col_left,  t1_data, team1_name, team1_color),
        (col_right, t2_data, team2_name, team2_color),
    ]:
        sa = data["season_avgs"]
        l10 = data["last10_avgs"]
        with col:
            st.markdown(section_header(name.split()[-1] + " — Season Overview", "📊"), unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PPG", f"{sa.get('PTS', 0):.1f}")
            m2.metric("Off Rtg", f"{sa.get('E_OFF_RATING', sa.get('OFF_RATING', 0)):.1f}")
            m3.metric("Def Rtg", f"{sa.get('E_DEF_RATING', sa.get('DEF_RATING', 0)):.1f}")
            m4.metric("Net Rtg", f"{sa.get('E_NET_RATING', sa.get('NET_RATING', 0)):.1f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("Last 10 Games", "🔥"), unsafe_allow_html=True)
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Win %", f"{l10.get('WIN_PCT_L10', 0)*100:.0f}%")
            m6.metric("PPG", f"{l10.get('PTS', 0):.1f}")
            m7.metric("FG%", f"{l10.get('FG_PCT', 0)*100:.1f}%")
            m8.metric("3P%", f"{l10.get('FG3_PCT', 0)*100:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("Schedule & Fatigue", "📅"), unsafe_allow_html=True)
            sched_html = (
                stat_badge("Rest Days", str(data["rest_days"]),
                           "green" if data["rest_days"] >= 2 else "red") +
                stat_badge("Games in 10d", str(data["games_in_10d"]),
                           "orange" if data["games_in_10d"] >= 4 else "blue") +
                stat_badge("AST", f"{l10.get('AST', 0):.1f}", "teal") +
                stat_badge("TOV", f"{l10.get('TOV', 0):.1f}",
                           "red" if l10.get("TOV", 0) > 14 else "green") +
                stat_badge("REB", f"{l10.get('REB', 0):.1f}", "purple")
            )
            st.markdown(sched_html, unsafe_allow_html=True)

    st.divider()

    # Factor edge table
    st.markdown(section_header("Factor-by-Factor Edge Analysis", "⚡"), unsafe_allow_html=True)
    factors = result["factor_scores"]
    rows = []
    for key, label in FACTOR_LABELS.items():
        score = factors.get(key, 50)
        opp_score = 100 - score
        edge = team1_name.split()[-1] if score > 50 else (team2_name.split()[-1] if score < 50 else "Even")
        margin = abs(score - 50) * 2  # 0–100
        rows.append({
            "Factor": label,
            f"{team1_name.split()[-1]} Score": f"{score:.1f}",
            f"{team2_name.split()[-1]} Score": f"{opp_score:.1f}",
            "Edge": edge,
            "Margin": f"{margin:.1f}",
        })
    df_factors = pd.DataFrame(rows)
    st.dataframe(
        df_factors,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Margin": st.column_config.ProgressColumn(
                "Edge Margin", min_value=0, max_value=100, format="%.1f"
            )
        },
    )


# ── Tab 2: Team Stats ────────────────────────────────────────────────────────
with tab2:
    col_l, col_r = st.columns(2)
    for col, data, name, color in [
        (col_l, t1_data, team1_name, team1_color),
        (col_r, t2_data, team2_name, team2_color),
    ]:
        sa = data["season_avgs"]
        l10 = data["last10_avgs"]
        gl = data["last10_log"]
        clutch = data.get("clutch", {})
        with col:
            st.markdown(f"<h3 style='color:{color};'>{name}</h3>", unsafe_allow_html=True)

            stats = {
                "Points per Game":        f"{sa.get('PTS', 0):.1f}",
                "FG%":                     f"{sa.get('FG_PCT', 0)*100:.1f}%",
                "3P%":                     f"{sa.get('FG3_PCT', 0)*100:.1f}%",
                "FT%":                     f"{sa.get('FT_PCT', 0)*100:.1f}%",
                "Assists per Game":        f"{sa.get('AST', 0):.1f}",
                "Rebounds per Game":       f"{sa.get('REB', 0):.1f}",
                "Turnovers per Game":      f"{sa.get('TOV', 0):.1f}",
                "Steals per Game":         f"{sa.get('STL', 0):.1f}",
                "Blocks per Game":         f"{sa.get('BLK', 0):.1f}",
                "Offensive Rating":        f"{sa.get('E_OFF_RATING', sa.get('OFF_RATING', 0)):.1f}",
                "Defensive Rating":        f"{sa.get('E_DEF_RATING', sa.get('DEF_RATING', 0)):.1f}",
                "Net Rating":              f"{sa.get('E_NET_RATING', sa.get('NET_RATING', 0)):.1f}",
                "Pace":                    f"{sa.get('PACE', sa.get('E_PACE', 0)):.1f}",
                "TS% (L10)":              f"{(l10.get('PTS', 0) / max(2*(l10.get('FGA', 1) + 0.44*l10.get('FTA', 0)), 1))*100:.1f}%",
                "Clutch +/-":             f"{clutch.get('CLUTCH_PLUS_MINUS', 'N/A')}",
            }

            for stat, val in stats.items():
                st.markdown(
                    f"""<div style="display:flex; justify-content:space-between; align-items:center;
                                   padding:0.45rem 0.8rem; border-bottom:1px solid {BORDER_COLOR};
                                   font-size:0.88rem;">
                        <span style="color:{TEXT_MUTED};">{stat}</span>
                        <span style="color:{color}; font-weight:700;">{val}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            if not gl.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(section_header("Recent Game Log", "📋"), unsafe_allow_html=True)
                display_cols = ["GAME_DATE", "MATCHUP", "WL", "PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "PLUS_MINUS"]
                available = [c for c in display_cols if c in gl.columns]
                log_display = gl[available].copy()
                if "GAME_DATE" in log_display.columns:
                    log_display["GAME_DATE"] = log_display["GAME_DATE"].dt.strftime("%b %d")
                if "FG_PCT" in log_display.columns:
                    log_display["FG_PCT"] = (log_display["FG_PCT"] * 100).round(1).astype(str) + "%"
                if "FG3_PCT" in log_display.columns:
                    log_display["FG3_PCT"] = (log_display["FG3_PCT"] * 100).round(1).astype(str) + "%"
                st.dataframe(log_display, use_container_width=True, hide_index=True)


# ── Tab 3: Charts ────────────────────────────────────────────────────────────
with tab3:
    st.plotly_chart(
        factor_radar(result["factor_scores"], team1_name, team2_name, team1_color, team2_color),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        factor_bar_comparison(result["factor_scores"], team1_name, team2_name, team1_color, team2_color),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    c1, c2 = st.columns(2)
    with c1:
        if not t1_data["last10_log"].empty:
            st.plotly_chart(
                game_log_sparkline(t1_data["last10_log"], team1_name, team1_color),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.plotly_chart(
                shooting_breakdown(t1_data["last10_avgs"], team1_name, team1_color),
                use_container_width=True, config={"displayModeBar": False},
            )
    with c2:
        if not t2_data["last10_log"].empty:
            st.plotly_chart(
                game_log_sparkline(t2_data["last10_log"], team2_name, team2_color),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.plotly_chart(
                shooting_breakdown(t2_data["last10_avgs"], team2_name, team2_color),
                use_container_width=True, config={"displayModeBar": False},
            )


# ── Tab 4: Head-to-Head ──────────────────────────────────────────────────────
with tab4:
    st.markdown(section_header(f"{team1_name} vs {team2_name} — This Season", "🔄"), unsafe_allow_html=True)
    with st.spinner("Loading head-to-head data…"):
        try:
            h2h = get_head_to_head(t1_data["team_id"], t2_data["team_id"], season=CURRENT_SEASON)
            if h2h.empty:
                st.info("No head-to-head games found for the current season yet.")

            if not h2h.empty:
                wins  = (h2h["WL"] == "W").sum()
                losses = (h2h["WL"] == "L").sum()
                m1, m2, m3 = st.columns(3)
                m1.metric(f"{team1_name.split()[-1]} Wins", wins)
                m2.metric(f"{team2_name.split()[-1]} Wins", losses)
                m3.metric("Games Played", len(h2h))

                display_cols = ["GAME_DATE", "MATCHUP", "WL", "PTS", "REB", "AST", "PLUS_MINUS"]
                available = [c for c in display_cols if c in h2h.columns]
                st.dataframe(h2h[available].head(10), use_container_width=True, hide_index=True)
            else:
                st.info("No matchup data found for this season.")
        except Exception as e:
            st.warning(f"Could not load head-to-head data: {e}")

    st.divider()
    st.markdown(section_header("Prediction Summary", "🎯"), unsafe_allow_html=True)
    winner_color = team1_color if result["predicted_winner"] == team1_name else team2_color

    st.markdown(
        card(f"""
            <div style="text-align:center; padding:1rem 0;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">🏆</div>
                <h2 style="color:{winner_color}; margin:0; font-size:1.6rem;">
                    {result['predicted_winner']}
                </h2>
                <p style="color:{TEXT_MUTED}; margin:0.3rem 0;">Predicted to win</p>
                <div style="margin-top:0.8rem; display:flex; justify-content:center; gap:2rem;">
                    <div>
                        <div style="font-size:1.8rem; font-weight:800; color:{team1_color};">
                            {result['team_win_prob']*100:.1f}%
                        </div>
                        <div style="color:{TEXT_MUTED}; font-size:0.78rem;">{team1_name}</div>
                    </div>
                    <div style="color:{TEXT_MUTED}; font-size:1.5rem; padding-top:0.2rem;">—</div>
                    <div>
                        <div style="font-size:1.8rem; font-weight:800; color:{team2_color};">
                            {result['opp_win_prob']*100:.1f}%
                        </div>
                        <div style="color:{TEXT_MUTED}; font-size:0.78rem;">{team2_name}</div>
                    </div>
                </div>
                <div style="margin-top:0.8rem; padding:0.5rem 1.5rem;
                            background:#1a1a1a; border-radius:8px; display:inline-block;">
                    <span style="color:{TEXT_MUTED}; font-size:0.8rem;">Model Confidence: </span>
                    <span style="color:{ACCENT_COLORS['green'] if 'High' in result['confidence'] else ACCENT_COLORS['orange']};
                                 font-weight:700;">
                        {result['confidence']}
                    </span>
                </div>
            </div>
        """),
        unsafe_allow_html=True,
    )
