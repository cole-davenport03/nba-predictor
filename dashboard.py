"""Momentum AI — model predictions dashboard.

A polished dark-theme Streamlit dashboard that surfaces predictions from both
trained models:
  • player_stats        — tabular transformer that predicts PTS from 11 features
  • random_masking      — BERT-style transformer that reconstructs ANY masked column

Colour rules (configurable):
  • PTS                                  green when |err| ≤ 3
  • FG_PCT / FG3_PCT / FT_PCT            green when |err| ≤ 0.15
  • Other counting stats                 green when |err| ≤ 1
  otherwise red.

Run with:  .venv/bin/streamlit run dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_data import (
    get_player_stats_results,
    get_random_masking_results,
    list_players,
    list_dates_for,
    retrain_player_stats,
    retrain_random_masking,
    DEFAULT_PLAYER,
    DEFAULT_DATES,
    DEFAULT_BONUS_COLS,
)


# --- Page setup -------------------------------------------------------------
st.set_page_config(
    page_title="Momentum AI · Model Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent
LOGO_PATH = ROOT / "assets" / "momentum_ai.png"


# --- Palette ----------------------------------------------------------------
BG          = "#070A0F"
PANEL       = "#0E141C"
PANEL_HI    = "#141C26"
BORDER      = "#1F2A38"
BORDER_HI   = "#2D3E55"
TEXT        = "#E8EEF7"
TEXT_MUTED  = "#7C8AA0"
TEXT_DIM    = "#566275"
ACCENT      = "#3B9BFF"      # momentum blue
ACCENT_GLOW = "#54B6FF"
GREEN       = "#22E58B"
GREEN_DIM   = "#0B4F2E"
RED         = "#FF5C77"
RED_DIM     = "#4A1623"


# --- HTML helper ------------------------------------------------------------
def _h(html_str: str) -> str:
    """Collapse multi-line indented HTML to one line so st.markdown won't
    treat indented blocks (4+ spaces after a blank line) as code blocks."""
    return "".join(line.strip() for line in html_str.splitlines() if line.strip())


# --- Color rule -------------------------------------------------------------
def is_green(stat: str, error: float) -> bool:
    """Return True if the prediction is "close enough" to count as a hit."""
    abs_err = abs(error)
    if stat == "PTS":
        return abs_err <= 3
    if stat in ("FG_PCT", "FG3_PCT", "FT_PCT"):
        return abs_err <= 0.15
    if stat in ("MIN", "REB"):
        return abs_err <= 2
    return abs_err <= 1  # AST, STL, BLK, TOV, PF, etc.


def color_for(stat: str, error: float) -> str:
    return GREEN if is_green(stat, error) else RED


# --- Global CSS -------------------------------------------------------------
GLOBAL_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1400px;
    }}

    /* Subtle radial gradient backdrop */
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        inset: 0;
        background:
            radial-gradient(circle at 15% 0%, rgba(59, 155, 255, 0.08), transparent 40%),
            radial-gradient(circle at 85% 100%, rgba(34, 229, 139, 0.05), transparent 45%);
        pointer-events: none;
        z-index: 0;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT} !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.01em;
        font-weight: 700;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer, [data-testid="stStatusWidget"] {{ visibility: hidden !important; }}

    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 6px;
        background: transparent;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 0;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        background: transparent;
        color: {TEXT_MUTED};
        padding: 10px 22px;
        border: none;
        border-bottom: 2px solid transparent;
        font-weight: 600;
        font-size: 0.92rem;
        letter-spacing: 0.02em;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: {TEXT} !important;
        border-bottom: 2px solid {ACCENT} !important;
        background: transparent !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-panel"] {{ padding-top: 1.5rem; }}

    .stDataFrame {{ border-radius: 12px; overflow: hidden; border: 1px solid {BORDER}; }}

    div[data-testid="stMarkdownContainer"] code {{
        background: {PANEL};
        color: {ACCENT_GLOW};
        padding: 2px 7px;
        border-radius: 5px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85em;
    }}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# --- Cached data load -------------------------------------------------------
# Cache keyed on (player, tuple-of-dates, bonus_date) — switching selection
# triggers a fresh compute; switching back returns the cached result.
@st.cache_data(show_spinner=False)
def _cached_player_stats(player: str, dates_tuple: tuple):
    return get_player_stats_results(player=player, dates=list(dates_tuple))


@st.cache_data(show_spinner=False)
def _cached_random_masking(player: str, dates_tuple: tuple, bonus_date: str):
    return get_random_masking_results(
        player=player, dates=list(dates_tuple), bonus_date=bonus_date
    )


@st.cache_data(show_spinner=False)
def _cached_player_list():
    return list_players()


@st.cache_data(show_spinner=False)
def _cached_dates_for(player: str):
    return list_dates_for(player)


def load_all_results(player: str, dates: list[str], bonus_date: str) -> dict:
    return {
        "player_stats": _cached_player_stats(player, tuple(dates)),
        "random_masking": _cached_random_masking(player, tuple(dates), bonus_date),
    }


# --- Header -----------------------------------------------------------------
def render_header() -> None:
    col_logo, col_title = st.columns([1, 6], gap="medium")
    with col_logo:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=120)
        else:
            st.markdown(
                f"<div style='width:120px; height:120px; border:1px dashed {BORDER_HI};"
                f"border-radius:14px; display:flex; align-items:center; justify-content:center;"
                f"color:{TEXT_DIM}; font-size:0.7rem; text-align:center;'>save logo to<br>assets/momentum_ai.png</div>",
                unsafe_allow_html=True,
            )
    with col_title:
        st.markdown(
            f"""
            <div style="padding-top: 0.4rem;">
              <div style="font-size:0.75rem; letter-spacing:0.24em; color:{ACCENT};
                          text-transform:uppercase; font-weight:700; margin-bottom:6px;">
                Momentum AI · Sandbox
              </div>
              <h1 style="font-size:2.6rem; margin:0; font-weight:800; line-height:1.1;
                         background: linear-gradient(135deg, {TEXT} 0%, {ACCENT_GLOW} 100%);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                NBA Predictor Dashboard
              </h1>
              <p style="color:{TEXT_MUTED}; font-size:0.95rem; margin-top:6px; max-width:780px;">
                Tabular transformers trained on player box-score data. PyTorch · self-attention
                over feature tokens · BERT-style masked reconstruction.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div style='height:1px; background:linear-gradient(90deg, transparent, {BORDER_HI}, transparent);"
        f"margin: 1.8rem 0 1.6rem 0;'></div>",
        unsafe_allow_html=True,
    )


# --- Reusable components ----------------------------------------------------
def kpi_card(label: str, value: str, sub: str = "", accent: str = ACCENT) -> str:
    sub_html = f'<div style="color:{TEXT_MUTED}; font-size:0.78rem; margin-top:6px;">{sub}</div>' if sub else ''
    return _h(f"""
    <div style="background: linear-gradient(180deg, {PANEL} 0%, {BG} 120%);
                border:1px solid {BORDER}; border-radius:14px; padding:1.1rem 1.3rem;
                position:relative; overflow:hidden; height:100%;">
      <div style="position:absolute; top:0; left:0; right:0; height:2px;
                  background:linear-gradient(90deg, {accent}, transparent);"></div>
      <div style="color:{TEXT_DIM}; font-size:0.7rem; letter-spacing:0.16em;
                  text-transform:uppercase; font-weight:600; margin-bottom:8px;">{label}</div>
      <div style="font-family:'JetBrains Mono', monospace; font-size:2rem;
                  color:{TEXT}; font-weight:700; line-height:1;">{value}</div>
      {sub_html}
    </div>
    """)


def split_pill(split: str) -> str:
    if split == "test":
        bg, fg = "#0B3D5C", ACCENT_GLOW
        label = "TEST SPLIT"
    else:
        bg, fg = "#3A2D14", "#FFB347"
        label = "TRAIN (SEEN)"
    return (
        f'<span style="background:{bg}; color:{fg}; padding:2px 9px; border-radius:6px;'
        f'font-size:0.65rem; font-weight:700; letter-spacing:0.12em;">{label}</span>'
    )


def format_stat(stat: str, value: float) -> str:
    return f"{value:.3f}" if stat.endswith("_PCT") else f"{value:.2f}"


def prediction_card(g: dict) -> str:
    """A game card showing predicted vs actual side by side."""
    stat = g["stat"]
    pred_color = color_for(stat, g["error"])
    is_hit = is_green(stat, g["error"])
    status_label = "HIT" if is_hit else "MISS"
    accent = GREEN if is_hit else RED
    err_str = f"{g['error']:+.3f}" if stat.endswith("_PCT") else f"{g['error']:+.2f}"

    return _h(f"""
    <div style="background:linear-gradient(180deg, {PANEL} 0%, {PANEL_HI} 100%);
                border:1px solid {BORDER}; border-radius:14px; padding:1.2rem 1.3rem;
                position:relative; overflow:hidden; height:100%;
                transition: transform 0.15s ease, border-color 0.15s ease;">
      <div style="position:absolute; top:0; left:0; right:0; height:2px;
                  background:linear-gradient(90deg, {accent}, transparent);"></div>
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
          <div style="color:{TEXT_DIM}; font-size:0.7rem; letter-spacing:0.16em;
                      text-transform:uppercase; font-weight:600;">
            {g['date']} &nbsp;·&nbsp; {g.get('matchup', '')}
          </div>
          <div style="color:{TEXT}; font-size:1.05rem; font-weight:700; margin-top:4px;">
            Masked stat: <span style="color:{ACCENT_GLOW}">{stat}</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="background:{accent}22; color:{accent}; padding:3px 10px;
                      border-radius:6px; font-size:0.7rem; font-weight:700;
                      letter-spacing:0.14em; display:inline-block;">{status_label}</div>
          <div style="margin-top:6px;">{split_pill(g['split'])}</div>
        </div>
      </div>
      <div style="display:flex; align-items:flex-end; gap:1.6rem; margin-top:1rem;">
        <div style="flex:1;">
          <div style="color:{TEXT_DIM}; font-size:0.68rem; letter-spacing:0.14em;
                      text-transform:uppercase;">Predicted</div>
          <div style="font-family:'JetBrains Mono', monospace; font-size:2.1rem;
                      color:{pred_color}; font-weight:700; line-height:1; margin-top:6px;">
            {format_stat(stat, g['predicted'])}
          </div>
        </div>
        <div style="color:{TEXT_DIM}; font-size:1.4rem; padding-bottom:6px;">→</div>
        <div style="flex:1; text-align:right;">
          <div style="color:{TEXT_DIM}; font-size:0.68rem; letter-spacing:0.14em;
                      text-transform:uppercase;">Actual</div>
          <div style="font-family:'JetBrains Mono', monospace; font-size:2.1rem;
                      color:{TEXT}; font-weight:700; line-height:1; margin-top:6px;">
            {format_stat(stat, g['actual'])}
          </div>
        </div>
      </div>
      <div style="margin-top:1rem; padding-top:0.7rem; border-top:1px solid {BORDER};
                  display:flex; justify-content:space-between; font-size:0.78rem;">
        <span style="color:{TEXT_MUTED};">Signed error</span>
        <span style="font-family:'JetBrains Mono', monospace; color:{pred_color};
                     font-weight:600;">{err_str}</span>
      </div>
    </div>
    """)


def prediction_card_compact(g: dict) -> str:
    """Narrow vertical variant — pred stacked above actual. Designed to fit
    inside the 5-across "different masks" grid without overflowing."""
    stat = g["stat"]
    pred_color = color_for(stat, g["error"])
    is_hit = is_green(stat, g["error"])
    status_label = "HIT" if is_hit else "MISS"
    accent = GREEN if is_hit else RED
    err_str = f"{g['error']:+.3f}" if stat.endswith("_PCT") else f"{g['error']:+.2f}"

    return _h(f"""
    <div style="background:linear-gradient(180deg, {PANEL} 0%, {PANEL_HI} 100%);
                border:1px solid {BORDER}; border-radius:12px; padding:0.9rem 0.9rem;
                position:relative; overflow:hidden; height:100%;">
      <div style="position:absolute; top:0; left:0; right:0; height:2px;
                  background:linear-gradient(90deg, {accent}, transparent);"></div>
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
        <div style="color:{ACCENT_GLOW}; font-family:'JetBrains Mono', monospace;
                    font-size:0.82rem; font-weight:700; overflow:hidden; text-overflow:ellipsis;
                    white-space:nowrap;">{stat}</div>
        <div style="background:{accent}22; color:{accent}; padding:2px 7px;
                    border-radius:5px; font-size:0.6rem; font-weight:700;
                    letter-spacing:0.12em; flex-shrink:0;">{status_label}</div>
      </div>
      <div style="margin-top:0.7rem;">
        <div style="color:{TEXT_DIM}; font-size:0.6rem; letter-spacing:0.12em;
                    text-transform:uppercase;">Predicted</div>
        <div style="font-family:'JetBrains Mono', monospace; font-size:1.35rem;
                    color:{pred_color}; font-weight:700; line-height:1.1; margin-top:2px;
                    word-break:break-all; overflow-wrap:anywhere;">
          {format_stat(stat, g['predicted'])}
        </div>
      </div>
      <div style="margin-top:0.55rem;">
        <div style="color:{TEXT_DIM}; font-size:0.6rem; letter-spacing:0.12em;
                    text-transform:uppercase;">Actual</div>
        <div style="font-family:'JetBrains Mono', monospace; font-size:1.35rem;
                    color:{TEXT}; font-weight:700; line-height:1.1; margin-top:2px;
                    word-break:break-all; overflow-wrap:anywhere;">
          {format_stat(stat, g['actual'])}
        </div>
      </div>
      <div style="margin-top:0.7rem; padding-top:0.5rem; border-top:1px solid {BORDER};
                  display:flex; justify-content:space-between; font-size:0.7rem;">
        <span style="color:{TEXT_MUTED};">err</span>
        <span style="font-family:'JetBrains Mono', monospace; color:{pred_color};
                     font-weight:600;">{err_str}</span>
      </div>
    </div>
    """)


def section_title(eyebrow: str, title: str, desc: str = "") -> str:
    desc_html = f'<p style="color:{TEXT_MUTED}; margin:6px 0 0 0; font-size:0.9rem;">{desc}</p>' if desc else ''
    return _h(f"""
    <div style="margin: 0.5rem 0 1.3rem 0;">
      <div style="color:{ACCENT}; font-size:0.72rem; letter-spacing:0.22em;
                  text-transform:uppercase; font-weight:700; margin-bottom:5px;">{eyebrow}</div>
      <h2 style="margin:0; font-size:1.6rem; font-weight:700;">{title}</h2>
      {desc_html}
    </div>
    """)


# --- Charts -----------------------------------------------------------------
def error_chart(games, title: str) -> go.Figure:
    """Horizontal bar chart of signed errors per game, coloured by hit/miss."""
    games = list(games)
    fig = go.Figure()
    labels = [f"{g['date']}  ·  {g.get('matchup', '')}" for g in games]
    errors = [g["error"] for g in games]
    colors = [color_for(g["stat"], g["error"]) for g in games]

    fig.add_trace(go.Bar(
        x=errors,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>error = %{x:+.2f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color=BORDER_HI, line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=14), x=0.0, xanchor="left"),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        height=max(180, 70 + 55 * len(games)),
        margin=dict(l=10, r=20, t=40, b=30),
        font=dict(color=TEXT_MUTED, family="Inter"),
        xaxis=dict(
            title="signed error (pred − actual)",
            gridcolor=BORDER, zerolinecolor=BORDER_HI,
            tickfont=dict(color=TEXT_MUTED, size=11),
            title_font=dict(color=TEXT_DIM, size=11),
        ),
        yaxis=dict(
            gridcolor=BORDER, tickfont=dict(color=TEXT, size=11),
            autorange="reversed",
        ),
        showlegend=False,
    )
    return fig


# --- Build the page ---------------------------------------------------------
render_header()


# --- Controls ---------------------------------------------------------------
# Pick a player, then the games to predict, then a "Run Prediction" button.
# Selection is stored in session_state so the cache key is stable across reruns.
players = _cached_player_list()
if DEFAULT_PLAYER in players:
    default_player_idx = players.index(DEFAULT_PLAYER)
else:
    default_player_idx = 0

st.markdown(
    section_title("Controls", "Choose a matchup",
                  "Pick any player in the dataset and which game dates to predict. "
                  "Hit Run Prediction to recompute."),
    unsafe_allow_html=True,
)

ctrl_cols = st.columns([1.2, 2.4, 0.9, 0.9], gap="medium")
with ctrl_cols[0]:
    chosen_player = st.selectbox("Player", players, index=default_player_idx, key="player_sel")

dates_for_player = _cached_dates_for(chosen_player)
default_picks = [d for d in DEFAULT_DATES if d in dates_for_player]
if not default_picks:
    default_picks = dates_for_player[:4]

with ctrl_cols[1]:
    chosen_dates = st.multiselect(
        "Game dates (pick 1–8)", dates_for_player,
        default=default_picks, key="dates_sel",
        max_selections=8,
    )

with ctrl_cols[2]:
    bonus_options = chosen_dates if chosen_dates else dates_for_player[:1]
    chosen_bonus = st.selectbox(
        "Bonus-mask game", bonus_options,
        index=len(bonus_options) - 1 if bonus_options else 0,
        key="bonus_sel",
        help="Which selected game to show under \"different masked columns\".",
    )

with ctrl_cols[3]:
    st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run Prediction", type="primary", width="stretch")

# If user just clicked Run, clear cached results so we re-execute.
if run_clicked:
    _cached_player_stats.clear()
    _cached_random_masking.clear()


# --- Re-train models --------------------------------------------------------
with st.expander("Re-train models (slow — random init each time)", expanded=False):
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin:-6px 0 12px 0;'>"
        f"Runs the same 500-epoch loop as <code>player_stats/train.py</code> and "
        f"<code>random_masking/train.py</code>. Because there's no fixed seed in the "
        f"training scripts, every run produces different weights → different predictions. "
        f"Overwrites <code>trained_model.pt</code> in each folder.</p>",
        unsafe_allow_html=True,
    )
    rt_cols = st.columns([1, 1, 2])
    with rt_cols[0]:
        retrain_ps_clicked = st.button("Re-train player_stats", width="stretch")
    with rt_cols[1]:
        retrain_rm_clicked = st.button("Re-train random_masking", width="stretch")
    with rt_cols[2]:
        retrain_both_clicked = st.button("Re-train both", type="primary", width="stretch")

    do_ps = retrain_ps_clicked or retrain_both_clicked
    do_rm = retrain_rm_clicked or retrain_both_clicked

    if do_ps or do_rm:
        # Clear prediction caches so the next render reflects the new weights.
        _cached_player_stats.clear()
        _cached_random_masking.clear()

        if do_ps:
            with st.status("Re-training player_stats…", expanded=True) as status:
                progress_bar = st.progress(0.0)
                log_lines: list[str] = []
                log_slot = st.empty()

                def ps_cb(ev: dict) -> None:
                    progress_bar.progress((ev["epoch"] + 1) / ev["epochs"])
                    star = " *" if ev["is_best"] else ""
                    log_lines.append(
                        f"epoch {ev['epoch']:3d}  train_loss={ev['train_loss']:.4f}  "
                        f"test MAE={ev['test_mae']:.2f}{star}"
                    )
                    log_slot.code("\n".join(log_lines[-12:]), language="text")

                result = retrain_player_stats(progress_cb=ps_cb)
                progress_bar.progress(1.0)
                status.update(
                    label=f"player_stats done — best MAE {result['best_mae']:.2f} "
                          f"at epoch {result['best_epoch']}",
                    state="complete",
                )

        if do_rm:
            with st.status("Re-training random_masking…", expanded=True) as status:
                progress_bar = st.progress(0.0)
                log_lines = []
                log_slot = st.empty()

                def rm_cb(ev: dict) -> None:
                    progress_bar.progress((ev["epoch"] + 1) / ev["epochs"])
                    star = " *" if ev["is_best"] else ""
                    log_lines.append(
                        f"epoch {ev['epoch']:3d}  train_loss={ev['train_loss']:.4f}  "
                        f"test PTS MAE={ev['test_mae']:.2f}{star}"
                    )
                    log_slot.code("\n".join(log_lines[-12:]), language="text")

                result = retrain_random_masking(progress_cb=rm_cb)
                progress_bar.progress(1.0)
                status.update(
                    label=f"random_masking done — best PTS MAE {result['best_mae']:.2f} "
                          f"at epoch {result['best_epoch']}",
                    state="complete",
                )

        st.success("Done. The dashboard below now uses the new weights.")


# --- Compute / load results -------------------------------------------------
if not chosen_dates:
    st.info("Pick at least one game date above to run the prediction.")
    st.stop()

with st.spinner("Running predictions…"):
    results = load_all_results(chosen_player, chosen_dates, chosen_bonus)

ps = results["player_stats"]
rm = results["random_masking"]

player_last = chosen_player.split()[-1]


# Top KPI strip
st.markdown(section_title("Overview", "Model performance at a glance",
                          f"Test-set MAE plus how the selected {player_last} games landed."),
            unsafe_allow_html=True)

kpi_cols = st.columns(4, gap="medium")
ps_hits = sum(1 for g in ps["games"] if is_green("PTS", g["error"]))
rm_hits = sum(1 for g in rm["pts_games"] if is_green("PTS", g["error"]))
kpi_cols[0].markdown(kpi_card("player_stats · test MAE", f"{ps['overall_mae']:.2f}",
                              "points off, on average", ACCENT), unsafe_allow_html=True)
kpi_cols[1].markdown(kpi_card(f"player_stats · {player_last} hits",
                              f"{ps_hits} / {max(len(ps['games']), 1)}",
                              "games within ±3 PTS",
                              GREEN if ps_hits >= max(len(ps['games']), 1) / 2 else RED),
                     unsafe_allow_html=True)
kpi_cols[2].markdown(kpi_card("random_masking · PTS MAE", f"{rm['overall_mae']:.2f}",
                              "BERT-style reconstruction", ACCENT), unsafe_allow_html=True)
kpi_cols[3].markdown(kpi_card(f"random_masking · {player_last} hits",
                              f"{rm_hits} / {max(len(rm['pts_games']), 1)}",
                              "games within ±3 PTS",
                              GREEN if rm_hits >= max(len(rm['pts_games']), 1) / 2 else RED),
                     unsafe_allow_html=True)


st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)


# Tabs
tab_ps, tab_rm = st.tabs(["  Player Stats Model  ", "  Random Masking Model  "])


# ── Tab 1: player_stats ────────────────────────────────────────────────────
with tab_ps:
    st.markdown(
        section_title(
            "Model · player_stats",
            "Tabular transformer · PTS regression",
            "11 box-score features tokenised into a learned embedding space, a "
            "[CLS] token attends across them, and a linear head outputs predicted points.",
        ),
        unsafe_allow_html=True,
    )

    cards = st.columns(2, gap="medium")
    for i, g in enumerate(ps["games"]):
        g_with_stat = {**g, "stat": "PTS"}
        with cards[i % 2]:
            st.markdown(prediction_card(g_with_stat), unsafe_allow_html=True)
            st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    chart_games = [{**g, "stat": "PTS"} for g in ps["games"]]
    st.plotly_chart(
        error_chart(chart_games, "Signed error per game (pred − actual)"),
        width="stretch",
        config={"displayModeBar": False},
    )

    # Feature inspection table
    st.markdown(
        section_title("Inputs", "Raw input features the model saw",
                      "Standardised before the transformer; shown here unscaled for readability."),
        unsafe_allow_html=True,
    )
    feature_rows = []
    for g in ps["games"]:
        row = {"date": g["date"], "matchup": g.get("matchup", ""), **g["features"]}
        feature_rows.append(row)
    df_features = pd.DataFrame(feature_rows)
    st.dataframe(df_features, width="stretch", hide_index=True)


# ── Tab 2: random_masking ──────────────────────────────────────────────────
with tab_rm:
    st.markdown(
        section_title(
            "Model · random_masking",
            "BERT-style masked-column reconstruction",
            "Same tabular-transformer backbone, but during training a random column is "
            "replaced with a learned [MASK] token. The model learns to recover any "
            "stat from the remaining ones — one model, eleven prediction heads.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; margin:-0.6rem 0 1rem 0;'>"
        f"Below: <b style='color:{TEXT};'>PTS</b> reconstructed for the selected "
        f"{player_last} games.</div>",
        unsafe_allow_html=True,
    )
    cards = st.columns(2, gap="medium")
    for i, g in enumerate(rm["pts_games"]):
        with cards[i % 2]:
            st.markdown(prediction_card(g), unsafe_allow_html=True)
            st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

    st.plotly_chart(
        error_chart(rm["pts_games"], "PTS reconstruction error per game"),
        width="stretch",
        config={"displayModeBar": False},
    )

    # Bonus: different masked columns on the same game
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    bonus_date = rm["bonus_games"][0]["date"] if rm["bonus_games"] else ""
    bonus_matchup = rm["bonus_games"][0].get("matchup", "") if rm["bonus_games"] else ""
    st.markdown(
        section_title(
            "Same game · different masks",
            f"Hiding each stat one at a time · {bonus_date} {bonus_matchup}",
            "Demonstrates the same trained model reconstructing different columns of "
            "the same row. PTS ±3, MIN ±2, REB ±2, FG% ±0.15, everything else ±1.",
        ),
        unsafe_allow_html=True,
    )
    if rm["bonus_games"]:
        bonus_cols = st.columns(len(rm["bonus_games"]), gap="small")
        for i, g in enumerate(rm["bonus_games"]):
            with bonus_cols[i]:
                st.markdown(prediction_card_compact(g), unsafe_allow_html=True)
    else:
        st.info("No bonus-mask game available for the current selection.")


# Footer
st.markdown(
    f"<div style='margin-top:3rem; padding-top:1.4rem; border-top:1px solid {BORDER};"
    f"text-align:center; color:{TEXT_DIM}; font-size:0.75rem; letter-spacing:0.1em;'>"
    f"NBA PREDICTOR · MOMENTUM AI SANDBOX</div>",
    unsafe_allow_html=True,
)
