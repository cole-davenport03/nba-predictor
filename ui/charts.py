"""
Plotly chart builders with dark theme and vivid colors.
"""

import plotly.graph_objects as go
import pandas as pd
from ui.styles import CARD_BG, CARD_BG2, BORDER_COLOR, TEXT_PRIMARY, TEXT_MUTED, ACCENT_COLORS

# Base layout — deliberately excludes xaxis, yaxis, and margin so each chart
# can pass those freely without "multiple values for keyword argument" errors.
_LAYOUT_BASE = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="Segoe UI, sans-serif", color=TEXT_PRIMARY, size=13),
)

# Reusable defaults referenced explicitly per chart
_MARGIN   = dict(l=10, r=10, t=40, b=10)
_MARGIN_0 = dict(l=0,  r=0,  t=0,  b=0)
_XAXIS    = dict(gridcolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR, tickfont=dict(color=TEXT_MUTED))
_YAXIS    = dict(gridcolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR, tickfont=dict(color=TEXT_MUTED))


def _rgba(hex_color: str, alpha: float = 0.18) -> str:
    """Convert a 6-digit hex color to rgba() string. Plotly rejects 8-digit hex."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color  # pass through if already in another format
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def win_probability_gauge(team_name: str, opp_name: str, team_prob: float, opp_prob: float,
                           team_color: str = "#78BE20", opp_color: str = "#C4CED4") -> go.Figure:
    """Split horizontal bar showing win probability split."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[team_prob * 100], y=[""], orientation="h",
        marker_color=team_color, name=team_name,
        text=f"  {team_name}  {team_prob*100:.1f}%",
        textposition="inside",
        textfont=dict(size=15, color="#000", family="Segoe UI, sans-serif"),
        hovertemplate=f"{team_name}: {team_prob*100:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[opp_prob * 100], y=[""], orientation="h",
        marker_color=opp_color, name=opp_name,
        text=f"  {opp_prob*100:.1f}%  {opp_name}  ",
        textposition="inside",
        textfont=dict(size=15, color="#000" if opp_color != "#C4CED4" else "#111",
                      family="Segoe UI, sans-serif"),
        hovertemplate=f"{opp_name}: {opp_prob*100:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        margin=_MARGIN_0,
        barmode="stack",
        height=90,
        showlegend=False,
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def factor_radar(factor_scores: dict, team_name: str, opp_name: str,
                 team_color: str = "#78BE20", opp_color: str = "#C4CED4") -> go.Figure:
    """Radar chart showing factor scores for both teams."""
    labels = {
        "net_rating":         "Net Rating",
        "off_rating_l10":     "Offense L10",
        "def_rating_l10":     "Defense L10",
        "win_pct_l10":        "Win % L10",
        "true_shooting_l10":  "True Shot%",
        "rest_advantage":     "Rest",
        "pace_adjusted_pts":  "Pace Pts",
        "turnover_rate":      "Ball Security",
        "rebound_rate":       "Rebounding",
        "clutch_plus_minus":  "Clutch",
        "assist_to_tov":      "AST/TOV",
        "bench_contribution": "Bench",
    }
    cats      = list(labels.values())
    team_vals = [factor_scores.get(k, 50) for k in labels]
    opp_vals  = [100 - v for v in team_vals]

    cats_closed  = cats       + [cats[0]]
    team_closed  = team_vals  + [team_vals[0]]
    opp_closed   = opp_vals   + [opp_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=team_closed, theta=cats_closed, fill="toself", name=team_name,
        line=dict(color=team_color, width=2), fillcolor=_rgba(team_color, 0.18),
    ))
    fig.add_trace(go.Scatterpolar(
        r=opp_closed, theta=cats_closed, fill="toself", name=opp_name,
        line=dict(color=opp_color, width=2), fillcolor=_rgba(opp_color, 0.18),
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        margin=_MARGIN,
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(color=TEXT_MUTED, size=9),
                            gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR),
            angularaxis=dict(tickfont=dict(color=TEXT_PRIMARY, size=11),
                             gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR),
        ),
        showlegend=True,
        legend=dict(font=dict(color=TEXT_PRIMARY), bgcolor=CARD_BG2,
                    bordercolor=BORDER_COLOR, borderwidth=1),
        height=420,
        title=dict(text="Factor Advantage Radar", font=dict(size=14, color=TEXT_PRIMARY)),
    )
    return fig


def factor_bar_comparison(factor_scores: dict, team_name: str, opp_name: str,
                           team_color: str = "#78BE20", opp_color: str = "#C4CED4") -> go.Figure:
    """Horizontal diverging bar chart comparing each factor."""
    labels = {
        "net_rating":         "Net Rating",
        "off_rating_l10":     "Offense (L10)",
        "def_rating_l10":     "Defense (L10)",
        "win_pct_l10":        "Win % (L10)",
        "true_shooting_l10":  "True Shooting %",
        "rest_advantage":     "Rest Advantage",
        "pace_adjusted_pts":  "Pace-Adj Points",
        "turnover_rate":      "Ball Security",
        "rebound_rate":       "Rebounding",
        "clutch_plus_minus":  "Clutch Perf.",
        "assist_to_tov":      "AST/TOV Ratio",
        "bench_contribution": "Bench Depth",
    }
    cats      = list(labels.values())
    team_vals = [factor_scores.get(k, 50) for k in labels]
    opp_vals  = [100 - v for v in team_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cats, x=team_vals, orientation="h",
        name=team_name, marker_color=team_color,
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=cats, x=[-v for v in opp_vals], orientation="h",
        name=opp_name, marker_color=opp_color,
        hovertemplate="%{y}: %{customdata:.1f}<extra></extra>",
        customdata=opp_vals,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        margin=_MARGIN,
        barmode="overlay",
        height=480,
        xaxis=dict(range=[-100, 100],
                   tickvals=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
                   ticktext=["100", "75", "50", "25", "0", "25", "50", "75", "100"],
                   gridcolor=BORDER_COLOR, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(tickfont=dict(color=TEXT_PRIMARY, size=12), gridcolor=BORDER_COLOR),
        showlegend=True,
        legend=dict(font=dict(color=TEXT_PRIMARY), bgcolor=CARD_BG2,
                    bordercolor=BORDER_COLOR, borderwidth=1),
        title=dict(text="Head-to-Head Factor Comparison", font=dict(size=14, color=TEXT_PRIMARY)),
        shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(cats) - 0.5,
                     line=dict(color="#444", width=1, dash="dot"))],
    )
    return fig


def game_log_sparkline(game_log: pd.DataFrame, team_name: str, color: str = "#78BE20") -> go.Figure:
    """Points scored per game with win/loss markers."""
    df     = game_log.copy().sort_values("GAME_DATE")
    wins   = df[df["WL"] == "W"]
    losses = df[df["WL"] == "L"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["GAME_DATE"], y=df["PTS"],
        mode="lines", line=dict(color=color, width=2),
        name="Points", showlegend=False,
        hovertemplate="%{x|%b %d}: %{y} pts<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=wins["GAME_DATE"], y=wins["PTS"],
        mode="markers", marker=dict(color=ACCENT_COLORS["green"], size=9, symbol="circle"),
        name="Win", hovertemplate="%{x|%b %d}: W — %{y} pts<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=losses["GAME_DATE"], y=losses["PTS"],
        mode="markers", marker=dict(color=ACCENT_COLORS["red"], size=9, symbol="x"),
        name="Loss", hovertemplate="%{x|%b %d}: L — %{y} pts<extra></extra>",
    ))
    avg = df["PTS"].mean()
    fig.add_hline(y=avg, line=dict(color="#555", dash="dot"),
                  annotation_text=f"Avg {avg:.1f}", annotation_font_color=TEXT_MUTED)
    fig.update_layout(
        **_LAYOUT_BASE,
        margin=_MARGIN,
        height=240,
        xaxis=_XAXIS,
        yaxis=_YAXIS,
        title=dict(text=f"{team_name} — Last {len(df)} Games",
                   font=dict(size=13, color=TEXT_PRIMARY)),
        legend=dict(font=dict(color=TEXT_PRIMARY, size=11), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def shooting_breakdown(last10_avgs: dict, team_name: str, color: str = "#78BE20") -> go.Figure:
    """Donut of shooting split: 2PT, 3PT, FT makes."""
    fg2m = max(last10_avgs.get("FGM", 0) - last10_avgs.get("FG3M", 0), 0)
    fg3m = last10_avgs.get("FG3M", 0)
    ftm  = last10_avgs.get("FTM", 0)

    fig = go.Figure(go.Pie(
        labels=["2-PT Makes", "3-PT Makes", "FT Makes"],
        values=[fg2m, fg3m, ftm],
        hole=0.55,
        marker=dict(colors=[color, ACCENT_COLORS["blue"], ACCENT_COLORS["orange"]]),
        textinfo="label+percent",
        textfont=dict(color=TEXT_PRIMARY, size=12),
        hovertemplate="%{label}: %{value:.1f} avg<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        margin=_MARGIN,
        height=280,
        title=dict(text=f"{team_name} Shot Mix (L10)", font=dict(size=13, color=TEXT_PRIMARY)),
        showlegend=False,
    )
    return fig
