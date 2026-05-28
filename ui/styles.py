"""
Global CSS injected into the Streamlit app for a dark theme with vivid accent colors.
"""

TEAM_COLORS = {
    "Minnesota Timberwolves": {"primary": "#0C2340", "accent": "#236192", "highlight": "#78BE20"},
    "San Antonio Spurs":       {"primary": "#061922", "accent": "#C4CED4", "highlight": "#000000"},
    # Fallback
    "default":                 {"primary": "#1a1a2e", "accent": "#e94560", "highlight": "#f5a623"},
}

ACCENT_COLORS = {
    "green":  "#00FF87",
    "blue":   "#4FC3F7",
    "orange": "#FFB347",
    "red":    "#FF6B6B",
    "purple": "#CE93D8",
    "teal":   "#4DD0E1",
    "yellow": "#FFF176",
    "white":  "#FFFFFF",
}

DARK_BG       = "#0D0D0D"
CARD_BG       = "#161616"
CARD_BG2      = "#1E1E1E"
BORDER_COLOR  = "#2A2A2A"
TEXT_PRIMARY  = "#F0F0F0"
TEXT_MUTED    = "#888888"


def get_css() -> str:
    return f"""
<style>
    /* ---- Page & background ---- */
    [data-testid="stAppViewContainer"] {{
        background-color: {DARK_BG};
        color: {TEXT_PRIMARY};
    }}
    [data-testid="stHeader"] {{
        background-color: {DARK_BG};
    }}
    [data-testid="stSidebar"] {{
        background-color: #111111;
        border-right: 1px solid {BORDER_COLOR};
    }}
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {TEXT_MUTED};
    }}

    /* ---- Typography ---- */
    h1, h2, h3, h4 {{
        color: {TEXT_PRIMARY} !important;
        font-family: 'Segoe UI', sans-serif !important;
        letter-spacing: 0.02em;
    }}
    p, li, label, span {{
        color: {TEXT_PRIMARY};
    }}

    /* ---- Metric cards ---- */
    [data-testid="metric-container"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {ACCENT_COLORS["green"]} !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* ---- Tabs ---- */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background-color: {CARD_BG};
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        background-color: transparent;
        color: {TEXT_MUTED};
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        transition: all 0.2s;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        background-color: {CARD_BG2} !important;
        color: {ACCENT_COLORS["green"]} !important;
    }}

    /* ---- Dividers ---- */
    hr {{
        border-color: {BORDER_COLOR};
        margin: 1rem 0;
    }}

    /* ---- Selectbox & inputs ---- */
    [data-testid="stSelectbox"] > div {{
        background-color: {CARD_BG} !important;
        border-color: {BORDER_COLOR} !important;
        color: {TEXT_PRIMARY} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSelectbox"] svg {{
        fill: {TEXT_MUTED};
    }}

    /* ---- Buttons ---- */
    [data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, #00C06B, #00FF87) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        letter-spacing: 0.05em;
        transition: all 0.2s;
    }}
    [data-testid="baseButton-secondary"] {{
        background-color: {CARD_BG} !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 8px !important;
    }}

    /* ---- Progress bar ---- */
    [data-testid="stProgress"] > div > div {{
        background-color: {ACCENT_COLORS["green"]} !important;
    }}

    /* ---- Dataframe / table ---- */
    [data-testid="stDataFrame"] {{
        background-color: {CARD_BG};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* ---- Spinner ---- */
    [data-testid="stSpinner"] {{
        color: {ACCENT_COLORS["green"]} !important;
    }}

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {DARK_BG};
    }}
    ::-webkit-scrollbar-thumb {{
        background: #333;
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #555;
    }}
</style>
"""


def card(content_html: str, padding: str = "1.2rem 1.4rem") -> str:
    """Wrap content in a dark card div."""
    return f"""
<div style="
    background: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: {padding};
    margin-bottom: 0.8rem;
">
{content_html}
</div>
"""


def stat_badge(label: str, value: str, color: str = "green") -> str:
    hex_color = ACCENT_COLORS.get(color, color)
    return f"""
<span style="
    display: inline-block;
    background: {hex_color}18;
    border: 1px solid {hex_color}55;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 3px;
    font-size: 0.82rem;
    color: {hex_color};
    font-weight: 600;
">{label}: <strong style="color:{hex_color}">{value}</strong></span>
"""


def section_header(title: str, icon: str = "") -> str:
    return f"""
<div style="
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {BORDER_COLOR};
">
    <span style="font-size: 1.3rem;">{icon}</span>
    <h3 style="margin: 0; font-size: 1rem; font-weight: 700;
               text-transform: uppercase; letter-spacing: 0.1em;
               color: {TEXT_PRIMARY};">{title}</h3>
</div>
"""
