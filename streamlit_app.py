
import hashlib
import io
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from supabase_store import authenticate, import_csv, load_all
from mlb_movement_reference import retrieve as retrieve_mlb, pooled as pooled_mlb, velocity_grade


# ==================================================
# APP CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="TrackMan Scouting Analytics",
    page_icon="⚾",
    layout="wide",
)

sb = authenticate()

st.title("⚾ TrackMan Scouting Analytics")
st.caption("International Scouting | Video & Data")

if "trackman_data" not in st.session_state:
    st.session_state.trackman_data = load_all(sb)


@st.cache_data(ttl=86400, show_spinner=False)
def mlb_reference_seasons():
    frames, errors = [], []
    for year in (2025, 2026):
        try:
            frame = retrieve_mlb(year)
            frames.append(frame)
            if frame.attrs.get("missing_pitch_types"):
                errors.append(f"{year} missing pitch types: {', '.join(frame.attrs['missing_pitch_types'])}")
        except Exception as exc:
            errors.append(f"{year}: {type(exc).__name__}")
    return frames, errors


# ==================================================
# CONSTANTS
# ==================================================

HARD_EV = 90.0
MEDIUM_EV = 80.0
ZONE_HALF_WIDTH = 17 / 24

LEVELS = [
    "MLB",
    "Triple-A",
    "Double-A",
    "High-A",
    "Single-A",
    "Rookie",
    "DSL",
]

PITCH_METRICS = {
    "RelSpeed": ("Velocity", "mph"),
    "SpinRate": ("Spin Rate", "rpm"),
    "SpinEfficiency": ("Spin Efficiency", "%"),
    "InducedVertBreak": ("IVB", "in"),
    "HorzBreak": ("Horizontal Break", "in"),
    "VertApprAngle": ("VAA", "degrees"),
    "Extension": ("Extension", "ft"),
    "RelHeight": ("Release Height", "ft"),
    "RelSide": ("Release Side", "ft"),
}

HITTER_METRICS = {
    "ExitSpeed": ("Exit Velocity", "mph"),
    "Angle": ("Launch Angle", "degrees"),
    "Distance": ("Distance", "ft"),
}

PITCH_COLORS = {
    "Fastball": "#2563eb",
    "Four-Seam": "#2563eb",
    "FourSeamFastBall": "#2563eb",
    "FourSeamFastball": "#2563eb",
    "Sinker": "#0ea5e9",
    "Two-Seam": "#0ea5e9",
    "Slider": "#ef4444",
    "Curveball": "#a855f7",
    "ChangeUp": "#16a34a",
    "Changeup": "#16a34a",
    "Cutter": "#f59e0b",
    "Splitter": "#14b8a6",
}

WHIFF_CALLS = {
    "strikeswinging",
    "swingingstrike",
    "swingingstrikewildpitch",
    "swingingstrikeblocked",
}

FOUL_CALLS = {
    "foulball",
    "foul",
    "foultip",
    "foulballfieldable",
    "foulballnotfieldable",
    "foulbunt",
}

IN_PLAY_CALLS = {
    "inplay",
    "ballinplay",
    "inplayout",
    "inplayrun",
    "inplaynoout",
}

CALLED_STRIKES = {
    "strikecalled",
    "calledstrike",
}

BALL_CALLS = {
    "ballcalled",
    "calledball",
    "ballindirt",
    "ball",
    "intentball",
    "intentionalball",
    "pitchout",
}


# ==================================================
# DATA HELPERS
# ==================================================

def column(data, name):
    if name in data.columns:
        return data[name]
    return pd.Series(None, index=data.index)


def numeric(data, name):
    return pd.to_numeric(
        column(data, name),
        errors="coerce",
    )


def normalized(data, name):
    return (
        column(data, name)
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(
            r"[^a-z0-9]",
            "",
            regex=True,
        )
    )


def pct(numerator, denominator):
    if denominator is None or denominator == 0:
        return None
    return 100 * numerator / denominator


def fmt(value, percent=False):
    if value is None or pd.isna(value):
        return "N/A"
    if percent:
        return f"{value:.1f}%"
    return f"{int(value):,}"


def choose_player(data, field, key):
    if field not in data.columns:
        st.warning(f"{field} column unavailable.")
        st.stop()

    names = sorted(
        x
        for x in data[field].dropna().astype(str).unique()
        if x.strip()
    )

    if not names:
        st.warning("No players available.")
        st.stop()

    return st.selectbox(
        f"Select {field}",
        names,
        key=key,
    )


def date_filter(data, key):
    if "Date" not in data.columns:
        return data

    dates = sorted(
        data["Date"].dropna().astype(str).unique()
    )

    selected = st.selectbox(
        "Session Date",
        ["All dates"] + dates,
        key=key,
    )

    if selected == "All dates":
        return data

    return data[
        data["Date"].astype(str) == selected
    ].copy()


def pitch_filter(data, key):
    if "TaggedPitchType" not in data.columns:
        return data, "All pitches"

    types = sorted(
        data["TaggedPitchType"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected = st.selectbox(
        "Pitch Type",
        ["All pitches"] + types,
        key=key,
    )

    if selected == "All pitches":
        return data, selected

    return (
        data[
            data["TaggedPitchType"].astype(str)
            == selected
        ].copy(),
        selected,
    )


# ==================================================
# PITCH CLASSIFICATION
# ==================================================

def classify(data, zone_bottom, zone_top):
    d = data.copy()
    calls = normalized(d, "PitchCall")

    d["_whiff"] = calls.isin(WHIFF_CALLS)
    d["_foul"] = calls.isin(FOUL_CALLS)
    d["_inplay"] = calls.isin(IN_PLAY_CALLS)

    d["_swing"] = (
        d["_whiff"]
        | d["_foul"]
        | d["_inplay"]
    )

    d["_contact"] = (
        d["_foul"] | d["_inplay"]
    )

    d["_strike"] = (
        d["_whiff"]
        | d["_foul"]
        | d["_inplay"]
        | calls.isin(CALLED_STRIKES)
    )

    d["_ball"] = calls.isin(BALL_CALLS)
    d["_known"] = d["_strike"] | d["_ball"]

    side = numeric(d, "PlateLocSide")
    height = numeric(d, "PlateLocHeight")

    d["_located"] = (
        side.notna() & height.notna()
    )

    d["_zone"] = (
        d["_located"]
        & side.between(
            -ZONE_HALF_WIDTH,
            ZONE_HALF_WIDTH,
        )
        & height.between(
            zone_bottom,
            zone_top,
        )
    )

    d["_outside"] = (
        d["_located"] & ~d["_zone"]
    )

    ev = numeric(d, "ExitSpeed")

    d["_valid_ev"] = (
        d["_inplay"] & ev.notna()
    )

    d["_hard"] = (
        d["_valid_ev"] & ev.ge(HARD_EV)
    )

    d["_medium"] = (
        d["_valid_ev"]
        & ev.ge(MEDIUM_EV)
        & ev.lt(HARD_EV)
    )

    d["_weak"] = (
        d["_valid_ev"] & ev.lt(MEDIUM_EV)
    )

    return d


# ==================================================
# BATTING RATES
# ==================================================

def batting_rates(d):
    swings = int(d["_swing"].sum())
    whiffs = int(d["_whiff"].sum())
    contacts = int(d["_contact"].sum())

    outside = int(d["_outside"].sum())

    outside_swings = int(
        (d["_outside"] & d["_swing"]).sum()
    )

    outside_contacts = int(
        (d["_outside"] & d["_contact"]).sum()
    )

    zone_swings = int(
        (d["_zone"] & d["_swing"]).sum()
    )

    zone_contacts = int(
        (d["_zone"] & d["_contact"]).sum()
    )

    ev_count = int(d["_valid_ev"].sum())

    return {
        "Swing %": pct(swings, len(d)),
        "Whiff %": pct(whiffs, swings),
        "Contact %": pct(contacts, swings),
        "Chase %": pct(
            outside_swings,
            outside,
        ),
        "Z-Contact %": pct(
            zone_contacts,
            zone_swings,
        ),
        "O-Contact %": pct(
            outside_contacts,
            outside_swings,
        ),
        "Hard Contact %": pct(
            int(d["_hard"].sum()),
            ev_count,
        ),
        "Medium Contact %": pct(
            int(d["_medium"].sum()),
            ev_count,
        ),
        "Weak Contact %": pct(
            int(d["_weak"].sum()),
            ev_count,
        ),
    }


# ==================================================
# FIRST PITCH STRIKE
# ==================================================

def first_pitch_strike(d):
    if "PitchofPA" not in d.columns:
        return None

    first = d[
        numeric(d, "PitchofPA") == 1
    ]

    first = first[first["_known"]]

    if first.empty:
        return None

    return pct(
        int(first["_strike"].sum()),
        len(first),
    )


# ==================================================
# PLATE APPEARANCE RESULTS
# ==================================================

def pa_results(d):
    unavailable = {
        "K": None,
        "BB": None,
        "K %": None,
        "BB %": None,
    }

    if "KorBB" not in d.columns:
        return unavailable

    possible = [
        "PlateAppearanceID",
        "PlateAppearanceId",
        "PAofGame",
        "PAofInning",
    ]

    pa_field = next(
        (x for x in possible if x in d.columns),
        None,
    )

    if pa_field is None:
        return unavailable

    if pa_field == "PAofInning":
        if "Inning" not in d.columns:
            return unavailable
        keys = ["Inning", "PAofInning"]
    else:
        keys = [pa_field]

    for extra in [
        "GameUID",
        "GameID",
        "SessionId",
        "Date",
        "Top/Bottom",
    ]:
        if extra in d.columns:
            keys.insert(0, extra)

    for field in ["Batter", "Pitcher"]:
        if field in d.columns:
            keys.append(field)

    strikeouts = 0
    walks = 0

    for _, group in d.groupby(
        keys,
        dropna=False,
    ):
        outcomes = normalized(group, "KorBB")

        if outcomes.isin(
            {"strikeout", "k", "so"}
        ).any():
            strikeouts += 1

        elif outcomes.isin(
            {"walk", "bb", "baseonballs"}
        ).any():
            walks += 1

    return {
        "K": strikeouts,
        "BB": walks,
        "K %": None,
        "BB %": None,
    }


# ==================================================
# BENCHMARKS
# ==================================================

BENCHMARK_COLUMNS = [
    "season",
    "level",
    "role",
    "metric",
    "average",
    "tolerance",
    "direction",
    "source",
]


@st.cache_data
def benchmark_table():
    path = Path(__file__).with_name(
        "benchmarks.csv"
    )

    if not path.exists():
        return pd.DataFrame(
            columns=BENCHMARK_COLUMNS
        )

    try:
        table = pd.read_csv(path)

        if not set(BENCHMARK_COLUMNS).issubset(
            table.columns
        ):
            return pd.DataFrame(
                columns=BENCHMARK_COLUMNS
            )

        return table

    except Exception:
        return pd.DataFrame(
            columns=BENCHMARK_COLUMNS
        )


def reference(metric, role):
    """Fail safely on unavailable or malformed benchmark data; never crash player analytics."""
    try:
        table = benchmark_table()
        if table.empty:
            return None
        required = {"season", "level", "role", "metric", "average", "tolerance", "direction", "source"}
        if not required.issubset(table.columns) or not table.columns.is_unique:
            return None
        years = ["2025", "2026"] if benchmark_season_mode == "2025 + 2026 (available data)" else [str(benchmark_season)]
        matches = table.loc[
            table["level"].astype("string").eq(str(benchmark_level))
            & table["season"].astype("string").isin(years)
            & table["role"].astype("string").str.lower().eq(str(role).lower())
            & table["metric"].astype("string").eq(str(metric))
        ].copy()
        if matches.empty:
            return None
        matches["average"] = pd.to_numeric(matches["average"], errors="coerce")
        matches["tolerance"] = pd.to_numeric(matches["tolerance"], errors="coerce")
        matches = matches.dropna(subset=["average", "tolerance"])
        matches = matches.loc[
            matches["direction"].astype("string").str.lower().isin(["higher", "lower"])
            & matches["source"].fillna("").astype("string").str.strip().ne("")
        ]
        if matches.empty or matches["direction"].astype("string").str.lower().nunique() != 1:
            return None
        return (float(matches["average"].mean()),
                float(matches["tolerance"].abs().mean()),
                str(matches.iloc[0]["direction"]).lower(),
                "; ".join(sorted(set(matches["source"].astype(str)))))
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None


def rating(metric, role, value):
    ref = reference(metric, role)

    if (
        ref is None
        or value is None
        or pd.isna(value)
    ):
        return "No benchmark", "#64748b", None

    avg, tolerance, direction, source = ref

    if abs(value - avg) <= tolerance:
        return "Average", "#eab308", avg

    good = (
        value > avg
        if direction == "higher"
        else value < avg
    )

    if good:
        return "Good", "#dc2626", avg

    return "Poor", "#2563eb", avg


# ==================================================
# METRIC CARDS
# ==================================================

def visual_grade(value, metric, role):
    """Clearly labeled sample thresholds only when demo toggle is enabled."""
    if demo_mode:
        examples = {"Strike %": (65, 8, "higher"), "Whiff %": (25, 7, "higher"),
                    "Chase %": (30, 7, "higher"), "Ball %": (35, 7, "lower"),
                    "First Pitch Strike %": (60, 8, "higher")}
        if metric in examples and value is not None and pd.notna(value):
            avg, tolerance, direction = examples[metric]
            delta = (float(value) - avg) * (1 if direction == "higher" else -1)
            grade, color = ("GOOD", "#dc2626") if delta > tolerance else (("BELOW", "#2563eb") if delta < -tolerance else ("AVERAGE", "#eab308"))
            return grade + " · DEMO", color, avg
    return rating(metric, role, value)


def metric_cards(items, role):
    entries = list(items.items())

    for start in range(0, len(entries), 4):
        cols = st.columns(4)

        for c, (name, value) in zip(
            cols,
            entries[start:start + 4],
        ):
            c.metric(name, value)

            try:
                number = float(
                    str(value)
                    .replace("%", "")
                    .replace(",", "")
                )
            except (ValueError, TypeError):
                number = None

            label, color, avg = visual_grade(
                number,
                name,
                role,
            )

            if avg is None:
                c.caption("Measured · no verified comparison")
            else:
                c.markdown(
                    f'<div style="border-left:5px solid {color};'
                    f'background:{color}18;padding:9px 11px;border-radius:6px">'
                    f'<strong style="color:{color}">{label}</strong>'
                    f' · {"SAMPLE" if demo_mode and "DEMO" in label else benchmark_level} reference: {avg:.1f}</div>',
                    unsafe_allow_html=True,
                )


def metric_table(data, definitions):
    rows = []
    for field, (label, unit) in definitions.items():
        values = numeric(data, field).dropna()
        if values.empty:
            continue
        rows.append({"Metric": label, "Average": f"{values.mean():.1f} {unit}",
                     "Range": f"{values.min():.1f}–{values.max():.1f}", "Pitches": len(values)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("These are measurements, not grades. Pitch quality depends on pitch type and context.")
    else:
        st.info("No numeric metrics available.")


# ==================================================
# PITCHER AND BATTER HEADERS
# ==================================================

def hitting_stats(data, bottom, top):
    d = classify(data, bottom, top)
    rates = batting_rates(d)
    pa = pa_results(d)

    return {
        "Pitches Seen": fmt(len(d)),
        "K": fmt(pa["K"]),
        "BB": fmt(pa["BB"]),
        **{
            name: fmt(value, True)
            for name, value in rates.items()
        },
    }


def pitching_stats(data, bottom, top):
    d = classify(data, bottom, top)
    rates = batting_rates(d)
    pa = pa_results(d)

    known = int(d["_known"].sum())

    return {
        "Total Pitches": fmt(len(d)),
        "Strike %": fmt(
            pct(int(d["_strike"].sum()), known),
            True,
        ),
        "Ball %": fmt(
            pct(int(d["_ball"].sum()), known),
            True,
        ),
        "First Pitch Strike %": fmt(
            first_pitch_strike(d),
            True,
        ),
        "Whiff %": fmt(
            rates["Whiff %"],
            True,
        ),
        "Chase %": fmt(
            rates["Chase %"],
            True,
        ),
        "K %": fmt(pa["K %"], True),
        "BB %": fmt(pa["BB %"], True),
        "Weak Contact Generated %": fmt(
            rates["Weak Contact %"],
            True,
        ),
    }


def hitting_headers(data, bottom, top):
    metric_cards(
        hitting_stats(data, bottom, top),
        "Batter",
    )

    st.caption(
        "Contact includes fouls. "
        "Hard Contact: EV >= 90 mph. "
        "Medium Contact: 80 to <90 mph. "
        "Weak Contact: EV <80 mph. "
        "Zone rates use the approximate strike zone."
    )


def pitching_headers(data, bottom, top):
    metric_cards(
        pitching_stats(data, bottom, top),
        "Pitcher",
    )

    st.caption(
        "Strike and Ball percentages use recognized "
        "pitch calls. K% and BB% remain N/A when "
        "completed plate appearances cannot be verified."
    )


# ==================================================
# MOVEMENT PROFILE
# ==================================================

def movement_figure(data):
    required = [
        "HorzBreak",
        "InducedVertBreak",
    ]

    if not all(
        field in data.columns
        for field in required
    ):
        return None

    d = data.copy()

    for field in required:
        d[field] = numeric(d, field)

    d = d.dropna(subset=required)

    if d.empty:
        return None

    if "TaggedPitchType" not in d.columns:
        d["TaggedPitchType"] = "Unknown"

    d["TaggedPitchType"] = (
        d["TaggedPitchType"]
        .fillna("Unknown")
        .astype(str)
    )

    fig = px.scatter(
        d,
        x="HorzBreak",
        y="InducedVertBreak",
        color="TaggedPitchType",
        color_discrete_map=PITCH_COLORS,
        labels={
            "HorzBreak": "Horizontal Break (in)",
            "InducedVertBreak": "IVB (in)",
        },
    )

    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.85,
            line=dict(
                color="white",
                width=1,
            ),
        )
    )

    fig.update_xaxes(
        range=[-30, 30],
        dtick=5,
        tickfont=dict(
            size=17,
            color="#111827",
        ),
        ticks="outside",
        ticklen=7,
        zeroline=True,
        zerolinewidth=3,
        zerolinecolor="#111827",
        showgrid=True,
        gridcolor="#e5e7eb",
        title="Horizontal Break (in)",
    )

    fig.update_yaxes(
        range=[-30, 30],
        dtick=5,
        tickfont=dict(
            size=17,
            color="#111827",
        ),
        ticks="outside",
        ticklen=7,
        zeroline=True,
        zerolinewidth=3,
        zerolinecolor="#111827",
        showgrid=True,
        gridcolor="#e5e7eb",
        scaleanchor="x",
        scaleratio=1,
        title="Induced Vertical Break (in)",
    )

    fig.update_layout(
        height=760,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            color="#111827",
            size=15,
        ),
        legend_title="Pitch Type",
        legend=dict(
            orientation="h",
            y=1.08,
        ),
    )

    return fig


def movement_profile(data):
    fig = movement_figure(data)

    if fig is None:
        st.info("Movement data unavailable.")
        return

    st.subheader(
        "Movement Profile | Pitcher View"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Both axes are in inches. "
        "Major tick marks are 5 inches apart."
    )

    d = data.copy()

    d["HorzBreak"] = numeric(
        d,
        "HorzBreak",
    )

    d["InducedVertBreak"] = numeric(
        d,
        "InducedVertBreak",
    )

    d = d.dropna(
        subset=[
            "HorzBreak",
            "InducedVertBreak",
        ]
    )

    if "TaggedPitchType" not in d.columns:
        d["TaggedPitchType"] = "Unknown"

    averages = (
        d.groupby("TaggedPitchType", dropna=False)
        .agg(
            Pitches=("HorzBreak", "size"),
            Avg_HB=("HorzBreak", "mean"),
            Avg_IVB=("InducedVertBreak", "mean"),
        )
        .reset_index()
    )

    st.dataframe(
        averages.round(1),
        hide_index=True,
        use_container_width=True,
    )


# ==================================================
# STRIKE ZONE
# ==================================================

def strike_zone_figure(data, bottom, top):
    required = [
        "PlateLocSide",
        "PlateLocHeight",
    ]

    if not all(
        field in data.columns
        for field in required
    ):
        return None

    d = data.copy()

    for field in required:
        d[field] = numeric(d, field)

    d = d.dropna(subset=required)

    if d.empty:
        return None

    classified = classify(d, bottom, top)

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=-ZONE_HALF_WIDTH,
        x1=ZONE_HALF_WIDTH,
        y0=bottom,
        y1=top,
        line=dict(
            color="#334155",
            width=3,
        ),
        fillcolor="rgba(148,163,184,0.10)",
    )

    for i in [1, 2]:
        x = (
            -ZONE_HALF_WIDTH
            + 2 * ZONE_HALF_WIDTH * i / 3
        )

        y = bottom + (
            top - bottom
        ) * i / 3

        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=bottom,
            y1=top,
            line=dict(
                color="#94a3b8",
                dash="dot",
            ),
        )

        fig.add_shape(
            type="line",
            x0=-ZONE_HALF_WIDTH,
            x1=ZONE_HALF_WIDTH,
            y0=y,
            y1=y,
            line=dict(
                color="#94a3b8",
                dash="dot",
            ),
        )

    d["_zone_label"] = classified["_zone"].map({
        True: "In Zone",
        False: "Out of Zone",
    })

    for label, color in {
        "In Zone": "#2563eb",
        "Out of Zone": "#ef4444",
    }.items():
        subset = d[d["_zone_label"] == label]

        fig.add_trace(
            go.Scatter(
                x=subset["PlateLocSide"],
                y=subset["PlateLocHeight"],
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=10,
                    opacity=0.85,
                    line=dict(
                        color="white",
                        width=1,
                    ),
                ),
            )
        )

    fig.update_layout(
        height=620,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="Horizontal Location (ft)",
            range=[-2.5, 2.5],
            showgrid=False,
        ),
        yaxis=dict(
            title="Height (ft)",
            range=[0, 5],
            showgrid=False,
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    return fig


def strike_zone_chart(data, bottom, top):
    fig = strike_zone_figure(
        data,
        bottom,
        top,
    )

    if fig is None:
        st.info("Pitch locations unavailable.")
        return

    classified = classify(
        data,
        bottom,
        top,
    )

    located = classified[
        classified["_located"]
    ]

    inside = int(located["_zone"].sum())
    outside = len(located) - inside

    st.subheader(
        "Pitch Location | Strike Zone"
    )

    a, b, c = st.columns(3)

    a.metric("In Zone", inside)
    b.metric("Out of Zone", outside)
    c.metric(
        "Zone %",
        fmt(pct(inside, len(located)), True),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Blue = inside approximate zone. "
        "Red = outside. "
        "Not an official umpire call."
    )


# ==================================================
# PDF REPORT
# ==================================================

def pdf_report(player, role, data, bottom, top):
    """Short, beginner-friendly one-page report. Never claim demo data is real."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from pitch_scouting_engine import pitch_rows
    from scouting_quick_summary import shape_summary, overall_summary

    navy = colors.HexColor("#142d49")
    muted = colors.HexColor("#64748b")
    pale = colors.HexColor("#f1f5f9")
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=letter, leftMargin=38, rightMargin=38,
                            topMargin=38, bottomMargin=38)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SimpleTitle", parent=styles["Title"], fontSize=18, leading=22, textColor=navy, spaceAfter=6))
    styles.add(ParagraphStyle(name="SimpleHead", parent=styles["Heading2"], fontSize=11, leading=14, textColor=navy, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="SimpleText", parent=styles["Normal"], fontSize=9, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="SimpleSmall", parent=styles["Normal"], fontSize=8, leading=11, textColor=muted))
    elements = [Paragraph(escape(str(player)), styles["SimpleTitle"]),
                Paragraph(f"{escape(role)} | {len(data)} pitches | TrackMan session report", styles["SimpleSmall"]),
                Spacer(1, 9)]
    if demo_mode:
        elements.append(Paragraph("<b>DEMO PREVIEW — sample grading thresholds, not official MLB/MiLB data</b>", styles["SimpleText"]))
    if role == "Pitcher":
        stats = pitching_stats(data, bottom, top)
        elements.append(Paragraph("AT A GLANCE", styles["SimpleHead"]))
        key_names = ["Strike %", "Ball %", "Whiff %", "Chase %"]
        cells = []
        for name in key_names:
            value = stats.get(name, "N/A")
            try:
                num = float(str(value).replace("%", "").replace(",", ""))
            except (TypeError, ValueError):
                num = None
            label, shade, avg = visual_grade(num, name, "Pitcher")
            note = label if avg is not None else "Measured"
            cells.append(Paragraph(f'<font color="{shade}"><b>{escape(str(value))}</b></font><font size="8"><br/>{escape(name)}<br/>{escape(note)}</font>', styles["SimpleText"]))
        card_table = Table([cells], colWidths=[134]*4)
        card_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),pale),("VALIGN",(0,0),(-1,-1),"TOP"),
                                        ("LEFTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),12),
                                        ("BOTTOMPADDING",(0,0),(-1,-1),9)]))
        elements.append(card_table)
        elements.append(Paragraph("PITCH ARSENAL", styles["SimpleHead"]))
        pitches = pitch_rows(data)
        rows = [["PITCH", "THROWN", "SPEED", "IVB", "SIDE BREAK", "WHIFF"]]
        for p in pitches:
            v=p["v"]
            fmtval=lambda k: "—" if v.get(k) is None else f'{v[k]:.1f}'
            whiff="—" if p["whiff_pct"] is None else f'{p["whiff_pct"]:.0f}%'
            rows.append([p["name"], str(p["n"]),fmtval("RelSpeed"),fmtval("InducedVertBreak"),fmtval("HorzBreak"),whiff])
        table=Table(rows,colWidths=[143,62,73,72,99,87],repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),navy),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                   ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,pale]),
                                   ("FONTSIZE",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),8),
                                   ("BOTTOMPADDING",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        elements.append(table)
        elements.append(Paragraph("OVERALL SUMMARY", styles["SimpleHead"]))
        elements.append(Paragraph(escape(overall_summary(pitches)), styles["SimpleText"]))
        elements.append(Paragraph("PITCH-BY-PITCH SCOUTING", styles["SimpleHead"]))
        for p in pitches:
            vertical, horizontal, separation = shape_summary(p, pitches)
            elements.append(KeepTogether([
                Paragraph(f'<b>{escape(p["name"])} ({p["n"]} pitches)</b>', styles["SimpleText"]),
                Paragraph(escape(vertical), styles["SimpleText"]),
                Paragraph(escape(horizontal), styles["SimpleText"]),
                Paragraph(escape(separation), styles["SimpleText"]),
            ]))
        elements.append(Paragraph("QUICK DEFINITIONS", styles["SimpleHead"]))
        elements.append(Paragraph("<b>IVB:</b> how much a pitch stays up or drops versus gravity. <b>Side break:</b> left/right movement. <b>Whiff:</b> missed swings.",styles["SimpleText"]))
    else:
        stats=hitting_stats(data,bottom,top)
        elements.append(Paragraph("HITTING SNAPSHOT",styles["SimpleHead"]))
        for name,value in list(stats.items())[:8]:
            elements.append(Paragraph(f"<b>{escape(str(name))}:</b> {escape(str(value))}",styles["SimpleText"]))
        ev=numeric(data,"ExitSpeed").dropna()
        if len(ev):
            elements.append(Paragraph(f"<b>Average exit speed:</b> {ev.mean():.1f} mph from {len(ev)} recorded contacts.",styles["SimpleText"]))
    elements.append(Spacer(1,10))
    elements.append(Paragraph("This report describes only the selected session. Few pitches or swings can give misleading percentages. Red = above demo reference, yellow = near, blue = below (only when DEMO is enabled).",styles["SimpleSmall"]))
    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(38,30,574,30)
        canvas.setFont("Helvetica",8)
        canvas.setFillColor(muted)
        canvas.drawString(38,19,"TRACKMAN SCOUTING | QUICK READ")
        canvas.drawRightString(574,19,f"Page {document.page}")
        canvas.restoreState()
    doc.build(elements,onFirstPage=footer,onLaterPages=footer)
    return buff.getvalue()


def download_report(
    player,
    role,
    data,
    bottom,
    top,
):
    try:
        output = pdf_report(
            player,
            role,
            data,
            bottom,
            top,
        )

        safe_name = "".join(
            c if c.isalnum() or c in "-_"
            else "_"
            for c in str(player)
        )

        st.download_button(
            "Download Full Scouting Report (PDF)",
            data=output,
            file_name=(
                f"{safe_name}_"
                f"{role.lower()}_report.pdf"
            ),
            mime="application/pdf",
        )

    except Exception as exc:
        st.error(
            f"PDF generation failed: {exc}"
        )


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Analysis Settings")

demo_mode = st.sidebar.toggle("Preview performance colors (DEMO)", value=True, help="Illustrative only: NOT MLB/MiLB statistics or scouting grades.")

benchmark_level = st.sidebar.selectbox(
    "Benchmark Level",
    LEVELS,
)

benchmark_season_mode = st.sidebar.selectbox(
    "Benchmark Period",
    ["2025 + 2026 (available data)", "Single season"],
)
benchmark_season = st.sidebar.selectbox(
    "Single Benchmark Season",
    [2025, 2026],
    index=0,
    disabled=benchmark_season_mode != "Single season",
)
available_years = sorted(set(
    benchmark_table().loc[
        (benchmark_table()["level"].astype(str) == benchmark_level)
        & (benchmark_table()["season"].astype(str).isin(["2025", "2026"])),
        "season",
    ].astype(str)
)) if not benchmark_table().empty else []
st.sidebar.caption(
    "Reference seasons available for this level: "
    + (", ".join(available_years) if available_years else "none")
    + ". Combined mode averages available seasonal reference values equally; missing seasons are never estimated."
)

if benchmark_table().empty:
    st.sidebar.caption(
        "No sourced benchmark data installed yet. "
        "Ratings will show No benchmark."
    )

zone_bottom = st.sidebar.number_input(
    "Strike Zone Bottom (ft)",
    min_value=0.0,
    max_value=5.0,
    value=1.50,
    step=0.05,
)

zone_top = st.sidebar.number_input(
    "Strike Zone Top (ft)",
    min_value=0.0,
    max_value=6.0,
    value=3.50,
    step=0.05,
)

if zone_top <= zone_bottom:
    st.error(
        "Strike zone top must exceed bottom."
    )
    st.stop()

pages = [
    "Overview",
    "Pitcher Analytics",
    "Hitter Analytics",
    "Player Profiles",
    "Strike Zone",
    "CSV Upload",
]

page = st.sidebar.radio(
    "Navigation",
    pages,
)


# ==================================================
# CSV UPLOAD
# ==================================================

if page == "CSV Upload":
    st.header("Upload TrackMan CSV")

    uploaded = st.file_uploader(
        "Choose a TrackMan CSV file",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            file_bytes = uploaded.getvalue()

            fingerprint = hashlib.sha256(
                file_bytes
            ).hexdigest()

            if (
                st.session_state.get(
                    "last_upload_attempt"
                )
                != fingerprint
            ):
                status, count = import_csv(
                    sb,
                    file_bytes,
                    uploaded.name,
                )

                st.session_state.last_upload_attempt = (
                    fingerprint
                )

                st.session_state.trackman_data = (
                    load_all(sb)
                )

                if status == "duplicate":
                    st.info(
                        "CSV already exists. "
                        "No duplicates added."
                    )
                else:
                    st.success(
                        f"CSV saved: {count:,} pitches."
                    )

            else:
                st.info(
                    "File already processed "
                    "in this session."
                )

            saved = (
                st.session_state.trackman_data
            )

            if saved is not None:
                st.dataframe(
                    saved.tail(20),
                    use_container_width=True,
                )

        except Exception as error:
            st.error(
                f"CSV import failed: {error}"
            )


# ==================================================
# LOAD DATA
# ==================================================

df = st.session_state.trackman_data

if page != "CSV Upload":
    if df is None or df.empty:
        st.info(
            "No saved TrackMan records yet. "
            "Go to CSV Upload."
        )
        st.stop()


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":
    st.header("Overview")

    a, b, c = st.columns(3)

    a.metric(
        "Total Records",
        len(df),
    )

    b.metric(
        "Pitchers",
        df["Pitcher"].nunique()
        if "Pitcher" in df.columns
        else 0,
    )

    c.metric(
        "Batters",
        df["Batter"].nunique()
        if "Batter" in df.columns
        else 0,
    )

    if "TaggedPitchType" in df.columns:
        st.subheader("Pitch Distribution")

        counts = (
            df["TaggedPitchType"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Pitch Type")
            .reset_index(name="Count")
        )

        fig = px.bar(
            counts,
            x="Pitch Type",
            y="Count",
            color="Pitch Type",
            color_discrete_map=PITCH_COLORS,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ==================================================
# PITCHER ANALYTICS
# ==================================================

elif page == "Pitcher Analytics":
    st.header("Pitcher Analytics")

    pitcher = choose_player(
        df,
        "Pitcher",
        "pitcher_analytics",
    )

    d = df[
        df["Pitcher"].astype(str) == pitcher
    ].copy()

    d = date_filter(
        d,
        "pitcher_date",
    )

    report_data = d.copy()  # All pitch types for the selected player/date.
    d, pitch_type = pitch_filter(
        d,
        "pitcher_pitch_type",
    )

    st.subheader(pitcher)

    pitching_headers(
        d,
        zone_bottom,
        zone_top,
    )

    st.subheader(
        f"Pitch Metrics | {pitch_type}"
    )

    metric_table(
        d,
        PITCH_METRICS,
    )

    if "RelSpeed" in d.columns:
        velocity = d.copy()

        velocity["RelSpeed"] = numeric(
            velocity,
            "RelSpeed",
        )

        velocity = velocity.dropna(
            subset=["RelSpeed"]
        )

        if not velocity.empty:
            if "TaggedPitchType" not in velocity.columns:
                velocity["TaggedPitchType"] = "Unknown"

            st.subheader(
                "Velocity by Pitch Type"
            )

            fig = px.box(
                velocity,
                x="TaggedPitchType",
                y="RelSpeed",
                color="TaggedPitchType",
                color_discrete_map=PITCH_COLORS,
                points="all",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    movement_profile(d)

    # Arsenal-level visual summary remains visible even when a pitch filter is active.
    from pitch_scouting_engine import pitch_rows
    arsenal = pitch_rows(report_data)
    if arsenal:
        st.subheader("Full Arsenal | Pitch-by-Pitch")
        mlb_frames, mlb_errors = mlb_reference_seasons()
        st.caption("MLB Statcast 2025–2026 | pitch-count-weighted reference by pitch type. Velocity colors describe velocity only, not overall pitch quality.")
        if mlb_errors:
            st.warning("MLB reference download unavailable for: " + ", ".join(mlb_errors) + ". Unavailable years are not estimated.")
        st.caption("Pitch colors identify pitch types; they are not performance grades. Red = good, yellow = average, blue = poor only for verified comparable benchmarks.")
        for p in arsenal:
            vals = p["v"]
            label = escape(p["name"])
            pitch_color = PITCH_COLORS.get(p["name"], PITCH_COLORS.get(p["name"].replace("Four-Seam Fastball", "Fastball"), "#64748b"))
            st.markdown(f'<div style="border-left:6px solid {pitch_color};padding:7px 12px;margin:9px 0;background:rgba(128,128,128,0.08);border-radius:6px"><b>{label}</b> · {p["n"]} pitches</div>', unsafe_allow_html=True)
            hand_raw = str(report_data["PitcherThrows"].dropna().iloc[0]).strip().upper() if "PitcherThrows" in report_data and report_data["PitcherThrows"].notna().any() else ""
            hand = "R" if hand_raw in ("R","RIGHT") else "L" if hand_raw in ("L","LEFT") else None
            ref = pooled_mlb(mlb_frames, p["name"], hand=hand)
            if ref and ref.get("RelSpeed") is not None and vals["RelSpeed"] is not None:
                grade, grade_color = velocity_grade(vals["RelSpeed"], ref["RelSpeed"])
                st.markdown(f'<div style="padding:9px 12px;background:{grade_color}20;border-left:6px solid {grade_color};border-radius:5px"><b style="color:{grade_color}">{grade}</b> · Player {vals["RelSpeed"]:.1f} mph / MLB {ref["RelSpeed"]:.1f} mph · {", ".join(map(str,ref["seasons"]))} · {ref["RelSpeed_n"]:,} MLB pitches</div>', unsafe_allow_html=True)
                st.caption("MLB IVB and horizontal movement: " + (f'{ref.get("InducedVertBreak",float("nan")):.1f} / {ref.get("HorzBreak",float("nan")):.1f} inches. Movement sign conventions must be verified before grading shape.'))
            elif demo_mode and vals["RelSpeed"] is not None:
                demo_velocities = {"Four-Seam Fastball": 93.5, "Sinker": 93, "Cutter": 89,
                                   "Slider": 85, "Sweeper": 82, "Curveball": 79,
                                   "Changeup": 85, "Splitter": 86}
                demo_avg = demo_velocities.get(p["name"])
                if demo_avg is not None:
                    grade, grade_color = velocity_grade(vals["RelSpeed"], demo_avg)
                    st.markdown(f'<div style="padding:10px;border-left:6px solid {grade_color};background:{grade_color}20"><b style="color:{grade_color}">{grade} · DEMO ONLY</b> · Player {vals["RelSpeed"]:.1f} mph / sample {demo_avg:.1f} mph</div>', unsafe_allow_html=True)
                    st.caption("Illustrative reference only — NOT actual MLB or MiLB averages.")
                else:
                    st.caption("No demo example for this pitch type.")
            else:
                st.caption("MLB comparison unavailable. Turn on DEMO in the sidebar to preview the colors.")
            cols = st.columns(5)
            for c, field, title, unit in zip(cols, ["RelSpeed","InducedVertBreak","HorzBreak","SpinRate","VertApprAngle"], ["Velocity","IVB","HB","Spin","VAA"], ["mph","in","in","rpm","°"]):
                val = vals[field]
                c.metric(title, "N/A" if val is None else f"{val:.1f} {unit}")
            if p["whiff_pct"] is not None:
                st.caption(f'Observed Whiff: {p["whiff_pct"]:.1f}% ({p["whiffs"]}/{p["swings"]} swings). ' + ("Small sample." if p["swings"] < 20 else ""))
        st.caption("MLB/MiLB grades require verified pitch-type, handedness, season and measurement-matched reference distributions.")

    st.caption("The PDF includes the full pitch arsenal for the selected player and session dates, even when a single pitch type is selected in the chart filter.")
    download_report(
        pitcher,
        "Pitcher",
        report_data,
        zone_bottom,
        zone_top,
    )


# ==================================================
# HITTER ANALYTICS
# ==================================================

elif page == "Hitter Analytics":
    st.header("Hitter Analytics")

    batter = choose_player(
        df,
        "Batter",
        "hitter_analytics",
    )

    d = df[
        df["Batter"].astype(str) == batter
    ].copy()

    d = date_filter(
        d,
        "hitter_date",
    )

    st.subheader(batter)

    hitting_headers(
        d,
        zone_bottom,
        zone_top,
    )

    st.subheader("Hitting Metrics")

    metric_table(
        d,
        HITTER_METRICS,
    )

    download_report(
        batter,
        "Batter",
        d,
        zone_bottom,
        zone_top,
    )


# ==================================================
# PLAYER PROFILES
# ==================================================

elif page == "Player Profiles":
    st.header("Player Profiles")

    role = st.radio(
        "Player Role",
        ["Pitcher", "Batter"],
        horizontal=True,
        key="profile_role",
    )

    player = choose_player(
        df,
        role,
        f"profile_{role.lower()}",
    )

    d = df[
        df[role].astype(str) == player
    ].copy()

    d = date_filter(
        d,
        f"profile_date_{role.lower()}",
    )

    st.header(player)

    if role == "Pitcher":
        st.subheader("Pitching Profile")

        pitching_headers(
            d,
            zone_bottom,
            zone_top,
        )

        metric_table(
            d,
            PITCH_METRICS,
        )

        movement_profile(d)

    else:
        st.subheader("Hitting Profile")

        hitting_headers(
            d,
            zone_bottom,
            zone_top,
        )

        metric_table(
            d,
            HITTER_METRICS,
        )

    download_report(
        player,
        role,
        d,
        zone_bottom,
        zone_top,
    )


# ==================================================
# STRIKE ZONE
# ==================================================

elif page == "Strike Zone":
    st.header("Strike Zone")

    pitcher = choose_player(
        df,
        "Pitcher",
        "strike_zone_pitcher",
    )

    d = df[
        df["Pitcher"].astype(str) == pitcher
    ].copy()

    d = date_filter(
        d,
        "strike_zone_date",
    )

    d, pitch_type = pitch_filter(
        d,
        "strike_zone_pitch_type",
    )

    st.metric(
        "Selected Pitches",
        len(d),
    )

    strike_zone_chart(
        d,
        zone_bottom,
        zone_top,
    )
