
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from supabase_store import authenticate, import_csv, load_all

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

HARD_EV = 95.0
WEAK_EV = 70.0
ZONE_HALF_WIDTH = 17 / 24


# ==========================================
# GENERAL HELPERS
# ==========================================

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
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def pct(num, den):
    if den is None or den == 0:
        return None
    return 100 * num / den


def fmt(value, percent=False):
    if value is None or pd.isna(value):
        return "N/A"
    if percent:
        return f"{value:.1f}%"
    return f"{int(value):,}"


def metric_cards(items, per_row=4):
    entries = list(items.items())
    for start in range(0, len(entries), per_row):
        cols = st.columns(per_row)
        for c, (name, value) in zip(
            cols, entries[start:start + per_row]
        ):
            c.metric(name, value)


def metric_table(data, definitions):
    rows = []
    for field, (label, unit) in definitions.items():
        values = numeric(data, field).dropna()
        if values.empty:
            continue
        rows.append({
            "Metric": f"{label} ({unit})",
            "AVG": round(values.mean(), 2),
            "MAX": round(values.max(), 2),
            "MIN": round(values.min(), 2),
            "Count": len(values),
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No numeric metrics available.")


def choose_player(data, field, key):
    if field not in data.columns:
        st.warning(f"{field} column not available.")
        st.stop()

    names = sorted(
        x for x in data[field].dropna().astype(str).unique()
        if x.strip()
    )
    if not names:
        st.warning("No players available.")
        st.stop()

    return st.selectbox(
        f"Select {field}", names, key=key
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
        .dropna().astype(str).unique()
    )
    selected = st.selectbox(
        "Pitch Type",
        ["All pitches"] + types,
        key=key,
    )
    if selected == "All pitches":
        return data, selected

    return data[
        data["TaggedPitchType"].astype(str) == selected
    ].copy(), selected


# ==========================================
# PITCH CALL CLASSIFICATION
# ==========================================

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

# Foul bunts with two strikes and special scoring
# situations may need separate treatment.


def classify(data, zone_bottom, zone_top):
    d = data.copy()
    calls = normalized(d, "PitchCall")

    d["_whiff"] = calls.isin(WHIFF_CALLS)
    d["_foul"] = calls.isin(FOUL_CALLS)
    d["_inplay"] = calls.isin(IN_PLAY_CALLS)

    d["_swing"] = (
        d["_whiff"] | d["_foul"] | d["_inplay"]
    )
    d["_contact"] = d["_foul"] | d["_inplay"]

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

    d["_located"] = side.notna() & height.notna()
    d["_zone"] = (
        d["_located"]
        & side.between(
            -ZONE_HALF_WIDTH, ZONE_HALF_WIDTH
        )
        & height.between(zone_bottom, zone_top)
    )
    d["_outside"] = d["_located"] & ~d["_zone"]

    ev = numeric(d, "ExitSpeed")
    d["_valid_ev"] = d["_inplay"] & ev.notna()
    d["_hard"] = d["_valid_ev"] & (ev >= HARD_EV)
    d["_weak"] = d["_valid_ev"] & (ev < WEAK_EV)

    return d


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
        "Chase %": pct(outside_swings, outside),
        "Z-Contact %": pct(zone_contacts, zone_swings),
        "O-Contact %": pct(
            outside_contacts, outside_swings
        ),
        "Hard Contact %": pct(
            int(d["_hard"].sum()), ev_count
        ),
        "Weak Contact %": pct(
            int(d["_weak"].sum()), ev_count
        ),
    }


# ==========================================
# FIRST PITCH STRIKE
# ==========================================

def first_pitch_strike(d):
    if "PitchofPA" not in d.columns:
        return None

    first = d[numeric(d, "PitchofPA") == 1]
    if first.empty:
        return None

    first = first[first["_known"]]
    if first.empty:
        return None

    return pct(
        int(first["_strike"].sum()),
        len(first),
    )


# ==========================================
# STRIKEOUTS AND WALKS
# ==========================================

def pa_results(d):
    unavailable = {
        "K": None,
        "BB": None,
        "K %": None,
        "BB %": None,
    }

    if "KorBB" not in d.columns:
        return unavailable

    # Require a reliable PA identifier.
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

    for _, group in d.groupby(keys, dropna=False):
        outcomes = normalized(group, "KorBB")

        if outcomes.isin(
            {"strikeout", "k", "so"}
        ).any():
            strikeouts += 1
        elif outcomes.isin(
            {"walk", "bb", "baseonballs"}
        ).any():
            walks += 1

    # PA completion cannot be verified from every
    # type of TrackMan CSV, so K% and BB% remain N/A.
    return {
        "K": strikeouts,
        "BB": walks,
        "K %": None,
        "BB %": None,
    }


# ==========================================
# HITTER HEADERS
# ==========================================

def hitting_headers(data, bottom, top):
    d = classify(data, bottom, top)
    rates = batting_rates(d)

    cards = {
        "Pitches Seen": fmt(len(d)),
        "Swing %": fmt(rates["Swing %"], True),
        "Whiff %": fmt(rates["Whiff %"], True),
        "Chase %": fmt(rates["Chase %"], True),
        "Contact %": fmt(rates["Contact %"], True),
        "Z-Contact %": fmt(rates["Z-Contact %"], True),
        "O-Contact %": fmt(rates["O-Contact %"], True),
        "Hard Contact %": fmt(
            rates["Hard Contact %"], True
        ),
        "Weak Contact %": fmt(
            rates["Weak Contact %"], True
        ),
    }

    metric_cards(cards)

    st.caption(
        "Contact includes fouls. "
        "Hard Contact: EV >= 95 mph. "
        "Weak Contact: EV < 70 mph. "
        "Zone rates use the approximate strike zone."
    )


# ==========================================
# PITCHER HEADERS
# ==========================================

def pitching_headers(data, bottom, top):
    d = classify(data, bottom, top)
    rates = batting_rates(d)
    pa = pa_results(d)

    known = int(d["_known"].sum())
    strikes = int(d["_strike"].sum())
    balls = int(d["_ball"].sum())

    cards = {
        "Total Pitches": fmt(len(d)),
        "Strike %": fmt(pct(strikes, known), True),
        "Ball %": fmt(pct(balls, known), True),
        "First Pitch Strike %": fmt(
            first_pitch_strike(d), True
        ),
        "Whiff %": fmt(rates["Whiff %"], True),
        "Chase %": fmt(rates["Chase %"], True),
        "Strikeouts (K)": fmt(pa["K"]),
        "Walks (BB)": fmt(pa["BB"]),
        "K %": fmt(pa["K %"], True),
        "BB %": fmt(pa["BB %"], True),
        "Hard Contact Allowed %": fmt(
            rates["Hard Contact %"], True
        ),
        "Weak Contact Allowed %": fmt(
            rates["Weak Contact %"], True
        ),
    }

    metric_cards(cards)

    st.caption(
        "Strike and Ball percentages use recognized "
        "pitch calls. K and BB require reliable "
        "plate-appearance identifiers and KorBB. "
        "Unavailable statistics show N/A."
    )


# ==========================================
# MOVEMENT PROFILE
# ==========================================

def movement_profile(data):
    required = ["HorzBreak", "InducedVertBreak"]

    if not all(x in data.columns for x in required):
        st.info("Movement data unavailable.")
        return

    d = data.copy()
    for field in required:
        d[field] = numeric(d, field)

    d = d.dropna(subset=required)
    if d.empty:
        st.info("No movement data available.")
        return

    if "TaggedPitchType" not in d.columns:
        d["TaggedPitchType"] = "Unknown"

    d["TaggedPitchType"] = (
        d["TaggedPitchType"]
        .fillna("Unknown")
        .astype(str)
    )

    st.subheader("Movement Profile | Pitcher View")

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
            line=dict(color="white", width=1),
        )
    )

    # Each keyword appears only once.
    fig.update_xaxes(
        range=[-30, 30],
        dtick=5,
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
        height=680,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#111827"),
        legend_title="Pitch Type",
        legend=dict(orientation="h", y=1.08),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ==========================================
# STRIKE ZONE
# ==========================================

def strike_zone_chart(data, bottom, top):
    required = ["PlateLocSide", "PlateLocHeight"]

    if not all(x in data.columns for x in required):
        st.info("Pitch locations unavailable.")
        return

    d = data.copy()
    for field in required:
        d[field] = numeric(d, field)

    d = d.dropna(subset=required)

    if d.empty:
        st.info("No pitch locations available.")
        return

    classified = classify(d, bottom, top)
    inside = classified["_zone"]

    st.subheader("Pitch Location | Strike Zone")

    a, b, c = st.columns(3)
    a.metric("In Zone", int(inside.sum()))
    b.metric("Out of Zone", int((~inside).sum()))
    c.metric(
        "Zone %",
        fmt(pct(int(inside.sum()), len(d)), True),
    )

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=-ZONE_HALF_WIDTH,
        x1=ZONE_HALF_WIDTH,
        y0=bottom,
        y1=top,
        line=dict(color="#334155", width=3),
        fillcolor="rgba(148,163,184,0.10)",
    )

    for i in [1, 2]:
        x = -ZONE_HALF_WIDTH + (
            2 * ZONE_HALF_WIDTH * i / 3
        )
        y = bottom + (top - bottom) * i / 3

        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=bottom,
            y1=top,
            line=dict(color="#94a3b8", dash="dot"),
        )

        fig.add_shape(
            type="line",
            x0=-ZONE_HALF_WIDTH,
            x1=ZONE_HALF_WIDTH,
            y0=y,
            y1=y,
            line=dict(color="#94a3b8", dash="dot"),
        )

    d["_zone_label"] = inside.map({
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
                    line=dict(color="white", width=1),
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

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Blue = inside approximate zone. "
        "Red = outside. Not an official umpire call."
    )


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Analysis Settings")

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

page = st.sidebar.radio("Navigation", pages)


# ==========================================
# CSV UPLOAD
# ==========================================

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
                st.session_state.get("last_upload_attempt")
                != fingerprint
            ):
                status, count = import_csv(
                    sb, file_bytes, uploaded.name
                )

                st.session_state.last_upload_attempt = (
                    fingerprint
                )
                st.session_state.trackman_data = load_all(sb)

                if status == "duplicate":
                    st.info(
                        "CSV already exists. No duplicates added."
                    )
                else:
                    st.success(
                        f"CSV saved: {count:,} pitches."
                    )
            else:
                st.info(
                    "File already processed in this session."
                )

            saved = st.session_state.trackman_data
            if saved is not None:
                st.dataframe(
                    saved.tail(20),
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"CSV import failed: {error}")


df = st.session_state.trackman_data

if page != "CSV Upload":
    if df is None or df.empty:
        st.info(
            "No saved TrackMan records yet. "
            "Go to CSV Upload."
        )
        st.stop()


# ==========================================
# OVERVIEW
# ==========================================

if page == "Overview":
    st.header("Overview")

    a, b, c = st.columns(3)
    a.metric("Total Records", len(df))
    b.metric(
        "Pitchers",
        df["Pitcher"].nunique()
        if "Pitcher" in df.columns else 0,
    )
    c.metric(
        "Batters",
        df["Batter"].nunique()
        if "Batter" in df.columns else 0,
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


# ==========================================
# PITCHER ANALYTICS
# ==========================================

elif page == "Pitcher Analytics":
    st.header("Pitcher Analytics")

    pitcher = choose_player(
        df, "Pitcher", "pitcher_analytics"
    )

    d = df[
        df["Pitcher"].astype(str) == pitcher
    ].copy()

    d = date_filter(d, "pitcher_date")
    d, pitch_type = pitch_filter(
        d, "pitcher_pitch_type"
    )

    st.subheader(pitcher)
    pitching_headers(d, zone_bottom, zone_top)

    st.subheader(f"Pitch Metrics | {pitch_type}")
    metric_table(d, PITCH_METRICS)

    if "RelSpeed" in d.columns:
        velocity = d.copy()
        velocity["RelSpeed"] = numeric(
            velocity, "RelSpeed"
        )
        velocity = velocity.dropna(
            subset=["RelSpeed"]
        )

        if not velocity.empty:
            if "TaggedPitchType" not in velocity.columns:
                velocity["TaggedPitchType"] = "Unknown"

            st.subheader("Velocity by Pitch Type")

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


# ==========================================
# HITTER ANALYTICS
# ==========================================

elif page == "Hitter Analytics":
    st.header("Hitter Analytics")

    batter = choose_player(
        df, "Batter", "hitter_analytics"
    )

    d = df[
        df["Batter"].astype(str) == batter
    ].copy()

    d = date_filter(d, "hitter_date")

    st.subheader(batter)
    hitting_headers(d, zone_bottom, zone_top)

    st.subheader("Hitting Metrics")
    metric_table(d, HITTER_METRICS)


# ==========================================
# PLAYER PROFILES
# ==========================================

elif page == "Player Profiles":
    st.header("Player Profiles")

    role = st.radio(
        "Player Role",
        ["Pitcher", "Batter"],
        horizontal=True,
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
        pitching_headers(d, zone_bottom, zone_top)
        metric_table(d, PITCH_METRICS)
        movement_profile(d)
    else:
        st.subheader("Hitting Profile")
        hitting_headers(d, zone_bottom, zone_top)
        metric_table(d, HITTER_METRICS)


# ==========================================
# STRIKE ZONE
# ==========================================

elif page == "Strike Zone":
    st.header("Strike Zone")

    pitcher = choose_player(
        df, "Pitcher", "strike_zone_pitcher"
    )

    d = df[
        df["Pitcher"].astype(str) == pitcher
    ].copy()

    d = date_filter(d, "strike_zone_date")
    d, pitch_type = pitch_filter(
        d, "strike_zone_pitch_type"
    )

    st.metric("Selected Pitches", len(d))
    strike_zone_chart(d, zone_bottom, zone_top)
