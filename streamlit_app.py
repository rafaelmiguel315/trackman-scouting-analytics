
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

HARD_CONTACT_MPH = 95.0
WEAK_CONTACT_MPH = 70.0
ZONE_HALF_WIDTH = 17 / 24


# ==========================================
# HELPERS
# ==========================================

def col(data, name):
    if name in data.columns:
        return data[name]

    return pd.Series(
        [None] * len(data),
        index=data.index,
    )


def numeric(data, name):
    return pd.to_numeric(
        col(data, name),
        errors="coerce",
    )


def normalized(data, name):
    return (
        col(data, name)
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "",
            regex=True,
        )
    )


def percentage(numerator, denominator):
    if denominator is None or denominator == 0:
        return None

    return 100.0 * numerator / denominator


def display_value(value, percent=False):
    if value is None or pd.isna(value):
        return "N/A"

    if percent:
        return f"{value:.1f}%"

    return f"{int(value):,}"


def show_cards(metrics, columns=4):
    items = list(metrics.items())

    for start in range(0, len(items), columns):
        row = st.columns(columns)

        for container, (label, value) in zip(
            row,
            items[start:start + columns],
        ):
            container.metric(label, value)


def metric_table(data, metrics):
    rows = []

    for field, (label, unit) in metrics.items():
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


def player_selector(data, field, key):
    if field not in data.columns:
        st.warning(f"{field} column not found.")
        st.stop()

    names = sorted(
        data[field]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.strip() != ""]
        .unique()
    )

    if not names:
        st.warning(f"No {field.lower()} records found.")
        st.stop()

    return st.selectbox(
        f"Select {field}",
        names,
        key=key,
    )


def filter_date(data, key):
    if "Date" not in data.columns:
        return data

    dates = sorted(
        data["Date"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected = st.selectbox(
        "Session Date",
        ["All dates"] + dates,
        key=key,
    )

    if selected != "All dates":
        return data[
            data["Date"].astype(str) == selected
        ].copy()

    return data


def filter_pitch(data, key):
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

    if selected != "All pitches":
        data = data[
            data["TaggedPitchType"].astype(str)
            == selected
        ].copy()

    return data, selected


# ==========================================
# PITCH CLASSIFICATION
# ==========================================

def classify_pitches(data, bottom, top):
    """
    Build common hitting and pitching flags.

    Contact includes fouls.

    Zone-based percentages require valid
    pitch locations.

    Missing data is not treated as a ball,
    strike, swing, or contact.
    """

    result = data.copy()

    calls = normalized(result, "PitchCall")

    swing_miss_calls = {
        "strikeswinging",
        "swingingstrike",
        "swingingstrikewildpitch",
        "swingingstrikeblocked",
    }

    foul_calls = {
        "foulball",
        "foul",
        "foultip",
        "foulballfieldable",
        "foulballnotfieldable",
        "foulbunt",
    }

    in_play_calls = {
        "inplay",
        "ballinplay",
        "inplayout",
        "inplayrun",
        "inplaynoout",
    }

    called_strikes = {
        "strikecalled",
        "calledstrike",
    }

    ball_calls = {
        "ballcalled",
        "calledball",
        "ballindirt",
        "ball",
        "intentball",
        "intentionalball",
        "pitchout",
        "hitbypitch",
    }

    result["_whiff"] = calls.isin(swing_miss_calls)

    result["_foul"] = calls.isin(foul_calls)

    result["_in_play"] = calls.isin(in_play_calls)

    result["_swing"] = (
        result["_whiff"]
        | result["_foul"]
        | result["_in_play"]
    )

    result["_contact"] = (
        result["_foul"]
        | result["_in_play"]
    )

    result["_strike"] = (
        result["_whiff"]
        | result["_foul"]
        | result["_in_play"]
        | calls.isin(called_strikes)
    )

    result["_ball"] = calls.isin(ball_calls)

    result["_known_call"] = (
        result["_strike"]
        | result["_ball"]
    )

    side = numeric(result, "PlateLocSide")
    height = numeric(result, "PlateLocHeight")

    result["_valid_location"] = (
        side.notna() & height.notna()
    )

    result["_in_zone"] = (
        result["_valid_location"]
        & side.between(
            -ZONE_HALF_WIDTH,
            ZONE_HALF_WIDTH,
        )
        & height.between(bottom, top)
    )

    result["_out_zone"] = (
        result["_valid_location"]
        & ~result["_in_zone"]
    )

    result["_exit_speed"] = numeric(
        result,
        "ExitSpeed",
    )

    result["_hard"] = (
        result["_in_play"]
        & (
            result["_exit_speed"]
            >= HARD_CONTACT_MPH
        )
    )

    result["_weak"] = (
        result["_in_play"]
        & (
            result["_exit_speed"]
            < WEAK_CONTACT_MPH
        )
    )

    result["_valid_ev"] = (
        result["_in_play"]
        & result["_exit_speed"].notna()
    )

    return result


# ==========================================
# COMMON PERCENTAGES
# ==========================================

def common_stats(data):
    total = len(data)

    swings = int(data["_swing"].sum())
    whiffs = int(data["_whiff"].sum())
    contacts = int(data["_contact"].sum())

    valid_locations = int(
        data["_valid_location"].sum()
    )

    out_zone = int(data["_out_zone"].sum())

    chase_swings = int(
        (
            data["_out_zone"]
            & data["_swing"]
        ).sum()
    )

    zone_swings = int(
        (
            data["_in_zone"]
            & data["_swing"]
        ).sum()
    )

    zone_contacts = int(
        (
            data["_in_zone"]
            & data["_contact"]
        ).sum()
    )

    out_contacts = int(
        (
            data["_out_zone"]
            & data["_contact"]
        ).sum()
    )

    valid_ev = int(data["_valid_ev"].sum())

    hard = int(data["_hard"].sum())
    weak = int(data["_weak"].sum())

    return {
        "Pitches": total,
        "Swing %": percentage(swings, total),
        "Whiff %": percentage(whiffs, swings),
        "Contact %": percentage(contacts, swings),
        "Chase %": percentage(
            chase_swings,
            out_zone,
        ),
        "Z-Contact %": percentage(
            zone_contacts,
            zone_swings,
        ),
        "O-Contact %": percentage(
            out_contacts,
            chase_swings,
        ),
        "Hard Contact %": percentage(
            hard,
            valid_ev,
        ),
        "Weak Contact %": percentage(
            weak,
            valid_ev,
        ),
        "Valid Locations": valid_locations,
        "EV Samples": valid_ev,
    }


# ==========================================
# PLATE APPEARANCE RESULTS
# ==========================================

def plate_appearance_stats(data):
    """
    Count completed plate appearances only
    when the CSV provides a reliable PA key
    and an explicit terminal result.

    Do not estimate K or BB from isolated
    pitch calls or partial sessions.
    """

    pa_fields = [
        "PAofInning",
        "PAofGame",
        "PlateAppearanceId",
        "PlateAppearanceID",
    ]

    pa_field = next(
        (
            name for name in pa_fields
            if name in data.columns
        ),
        None,
    )

    if pa_field is None or "KorBB" not in data.columns:
        return {
            "Strikeouts": None,
            "Walks": None,
            "K %": None,
            "BB %": None,
        }

    if pa_field == "PAofInning":
        if "Inning" not in data.columns:
            return {
                "Strikeouts": None,
                "Walks": None,
                "K %": None,
                "BB %": None,
            }

        group_fields = ["Inning", pa_field]

    else:
        group_fields = [pa_field]

    for extra in (
        "GameUID",
        "SessionId",
        "Date",
        "Top/Bottom",
    ):
        if extra in data.columns:
            group_fields.insert(0, extra)

    if "Batter" in data.columns:
        group_fields.append("Batter")

    if "Pitcher" in data.columns:
        group_fields.append("Pitcher")

    results = []

    for _, group in data.groupby(
        group_fields,
        dropna=False,
    ):
        outcomes = normalized(
            group,
            "KorBB",
        )

        if outcomes.isin({
            "strikeout",
            "k",
            "so",
        }).any():
            results.append("K")

        elif outcomes.isin({
            "walk",
            "bb",
            "baseonballs",
        }).any():
            results.append("BB")

        else:
            results.append("OTHER")

    return {
        "Strikeouts": results.count("K"),
        "Walks": results.count("BB"),
        "K %": None,
        "BB %": None,
    }


# ==========================================
# FIRST PITCH STRIKE
# ==========================================

def first_pitch_strike(data):
    if "PitchofPA" not in data.columns:
        return None

    pitch_number = numeric(
        data,
        "PitchofPA",
    )

    first = data.loc[
        pitch_number == 1
    ]

    if first.empty:
        return None

    known = first[
        first["_known_call"]
    ]

    if known.empty:
        return None

    return percentage(
        int(known["_strike"].sum()),
        len(known),
    )


# ==========================================
# HITTING HEADERS
# ==========================================

def show_hitting_headers(data, bottom, top):
    stats = common_stats(
        classify_pitches(
            data,
            bottom,
            top,
        )
    )

    metrics = {
        "Pitches Seen": display_value(
            stats["Pitches"]
        ),
        "Swing %": display_value(
            stats["Swing %"],
            True,
        ),
        "Whiff %": display_value(
            stats["Whiff %"],
            True,
        ),
        "Chase %": display_value(
            stats["Chase %"],
            True,
        ),
        "Contact %": display_value(
            stats["Contact %"],
            True,
        ),
        "Z-Contact %": display_value(
            stats["Z-Contact %"],
            True,
        ),
        "O-Contact %": display_value(
            stats["O-Contact %"],
            True,
        ),
        "Hard Contact %": display_value(
            stats["Hard Contact %"],
            True,
        ),
        "Weak Contact %": display_value(
            stats["Weak Contact %"],
            True,
        ),
    }

    show_cards(metrics)

    st.caption(
        "Contact includes fouls. "
        "Hard Contact: EV >= 95 mph. "
        "Weak Contact: EV < 70 mph. "
        "Zone metrics use the selected "
        "approximate strike zone."
    )


# ==========================================
# PITCHING HEADERS
# ==========================================

def show_pitching_headers(data, bottom, top):
    classified = classify_pitches(
        data,
        bottom,
        top,
    )

    stats = common_stats(classified)
    pa = plate_appearance_stats(classified)

    known_calls = int(
        classified["_known_call"].sum()
    )

    strikes = int(
        classified["_strike"].sum()
    )

    balls = int(
        classified["_ball"].sum()
    )

    fps = first_pitch_strike(classified)

    # Extension is measured in feet.
    extension_values = numeric(
        data,
        "Extension",
    ).dropna()

    if not extension_values.empty:
        avg_extension = (
            f"{extension_values.mean():.2f} ft"
        )
        max_extension = (
            f"{extension_values.max():.2f} ft"
        )
    else:
        avg_extension = "N/A"
        max_extension = "N/A"

    metrics = {
        "Total Pitches": display_value(
            len(classified)
        ),
        "Strike %": display_value(
            percentage(strikes, known_calls),
            True,
        ),
        "Ball %": display_value(
            percentage(balls, known_calls),
            True,
        ),
        "First Pitch Strike %": display_value(
            fps,
            True,
        ),
        "Whiff %": display_value(
            stats["Whiff %"],
            True,
        ),
        "Chase %": display_value(
            stats["Chase %"],
            True,
        ),
        "Strikeouts (K)": display_value(
            pa["Strikeouts"]
        ),
        "Walks (BB)": display_value(
            pa["Walks"]
        ),
        "K %": display_value(
            pa["K %"],
            True,
        ),
        "BB %": display_value(
            pa["BB %"],
            True,
        ),
        "Hard Contact Allowed %": display_value(
            stats["Hard Contact %"],
            True,
        ),
        "Weak Contact Allowed %": display_value(
            stats["Weak Contact %"],
            True,
        ),
        "AVG Extension": avg_extension,
        "MAX Extension": max_extension,
    }

    show_cards(metrics)

    st.caption(
        "Strike % and Ball % use recognized "
        "pitch calls. First Pitch Strike % "
        "requires PitchofPA. K and BB require "
        "reliable PA identifiers and KorBB. "
        "Extension is displayed in feet. "
        "Unavailable metrics display N/A."
    )


# ==========================================
# MOVEMENT PROFILE
# ==========================================

def show_movement_profile(data):
    required = [
        "HorzBreak",
        "InducedVertBreak",
    ]

    if not all(
        field in data.columns
        for field in required
    ):
        st.warning(
            "Movement columns not available."
        )
        return

    movement = data.copy()

    for field in required:
        movement[field] = numeric(
            movement,
            field,
        )

    movement = movement.dropna(
        subset=required
    )

    if movement.empty:
        st.info("No movement data available.")
        return

    if "TaggedPitchType" not in movement.columns:
        movement["TaggedPitchType"] = "Unknown"

    movement["TaggedPitchType"] = (
        movement["TaggedPitchType"]
        .fillna("Unknown")
        .astype(str)
    )

    st.subheader(
        "Pitch Movement Profile | Pitcher View"
    )

    fig = px.scatter(
        movement,
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
        zeroline=True,
        zerolinewidth=3,
        zerolinecolor="#111827",
        showgrid=True,
        gridcolor="#e5e7eb",
    )

    fig.update_yaxes(
        range=[-30, 30],
        dtick=5,
        zeroline=True,
        zerolinewidth=3,
        zerolinecolor="#111827",
        zerolinewidth=3,
        showgrid=True,
        gridcolor="#e5e7eb",
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        height=680,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#111827"),
        legend_title="Pitch Type",
        legend=dict(
            orientation="h",
            y=1.08,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ==========================================
# STRIKE ZONE
# ==========================================

def show_strike_zone(data, bottom, top):
    if not all(
        field in data.columns
        for field in [
            "PlateLocSide",
            "PlateLocHeight",
        ]
    ):
        st.warning(
            "Pitch location columns not found."
        )
        return

    locations = data.copy()

    locations["PlateLocSide"] = numeric(
        locations,
        "PlateLocSide",
    )

    locations["PlateLocHeight"] = numeric(
        locations,
        "PlateLocHeight",
    )

    locations = locations.dropna(
        subset=[
            "PlateLocSide",
            "PlateLocHeight",
        ]
    )

    if locations.empty:
        st.info("No pitch locations available.")
        return

    classified = classify_pitches(
        locations,
        bottom,
        top,
    )

    inside = classified["_in_zone"]

    st.subheader(
        "Pitch Location | Strike Zone"
    )

    a, b, c = st.columns(3)

    a.metric(
        "In Zone",
        int(inside.sum()),
    )

    b.metric(
        "Out of Zone",
        int((~inside).sum()),
    )

    c.metric(
        "Zone %",
        display_value(
            percentage(
                int(inside.sum()),
                len(locations),
            ),
            True,
        ),
    )

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

        y = bottom + (top - bottom) * i / 3

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

    locations["_zone"] = inside.map({
        True: "In Zone",
        False: "Out of Zone",
    })

    for label, color in {
        "In Zone": "#2563eb",
        "Out of Zone": "#ef4444",
    }.items():
        subset = locations[
            locations["_zone"] == label
        ]

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

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Blue = inside approximate zone. "
        "Red = outside approximate zone. "
        "This is not an official umpire call."
    )

    if "PitchCall" in locations.columns:
        st.subheader("Recorded Pitch Calls")

        calls = (
            locations["PitchCall"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Pitch Call")
            .reset_index(name="Count")
        )

        st.dataframe(
            calls,
            hide_index=True,
            use_container_width=True,
        )


# ==========================================
# GLOBAL SETTINGS
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
        "Strike zone top must be greater "
        "than strike zone bottom."
    )
    st.stop()


# ==========================================
# NAVIGATION
# ==========================================

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
                st.session_state.get(
                    "last_upload_attempt"
                ) != fingerprint
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
                        "This CSV is already stored. "
                        "No duplicate records added."
                    )
                else:
                    st.success(
                        f"CSV saved: {count:,} pitches."
                    )
            else:
                st.info(
                    "File already processed in "
                    "this session."
                )

            if (
                st.session_state.trackman_data
                is not None
            ):
                st.dataframe(
                    st.session_state
                    .trackman_data
                    .tail(20),
                    use_container_width=True,
                )

        except Exception as error:
            st.error(
                f"CSV import failed: {error}"
            )


df = st.session_state.trackman_data

if page != "CSV Upload":
    if df is None or df.empty:
        st.info(
            "No TrackMan records stored yet. "
            "Go to CSV Upload to import data."
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
        (
            df["Pitcher"].nunique()
            if "Pitcher" in df.columns
            else 0
        ),
    )

    c.metric(
        "Batters",
        (
            df["Batter"].nunique()
            if "Batter" in df.columns
            else 0
        ),
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

    pitcher = player_selector(
        df,
        "Pitcher",
        "analytics_pitcher",
    )

    pitcher_data = df[
        df["Pitcher"].astype(str) == pitcher
    ].copy()

    pitcher_data = filter_date(
        pitcher_data,
        "analytics_date",
    )

    pitcher_data, selected_type = filter_pitch(
        pitcher_data,
        "analytics_pitch_type",
    )

    st.subheader(pitcher)

    show_pitching_headers(
        pitcher_data,
        zone_bottom,
        zone_top,
    )

    st.subheader(
        f"Pitching Metrics | {selected_type}"
    )

    metric_table(
        pitcher_data,
        PITCH_METRICS,
    )

    if "RelSpeed" in pitcher_data.columns:
        velocity = pitcher_data.copy()

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

    show_movement_profile(pitcher_data)


# ==========================================
# HITTER ANALYTICS
# ==========================================

elif page == "Hitter Analytics":
    st.header("Hitter Analytics")

    batter = player_selector(
        df,
        "Batter",
        "analytics_batter",
    )

    hitter_data = df[
        df["Batter"].astype(str) == batter
    ].copy()

    hitter_data = filter_date(
        hitter_data,
        "hitter_date",
    )

    st.subheader(batter)

    show_hitting_headers(
        hitter_data,
        zone_bottom,
        zone_top,
    )

    st.subheader("Hitting Metrics")

    metric_table(
        hitter_data,
        HITTER_METRICS,
    )


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

    player = player_selector(
        df,
        role,
        f"profile_{role.lower()}",
    )

    player_data = df[
        df[role].astype(str) == player
    ].copy()

    player_data = filter_date(
        player_data,
        f"profile_date_{role.lower()}",
    )

    st.header(player)

    if role == "Pitcher":
        st.subheader("Pitching Profile")

        show_pitching_headers(
            player_data,
            zone_bottom,
            zone_top,
        )

        metric_table(
            player_data,
            PITCH_METRICS,
        )

        show_movement_profile(player_data)

    else:
        st.subheader("Hitting Profile")

        show_hitting_headers(
            player_data,
            zone_bottom,
            zone_top,
        )

        metric_table(
            player_data,
            HITTER_METRICS,
        )


# ==========================================
# STRIKE ZONE
# ==========================================

elif page == "Strike Zone":
    st.header("Strike Zone")

    pitcher = player_selector(
        df,
        "Pitcher",
        "strike_pitcher",
    )

    strike_data = df[
        df["Pitcher"].astype(str) == pitcher
    ].copy()

    strike_data = filter_date(
        strike_data,
        "strike_date",
    )

    strike_data, selected_type = filter_pitch(
        strike_data,
        "strike_pitch_type",
    )

    st.metric(
        "Selected Pitches",
        len(strike_data),
    )

    show_strike_zone(
        strike_data,
        zone_bottom,
        zone_top,
    )
