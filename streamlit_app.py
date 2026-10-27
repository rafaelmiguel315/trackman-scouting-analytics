
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase_store import authenticate, import_csv, load_all

st.set_page_config(
    page_title="TrackMan Scouting Analytics",
    page_icon="⚾",
    layout="wide"
)

sb = authenticate()
st.title("⚾ TrackMan Scouting Analytics")
st.caption("International Scouting | Video & Data")

# ---------------------------
# SESSION DATA
# ---------------------------

if "trackman_data" not in st.session_state:
    st.session_state.trackman_data = load_all(sb)

PITCH_METRICS = {
    "RelSpeed": ("Velocity", "mph"),
    "SpinRate": ("Spin Rate", "rpm"),
    "InducedVertBreak": ("IVB", "in"),
    "HorzBreak": ("Horizontal Break", "in"),
    "VertApprAngle": ("VAA", "degrees"),
    "Extension": ("Extension", "ft"),
    "RelHeight": ("Release Height", "ft")
}

HITTER_METRICS = {
    "ExitSpeed": ("Exit Velocity", "mph"),
    "Angle": ("Launch Angle", "degrees"),
    "Distance": ("Distance", "ft")
}

PITCH_COLORS = {
    "Fastball": "#2563eb",
    "Four-Seam": "#2563eb",
    "FourSeamFastBall": "#2563eb",
    "Sinker": "#0ea5e9",
    "Two-Seam": "#0ea5e9",
    "Slider": "#ef4444",
    "Curveball": "#a855f7",
    "ChangeUp": "#16a34a",
    "Changeup": "#16a34a",
    "Cutter": "#f59e0b",
    "Splitter": "#14b8a6"
}

# ---------------------------
# HELPERS
# ---------------------------

def numeric_column(data, column):
    return pd.to_numeric(
        data[column],
        errors="coerce"
    )


def metric_summary(data, metrics):
    rows = []

    for column, (name, unit) in metrics.items():
        if column not in data.columns:
            continue

        values = numeric_column(
            data, column
        ).dropna()

        if values.empty:
            continue

        rows.append({
            "Metric": f"{name} ({unit})",
            "AVG": round(values.mean(), 2),
            "MAX": round(values.max(), 2),
            "MIN": round(values.min(), 2),
            "Count": len(values)
        })

    return pd.DataFrame(rows)


def show_metric_table(data, metrics):
    summary = metric_summary(data, metrics)

    if summary.empty:
        st.info("No numeric metrics available.")
        return

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Descriptive statistics only. "
        "MLB benchmarks and grades will be added later."
    )


def select_pitcher(data, key):
    if "Pitcher" not in data.columns:
        st.error("Pitcher column not found.")
        st.stop()

    pitchers = sorted(
        data["Pitcher"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not pitchers:
        st.warning("No pitchers found.")
        st.stop()

    return st.selectbox(
        "Select Pitcher",
        pitchers,
        key=key
    )


def filter_pitch_type(data, key):
    if "TaggedPitchType" not in data.columns:
        return data, "All pitches"

    pitch_types = sorted(
        data["TaggedPitchType"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected = st.selectbox(
        "Pitch Type",
        ["All pitches"] + pitch_types,
        key=key
    )

    if selected != "All pitches":
        data = data[
            data["TaggedPitchType"].astype(str)
            == selected
        ].copy()

    return data, selected


def filter_session_date(data, key):
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
        key=key
    )

    if selected != "All dates":
        data = data[
            data["Date"].astype(str) == selected
        ].copy()

    return data


def pitch_type_colors(data):
    if "TaggedPitchType" not in data.columns:
        return {}

    types = data["TaggedPitchType"].unique()

    return {
        pitch: PITCH_COLORS.get(
            pitch,
            "#64748b"
        )
        for pitch in types
    }


# ---------------------------
# MOVEMENT PROFILE
# ---------------------------

def show_movement_profile(data):
    required = [
        "HorzBreak",
        "InducedVertBreak"
    ]

    if not all(c in data.columns for c in required):
        st.warning(
            "Horizontal Break or IVB columns not found."
        )
        return

    movement = data.copy()

    for column in required:
        movement[column] = numeric_column(
            movement, column
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
        color_discrete_map=pitch_type_colors(
            movement
        ),
        labels={
            "HorzBreak": "Horizontal Break (in)",
            "InducedVertBreak": "IVB (in)",
            "TaggedPitchType": "Pitch Type"
        },
        hover_data={
            "HorzBreak": ":.2f",
            "InducedVertBreak": ":.2f"
        }
    )

    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.85,
            line=dict(
                color="white",
                width=1
            )
        )
    )

    # Cartesian plane centered at zero.
    # Fixed equal scales prevent distortion.
    fig.update_xaxes(
        range=[-30, 30],
        dtick=5,
        zeroline=True,
        zerolinecolor="#111827",
        zerolinewidth=3,
        showgrid=True,
        gridcolor="#e5e7eb",
        title="Horizontal Break (in)"
    )

    fig.update_yaxes(
        range=[-30, 30],
        dtick=5,
        zeroline=True,
        zerolinecolor="#111827",
        zerolinewidth=3,
        showgrid=True,
        gridcolor="#e5e7eb",
        scaleanchor="x",
        scaleratio=1,
        title="Induced Vertical Break (in)"
    )

    fig.update_layout(
        height=720,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#111827"),
        legend_title="Pitch Type",
        legend=dict(
            orientation="h",
            y=1.08,
            x=0
        ),
        margin=dict(
            l=35,
            r=35,
            t=65,
            b=40
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Each point represents one pitch. "
        "Horizontal and vertical axes use the same "
        "scale in inches. The horizontal direction "
        "follows the CSV's HorzBreak sign convention."
    )


# ---------------------------
# STRIKE ZONE
# ---------------------------

def show_strike_zone(data):
    required = [
        "PlateLocSide",
        "PlateLocHeight"
    ]

    if not all(c in data.columns for c in required):
        st.error(
            "Pitch location columns not found."
        )
        return

    locations = data.copy()

    for column in required:
        locations[column] = numeric_column(
            locations, column
        )

    locations = locations.dropna(
        subset=required
    )

    st.subheader("Pitch Location | Strike Zone")

    st.caption(
        "Approximate strike zone. "
        "Adjust the top and bottom to match "
        "the batter's strike zone."
    )

    c1, c2 = st.columns(2)

    with c1:
        zone_bottom = st.number_input(
            "Zone Bottom (ft)",
            min_value=0.0,
            max_value=5.0,
            value=1.50,
            step=0.05
        )

    with c2:
        zone_top = st.number_input(
            "Zone Top (ft)",
            min_value=0.0,
            max_value=6.0,
            value=3.50,
            step=0.05
        )

    if zone_top <= zone_bottom:
        st.error(
            "Zone Top must be greater than Zone Bottom."
        )
        return

    if locations.empty:
        st.info("No pitch locations available.")
        return

    half_width = 17 / 24

    inside = (
        locations["PlateLocSide"].between(
            -half_width,
            half_width
        )
        &
        locations["PlateLocHeight"].between(
            zone_bottom,
            zone_top
        )
    )

    locations["Zone Location"] = (
        "Out of Zone"
    )

    locations.loc[
        inside,
        "Zone Location"
    ] = "In Zone"

    in_count = int(inside.sum())
    out_count = len(locations) - in_count

    a, b, c = st.columns(3)

    a.metric("In Zone", in_count)
    b.metric("Out of Zone", out_count)
    c.metric(
        "Zone %",
        f"{in_count / len(locations) * 100:.1f}%"
    )

    fig = go.Figure()

    # Rectangular strike zone
    fig.add_shape(
        type="rect",
        x0=-half_width,
        x1=half_width,
        y0=zone_bottom,
        y1=zone_top,
        line=dict(
            color="#334155",
            width=3
        ),
        fillcolor="rgba(148,163,184,0.10)"
    )

    # 3x3 strike-zone grid
    for i in [1, 2]:
        x = (
            -half_width
            + (2 * half_width * i / 3)
        )

        y = (
            zone_bottom
            + (zone_top - zone_bottom) * i / 3
        )

        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=zone_bottom,
            y1=zone_top,
            line=dict(
                color="#94a3b8",
                dash="dot",
                width=1
            )
        )

        fig.add_shape(
            type="line",
            x0=-half_width,
            x1=half_width,
            y0=y,
            y1=y,
            line=dict(
                color="#94a3b8",
                dash="dot",
                width=1
            )
        )

    zone_colors = {
        "In Zone": "#2563eb",
        "Out of Zone": "#ef4444"
    }

    for status, color in zone_colors.items():
        subset = locations[
            locations["Zone Location"] == status
        ]

        fig.add_trace(
            go.Scatter(
                x=subset["PlateLocSide"],
                y=subset["PlateLocHeight"],
                mode="markers",
                name=status,
                marker=dict(
                    color=color,
                    size=10,
                    opacity=0.85,
                    line=dict(
                        color="white",
                        width=1
                    )
                ),
                hovertemplate=(
                    "Horizontal: %{x:.2f} ft"
                    "<br>Height: %{y:.2f} ft"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        height=650,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#111827"),
        xaxis=dict(
            title="Horizontal Location (ft)",
            range=[-2.5, 2.5],
            zeroline=False,
            showgrid=False
        ),
        yaxis=dict(
            title="Height (ft)",
            range=[0, 5],
            zeroline=False,
            showgrid=False,
            scaleanchor="x",
            scaleratio=1
        ),
        legend=dict(
            orientation="h",
            y=1.08
        ),
        margin=dict(
            l=25,
            r=25,
            t=55,
            b=30
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Blue = location inside the approximate zone. "
        "Red = location outside. "
        "This does not determine the official ball/strike call."
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
            use_container_width=True
        )


# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------

pages = [
    "Overview",
    "Pitcher Analytics",
    "Hitter Analytics",
    "Player Profiles",
    "Strike Zone",
    "CSV Upload"
]

page = st.sidebar.radio(
    "Navigation",
    pages
)

# ---------------------------
# CSV UPLOAD
# ---------------------------

if page == "CSV Upload":
    st.header("Upload TrackMan CSV")

    uploaded = st.file_uploader(
        "Choose a TrackMan CSV file",
        type=["csv"]
    )

    if uploaded is not None:
        try:
            file_bytes = uploaded.getvalue()
            # The file uploader persists across reruns; hash prevents repeated writes.
            import hashlib
            fingerprint = hashlib.sha256(file_bytes).hexdigest()
            if st.session_state.get('last_upload_attempt') != fingerprint:
                status, count = import_csv(sb, file_bytes, uploaded.name)
                st.session_state.last_upload_attempt = fingerprint
                st.session_state.trackman_data = load_all(sb)
                if status == 'duplicate':
                    st.info('Este CSV ya existe en Supabase. No se duplicaron los datos.')
                else:
                    st.success(f'CSV guardado automáticamente: {count:,} lanzamientos.')
            else:
                st.info('Archivo procesado. Selecciona otro CSV para continuar.')
            if st.session_state.trackman_data is not None:
                st.dataframe(st.session_state.trackman_data.tail(20), use_container_width=True)
        except Exception as error:
            st.error(f'No se pudo importar el CSV: {error}')

df = st.session_state.trackman_data

if page != "CSV Upload" and df is None:
    st.info(
        "No hay lanzamientos guardados todavía. Ve a CSV Upload para importar un archivo."
    )
    st.stop()

# ---------------------------
# OVERVIEW
# ---------------------------

if page == "Overview":
    st.header("Overview")

    a, b, c = st.columns(3)

    a.metric("Total Records", len(df))

    b.metric(
        "Pitchers",
        df["Pitcher"].nunique()
        if "Pitcher" in df.columns else 0
    )

    c.metric(
        "Batters",
        df["Batter"].nunique()
        if "Batter" in df.columns else 0
    )

    if "TaggedPitchType" in df.columns:
        st.subheader("Pitch Distribution")

        counts = (
            df["TaggedPitchType"]
            .value_counts()
            .rename_axis("Pitch Type")
            .reset_index(name="Count")
        )

        fig = px.bar(
            counts,
            x="Pitch Type",
            y="Count",
            color="Pitch Type",
            color_discrete_map=PITCH_COLORS
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ---------------------------
# PITCHER ANALYTICS
# ---------------------------

elif page == "Pitcher Analytics":
    st.header("Pitcher Analytics")

    pitcher = select_pitcher(
        df,
        "analytics_pitcher"
    )

    pitcher_data = df[
        df["Pitcher"].astype(str) == pitcher
    ].copy()

    pitcher_data = filter_session_date(
        pitcher_data,
        "analytics_date"
    )

    pitcher_data, selected_type = filter_pitch_type(
        pitcher_data,
        "analytics_pitch_type"
    )

    st.metric(
        "Pitch Count",
        len(pitcher_data)
    )

    st.subheader(
        f"Pitching Metrics | {selected_type}"
    )

    show_metric_table(
        pitcher_data,
        PITCH_METRICS
    )

    # Velocity distribution
    if "RelSpeed" in pitcher_data.columns:
        velocity = pitcher_data.copy()

        velocity["RelSpeed"] = numeric_column(
            velocity,
            "RelSpeed"
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
                labels={
                    "TaggedPitchType": "Pitch Type",
                    "RelSpeed": "Velocity (mph)"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # Cartesian movement chart
    show_movement_profile(pitcher_data)

# ---------------------------
# HITTER ANALYTICS
# ---------------------------

elif page == "Hitter Analytics":
    st.header("Hitter Analytics")

    if "Batter" not in df.columns:
        st.warning("Batter column not found.")
        st.stop()

    batters = sorted(
        df["Batter"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not batters:
        st.warning("No batters found.")
        st.stop()

    batter = st.selectbox(
        "Select Batter",
        batters
    )

    hitter_data = df[
        df["Batter"].astype(str) == batter
    ].copy()

    hitter_data = filter_session_date(
        hitter_data,
        "hitter_date"
    )

    st.metric(
        "Records",
        len(hitter_data)
    )

    show_metric_table(
        hitter_data,
        HITTER_METRICS
    )

# ---------------------------
# PLAYER PROFILES
# ---------------------------

elif page == "Player Profiles":
    st.header("Player Profiles")

    players = set()

    for column in ["Pitcher", "Batter"]:
        if column in df.columns:
            players.update(
                df[column]
                .dropna()
                .astype(str)
                .unique()
            )

    if not players:
        st.warning("No players found.")
        st.stop()

    player = st.selectbox(
        "Select Player",
        sorted(players)
    )

    st.subheader(player)

    for role, column in [
        ("Pitching", "Pitcher"),
        ("Hitting", "Batter")
    ]:
        if column not in df.columns:
            continue

        player_data = df[
            df[column].astype(str) == player
        ]

        if player_data.empty:
            continue

        st.subheader(role)

        st.metric(
            "Records",
            len(player_data)
        )

        metrics = (
            PITCH_METRICS
            if role == "Pitching"
            else HITTER_METRICS
        )

        show_metric_table(
            player_data,
            metrics
        )

# ---------------------------
# STRIKE ZONE
# ---------------------------

elif page == "Strike Zone":
    st.header("Strike Zone")

    pitcher = select_pitcher(
        df,
        "strike_pitcher"
    )

    strike_data = df[
        df["Pitcher"].astype(str) == pitcher
    ].copy()

    strike_data = filter_session_date(
        strike_data,
        "strike_date"
    )

    strike_data, selected_type = filter_pitch_type(
        strike_data,
        "strike_pitch_type"
    )

    st.metric(
        "Selected Pitches",
        len(strike_data)
    )

    show_strike_zone(strike_data)
