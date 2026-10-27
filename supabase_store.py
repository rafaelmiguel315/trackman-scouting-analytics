
"""Authenticated TrackMan CSV persistence. Never use a service-role key here."""
import hashlib
import io
import json
import math
from datetime import date

import pandas as pd
import streamlit as st
from supabase import create_client


def client():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


def authenticate():
    """Require a valid Supabase login and authorized membership."""

    if "supabase_session" not in st.session_state:
        st.title("⚾ TrackMan Scouting Analytics")
        st.caption("Acceso autorizado | International Scouting")

        with st.form("supabase_login"):
            email = st.text_input("Correo electrónico")
            password = st.text_input(
                "Contraseña",
                type="password"
            )
            submitted = st.form_submit_button("Iniciar sesión")

        if submitted:
            try:
                result = client().auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                if not result.session:
                    raise ValueError("No se pudo iniciar sesión.")

                st.session_state.supabase_session = {
                    "access_token": result.session.access_token,
                    "refresh_token": result.session.refresh_token
                }
                st.rerun()

            except Exception:
                st.error(
                    "No se pudo iniciar sesión. Comprueba tus credenciales."
                )

        st.stop()

    sb = client()

    try:
        saved = st.session_state.supabase_session

        sb.auth.set_session(
            saved["access_token"],
            saved["refresh_token"]
        )

        session = sb.auth.get_session()

        if not session or not session.access_token:
            raise ValueError("Sesión caducada")

        user_response = sb.auth.get_user()
        user = user_response.user

        if not user:
            raise ValueError("Usuario no autenticado")

        st.session_state.supabase_session = {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token
        }

    except Exception:
        st.session_state.pop("supabase_session", None)
        st.rerun()

    try:
        membership = (
            sb.table("app_admins")
            .select("user_id")
            .eq("user_id", str(user.id))
            .limit(1)
            .execute()
            .data
        )

    except Exception:
        st.error(
            "No se pudo verificar tu autorización. "
            "Revisa los permisos de Supabase."
        )
        st.stop()

    if not membership:
        st.error(
            "Tu cuenta no tiene permiso de acceso a los datos."
        )

        if st.button("Cerrar sesión"):
            sb.auth.sign_out()
            st.session_state.clear()
            st.rerun()

        st.stop()

    if st.sidebar.button("Cerrar sesión"):
        sb.auth.sign_out()
        st.session_state.clear()
        st.rerun()

    return sb


def _clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (date, pd.Timestamp)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _text(value):
    value = _clean(value)
    return (
        str(value).strip()
        if value is not None and str(value).strip()
        else None
    )


def _float(row, key):
    try:
        v = _clean(row.get(key))
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _unique_players(df):
    players = {}

    for role, idcol, namecol in [
        ("pitcher", "PitcherId", "Pitcher"),
        ("batter", "BatterId", "Batter")
    ]:
        if idcol not in df.columns or namecol not in df.columns:
            continue

        for pid, name in zip(df[idcol], df[namecol]):
            key = _text(pid)
            if key and key not in players:
                players[key] = _text(name) or key

    return players


def _existing(sb, table, col, values):
    output = {}

    for i in range(0, len(values), 100):
        batch = values[i:i + 100]

        if batch:
            for r in (
                sb.table(table)
                .select("*")
                .in_(col, batch)
                .execute()
                .data
            ):
                output[str(r[col])] = r

    return output


def import_csv(sb, file_bytes, filename):
    """Deduplicate identical file content. Raise on failures."""

    digest = hashlib.sha256(file_bytes).hexdigest()

    existing = (
        sb.table("imports")
        .select("id")
        .eq("file_hash", digest)
        .limit(1)
        .execute()
        .data
    )

    if existing:
        return "duplicate", 0

    df = pd.read_csv(
        io.BytesIO(file_bytes),
        low_memory=False
    )

    df.columns = df.columns.astype(str).str.strip()

    required = {
        "Date",
        "Pitcher",
        "PitcherId",
        "PitchUID",
        "SessionId"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Columnas faltantes: " + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("El CSV está vacío.")

    if df["Date"].nunique() != 1:
        raise ValueError(
            "Por seguridad, cada CSV debe contener "
            "una sola fecha de sesión."
        )

    session_date = (
        pd.to_datetime(
            df["Date"].iloc[0],
            errors="raise"
        )
        .date()
        .isoformat()
    )

    session_key = _text(df["SessionId"].iloc[0])

    if df["SessionId"].astype(str).nunique() != 1:
        raise ValueError(
            "Por seguridad, este importador requiere "
            "un único SessionId por CSV."
        )

    note = "TrackMan SessionId: " + str(session_key)

    matches = (
        sb.table("sessions")
        .select("id")
        .eq("session_date", session_date)
        .eq("notes", note)
        .limit(1)
        .execute()
        .data
    )

    if matches:
        sid = matches[0]["id"]
    else:
        created = (
            sb.table("sessions")
            .insert({
                "session_date": session_date,
                "session_type": "TrackMan CSV",
                "notes": note
            })
            .execute()
            .data
        )
        sid = created[0]["id"]

    player_names = _unique_players(df)

    known = _existing(
        sb,
        "players",
        "external_id",
        list(player_names)
    )

    for external_id, name in player_names.items():
        if external_id not in known:
            created = (
                sb.table("players")
                .insert({
                    "external_id": external_id,
                    "name": name
                })
                .execute()
                .data
            )
            known[external_id] = created[0]

    created = (
        sb.table("imports")
        .insert({
            "session_id": sid,
            "filename": filename,
            "file_hash": digest
        })
        .execute()
        .data
    )

    import_id = created[0]["id"]
    records = []

    for idx, row in df.iterrows():
        raw = {
            str(k): _clean(v)
            for k, v in row.items()
        }

        pitcher = known.get(_text(row.get("PitcherId")))
        batter = known.get(_text(row.get("BatterId")))

        records.append({
            "session_id": sid,
            "import_id": import_id,
            "pitcher_id": pitcher["id"] if pitcher else None,
            "batter_id": batter["id"] if batter else None,
            "source_pitch_id": _text(row.get("PitchUID")),
            "pitch_number": (
                int(row["PitchNo"])
                if pd.notna(row.get("PitchNo"))
                else None
            ),
            "pitch_type": _text(row.get("TaggedPitchType")),
            "pitch_call": _text(row.get("PitchCall")),
            "play_result": _text(row.get("PlayResult")),
            "batter_side": _text(row.get("BatterSide")),
            "velocity_mph": _float(row, "RelSpeed"),
            "spin_rate": _float(row, "SpinRate"),
            "ivb_inches": _float(row, "InducedVertBreak"),
            "hb_inches": _float(row, "HorzBreak"),
            "vaa_degrees": _float(row, "VertApprAngle"),
            "extension_ft": _float(row, "Extension"),
            "release_height_ft": _float(row, "RelHeight"),
            "release_side_ft": _float(row, "RelSide"),
            "plate_loc_height_ft": _float(
                row, "PlateLocHeight"
            ),
            "plate_loc_side_ft": _float(
                row, "PlateLocSide"
            ),
            "raw_data": raw
        })

    try:
        for i in range(0, len(records), 100):
            (
                sb.table("pitches")
                .insert(records[i:i + 100])
                .execute()
            )

    except Exception as exc:
        raise RuntimeError(
            "Importación incompleta: no vuelvas a subir "
            "el archivo todavía. Se requiere revisar "
            "el registro de importación para evitar duplicados."
        ) from exc

    return "inserted", len(records)


def load_all(sb):
    """Paginate TrackMan rows, preserving dashboard column names."""

    rows = []
    offset = 0
    batch = 500

    while True:
        chunk = (
            sb.table("pitches")
            .select("raw_data")
            .order("created_at")
            .range(offset, offset + batch - 1)
            .execute()
            .data
        )

        rows.extend(
            x["raw_data"]
            for x in chunk
            if isinstance(x.get("raw_data"), dict)
        )

        if len(chunk) < batch:
            break

        offset += batch

    return pd.DataFrame(rows) if rows else None
