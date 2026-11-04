
"""TrackMan Scouting Analytics: authenticated CSV storage."""

import hashlib
import io
import math
from datetime import date

import pandas as pd
import streamlit as st
from supabase import create_client


# ==========================================
# SUPABASE CONNECTION
# ==========================================

def client():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


# ==========================================
# AUTHENTICATION
# ==========================================

def authenticate():
    """Require Supabase login and app_admins membership."""

    if "supabase_session" not in st.session_state:
        st.title("⚾ TrackMan Scouting Analytics")
        st.caption("Acceso autorizado | International Scouting")

        with st.form("supabase_login"):
            email = st.text_input("Correo electrónico")
            password = st.text_input(
                "Contraseña",
                type="password"
            )
            submitted = st.form_submit_button(
                "Iniciar sesión"
            )

        if submitted:
            try:
                result = client().auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                if not result.session:
                    raise ValueError("No session")

                st.session_state.supabase_session = {
                    "access_token": result.session.access_token,
                    "refresh_token": result.session.refresh_token
                }
                st.rerun()

            except Exception:
                st.error(
                    "No se pudo iniciar sesión. "
                    "Comprueba tus credenciales."
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
            raise ValueError("Session expired")

        user = sb.auth.get_user().user

        if not user:
            raise ValueError("Not authenticated")

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
            "Tu cuenta no tiene permiso "
            "de acceso a los datos."
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


# ==========================================
# DATA CLEANING
# ==========================================

def _clean(value):
    if value is None:
        return None

    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, (date, pd.Timestamp)):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def _text(value):
    value = _clean(value)

    if value is None:
        return None

    result = str(value).strip()
    return result if result else None


def _float(row, key):
    try:
        value = _clean(row.get(key))
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ==========================================
# PLAYER IDENTIFICATION
# ==========================================

def _player_external_id(role, pid, name):
    """
    Use real TrackMan player IDs.

    If the ID is zero or missing, use the
    normalized player name and role.
    """

    pid = _text(pid)
    name = _text(name)

    if not name:
        return None

    invalid_ids = {
        "0",
        "0.0",
        "nan",
        "none",
        "null",
        "-1"
    }

    if pid and pid.lower() not in invalid_ids:
        return f"trackman:{role}:{pid}"

    normalized = " ".join(
        name.casefold().split()
    )

    return f"name:{role}:{normalized}"


def _existing(sb, table, col, values):
    output = {}

    values = list(
        dict.fromkeys(
            v for v in values if v
        )
    )

    for i in range(0, len(values), 100):
        batch = values[i:i + 100]

        result = (
            sb.table(table)
            .select("*")
            .in_(col, batch)
            .execute()
            .data
        )

        for row in result:
            output[str(row[col])] = row

    return output


# ==========================================
# SESSION IDENTIFICATION
# ==========================================

def _session_identity(row, digest):
    """
    Recognize portable and stadium exports.
    """

    for column in (
        "SessionId",
        "GameUID",
        "GameID"
    ):
        value = _text(row.get(column))

        if value:
            return f"{column}: {value}"

    return f"CSV SHA256: {digest}"


# ==========================================
# DUPLICATE FILE PROTECTION
# ==========================================

def _verify_existing_import(sb, digest):
    """
    Check whether the file has already
    been imported.
    """

    existing = (
        sb.table("imports")
        .select("id")
        .eq("file_hash", digest)
        .limit(1)
        .execute()
        .data
    )

    if not existing:
        return False

    import_id = existing[0]["id"]

    pitches = (
        sb.table("pitches")
        .select("id")
        .eq("import_id", import_id)
        .limit(1)
        .execute()
        .data
    )

    if pitches:
        return True

    raise RuntimeError(
        "Existe un registro de importación "
        "sin lanzamientos. Revisa la tabla "
        "imports antes de reintentar."
    )


# ==========================================
# CSV IMPORT
# ==========================================

def import_csv(sb, file_bytes, filename):
    """
    Import stadium and portable TrackMan CSVs.

    Preserve all original columns in raw_data.
    """

    digest = hashlib.sha256(
        file_bytes
    ).hexdigest()

    if _verify_existing_import(sb, digest):
        return "duplicate", 0

    df = pd.read_csv(
        io.BytesIO(file_bytes),
        low_memory=False
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if df.empty:
        raise ValueError(
            "El CSV está vacío."
        )

    if df.columns.duplicated().any():
        raise ValueError(
            "El CSV contiene nombres "
            "de columnas duplicados."
        )

    required = {
        "Date",
        "Pitcher",
        "PitchUID"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Columnas faltantes: "
            + ", ".join(sorted(missing))
        )

    # ======================================
    # VALIDATE SESSION DATE
    # ======================================

    parsed_dates = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    if (
        parsed_dates.isna().any()
        or parsed_dates.dt.date.nunique() != 1
    ):
        raise ValueError(
            "Cada CSV debe tener una sola "
            "fecha válida en Date."
        )

    session_date = (
        parsed_dates.iloc[0]
        .date()
        .isoformat()
    )

    # ======================================
    # IDENTIFY SESSION
    # ======================================

    identities = df.apply(
        lambda row: _session_identity(
            row,
            digest
        ),
        axis=1
    )

    if identities.nunique() != 1:
        raise ValueError(
            "El CSV contiene varias sesiones "
            "o juegos. Sepáralos antes de importar."
        )

    note = identities.iloc[0]

    # ======================================
    # IDENTIFY PLAYERS
    # ======================================

    player_names = {}

    for role, idcol, namecol in (
        ("pitcher", "PitcherId", "Pitcher"),
        ("batter", "BatterId", "Batter")
    ):
        if namecol not in df.columns:
            continue

        ids = (
            df[idcol]
            if idcol in df.columns
            else [None] * len(df)
        )

        for pid, name in zip(
            ids,
            df[namecol]
        ):
            key = _player_external_id(
                role,
                pid,
                name
            )

            if key:
                player_names[key] = _text(name)

    known = _existing(
        sb,
        "players",
        "external_id",
        player_names.keys()
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

    # ======================================
    # CREATE OR FIND SESSION
    # ======================================

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

    # ======================================
    # PREPARE PITCH RECORDS
    # ======================================

    records = []

    for _, row in df.iterrows():
        raw = {
            str(k): _clean(v)
            for k, v in row.items()
        }

        pitcher_key = _player_external_id(
            "pitcher",
            row.get("PitcherId"),
            row.get("Pitcher")
        )

        batter_key = _player_external_id(
            "batter",
            row.get("BatterId"),
            row.get("Batter")
        )

        pitcher = known.get(pitcher_key)
        batter = known.get(batter_key)

        pitch_no = _float(row, "PitchNo")

        records.append({
            "session_id": sid,
            "pitcher_id": (
                pitcher["id"] if pitcher else None
            ),
            "batter_id": (
                batter["id"] if batter else None
            ),
            "source_pitch_id": _text(
                row.get("PitchUID")
            ),
            "pitch_number": (
                int(pitch_no)
                if pitch_no is not None
                and math.isfinite(pitch_no)
                else None
            ),
            "pitch_type": _text(
                row.get("TaggedPitchType")
            ),
            "pitch_call": _text(
                row.get("PitchCall")
            ),
            "play_result": _text(
                row.get("PlayResult")
            ),
            "batter_side": _text(
                row.get("BatterSide")
            ),
            "velocity_mph": _float(
                row, "RelSpeed"
            ),
            "spin_rate": _float(
                row, "SpinRate"
            ),
            "ivb_inches": _float(
                row, "InducedVertBreak"
            ),
            "hb_inches": _float(
                row, "HorzBreak"
            ),
            "vaa_degrees": _float(
                row, "VertApprAngle"
            ),
            "extension_ft": _float(
                row, "Extension"
            ),
            "release_height_ft": _float(
                row, "RelHeight"
            ),
            "release_side_ft": _float(
                row, "RelSide"
            ),
            "plate_loc_height_ft": _float(
                row, "PlateLocHeight"
            ),
            "plate_loc_side_ft": _float(
                row, "PlateLocSide"
            ),
            "raw_data": raw
        })

    # ======================================
    # REGISTER IMPORT
    # ======================================

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

    # ======================================
    # SAVE PITCHES IN BATCHES
    # ======================================

    try:
        for i in range(0, len(records), 100):
            batch = [
                dict(r, import_id=import_id)
                for r in records[i:i + 100]
            ]

            (
                sb.table("pitches")
                .insert(batch)
                .execute()
            )

    except Exception as exc:
        raise RuntimeError(
            "Importación posiblemente incompleta. "
            "No vuelvas a subir el CSV hasta "
            "revisar imports y pitches para "
            "evitar duplicados."
        ) from exc

    return "inserted", len(records)


# ==========================================
# LOAD HISTORICAL DATA
# ==========================================

def load_all(sb):
    """
    Load saved TrackMan records and restore
    their original dashboard column names.
    """

    rows = []
    offset = 0
    batch = 500

    while True:
        chunk = (
            sb.table("pitches")
            .select("raw_data")
            .order("created_at")
            .order("id")
            .range(
                offset,
                offset + batch - 1
            )
            .execute()
            .data
        )

        rows.extend(
            x["raw_data"]
            for x in chunk
            if isinstance(
                x.get("raw_data"),
                dict
            )
        )

        if len(chunk) < batch:
            break

        offset += batch

    return (
        pd.DataFrame(rows)
        if rows
        else None
    )
