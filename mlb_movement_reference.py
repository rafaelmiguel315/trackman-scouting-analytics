"""Public MLB Statcast movement reference, independent of private TrackMan storage.

Fetches official 2025/2026 leaderboard CSVs and pools pitcher-pitch observations
weighted by pitches thrown. Missing seasons are never fabricated.
"""
from io import StringIO
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

URL = "https://baseballsavant.mlb.com/leaderboard/pitch-movement?year={year}&pitch_type={pitch_type}&min=0&csv=true"
PITCH_TYPES = ("FF","SI","FC","SL","ST","CU","KC","CH","FS")
ALIASES = {"Four-Seam Fastball":"FF","Sinker":"SI","Cutter":"FC","Slider":"SL","Sweeper":"ST","Curveball":"CU","Knuckle Curve":"KC","Changeup":"CH","Splitter":"FS","Forkball":"FO"}
CODE_NAMES = {"FF":"Four-Seam Fastball","SI":"Sinker","FC":"Cutter","SL":"Slider","ST":"Sweeper","CU":"Curveball","KC":"Knuckle Curve","CH":"Changeup","FS":"Splitter","FO":"Forkball"}
FIELDS = {"RelSpeed":"avg_speed","InducedVertBreak":"pitcher_break_z_induced","HorzBreak":"pitcher_break_x"}

def _fetch_pitch(year, pitch_type):
    request = Request(URL.format(year=int(year), pitch_type=pitch_type),
                      headers={"User-Agent":"Mozilla/5.0 (compatible; scouting-analytics/1.0)"})
    with urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8-sig")
    frame = pd.read_csv(StringIO(body))
    required = {"pitch_type","pitch_hand","pitches_thrown","avg_speed",
                "pitcher_break_z_induced","pitcher_break_x"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{year} {pitch_type}: unexpected MLB CSV schema: {sorted(required-set(frame.columns))}")
    frame = frame[frame["pitch_type"].astype(str).str.upper().eq(pitch_type)].copy()
    frame["season"] = int(year)
    return frame

def retrieve(year):
    """Download each pitch category; the leaderboard defaults to four-seam only."""
    frames, failures = [], []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_pitch,year,code):code for code in PITCH_TYPES}
        for future in as_completed(futures):
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                failures.append(f"{futures[future]}: {type(exc).__name__}")
    if not frames:
        raise RuntimeError(f"No verified Baseball Savant movement CSVs for {year}: {', '.join(failures)}")
    result = pd.concat(frames, ignore_index=True)
    result.attrs["missing_pitch_types"] = failures
    return result

def pooled(frames, pitch_name, hand=None):
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)
    code = ALIASES.get(pitch_name, pitch_name)
    data = data[data["pitch_type"].astype(str).str.upper() == code].copy()
    if hand in ("R","L"):
        data = data[data["pitch_hand"].astype(str).str.upper() == hand]
    if data.empty:
        return None
    weight = pd.to_numeric(data["pitches_thrown"], errors="coerce")
    output = {"pitch":CODE_NAMES.get(code,pitch_name), "seasons":sorted(data["season"].unique().tolist()), "rows":len(data)}
    for metric, field in FIELDS.items():
        values = pd.to_numeric(data[field], errors="coerce")
        valid = values.notna() & weight.notna() & (weight > 0)
        if valid.any():
            output[metric] = float((values[valid]*weight[valid]).sum()/weight[valid].sum())
            output[metric+"_n"] = int(weight[valid].sum())
    return output

def velocity_grade(value, reference, tolerance=1.5):
    """Velocity-only comparison, not an overall quality or stuff grade."""
    if value is None or reference is None:
        return ("Reference unavailable", "#64748b")
    delta = float(value)-float(reference)
    if delta > tolerance:
        return ("Above MLB velocity avg", "#dc2626")
    if delta < -tolerance:
        return ("Below MLB velocity avg", "#2563eb")
    return ("Near MLB velocity avg", "#eab308")
