"""Public MLB Statcast movement reference, independent of private TrackMan storage.

Fetches official 2025/2026 leaderboard CSVs and pools pitcher-pitch observations
weighted by pitches thrown. Missing seasons are never fabricated.
"""
from io import StringIO
from urllib.request import Request, urlopen
import pandas as pd

URL = "https://baseballsavant.mlb.com/leaderboard/pitch-movement?year={year}&csv=true"
ALIASES = {"Four-Seam Fastball":"FF","Sinker":"SI","Cutter":"FC","Slider":"SL","Sweeper":"ST","Curveball":"CU","Knuckle Curve":"KC","Changeup":"CH","Splitter":"FS","Forkball":"FO"}
CODE_NAMES = {"FF":"Four-Seam Fastball","SI":"Sinker","FC":"Cutter","SL":"Slider","ST":"Sweeper","CU":"Curveball","KC":"Knuckle Curve","CH":"Changeup","FS":"Splitter","FO":"Forkball"}
FIELDS = {"RelSpeed":"avg_speed","InducedVertBreak":"pitcher_break_z_induced","HorzBreak":"pitcher_break_x"}

def retrieve(year):
    request = Request(URL.format(year=int(year)), headers={"User-Agent":"Mozilla/5.0 TrackManScoutingAnalytics/1.0"})
    with urlopen(request, timeout=18) as response:
        data = response.read().decode("utf-8-sig")
    frame = pd.read_csv(StringIO(data))
    needed = {"pitch_type","pitch_hand","pitches_thrown","avg_speed","pitcher_break_z_induced","pitcher_break_x"}
    if not needed.issubset(frame.columns):
        raise ValueError("Unexpected Baseball Savant CSV columns: " + ",".join(sorted(needed-set(frame.columns))))
    frame["season"] = int(year)
    return frame

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
