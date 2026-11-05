"""Evidence-limited descriptive pitch reports. Never manufacture league percentiles."""
import pandas as pd

NAMES={"FF":"Four-Seam Fastball","FA":"Four-Seam Fastball","SI":"Sinker","FT":"Sinker","FC":"Cutter","SL":"Slider","ST":"Sweeper","CB":"Curveball","KC":"Knuckle Curve","CH":"Changeup","FS":"Splitter","FO":"Forkball"}
METRICS=[("RelSpeed","Velocity","mph"),("InducedVertBreak","IVB","in"),("HorzBreak","HB","in"),("SpinRate","Spin Rate","rpm"),("VertApprAngle","VAA","deg"),("RelHeight","Release Height","ft"),("RelSide","Release Side","ft"),("Extension","Extension","ft")]
def mean(df,key):
    if key not in df:return None
    s=pd.to_numeric(df[key],errors="coerce").dropna()
    return float(s.mean()) if len(s) else None
def display(n,dec=1):
    return "N/A" if n is None else f"{n:.{dec}f}"
ALIASES={"FOURSEAMFASTBALL":"FF","FOURSEAM":"FF","FASTBALL":"FF","TWOSEAM":"SI","TWOSEAMFASTBALL":"SI","SINKER":"SI","CUTTER":"FC","SLIDER":"SL","SWEEPER":"ST","CURVEBALL":"CB","KNUCKLECURVE":"KC","CHANGEUP":"CH","SPLITTER":"FS"}
def name(code):
    raw=str(code).strip()
    key="".join(c for c in raw.upper() if c.isalnum())
    return NAMES.get(ALIASES.get(key,key),raw)
def pitch_rows(df):
    if "TaggedPitchType" not in df:return []
    out=[]
    for tag,group in df.groupby("TaggedPitchType",dropna=True):
        if not str(tag).strip() or str(tag).strip().lower() in ("nan","none","unknown"):continue
        vals={key:mean(group,key) for key,_,_ in METRICS}
        calls=group["PitchCall"].fillna("").astype(str).str.lower().str.replace(r"[^a-z]","",regex=True) if "PitchCall" in group else pd.Series([],dtype=str)
        swing=calls.isin(["strikeswinging","foulball","foulballnotfieldable","foultip","inplay","foulbunt","strikebunt"])
        whiff=calls.isin(["strikeswinging","foultip"])
        ns=int(swing.sum()); nw=int(whiff.sum())
        out.append({"name":name(tag),"n":len(group),"v":vals,"swings":ns,"whiffs":nw,"whiff_pct":100*nw/ns if ns else None})
    return sorted(out,key=lambda p:-p["n"])
def summaries(df):
    rows=pitch_rows(df)
    if not rows:return [],[]
    primary=next((p for p in rows if p["name"] in ("Four-Seam Fastball","Sinker")),rows[0])
    v=primary["v"]
    overall=[f"The selected sample contains {len(df):,} pitches and {len(rows)} tagged pitch types. The primary fastball selected for comparison is {primary['name']} ({primary['n']} pitches), averaging {display(v['RelSpeed'])} mph, {display(v['InducedVertBreak'])} inches IVB and {display(v['HorzBreak'])} inches HB."]
    for p in rows:
        if p is primary:continue
        q=p["v"];dif=[]
        for field,label,unit in [("RelSpeed","velocity","mph"),("InducedVertBreak","IVB","in"),("HorzBreak","HB","in")]:
            if v[field] is not None and q[field] is not None:
                dif.append(f"{v[field]-q[field]:+.1f} {unit} {label}")
        if dif:overall.append(f"{primary['name']} minus {p['name']}: "+", ".join(dif)+".")
    heights=[p["v"]["RelHeight"] for p in rows if p["v"]["RelHeight"] is not None]
    sides=[p["v"]["RelSide"] for p in rows if p["v"]["RelSide"] is not None]
    if len(heights)>1 and len(sides)>1:
        overall.append(f"Mean release-point spread across pitch types: {max(heights)-min(heights):.2f} ft vertically and {max(sides)-min(sides):.2f} ft laterally. Release point alone does not establish arm angle.")
    overall.append("Differences in speed, movement, and release may support deception potential, but do not prove deception or tunneling. Observed effectiveness must be evaluated with sufficient swing and contact outcomes.")
    detail=[]
    for p in rows:
        q=p["v"]
        desc=f"{p['n']} pitches. "+", ".join(f"{label}: {display(q[field])} {unit}" for field,label,unit in METRICS)+". "
        if p["name"]=="Four-Seam Fastball":
            desc+="Four-seam carry is represented by induced vertical break. A flat/low-carry designation requires a verified same-pitch reference; IVB alone cannot establish pitch effectiveness. "
        elif p["name"]=="Sinker":
            desc+="Sinker movement is evaluated for arm-side run and sink relative to a same-handed pitch-type reference; direction cannot be assigned without confirming the TrackMan horizontal sign convention. "
        elif p["name"]=="Cutter":
            desc+="Cutter movement is evaluated for glove-side cutting action and separation from the primary fastball; effectiveness requires observed outcomes. "
        elif p["name"] in ("Slider","Sweeper"):
            desc+="Breaking-ball shape is evaluated for horizontal sweep, vertical depth, and fastball separation; the observed outcome sample determines effectiveness evidence. "
        elif p["name"] in ("Curveball","Knuckle Curve"):
            desc+="Curveball shape is evaluated for vertical depth and lateral break; the magnitude of IVB should not be graded like a four-seam fastball. "
        elif p["name"] in ("Changeup","Splitter","Forkball"):
            desc+="Offspeed shape is evaluated for velocity separation, induced vertical break and horizontal fade or tumble; direction requires confirmed handedness conventions. "
        else:
            desc+="Movement shape requires a pitch-type-matched reference and verified coordinate conventions. "
        if p["whiff_pct"] is not None:
            desc+=f"Observed Whiff%: {p['whiff_pct']:.1f}% ({p['whiffs']} misses / {p['swings']} swings). "
        else:desc+="Observed Whiff%: unavailable (no recorded swings). "
        desc+="MLB/MiLB adjusted grade: insufficient validated reference data."
        detail.append((p["name"],desc))
    return overall,detail
