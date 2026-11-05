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
def name(code):
    return NAMES.get(str(code).strip().upper(),str(code))
def pitch_rows(df):
    if "TaggedPitchType" not in df:return []
    out=[]
    for tag,group in df.groupby("TaggedPitchType",dropna=True):
        if not str(tag).strip():continue
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
    overall=[f"The selected sample contains {len(df):,} pitches and {len(rows)} tagged pitch types. The most-used primary fastball is {primary['name']} ({primary['n']} pitches), averaging {display(v['RelSpeed'])} mph, {display(v['InducedVertBreak'])} inches IVB and {display(v['HorzBreak'])} inches HB."]
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
        if p["name"]=="Four-Seam Fastball" and q["InducedVertBreak"] is not None:
            if q["InducedVertBreak"]<14:
                desc+="The fastball shows relatively limited induced vertical break compared with typical MLB four-seam shapes (potentially flat). "
            elif q["InducedVertBreak"]>=18:
                desc+="The fastball shows substantial induced vertical break (carry); effectiveness is not established by IVB alone. "
            else:desc+="The fastball shows intermediate induced vertical break; a matched arm-slot comparison is required. "
        else:
            desc+="Evaluate its vertical and horizontal movement relative to the same pitch type and a comparable release profile; the movement alone cannot establish effectiveness. "
        if p["whiff_pct"] is not None:
            desc+=f"Observed Whiff%: {p['whiff_pct']:.1f}% ({p['whiffs']} misses / {p['swings']} swings). "
        else:desc+="Observed Whiff%: unavailable (no recorded swings). "
        desc+="MLB/MiLB adjusted grade: insufficient validated reference data."
        detail.append((p["name"],desc))
    return overall,detail
