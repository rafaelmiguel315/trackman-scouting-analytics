"""Short scouting interpretations; descriptive shape and separation, not unverified quality grades."""
from math import hypot

def _number(p, key):
    v=p.get("v",{}).get(key)
    try:
        return float(v) if v is not None else None
    except (ValueError,TypeError):
        return None

def shape_summary(p, all_pitches):
    name=p["name"]
    n=p["n"]
    ivb=_number(p,"InducedVertBreak")
    hb=_number(p,"HorzBreak")
    # IVB and HB are different axes. Their numerical equality is NOT a
    # universal definition of a flat pitch, nor is IVB >= 10 a good fastball.
    # Quality requires pitch type, handedness, arm slot, velocity and verified
    # pitch-type benchmarks. Until those are present, describe measurements only.
    vertical = ("Vertical: not measured." if ivb is None
                else f"Vertical (IVB): {ivb:.1f} in; quality not graded.")
    horizontal = ("Horizontal: not measured." if hb is None
                  else f"Horizontal (HB): {hb:+.1f} in; quality not graded.")
    if ivb is not None and hb is not None and "fastball" in name.lower():
        if ivb <= 12 and abs(hb) >= 8:
            vertical += " Low ride for a four-seam profile; check arm slot and velocity."
            horizontal += " Noticeable arm-side movement; compare with sinker/run profile."
    comparisons=[]
    for other in all_pitches:
        if other is p or other["name"]==name:
            continue
        a,b=_number(other,"InducedVertBreak"),_number(other,"HorzBreak")
        if ivb is not None and hb is not None and a is not None and b is not None:
            comparisons.append((hypot(ivb-a,hb-b),other["name"]))
    if comparisons:
        distance,other_name=min(comparisons)
        if distance >= 12:
            separation=f"Separation: clear ({distance:.1f} in) from {other_name}."
        elif distance >= 6:
            separation=f"Separation: moderate ({distance:.1f} in) from {other_name}."
        else:
            separation=f"Separation: similar movement ({distance:.1f} in) to {other_name}."
    else:
        separation="Separation: cannot compare with another measured pitch."
    caution=" Limited sample." if n<10 else ""
    return vertical,horizontal,separation+caution

def overall_summary(pitches):
    if not pitches:
        return "No pitch data to evaluate."
    if len(pitches)==1:
        return "One pitch type recorded. Arsenal separation cannot be assessed."
    counts=sum(p["n"] for p in pitches)
    primary=max(pitches,key=lambda p:p["n"])
    text=f"{len(pitches)} pitch types across {counts} pitches. Most used: {primary['name']} ({primary['n']} pitches). "
    distances=[]
    for i,a in enumerate(pitches):
        for b in pitches[i+1:]:
            ai,ah=_number(a,"InducedVertBreak"),_number(a,"HorzBreak")
            bi,bh=_number(b,"InducedVertBreak"),_number(b,"HorzBreak")
            if None not in (ai,ah,bi,bh):
                distances.append((hypot(ai-bi,ah-bh),a["name"],b["name"]))
    if distances:
        d,a,b=max(distances)
        text+=f"Largest movement gap: {a} vs {b} ({d:.1f} in). "
    text+="Movement differences may support pitch tunneling or deception, but do not prove deception without release, trajectory and hitter-response analysis."
    if counts<30 or any(p["n"]<10 for p in pitches):
        text+=" Small sample: conclusions are preliminary."
    return text
