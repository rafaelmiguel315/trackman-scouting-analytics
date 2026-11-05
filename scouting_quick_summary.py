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
    if ivb is None:
        vertical="Vertical: not measured."
    elif ivb >= 10:
        vertical=f"Vertical: {ivb:.1f} in; strong upward shape relative to gravity."
    elif ivb <= -8:
        vertical=f"Vertical: {ivb:.1f} in; pronounced downward shape."
    else:
        vertical=f"Vertical: {ivb:.1f} in; modest vertical movement."
    if hb is None:
        horizontal="Horizontal: not measured."
    elif abs(hb) >= 10:
        horizontal=f"Horizontal: {hb:+.1f} in; large side movement."
    elif abs(hb) >= 5:
        horizontal=f"Horizontal: {hb:+.1f} in; moderate side movement."
    else:
        horizontal=f"Horizontal: {hb:+.1f} in; limited side movement."
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
