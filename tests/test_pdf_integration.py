"""PDF integration tests without connecting to Supabase or starting Streamlit."""
import ast
import io
import unittest
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from pypdf import PdfReader


def load_report_function():
    source = Path(__file__).resolve().parents[1].joinpath("streamlit_app.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pdf_report")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {
        "io": io, "escape": escape, "pd": pd,
        "benchmark_level": "MLB", "benchmark_season": 2026,
        "PITCH_METRICS": {"RelSpeed": ("Velocity", "mph"), "InducedVertBreak": ("IVB", "in")},
        "HITTER_METRICS": {"ExitSpeed": ("Exit Velocity", "mph")},
        "pitching_stats": lambda d, bottom, top: {"Total Pitches": str(len(d))},
        "hitting_stats": lambda d, bottom, top: {"Pitches Seen": str(len(d))},
        "rating": lambda metric, role, value: ("No benchmark", "#64748b", None),
        "numeric": lambda d, key: pd.to_numeric(d[key] if key in d else pd.Series(index=d.index, dtype=float), errors="coerce"),
        "movement_figure": lambda d: None,
        "strike_zone_figure": lambda d, bottom, top: None,
        "classify": lambda d, bottom, top: d.assign(
            _swing=d["PitchCall"].isin(["StrikeSwinging", "FoulBall"]),
            _whiff=d["PitchCall"].eq("StrikeSwinging"),
            _outside=False, _contact=d["PitchCall"].eq("FoulBall"),
            _zone=False, _valid_ev=d.get("ExitSpeed", pd.Series(index=d.index, dtype=float)).notna(),
            _hard=False, _medium=False, _weak=False,
        ),
        "batting_rates": lambda d: {
            "Whiff %": 100*d["_whiff"].sum()/d["_swing"].sum() if d["_swing"].sum() else None,
            "Chase %": None, "Hard Contact %": None, "Weak Contact %": None,
        },
    }
    exec(compile(module, "streamlit_app.py", "exec"), namespace)
    return namespace["pdf_report"]


class PdfReportIntegrationTests(unittest.TestCase):
    def test_pitcher_pdf_contains_both_pitches_and_summary(self):
        report = load_report_function()
        df = pd.DataFrame({
            "TaggedPitchType": ["FF", "FF", "SL", "SL"],
            "PitchCall": ["StrikeSwinging", "BallCalled", "FoulBall", "StrikeSwinging"],
            "RelSpeed": [94, 95, 84, 85],
            "InducedVertBreak": [17, 18, 1, 0],
            "HorzBreak": [10, 11, -5, -6],
            "Date": ["2026-10-14"]*4,
        })
        payload = report("Sample Pitcher", "Pitcher", df, 1.5, 3.5)
        self.assertTrue(payload.startswith(b"%PDF"))
        extracted = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(payload)).pages)
        for phrase in ("Sample Pitcher", "Player Scouting Summary", "Four-Seam Fastball", "Slider", "Pitch-by-Pitch Movement Analysis"):
            self.assertIn(phrase, extracted)

    def test_batter_pdf_contains_hitting_summary(self):
        report = load_report_function()
        df = pd.DataFrame({
            "PitchCall": ["StrikeSwinging", "FoulBall"],
            "ExitSpeed": [None, 91.0],
            "Angle": [None, 18.0],
        })
        payload = report("Sample Batter", "Batter", df, 1.5, 3.5)
        self.assertTrue(payload.startswith(b"%PDF"))
        extracted = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(payload)).pages)
        self.assertIn("Overall Hitting Assessment", extracted)
        self.assertIn("91.0 mph", extracted)


if __name__ == "__main__":
    unittest.main()
