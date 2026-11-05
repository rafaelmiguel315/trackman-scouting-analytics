import unittest
import pandas as pd
from pitch_scouting_engine import summaries, pitch_rows

class ScoutingSummaryTests(unittest.TestCase):
    def test_arsenal_separation_and_whiff(self):
        df = pd.DataFrame({
            "TaggedPitchType":["FF","FF","SL","SL"],
            "RelSpeed":[94,96,84,86],
            "InducedVertBreak":[16,18,0,2],
            "HorzBreak":[10,12,-5,-7],
            "SpinRate":[2200,2300,2400,2500],
            "RelHeight":[6,6,6.1,6.1],
            "RelSide":[2,2,2.1,2.1],
            "PitchCall":["StrikeSwinging","BallCalled","StrikeSwinging","FoulBall"],
        })
        overall,detail=summaries(df)
        self.assertEqual(len(detail),2)
        self.assertTrue(any("+10.0 mph velocity" in line for line in overall))
        self.assertTrue(any("50.0%" in paragraph for _,paragraph in detail))
        self.assertTrue(any("deception" in line.lower() for line in overall))
    def test_missing_metrics_never_invented(self):
        df=pd.DataFrame({"TaggedPitchType":["CH"],"PitchCall":["BallCalled"]})
        overall,detail=summaries(df)
        self.assertEqual(len(detail),1)
        self.assertIn("N/A",detail[0][1])
        self.assertIn("unavailable",detail[0][1])
        self.assertIn("a validated pitch-type/level comparison",detail[0][1])
    def test_pitch_aliases_are_combined(self):
        df=pd.DataFrame({"TaggedPitchType":["FF","FourSeamFastBall","Four-Seam"],"PitchCall":["StrikeSwinging","BallCalled","FoulBall"]})
        rows=pitch_rows(df)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["n"],3)
        self.assertEqual(rows[0]["name"],"Four-Seam Fastball")
    def test_foul_tip_is_contact_not_whiff(self):
        df=pd.DataFrame({"TaggedPitchType":["SL","SL"],"PitchCall":["FoulTip","StrikeSwinging"]})
        rows=pitch_rows(df)
        self.assertEqual(rows[0]["swings"],2)
        self.assertEqual(rows[0]["whiffs"],1)
        self.assertEqual(rows[0]["whiff_pct"],50.0)
    def test_no_pitches(self):
        self.assertEqual(summaries(pd.DataFrame()),([],[]))
if __name__=="__main__":
    unittest.main()
