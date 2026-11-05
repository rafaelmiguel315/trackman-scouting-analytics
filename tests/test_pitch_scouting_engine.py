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
        self.assertIn("insufficient validated reference data",detail[0][1])
    def test_no_pitches(self):
        self.assertEqual(summaries(pd.DataFrame()),([],[]))
if __name__=="__main__":
    unittest.main()
