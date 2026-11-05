import unittest
import pandas as pd
from unittest.mock import patch
from mlb_movement_reference import retrieve
from mlb_movement_reference import pooled, velocity_grade

class MLBReferenceTests(unittest.TestCase):
    def test_weighted_2025_2026_pooling(self):
        a=pd.DataFrame([{"season":2025,"pitch_type":"FF","pitch_hand":"R","pitches_thrown":100,"avg_speed":94,"pitcher_break_z_induced":17,"pitcher_break_x":8}])
        b=pd.DataFrame([{"season":2026,"pitch_type":"FF","pitch_hand":"R","pitches_thrown":300,"avg_speed":96,"pitcher_break_z_induced":19,"pitcher_break_x":10}])
        r=pooled([a,b],"Four-Seam Fastball","R")
        self.assertAlmostEqual(r["RelSpeed"],95.5)
        self.assertAlmostEqual(r["InducedVertBreak"],18.5)
        self.assertEqual(r["seasons"],[2025,2026])
        self.assertEqual(r["RelSpeed_n"],400)

    def test_missing_year_not_invented(self):
        a=pd.DataFrame([{"season":2025,"pitch_type":"SL","pitch_hand":"L","pitches_thrown":10,"avg_speed":85,"pitcher_break_z_induced":0,"pitcher_break_x":-5}])
        r=pooled([a],"Slider","L")
        self.assertEqual(r["seasons"],[2025])
        self.assertEqual(velocity_grade(88,85)[0],"Above MLB velocity avg")

    def test_downloads_all_pitch_categories_not_only_default_fastball(self):
        def fake_fetch(year, pitch_type):
            return pd.DataFrame([{"season":year,"pitch_type":pitch_type,"pitch_hand":"R",
                "pitches_thrown":100,"avg_speed":90,"pitcher_break_z_induced":10,
                "pitcher_break_x":5}])
        with patch("mlb_movement_reference._fetch_pitch", side_effect=fake_fetch):
            result = retrieve(2025)
        self.assertIn("SL", set(result["pitch_type"]))
        self.assertIn("CH", set(result["pitch_type"]))
        self.assertIn("FF", set(result["pitch_type"]))

    def test_missing_pitch_type_no_grade(self):
        self.assertIsNone(pooled([pd.DataFrame({"season":[2025],"pitch_type":["FF"],"pitch_hand":["R"],"pitches_thrown":[20],"avg_speed":[90]})],"Slider"))
        self.assertEqual(velocity_grade(None,95)[0],"Reference unavailable")

if __name__=="__main__":
    unittest.main()
