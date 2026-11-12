"""Cross-check the real scalar data against independently saved research tables."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import example


class SavedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.photos, cls.results = example.recompute()

    def test_complete_twenty_photo_factorial_and_saved_terms(self):
        self.assertEqual(len(self.results), 60)
        rows = example.read_table("coalition_scores.csv")
        self.assertEqual(len(rows), 960)
        self.assertEqual(len({(r['photo'], r['layout'], r['coalition']) for r in rows}), 960)
        terms = example.read_table("expected_terms.csv")
        self.assertEqual(len(terms), 600)
        self.assertEqual(len({(r['photo'], r['layout'], r['term']) for r in terms}), 600)
        self.assertLess(self.summary["max_term_error_vs_saved"], 1e-10)
        self.assertTrue(self.summary["all_efficiency_checks_pass"])

    def test_photo_scores_are_full_coalition_scores(self):
        for row in self.photos:
            for layout in ("L0", "L1", "L2"):
                self.assertEqual(float(row[layout]), self.results[row["photo"], layout]["full"])

    def test_descriptive_findings_match_saved_cohort_summary(self):
        for layout, decreases, mean in (("L1", 20, -18.253187203407286), ("L2", 14, -7.133028441667557)):
            self.assertEqual(self.summary["layouts"][layout]["decreased"], decreases)
            self.assertAlmostEqual(self.summary["layouts"][layout]["mean_delta"], mean, places=12)
        self.assertEqual(self.summary["historical_visual_qc"], {"failed": 13, "pending": 7})


if __name__ == "__main__":
    unittest.main()
