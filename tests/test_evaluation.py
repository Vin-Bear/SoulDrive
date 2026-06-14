import unittest
from pathlib import Path

from core.evaluation import evaluate_retrieval_cases, load_retrieval_eval_cases


class EvaluationTests(unittest.TestCase):
    def test_retrieval_eval_fixture_scores_expected_hits(self):
        fixture = Path(__file__).parent / "fixtures" / "retrieval_eval.jsonl"

        cases = load_retrieval_eval_cases(fixture)
        report = evaluate_retrieval_cases(cases, top_k=2)

        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["hit_rate_at_k"], 1.0)
        self.assertGreaterEqual(report["mean_reciprocal_rank"], 0.5)
        self.assertIn("evidence_acceptance_rate", report)
        self.assertIn("average_top_score", report)
        self.assertEqual(report["low_evidence_cases"], [])


if __name__ == "__main__":
    unittest.main()
