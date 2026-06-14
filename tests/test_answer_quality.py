import unittest

from core.answer_quality import citation_coverage, evaluate_evidence_gate, refusal_answer


class AnswerQualityTests(unittest.TestCase):
    def test_evidence_gate_rejects_empty_or_low_confidence_evidence(self):
        empty = evaluate_evidence_gate([])
        low = evaluate_evidence_gate([{"id": "E1", "score": 0.001}])

        self.assertFalse(empty.allowed)
        self.assertFalse(low.allowed)
        self.assertEqual(low.reason, "evidence confidence below threshold")

    def test_evidence_gate_accepts_scored_evidence(self):
        gate = evaluate_evidence_gate([{"id": "E1", "score": 0.2}])

        self.assertTrue(gate.allowed)
        self.assertEqual(gate.reason, "evidence accepted")

    def test_refusal_answer_is_explicit_about_local_evidence(self):
        gate = evaluate_evidence_gate([])
        answer = refusal_answer("What is GraphRAG?", gate)

        self.assertIn("未找到足够可靠的相关证据", answer)

    def test_citation_coverage_reports_used_and_unsupported_citations(self):
        report = citation_coverage("结论来自 [E1]，但 [E9] 不存在。", [{"id": "E1"}, {"id": "E2"}])

        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual(report["used"], ["E1"])
        self.assertEqual(report["unsupported"], ["E9"])


if __name__ == "__main__":
    unittest.main()
