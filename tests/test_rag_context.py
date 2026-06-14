import unittest

from core.rag_engine import build_fast_evidence_answer, compact_context_text, normalize_technical_terms


class RagContextTests(unittest.TestCase):
    def test_compact_context_text_preserves_head_and_tail(self):
        text = " ".join([f"token{i}" for i in range(500)])

        compacted = compact_context_text(text, max_chars=240)

        self.assertLessEqual(len(compacted), 260)
        self.assertIn("token0", compacted)
        self.assertIn("token499", compacted)
        self.assertIn("context trimmed", compacted)

    def test_fast_transformer_answer_uses_citations(self):
        answer = build_fast_evidence_answer(
            "transformer机制是什么",
            [{"id": "E1"}, {"id": "E2"}, {"id": "E3"}],
        )

        self.assertIsNotNone(answer)
        self.assertIn("Transformer", answer)
        self.assertIn("[E1]", answer)
        self.assertIn("[E2]", answer)

    def test_normalize_technical_terms_keeps_transformer_name(self):
        self.assertEqual(normalize_technical_terms("变压器架构"), "Transformer 架构")


if __name__ == "__main__":
    unittest.main()
