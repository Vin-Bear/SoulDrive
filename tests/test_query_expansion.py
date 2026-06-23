import unittest

from core.query_expansion import expand_query_variants


class QueryExpansionTests(unittest.TestCase):
    def test_expand_query_variants_keeps_original_query_first(self):
        variants = expand_query_variants("这个方案怎么保证隐私？")

        self.assertGreaterEqual(len(variants), 1)
        self.assertEqual(variants[0], "这个方案怎么保证隐私？")

    def test_expand_query_variants_is_bounded_and_deduplicated(self):
        variants = expand_query_variants("GraphRAG 是什么？")

        self.assertLessEqual(len(variants), 3)
        self.assertEqual(len(variants), len(set(variants)))

    def test_expand_query_variants_preserves_technical_terms(self):
        variants = expand_query_variants("GraphRAG local search 怎么工作？")

        joined = " ".join(variants)
        self.assertIn("GraphRAG", joined)
        self.assertIn("local search", joined)

    def test_expand_query_variants_adds_lightweight_enterprise_term_aliases(self):
        variants = expand_query_variants("异常检测论文主要解决什么问题？")

        joined = " ".join(variants).lower()
        self.assertLessEqual(len(variants), 3)
        self.assertIn("anomaly detection", joined)


if __name__ == "__main__":
    unittest.main()
