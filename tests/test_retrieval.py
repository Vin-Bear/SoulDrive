import unittest

from core.retrieval import SearchCandidate, build_citations, rank_hybrid_candidates


class RetrievalTests(unittest.TestCase):
    def test_hybrid_rank_rewards_keyword_and_graph_hits(self):
        candidates = [
            SearchCandidate(
                id="dense-only",
                content="This paper discusses generic transformer pretraining.",
                metadata={"source_filename": "generic.pdf"},
                dense_rank=1,
            ),
            SearchCandidate(
                id="graph-rag",
                content="GraphRAG builds graph communities for paper summarization and local search.",
                metadata={"source_filename": "graphrag.pdf"},
                dense_rank=8,
                graph_score=3,
            ),
        ]

        ranked = rank_hybrid_candidates("GraphRAG paper local search", candidates, top_k=2)

        self.assertEqual(ranked[0].id, "graph-rag")
        self.assertGreater(ranked[0].keyword_score, 0)
        self.assertIn("final", ranked[0].score_breakdown)

    def test_build_citations_exposes_source_and_page(self):
        candidate = SearchCandidate(
            id="doc_0",
            content="A paper excerpt about retrieval augmented generation.",
            metadata={
                "source_filename": "rag.pdf",
                "source_path": "E:/rag.pdf",
                "page": 3,
                "chunk_index": 2,
                "Header 2": "Method",
            },
            final_score=0.5,
        )

        citations = build_citations([candidate])

        self.assertEqual(citations[0]["id"], "E1")
        self.assertEqual(citations[0]["source_filename"], "rag.pdf")
        self.assertNotIn("source_path", citations[0])
        self.assertEqual(citations[0]["page_label"], "3")
        self.assertEqual(citations[0]["section"], "Method")


if __name__ == "__main__":
    unittest.main()
