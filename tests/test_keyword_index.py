import tempfile
import unittest
from pathlib import Path

from core.keyword_index import KeywordIndex


class KeywordIndexTests(unittest.TestCase):
    def test_keyword_index_returns_matching_child_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index = KeywordIndex(str(Path(temp_dir) / "keyword.sqlite"))
            try:
                index.upsert_document(
                    doc_id="doc-1-child-0",
                    content="GraphRAG local search improves paper question answering.",
                    metadata={"source_filename": "graph.pdf"},
                )
                index.upsert_document(
                    doc_id="doc-2-child-0",
                    content="Vision transformer image classification baseline.",
                    metadata={"source_filename": "vision.pdf"},
                )

                results = index.search("GraphRAG paper search", limit=5)
            finally:
                index.close()

            self.assertEqual(results[0].doc_id, "doc-1-child-0")
            self.assertEqual(results[0].metadata["source_filename"], "graph.pdf")


if __name__ == "__main__":
    unittest.main()
