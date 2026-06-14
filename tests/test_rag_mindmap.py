import unittest

from core.rag_engine import build_evidence_mindmap, query_requests_mindmap


class RagMindmapTests(unittest.TestCase):
    def test_concept_mindmap_matches_transformer_architecture_query(self):
        evidence = [
            {
                "id": "E1",
                "source_filename": "Attention Is All You Need.pdf",
                "section": "3 Model Architecture",
                "page": 4,
                "snippet": "The Transformer follows this overall architecture using stacked self-attention and point-wise fully connected layers.",
            },
            {
                "id": "E2",
                "source_filename": "Attention Is All You Need.pdf",
                "section": "3.2 Attention",
                "page": 5,
                "snippet": "An attention function can be described as mapping a query and a set of key-value pairs to an output.",
            }
        ]

        mindmap = build_evidence_mindmap("可不可以使用知识图谱给我描述一下 Transformer 架构", evidence)

        self.assertTrue(query_requests_mindmap("可不可以使用知识图谱给我描述一下 Transformer 架构"))
        self.assertIn("# Transformer 架构", mindmap)
        self.assertIn("## 核心思想", mindmap)
        self.assertIn("## 结构组成", mindmap)
        self.assertIn("## 主要优势", mindmap)
        self.assertIn("## 论文依据", mindmap)
        self.assertIn("[E1]", mindmap)
        self.assertIn("[E2]", mindmap)
        self.assertNotIn("# Attention Is All You Need", mindmap)
        self.assertNotIn("来源:", mindmap)
        self.assertNotIn("线索:", mindmap)


if __name__ == "__main__":
    unittest.main()
