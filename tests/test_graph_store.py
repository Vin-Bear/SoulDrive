import tempfile
import unittest
from pathlib import Path

from core.graph_db import LocalGraphDB


class GraphStoreTests(unittest.TestCase):
    def test_sqlite_graph_store_requires_explicit_path(self):
        with self.assertRaisesRegex(ValueError, "db_path"):
            LocalGraphDB()

    def test_sqlite_graph_store_returns_context_for_matched_entity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = LocalGraphDB(db_path=str(Path(temp_dir) / "graph.sqlite"))
            try:
                db.add_entity("GraphRAG", "技术", "结合图谱和检索增强生成")
                db.add_entity("Local Search", "查询模式", "围绕实体邻域检索")
                db.add_relationship("GraphRAG", "Local Search", "支持")

                context = db.search_context("如何使用 GraphRAG 做论文问答？")
            finally:
                db.close()

        self.assertEqual(context[0], "已知逻辑关系: [GraphRAG] --(支持)--> [Local Search]")


if __name__ == "__main__":
    unittest.main()
