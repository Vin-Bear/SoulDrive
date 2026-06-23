import json
import os
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout

from scripts import eval_rag


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "eval" / "enterprise_zh_v0.2.jsonl"
SCRIPT_PATH = ROOT / "scripts" / "eval_rag.py"
ALLOWED_SOURCE_HINTS = {
    "大模型知识管理系统",
    "大模型基准测试体系研究报告",
    "数据分类分级规则",
    "生成式人工智能服务安全基本要求",
    "知识库问答场景中大语言模型私有化研究与应用实践",
    "大模型私有部署与基础应用落地",
    "城市+AI应用场景清单",
    "综合交通运输大模型智能体创新应用典型案例",
    "人工智能安全治理蓝皮书",
    "数据要素发展报告",
    "基于大模型和RAG的知识库问答系统",
    "企业级RAG方案",
    "MaxKB",
    "LazyLLM",
}


class EvalAssetsTests(unittest.TestCase):
    def test_enterprise_eval_dataset_has_readable_required_fields(self):
        records = _load_dataset()

        self.assertGreaterEqual(len(records), 80)
        ids = {record["id"] for record in records}
        self.assertIn("KM-01", ids)
        self.assertIn("NEG-01", ids)
        self.assertIn("LAZY-01", ids)
        self.assertIn("XSEC-01", ids)

        for record in records:
            self.assertIsInstance(record["id"], str)
            self.assertIsInstance(record["question"], str)
            self.assertIsInstance(record["expected_sources"], list)
            self.assertIsInstance(record["rubric"], list)
            self.assertIsInstance(record.get("expected_keywords", []), list)
            self.assertIsInstance(record.get("should_refuse", False), bool)
            self.assertIsInstance(record.get("requires_multi_doc", False), bool)
            self.assertIsInstance(record.get("graph_relevant", False), bool)
            self.assertIn(record["type"], {"single_doc", "cross_doc", "negative", "safety"})

    def test_enterprise_eval_dataset_targets_enterprise_private_rag_documents(self):
        records = _load_dataset()
        covered_sources = set()

        for record in records:
            for source in record["expected_sources"]:
                self.assertIn(source, ALLOWED_SOURCE_HINTS)
                covered_sources.add(source)

        self.assertGreaterEqual(len(covered_sources), 10)
        self.assertTrue(any(record.get("requires_multi_doc") for record in records))
        self.assertTrue(any(record.get("graph_relevant") for record in records))
        self.assertTrue(any(record.get("should_refuse") for record in records))

    def test_eval_script_prints_plain_chinese_metrics_from_fixture(self):
        fixture = [
            {
                "id": "PASS-01",
                "question": "企业知识库为什么需要审计日志？",
                "expected_sources": ["数据安全法"],
                "type": "single_doc",
                "rubric": ["追溯", "安全保护义务"],
                "expected_keywords": ["审计", "追溯"],
            },
            {
                "id": "NEG-01",
                "question": "SoulDrive 的创始人是谁？",
                "expected_sources": [],
                "type": "negative",
                "rubric": ["应拒答"],
                "should_refuse": True,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "fixture.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in fixture),
                encoding="utf-8",
            )
            result_path = Path(temp_dir) / "retrieval_results.jsonl"
            result_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "PASS-01",
                                "answer": "企业知识库需要审计日志以支持追溯 [E1]。",
                                "evidence": [
                                    {
                                        "id": "E1",
                                        "source_filename": "数据安全法.pdf",
                                        "score": 0.24,
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "id": "NEG-01",
                                "answer": "根据本地知识库，未找到足够可靠的相关证据。",
                                "evidence": [],
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = eval_rag.main(
                    [
                        "--dataset",
                        str(dataset_path),
                        "--from-results",
                        str(result_path),
                    ]
                )

        output = stdout.getvalue()
        self.assertEqual(return_code, 0)
        self.assertIn("评测样本数: 2", output)
        self.assertIn("检索全命中率@3: 100.0%", output)
        self.assertIn("检索任一命中率@3: 100.0%", output)
        self.assertIn("首条命中率@1: 100.0%", output)
        self.assertIn("平均倒数排名(MRR): 1.000", output)
        self.assertIn("平均来源覆盖率@3: 100.0%", output)
        self.assertIn("引用合法率: 100.0%", output)
        self.assertIn("拒答正确率: 100.0%", output)
        self.assertIn("答案关键词覆盖率: 100.0%", output)

    def test_source_matching_accepts_local_filename_aliases(self):
        self.assertTrue(eval_rag.source_matches("“城市+AI”应用场景清单（第五批）.pdf", "城市+AI应用场景清单"))
        self.assertTrue(eval_rag.source_matches("综合交通运输大模型智能体创新应用.pdf", "综合交通运输大模型智能体创新应用典型案例"))
        self.assertTrue(eval_rag.source_matches("基于大模型和RAG的知识库问答系统.pdf", "MaxKB"))

    def test_all_known_local_document_names_match_dataset_sources(self):
        local_names = [
            "LazyLLM企业级RAG方案-私有化部署与权限安全.pdf",
            "基于大模型和RAG的知识库问答系统.pdf",
            "数据要素发展报告.pdf",
            "人工智能安全治理蓝皮书.pdf",
            "综合交通运输大模型智能体创新应用.pdf",
            "“城市+AI”应用场景清单（第五批）.pdf",
            "大模型私有部署与基础应用落地.pdf",
            "知识库问答场景中大语言模型私有化研究与应用实践.pdf",
            "生成式人工智能服务安全基本要求.pdf",
            "数据安全技术 数据分类分级规则.pdf",
            "大模型基准测试体系研究报告.pdf",
            "大模型知识管理系统.pdf",
        ]
        expected_sources = {
            source
            for record in _load_dataset()
            for source in record["expected_sources"]
        }

        unmatched = [
            source
            for source in expected_sources
            if not any(eval_rag.source_matches(name, source) for name in local_names)
        ]

        self.assertEqual(unmatched, [])

    def test_eval_script_reports_missing_passphrase_for_encrypted_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "SoulDrive"
            (workspace / "config").mkdir(parents=True)
            (workspace / "index").mkdir()
            (workspace / "config" / "workspace.json").write_text("{}", encoding="utf-8")
            (workspace / "index" / "secure_vectors.sqlite").write_bytes(b"")

            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                eval_rag.main(["--dataset", str(DATASET_PATH), "--workspace", str(workspace)])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--passphrase", stderr.getvalue())

    def test_eval_script_can_resolve_workspace_from_runtime_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "SoulDrive"
            (workspace / "config").mkdir(parents=True)
            (workspace / "config" / "workspace.json").write_text("{}", encoding="utf-8")

            original_get_runtime_state = eval_rag.get_runtime_state
            try:
                eval_rag.get_runtime_state = lambda: {
                    "workspace_path": str(workspace),
                    "locked": False,
                    "software_unlocked": True,
                }

                self.assertEqual(eval_rag.resolve_workspace_arg(None), str(workspace))
            finally:
                eval_rag.get_runtime_state = original_get_runtime_state

    def test_eval_script_can_resolve_project_local_workspace(self):
        with patch.object(eval_rag, "find_unlocked_drive_workspace", return_value=None):
            self.assertEqual(eval_rag.resolve_workspace_arg(None), str(ROOT / "souldrive_db"))

    def test_eval_script_prefers_unlocked_drive_workspace_over_project_local_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            drive_root = Path(temp_dir)
            workspace = drive_root / "SoulDrive"
            (workspace / "config").mkdir(parents=True)
            (workspace / "runtime").mkdir()
            (workspace / "config" / "workspace.json").write_text("{}", encoding="utf-8")
            (workspace / "runtime" / "runtime_state.json").write_text(
                json.dumps({"locked": False, "workspace_path": str(workspace)}),
                encoding="utf-8",
            )

            fake_drive = type("Drive", (), {"mountpoint": str(drive_root) + os.sep})()
            with patch.object(eval_rag, "iter_filesystem_drives", return_value=[fake_drive]):
                self.assertEqual(eval_rag.resolve_workspace_arg(None), str(workspace))

    def test_eval_script_prepares_encrypted_workspace_from_passphrase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            from core.security_context import clear_workspace_keys, get_workspace_keys
            from core.workspace import SoulDriveWorkspace
            from core.workspace_crypto import initialize_keystore

            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            initialize_keystore(workspace, "secret-passphrase")

            try:
                self.assertTrue(eval_rag.prepare_workspace_keys(workspace.root_path, "secret-passphrase"))
                self.assertIsNotNone(get_workspace_keys(workspace.root_path))
            finally:
                clear_workspace_keys(workspace.root_path)


def _load_dataset():
    records = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    unittest.main()
