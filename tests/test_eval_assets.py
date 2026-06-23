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
DATASET_PATH = ROOT / "eval" / "enterprise_zh_v0.1.jsonl"
SCRIPT_PATH = ROOT / "scripts" / "eval_rag.py"
ALLOWED_SOURCE_HINTS = {
    "大模型知识管理系统",
    "大模型基准测试体系研究报告",
    "数据分类分级规则",
    "生成式人工智能服务安全基本要求",
}


class EvalAssetsTests(unittest.TestCase):
    def test_enterprise_eval_dataset_has_readable_required_fields(self):
        records = _load_dataset()

        self.assertGreaterEqual(len(records), 16)
        ids = {record["id"] for record in records}
        self.assertIn("KM-01", ids)
        self.assertIn("NEG-01", ids)

        for record in records:
            self.assertIsInstance(record["id"], str)
            self.assertIsInstance(record["question"], str)
            self.assertIsInstance(record["expected_sources"], list)
            self.assertIsInstance(record["rubric"], list)
            self.assertIn(record["type"], {"single_doc", "cross_doc", "negative", "safety"})

    def test_enterprise_eval_dataset_only_targets_imported_four_documents(self):
        records = _load_dataset()

        for record in records:
            for source in record["expected_sources"]:
                self.assertIn(source, ALLOWED_SOURCE_HINTS)

    def test_eval_script_prints_plain_chinese_metrics_from_fixture(self):
        fixture = [
            {
                "id": "PASS-01",
                "question": "企业知识库为什么需要审计日志？",
                "expected_sources": ["数据安全法"],
                "type": "single_doc",
                "rubric": ["追溯", "安全保护义务"],
            },
            {
                "id": "NEG-01",
                "question": "SoulDrive 的创始人是谁？",
                "expected_sources": [],
                "type": "negative",
                "rubric": ["应拒答"],
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
        self.assertIn("检索命中率@3: 100.0%", output)
        self.assertIn("引用合法率: 100.0%", output)
        self.assertIn("拒答正确率: 100.0%", output)

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
