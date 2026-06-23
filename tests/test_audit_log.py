import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.audit_log import AuditLogger, append_audit_event


class AuditLogTests(unittest.TestCase):
    def test_append_audit_event_hash_chains_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            state_path = Path(temp_dir) / "audit_state.json"

            first = append_audit_event(
                "runtime.unlock",
                {"hardware_sn": "SECRET-SN", "auth_level": "PRO"},
                audit_path=audit_path,
                state_path=state_path,
            )
            second = append_audit_event(
                "chat.completed",
                {"trace_id": "trace-1", "citations": [{"id": "E1"}]},
                audit_path=audit_path,
                state_path=state_path,
            )

            records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 2)
        self.assertEqual(second["previous_hash"], first["event_hash"])
        self.assertNotIn("SECRET-SN", json.dumps(records, ensure_ascii=False))
        self.assertIn("hardware_sn_hash", records[0]["payload"])

    def test_verify_chain_reports_valid_audit_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            state_path = Path(temp_dir) / "audit_state.json"
            logger = AuditLogger(audit_path, state_path)
            logger.append_event("runtime.unlock", {"auth_level": "PRO"})
            logger.append_event("chat.completed", {"citations": [{"id": "E1"}]})

            report = logger.verify_chain()

        self.assertTrue(report["ready"])
        self.assertEqual(report["event_count"], 2)
        self.assertEqual(report["checked_events"], 2)
        self.assertIsNone(report["broken_at"])
        self.assertNotIn("audit_path", report)
        self.assertTrue(report["audit_log"].startswith("audit_log:"))
        self.assertNotIn(str(audit_path), json.dumps(report, ensure_ascii=False))

    def test_verify_chain_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            state_path = Path(temp_dir) / "audit_state.json"
            logger = AuditLogger(audit_path, state_path)
            logger.append_event("runtime.unlock", {"auth_level": "PRO"})
            records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
            records[0]["payload"]["auth_level"] = "ADMIN"
            audit_path.write_text(json.dumps(records[0], ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

            report = logger.verify_chain()

        self.assertFalse(report["ready"])
        self.assertEqual(report["broken_at"]["reason"], "event_hash_mismatch")
        self.assertNotIn("audit_path", report)

    def test_verify_chain_detects_invalid_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            state_path = Path(temp_dir) / "audit_state.json"
            logger = AuditLogger(audit_path, state_path)
            logger.append_event("runtime.unlock", {"auth_level": "PRO"})
            with audit_path.open("a", encoding="utf-8") as file:
                file.write("{broken-json\n")

            report = logger.verify_chain()

        self.assertFalse(report["ready"])
        self.assertEqual(report["invalid_lines"], [2])
        self.assertNotIn("audit_path", report)

    def test_default_audit_logger_follows_app_runtime_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"SOULDRIVE_APP_ROOT": temp_dir}, clear=False):
                event = append_audit_event("runtime.unlock", {"auth_level": "PRO"})
                audit_path = Path(temp_dir) / "runtime" / "audit_log.jsonl"
                self.assertTrue(audit_path.exists())
                self.assertFalse((Path(temp_dir) / "souldrive_db").exists())

        self.assertEqual(event["event_type"], "runtime.unlock")


if __name__ == "__main__":
    unittest.main()
