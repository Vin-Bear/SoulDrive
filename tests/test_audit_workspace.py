import tempfile
import unittest
from pathlib import Path

from core.audit_log import AuditLogger
from core.workspace import SoulDriveWorkspace


class AuditWorkspaceTests(unittest.TestCase):
    def test_audit_logger_can_be_bound_to_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            logger = AuditLogger.for_workspace(workspace)
            logger.append_event("test.event", {"value": 1})

            self.assertTrue(Path(workspace.audit_log_path).exists())


if __name__ == "__main__":
    unittest.main()
