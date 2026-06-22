import base64
import json
import tempfile
import unittest
from pathlib import Path

from core.workspace import SoulDriveWorkspace
from core.workspace_crypto import (
    IncorrectPassphraseError,
    initialize_keystore,
    is_keystore_initialized,
    unlock_keystore,
)


class WorkspaceCryptoTests(unittest.TestCase):
    def test_initialize_keystore_writes_wrapped_key_without_plaintext_passphrase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            result = initialize_keystore(workspace, "correct horse battery staple")
            payload = json.loads(Path(workspace.keystore_path).read_text(encoding="utf-8"))
            initialized = is_keystore_initialized(workspace)

        self.assertTrue(result["initialized"])
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["kdf"], "pbkdf2-sha256")
        self.assertEqual(payload["key_wrap"], "aes-256-gcm")
        self.assertNotIn("correct horse battery staple", json.dumps(payload))
        self.assertGreater(len(base64.b64decode(payload["encrypted_workspace_data_key"])), 32)
        self.assertTrue(initialized)

    def test_unlock_keystore_rejects_wrong_passphrase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            initialize_keystore(workspace, "right-passphrase")

            with self.assertRaises(IncorrectPassphraseError):
                unlock_keystore(workspace, "wrong-passphrase")

    def test_unlock_keystore_returns_stable_purpose_keys_for_correct_passphrase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            initialize_keystore(workspace, "right-passphrase")

            first = unlock_keystore(workspace, "right-passphrase")
            second = unlock_keystore(workspace, "right-passphrase")

        self.assertEqual(first.document_key, second.document_key)
        self.assertEqual(first.index_key, second.index_key)
        self.assertEqual(first.graph_key, second.graph_key)
        self.assertEqual(first.audit_key, second.audit_key)
        self.assertEqual(len(first.workspace_data_key), 32)
        self.assertEqual(len(first.document_key), 32)


if __name__ == "__main__":
    unittest.main()
