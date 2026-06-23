import tempfile
import unittest
from pathlib import Path

from core.secure_document_store import SecureDocumentStore
from core.workspace import SoulDriveWorkspace
from core.workspace_crypto import initialize_keystore, unlock_keystore


class SecureDocumentStoreTests(unittest.TestCase):
    def test_import_document_stores_encrypted_payload_without_plaintext_pdf_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            source_pdf = Path(temp_dir) / "paper.pdf"
            source_pdf.write_bytes(b"%PDF encrypted payload")
            initialize_keystore(workspace, "secret-passphrase")
            keys = unlock_keystore(workspace, "secret-passphrase")
            store = SecureDocumentStore(workspace, keys)
            try:
                item = store.import_document(str(source_pdf))
                listed = store.list_documents()
                encrypted_files = list(Path(workspace.documents_path).rglob("*.sdoc"))
                plaintext_pdf = list(Path(workspace.documents_path).rglob("*.pdf"))
            finally:
                store.close()

        self.assertEqual(item["status"], "imported")
        self.assertEqual(item["name"], "paper.pdf")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["name"], "paper.pdf")
        self.assertEqual(len(encrypted_files), 1)
        self.assertEqual(len(plaintext_pdf), 0)


if __name__ == "__main__":
    unittest.main()
