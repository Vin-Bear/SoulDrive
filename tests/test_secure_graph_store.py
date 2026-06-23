import tempfile
import unittest
from pathlib import Path

from core.secure_graph_store import SecureGraphStore
from core.workspace import SoulDriveWorkspace
from core.workspace_crypto import initialize_keystore, unlock_keystore


class SecureGraphStoreTests(unittest.TestCase):
    def test_secure_graph_store_returns_context_without_plaintext_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            initialize_keystore(workspace, "secret-passphrase")
            keys = unlock_keystore(workspace, "secret-passphrase")
            store = SecureGraphStore(str(Path(workspace.index_path) / "secure_graph.sqlite"), keys)
            try:
                store.add_entity("GraphRAG", "technique", "graph retrieval")
                store.add_entity("Local Search", "mode", "local graph search")
                store.add_relationship("GraphRAG", "Local Search", "supports")
                context = store.search_context("How does GraphRAG work?")
            finally:
                store.close()

            persisted = Path(workspace.secure_graph_store_path).read_bytes()

        self.assertEqual(context[0], "宸茬煡閫昏緫鍏崇郴: [GraphRAG] --(supports)--> [Local Search]")
        self.assertNotIn(b"GraphRAG", persisted)
        self.assertNotIn(b"Local Search", persisted)
        self.assertNotIn(b"supports", persisted)


if __name__ == "__main__":
    unittest.main()
