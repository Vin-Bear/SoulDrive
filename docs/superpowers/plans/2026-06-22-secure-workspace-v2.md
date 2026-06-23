# Secure Workspace V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade SoulDrive from software unlock gating to encrypted-at-rest workspace protection for local private documents, indexes, graph data, and runtime workflows.

**Architecture:** Keep the existing offline sidecar architecture. Upgrade the keystore to Argon2id-wrapped workspace keys, then route imports and indexing through encrypted object stores. Retrieval loads only unlocked data into memory and keeps persisted PDF, chunk, vector, metadata, and graph content encrypted.

**Tech Stack:** Python 3.10, FastAPI, cryptography 48.0.1, AES-256-GCM, HKDF-SHA256, Argon2id, SQLite, unittest, React/Tauri.

---

## Files And Responsibilities

- `core/workspace_crypto.py`: keystore v1/v2 compatibility, Argon2id password KDF, purpose key derivation, passphrase rotation.
- `core/secure_object_store.py`: authenticated encryption envelope for binary and JSON workspace objects.
- `core/secure_document_store.py`: encrypted PDF/document manifest and document listing.
- `core/secure_vector_store.py`: encrypted chunk, metadata, and embedding persistence with in-memory vector search after unlock.
- `core/secure_graph_store.py`: encrypted graph entity/relationship persistence with in-memory graph traversal after unlock.
- `core/security_context.py`: process-local unlocked workspace key cache, lock cleanup hooks, and temporary plaintext session cleanup.
- `core/paper_importer.py`: encrypted import instead of plaintext PDF copy.
- `core/indexer.py`: parse encrypted documents through controlled temporary plaintext files and index into secure stores.
- `core/knowledge_base.py`: switch retrieval to secure vector store when workspace security v2 is active.
- `core/mcp_server.py`: expose migration/status behavior and ensure locked workflows cannot access secure stores.
- `tests/test_workspace_crypto.py`: keystore v2, v1 migration, wrong password, passphrase rotation.
- `tests/test_secure_object_store.py`: envelope encryption integrity tests.
- `tests/test_secure_document_store.py`: encrypted import/listing/no plaintext PDF tests.
- `tests/test_secure_vector_store.py`: encrypted vector persistence and unlocked retrieval tests.
- `tests/test_secure_graph_store.py`: encrypted graph persistence and traversal tests.
- `tests/test_secure_workspace_scan.py`: full-workspace plaintext leak regression.
- `tests/test_mcp_server_papers.py`: update import/list behavior expectations for encrypted documents.

## Task 1: Keystore V2

- [ ] Write failing tests for Argon2id v2 initialization, v1 unlock compatibility, v1-to-v2 migration, and passphrase rotation.
- [ ] Implement v2 keystore format using Argon2id and AES-256-GCM key wrap.
- [ ] Keep v1 PBKDF2 unlock compatibility.
- [ ] Add explicit migration helper that rewrites keystore after successful v1 unlock.
- [ ] Verify `conda run -n souldrive python -m unittest tests.test_workspace_crypto -v`.

## Task 2: Secure Object Store

- [ ] Write failing tests for binary roundtrip, JSON roundtrip, wrong key failure, ciphertext tamper failure, and AAD mismatch failure.
- [ ] Implement AES-256-GCM envelope with magic/version/purpose/object id/nonce/ciphertext.
- [ ] Keep file names opaque using HMAC-derived object ids where needed.
- [ ] Verify `conda run -n souldrive python -m unittest tests.test_secure_object_store -v`.

## Task 3: Encrypted Document Store

- [ ] Write failing tests proving imported PDFs are stored as encrypted objects and no `.pdf` plaintext appears in workspace managed document storage.
- [ ] Implement encrypted document manifest and document listing compatible with current `/documents/list` payload.
- [ ] Update importer to require unlocked workspace keys.
- [ ] Verify `conda run -n souldrive python -m unittest tests.test_secure_document_store tests.test_mcp_server_papers -v`.

## Task 4: Secure Vector Store

- [ ] Write failing tests proving persisted chunks, metadata, and vectors do not contain plaintext sensitive fixture text.
- [ ] Implement encrypted SQLite vector store and in-memory cosine search.
- [ ] Wire `LocalKnowledgeBase` to use secure store for v2 workspaces.
- [ ] Verify `conda run -n souldrive python -m unittest tests.test_secure_vector_store -v`.

## Task 5: Secure Graph Store

- [ ] Write failing tests proving persisted graph entities/relationships do not expose plaintext.
- [ ] Implement encrypted graph store with in-memory traversal after unlock.
- [ ] Wire graph extractor to secure graph store for v2 workspaces.
- [ ] Verify `conda run -n souldrive python -m unittest tests.test_secure_graph_store tests.test_graph_store -v`.

## Task 6: Indexer And Temporary Plaintext

- [ ] Write failing tests proving encrypted PDFs can be indexed through temporary files and temp files are deleted after success and failure.
- [ ] Implement controlled session temp directory for parser compatibility.
- [ ] Update indexer discovery to read encrypted document manifest instead of plaintext PDF glob when v2 is active.
- [ ] Verify `conda run -n souldrive python -m unittest tests.test_indexer_scope -v`.

## Task 7: Migration And Runtime Locking

- [ ] Write failing tests for old plaintext workspace detection, migration into encrypted stores, and lock cleanup.
- [ ] Implement migration helper for local testing workspaces.
- [ ] Update `/security/status` with `workspace_security_version` and migration flags.
- [ ] Verify locked endpoints reject secure data access.

## Task 8: Full Regression And Packaging

- [ ] Run `conda run -n souldrive python -m unittest discover -s tests -v`.
- [ ] Run `npm run build` in `souldrive-ui`.
- [ ] Run `.\scripts\package-sidecar.ps1`.
- [ ] Verify a packaged sidecar can initialize, import encrypted documents, lock, unlock, and query status.
