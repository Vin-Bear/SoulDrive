# Personal Encrypted Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-version personal workspace software unlock with a passphrase-wrapped workspace data key, no recovery, gatekeeping, audit events, and UI unlock flow.

**Architecture:** Create a focused `core/workspace_crypto.py` module for keystore lifecycle and key derivation. Extend runtime state with crypto/software-unlock fields, add `/security/*` routes in `core/mcp_server.py`, and gate sensitive workflows until software unlock succeeds. Update React UI to initialize or unlock the workspace with explicit no-recovery acknowledgement.

**Tech Stack:** Python `unittest`, FastAPI, `cryptography` AESGCM/HKDF/PBKDF2HMAC, React 19 + TypeScript, Tauri

---

## File Structure Map

**Create:**

- `D:\PycharmProjects\LangChainProjects\SoulDrive\core\workspace_crypto.py`
  Purpose: keystore creation, passphrase verification, data key wrapping/unwrapping, purpose-key derivation.
- `D:\PycharmProjects\LangChainProjects\SoulDrive\tests\test_workspace_crypto.py`
  Purpose: prove keystore and passphrase behavior.
- `D:\PycharmProjects\LangChainProjects\SoulDrive\tests\test_mcp_server_security.py`
  Purpose: prove `/security/*` APIs and gatekeeping.

**Modify:**

- `D:\PycharmProjects\LangChainProjects\SoulDrive\core\workspace.py`
  Purpose: expose `keystore_path`.
- `D:\PycharmProjects\LangChainProjects\SoulDrive\core\runtime_state.py`
  Purpose: add crypto/software-unlock state and helpers.
- `D:\PycharmProjects\LangChainProjects\SoulDrive\core\mcp_server.py`
  Purpose: add security APIs and gate chat/import/index.
- `D:\PycharmProjects\LangChainProjects\SoulDrive\souldrive-ui\src\App.tsx`
  Purpose: show setup/unlock panel and call security APIs.

### Task 1: Add Workspace Keystore Crypto Module

**Files:**
- Create: `D:\PycharmProjects\LangChainProjects\SoulDrive\tests\test_workspace_crypto.py`
- Create: `D:\PycharmProjects\LangChainProjects\SoulDrive\core\workspace_crypto.py`
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\core\workspace.py`

- [ ] **Step 1: Write failing crypto tests**

Create `tests/test_workspace_crypto.py`:

```python
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

        self.assertTrue(result["initialized"])
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["kdf"], "pbkdf2-sha256")
        self.assertEqual(payload["key_wrap"], "aes-256-gcm")
        self.assertNotIn("correct horse battery staple", json.dumps(payload))
        self.assertGreater(len(base64.b64decode(payload["encrypted_workspace_data_key"])), 32)
        self.assertTrue(is_keystore_initialized(workspace))

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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
conda run -n souldrive python -m unittest tests.test_workspace_crypto -v
```

Expected: import failure for `core.workspace_crypto` or missing `keystore_path`.

- [ ] **Step 3: Implement keystore path and crypto module**

Add to `SoulDriveWorkspace` in `core/workspace.py`:

```python
    @property
    def keystore_path(self):
        return str(Path(self.config_path) / "keystore.json")
```

Create `core/workspace_crypto.py`:

```python
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


KEYSTORE_VERSION = 1
KDF_NAME = "pbkdf2-sha256"
KEY_WRAP = "aes-256-gcm"
DATA_KEY_BYTES = 32
SALT_BYTES = 16
NONCE_BYTES = 12
PBKDF2_ITERATIONS = 600_000


class WorkspaceCryptoError(Exception):
    pass


class KeystoreAlreadyInitializedError(WorkspaceCryptoError):
    pass


class KeystoreNotInitializedError(WorkspaceCryptoError):
    pass


class IncorrectPassphraseError(WorkspaceCryptoError):
    pass


@dataclass(frozen=True)
class WorkspaceKeys:
    workspace_data_key: bytes
    document_key: bytes
    index_key: bytes
    graph_key: bytes
    audit_key: bytes


def is_keystore_initialized(workspace) -> bool:
    return Path(workspace.keystore_path).exists()


def initialize_keystore(workspace, passphrase: str) -> dict:
    if not passphrase:
        raise ValueError("passphrase is required")
    path = Path(workspace.keystore_path)
    if path.exists():
        raise KeystoreAlreadyInitializedError("workspace keystore already initialized")

    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    workspace_data_key = os.urandom(DATA_KEY_BYTES)
    password_key = _derive_password_key(passphrase, salt, PBKDF2_ITERATIONS)
    encrypted = AESGCM(password_key).encrypt(nonce, workspace_data_key, _aad())
    payload = {
        "version": KEYSTORE_VERSION,
        "kdf": KDF_NAME,
        "kdf_params": {
            "iterations": PBKDF2_ITERATIONS,
            "salt": _b64(salt),
        },
        "key_wrap": KEY_WRAP,
        "nonce": _b64(nonce),
        "encrypted_workspace_data_key": _b64(encrypted),
        "created_at": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"initialized": True}


def unlock_keystore(workspace, passphrase: str) -> WorkspaceKeys:
    if not passphrase:
        raise IncorrectPassphraseError("passphrase is required")
    path = Path(workspace.keystore_path)
    if not path.exists():
        raise KeystoreNotInitializedError("workspace keystore is not initialized")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != KEYSTORE_VERSION or payload.get("kdf") != KDF_NAME or payload.get("key_wrap") != KEY_WRAP:
        raise WorkspaceCryptoError("unsupported keystore format")

    params = payload.get("kdf_params") or {}
    salt = _unb64(params["salt"])
    iterations = int(params["iterations"])
    nonce = _unb64(payload["nonce"])
    encrypted = _unb64(payload["encrypted_workspace_data_key"])
    password_key = _derive_password_key(passphrase, salt, iterations)
    try:
        workspace_data_key = AESGCM(password_key).decrypt(nonce, encrypted, _aad())
    except InvalidTag as exc:
        raise IncorrectPassphraseError("incorrect passphrase") from exc

    return WorkspaceKeys(
        workspace_data_key=workspace_data_key,
        document_key=_derive_purpose_key(workspace_data_key, b"documents"),
        index_key=_derive_purpose_key(workspace_data_key, b"indexes"),
        graph_key=_derive_purpose_key(workspace_data_key, b"graph"),
        audit_key=_derive_purpose_key(workspace_data_key, b"audit"),
    )


def _derive_password_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=DATA_KEY_BYTES,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def _derive_purpose_key(workspace_data_key: bytes, purpose: bytes) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=DATA_KEY_BYTES,
        salt=None,
        info=b"souldrive:" + purpose,
    )
    return hkdf.derive(workspace_data_key)


def _aad() -> bytes:
    return b"SoulDrive workspace data key v1"


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))
```

- [ ] **Step 4: Run crypto tests and commit**

Run:

```powershell
conda run -n souldrive python -m unittest tests.test_workspace_crypto -v
```

Expected: `OK`.

Commit:

```powershell
git add core/workspace.py core/workspace_crypto.py tests/test_workspace_crypto.py
git commit -m "feat: add personal workspace keystore"
```

### Task 2: Add Runtime Security State And Gatekeeping

**Files:**
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\tests\test_runtime_state.py`
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\core\runtime_state.py`
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\core\mcp_server.py`
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\tests\test_mcp_server_indexing.py`

- [ ] **Step 1: Add failing runtime state tests**

Add to `RuntimeStateTests`:

```python
    def test_runtime_unlock_marks_hardware_only_until_software_unlock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            with patch.object(runtime_state, "STATE_PATH", state_path):
                state = runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)

        self.assertTrue(state["locked"])
        self.assertTrue(state["hardware_mounted"])
        self.assertFalse(state["software_unlocked"])
        self.assertEqual(state["security_reason"], "workspace passphrase required")

    def test_mark_software_unlocked_allows_sensitive_workflows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            with patch.object(runtime_state, "STATE_PATH", state_path):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                state = runtime_state.mark_software_unlocked()

        self.assertFalse(state["locked"])
        self.assertTrue(state["software_unlocked"])
        self.assertEqual(state["auth_level"], "HARDWARE_PLUS_PASSWORD")
```

- [ ] **Step 2: Run runtime tests and verify failure**

Run:

```powershell
conda run -n souldrive python -m unittest tests.test_runtime_state.RuntimeStateTests.test_runtime_unlock_marks_hardware_only_until_software_unlock tests.test_runtime_state.RuntimeStateTests.test_mark_software_unlocked_allows_sensitive_workflows -v
```

Expected: missing fields or missing `mark_software_unlocked`.

- [ ] **Step 3: Extend runtime state**

In `core/runtime_state.py`, add default fields:

```python
    "hardware_mounted": False,
    "crypto_initialized": False,
    "software_unlocked": False,
    "security_reason": "storage device required",
```

Change `unlock_runtime` so hardware mount does not fully unlock sensitive workflows:

```python
        locked=True,
        reason="workspace passphrase required",
        auth_level="HARDWARE_ONLY",
        hardware_mounted=True,
        crypto_initialized=False,
        software_unlocked=False,
        security_reason="workspace passphrase required",
```

Add:

```python
def mark_software_unlocked(reason: str = "workspace unlocked"):
    state = set_runtime_state(
        locked=False,
        reason=reason,
        auth_level="HARDWARE_PLUS_PASSWORD",
        hardware_mounted=True,
        software_unlocked=True,
        security_reason=reason,
    )
    _audit_logger_for_workspace_path(state.get("workspace_path")).append_event("security.unlock_succeeded", {})
    return state


def mark_software_locked(reason: str = "workspace passphrase required"):
    previous_state = get_runtime_state()
    state = set_runtime_state(
        locked=True,
        reason=reason,
        auth_level="HARDWARE_ONLY" if previous_state.get("hardware_mounted") else "NONE",
        software_unlocked=False,
        security_reason=reason,
    )
    _audit_logger_for_workspace_path(previous_state.get("workspace_path")).append_event("security.software_locked", {"reason": reason})
    return state
```

- [ ] **Step 4: Add gatekeeping tests for index**

Update `test_index_run_starts_worker_for_active_workspace` and `test_index_run_rejects_when_worker_is_already_running` to call `runtime_state.mark_software_unlocked()` after `unlock_runtime`.

Add:

```python
    def test_index_run_rejects_hardware_only_workspace_before_software_unlock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()

            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                response = TestClient(mcp_server.app).post(
                    "/index/run",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        self.assertEqual(response.status_code, 423)
        self.assertEqual(response.json()["reason"], "workspace passphrase required")
```

- [ ] **Step 5: Implement gate helper in mcp_server**

Add in `core/mcp_server.py`:

```python
def require_software_unlock():
    state = get_runtime_state()
    if state.get("locked") or not state.get("software_unlocked"):
        current_audit_logger().append_event("security.password_required", {
            "reason": state.get("security_reason") or state.get("reason") or "workspace passphrase required",
        })
        return JSONResponse(
            {
                "error": "SoulDrive workspace requires passphrase unlock",
                "status": "locked",
                "reason": "workspace passphrase required",
            },
            status_code=423,
        )
    return None
```

Use it in `/chat`, `_import_documents`, and `/index/run`.

- [ ] **Step 6: Run runtime/index tests and commit**

Run:

```powershell
conda run -n souldrive python -m unittest tests.test_runtime_state tests.test_mcp_server_indexing -v
```

Expected: `OK`.

Commit:

```powershell
git add core/runtime_state.py core/mcp_server.py tests/test_runtime_state.py tests/test_mcp_server_indexing.py
git commit -m "feat: require software unlock for sensitive workflows"
```

### Task 3: Add Security API Routes

**Files:**
- Create: `D:\PycharmProjects\LangChainProjects\SoulDrive\tests\test_mcp_server_security.py`
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\core\mcp_server.py`
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\core\runtime_state.py`

- [ ] **Step 1: Write failing security API tests**

Create `tests/test_mcp_server_security.py`:

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from core import mcp_server
import core.runtime_state as runtime_state
from core.workspace import SoulDriveWorkspace


class McpServerSecurityTests(unittest.TestCase):
    def test_security_init_requires_no_recovery_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict("os.environ", {"SOULDRIVE_API_TOKEN": "test-token"}, clear=False):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                response = TestClient(mcp_server.app).post(
                    "/security/init",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "secret-passphrase", "acknowledge_no_recovery": False},
                )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "no recovery acknowledgement required")

    def test_security_init_and_unlock_enable_sensitive_workflows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict("os.environ", {"SOULDRIVE_API_TOKEN": "test-token"}, clear=False):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                client = TestClient(mcp_server.app)
                init_response = client.post(
                    "/security/init",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "secret-passphrase", "acknowledge_no_recovery": True},
                )
                lock_response = client.post(
                    "/security/lock",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={},
                )
                unlock_response = client.post(
                    "/security/unlock",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "secret-passphrase"},
                )
                state = runtime_state.get_runtime_state()

        self.assertEqual(init_response.status_code, 200)
        self.assertEqual(lock_response.status_code, 200)
        self.assertEqual(unlock_response.status_code, 200)
        self.assertFalse(state["locked"])
        self.assertTrue(state["software_unlocked"])

    def test_security_unlock_rejects_wrong_passphrase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict("os.environ", {"SOULDRIVE_API_TOKEN": "test-token"}, clear=False):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                client = TestClient(mcp_server.app)
                client.post(
                    "/security/init",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "secret-passphrase", "acknowledge_no_recovery": True},
                )
                response = client.post(
                    "/security/unlock",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "wrong-passphrase"},
                )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"], "incorrect passphrase")
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
conda run -n souldrive python -m unittest tests.test_mcp_server_security -v
```

Expected: 404 for `/security/*`.

- [ ] **Step 3: Implement security routes**

In `core/mcp_server.py`, import workspace crypto and runtime helpers:

```python
from core.runtime_state import get_runtime_state, lock_runtime, mark_software_locked, mark_software_unlocked, unlock_runtime, update_indexing_status
from core.workspace_crypto import (
    IncorrectPassphraseError,
    KeystoreAlreadyInitializedError,
    KeystoreNotInitializedError,
    initialize_keystore,
    is_keystore_initialized,
    unlock_keystore,
)
```

Add models:

```python
class SecurityInitRequest(BaseModel):
    passphrase: str = Field(min_length=8, max_length=256)
    acknowledge_no_recovery: bool = False


class SecurityUnlockRequest(BaseModel):
    passphrase: str = Field(min_length=1, max_length=256)
```

Add routes:

```python
@app.get("/security/status")
async def security_status():
    workspace = current_workspace()
    state = get_runtime_state()
    return {
        "crypto_initialized": is_keystore_initialized(workspace),
        "software_unlocked": bool(state.get("software_unlocked")),
        "hardware_mounted": bool(state.get("hardware_mounted") or state.get("workspace_path")),
        "reason": state.get("security_reason") or state.get("reason"),
        "no_recovery": True,
    }


@app.post("/security/init")
async def security_init(request: SecurityInitRequest):
    if not request.acknowledge_no_recovery:
        return JSONResponse({"error": "no recovery acknowledgement required"}, status_code=400)
    workspace = current_workspace()
    try:
        initialize_keystore(workspace, request.passphrase)
        unlock_keystore(workspace, request.passphrase)
    except KeystoreAlreadyInitializedError:
        return JSONResponse({"error": "workspace keystore already initialized"}, status_code=409)
    current_audit_logger().append_event("security.keystore_initialized", {"no_recovery": True})
    state = mark_software_unlocked()
    return {"initialized": True, "software_unlocked": True, "state": public_runtime_state()}


@app.post("/security/unlock")
async def security_unlock(request: SecurityUnlockRequest):
    workspace = current_workspace()
    try:
        unlock_keystore(workspace, request.passphrase)
    except KeystoreNotInitializedError:
        return JSONResponse({"error": "workspace keystore is not initialized"}, status_code=409)
    except IncorrectPassphraseError:
        current_audit_logger().append_event("security.unlock_failed", {"reason": "incorrect passphrase"})
        return JSONResponse({"error": "incorrect passphrase"}, status_code=403)
    mark_software_unlocked()
    return {"software_unlocked": True, "state": public_runtime_state()}


@app.post("/security/lock")
async def security_lock():
    _stop_indexer_worker()
    cleanup_runtime()
    mark_software_locked()
    return {"software_unlocked": False, "state": public_runtime_state()}
```

- [ ] **Step 4: Run security tests and commit**

Run:

```powershell
conda run -n souldrive python -m unittest tests.test_mcp_server_security tests.test_mcp_server_indexing tests.test_runtime_state -v
```

Expected: `OK`.

Commit:

```powershell
git add core/mcp_server.py core/runtime_state.py tests/test_mcp_server_security.py
git commit -m "feat: add workspace security unlock api"
```

### Task 4: Add UI Setup And Unlock Flow

**Files:**
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\souldrive-ui\src\App.tsx`

- [ ] **Step 1: Add frontend security types and state**

Add:

```tsx
interface SecurityStatus {
  crypto_initialized: boolean;
  software_unlocked: boolean;
  hardware_mounted: boolean;
  reason?: string;
  no_recovery: boolean;
}
```

Add state:

```tsx
  const [securityStatus, setSecurityStatus] = useState<SecurityStatus | null>(null);
  const [passphrase, setPassphrase] = useState("");
  const [confirmPassphrase, setConfirmPassphrase] = useState("");
  const [acknowledgeNoRecovery, setAcknowledgeNoRecovery] = useState(false);
  const [securityMessage, setSecurityMessage] = useState("");
  const [securityBusy, setSecurityBusy] = useState(false);
```

- [ ] **Step 2: Fetch security status in polling loop**

Add a `refreshSecurityStatus` function in the existing runtime polling effect:

```tsx
    const refreshSecurityStatus = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/security/status`, {
          headers: apiToken ? { "X-SoulDrive-Token": apiToken } : undefined,
        });
        if (!response.ok) throw new Error("security unavailable");
        const data = await response.json();
        if (isMounted) setSecurityStatus(data);
      } catch {
        if (isMounted) setSecurityStatus(null);
      }
    };
```

Call it with the other refresh functions.

- [ ] **Step 3: Add setup/unlock actions**

Add:

```tsx
  const setupWorkspaceSecurity = async () => {
    if (securityBusy) return;
    if (passphrase.length < 8) {
      setSecurityMessage("口令至少需要 8 个字符");
      return;
    }
    if (passphrase !== confirmPassphrase) {
      setSecurityMessage("两次输入的口令不一致");
      return;
    }
    if (!acknowledgeNoRecovery) {
      setSecurityMessage("请先确认忘记口令不可恢复");
      return;
    }
    setSecurityBusy(true);
    setSecurityMessage("");
    try {
      const response = await fetch(`${apiBaseUrl}/security/init`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiToken ? { "X-SoulDrive-Token": apiToken } : {}),
        },
        body: JSON.stringify({ passphrase, acknowledge_no_recovery: acknowledgeNoRecovery }),
      });
      if (!response.ok) throw new Error("security init failed");
      setPassphrase("");
      setConfirmPassphrase("");
      setAcknowledgeNoRecovery(false);
      setSecurityMessage("工作区已初始化并解锁");
    } catch {
      setSecurityMessage("初始化失败，请检查本地服务状态");
    } finally {
      setSecurityBusy(false);
    }
  };

  const unlockWorkspaceSecurity = async () => {
    if (securityBusy || !passphrase) return;
    setSecurityBusy(true);
    setSecurityMessage("");
    try {
      const response = await fetch(`${apiBaseUrl}/security/unlock`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiToken ? { "X-SoulDrive-Token": apiToken } : {}),
        },
        body: JSON.stringify({ passphrase }),
      });
      if (!response.ok) throw new Error("security unlock failed");
      setPassphrase("");
      setSecurityMessage("工作区已解锁");
    } catch {
      setSecurityMessage("口令错误或工作区不可用");
    } finally {
      setSecurityBusy(false);
    }
  };
```

- [ ] **Step 4: Render security panel in system sidebar**

Add a panel before the document library block:

```tsx
          <div className="panel-block">
            <div className="panel-title">
              <ShieldCheck size={15} />
              工作区解锁
            </div>
            {!securityStatus?.crypto_initialized ? (
              <div className="prompt-stack">
                <p>该口令用于保护本地工作区主密钥。SoulDrive 不会保存口令，也不提供找回能力。忘记口令后，该 U 盘中的加密知识库将无法解锁。</p>
                <input type="password" value={passphrase} onChange={(event) => setPassphrase(event.target.value)} placeholder="设置工作区口令" />
                <input type="password" value={confirmPassphrase} onChange={(event) => setConfirmPassphrase(event.target.value)} placeholder="再次输入口令" />
                <label>
                  <input type="checkbox" checked={acknowledgeNoRecovery} onChange={(event) => setAcknowledgeNoRecovery(event.target.checked)} />
                  我已知晓忘记口令不可恢复
                </label>
                <button type="button" disabled={securityBusy} onClick={() => void setupWorkspaceSecurity()}>初始化并解锁</button>
              </div>
            ) : securityStatus.software_unlocked ? (
              <div className="diagnostic-summary ok">
                <div><strong>UNLOCKED</strong><span>工作区已通过口令解锁</span></div>
              </div>
            ) : (
              <div className="prompt-stack">
                <input type="password" value={passphrase} onChange={(event) => setPassphrase(event.target.value)} placeholder="输入工作区口令" />
                <button type="button" disabled={securityBusy || !passphrase} onClick={() => void unlockWorkspaceSecurity()}>解锁工作区</button>
              </div>
            )}
            {securityMessage && <p>{securityMessage}</p>}
          </div>
```

- [ ] **Step 5: Build and commit UI**

Run:

```powershell
npm run build
```

Expected: `vite build` succeeds.

Commit:

```powershell
git add souldrive-ui/src/App.tsx
git commit -m "feat: add personal workspace unlock UI"
```

### Task 5: Full Regression And Final Cleanup

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run full Python tests**

Run:

```powershell
conda run -n souldrive python -m unittest discover -s tests -v
```

Expected: `OK`.

- [ ] **Step 2: Run frontend build**

Run:

```powershell
npm run build
```

from `D:\PycharmProjects\LangChainProjects\SoulDrive\souldrive-ui`.

Expected: build succeeds.

- [ ] **Step 3: Scan for unsafe secret logging**

Run:

```powershell
rg -n "passphrase|workspace_data_key|password_key|document_key|index_key|graph_key|audit_key" core tests souldrive-ui/src/App.tsx
```

Expected:

- No audit payload includes raw `passphrase`.
- No runtime state stores raw keys.
- Tests may mention key names and test passphrases.

- [ ] **Step 4: Review git status**

Run:

```powershell
git status --short --branch
```

Expected:

- Only known pre-existing unrelated entries remain if any.

## Spec Coverage Check

- No-recovery decision: covered by Task 3 API acknowledgement and Task 4 UI warning.
- Personal symmetric key model: covered by Task 1.
- Runtime gatekeeping: covered by Task 2.
- Security API: covered by Task 3.
- UI setup/unlock: covered by Task 4.
- Full regression: covered by Task 5.

## Placeholder Scan

No `TBD`, `TODO`, or deferred implementation markers remain.

## Type Consistency Check

`software_unlocked`, `crypto_initialized`, `hardware_mounted`, and `security_reason` are used consistently across runtime state, API, and UI.
