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
