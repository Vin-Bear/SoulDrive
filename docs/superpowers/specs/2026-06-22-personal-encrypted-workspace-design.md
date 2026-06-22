# Personal Encrypted Workspace Design

## 1. Goal

Add a first-version personal encrypted workspace model to SoulDrive.

This version keeps the current removable-storage hardware check, then adds a software unlock step based on a user passphrase. Each workspace owns a random data key. The passphrase never encrypts documents directly; it derives a wrapping key that unlocks the workspace data key.

If the user forgets the passphrase, the workspace cannot be recovered. SoulDrive does not store recovery material in this version.

## 2. Scope

This version implements:

- A workspace keystore file at `SoulDrive/config/keystore.json`.
- Passphrase-based workspace initialization.
- Passphrase-based workspace unlock.
- Runtime state fields for crypto initialization and software unlock status.
- Gatekeeping for chat, document import, and indexing until software unlock succeeds.
- Audit events for initialization, unlock success, unlock failure, and software lock.
- UI copy that warns the user that passphrase recovery is not available.

This version does not implement:

- Enterprise authorization.
- Public/private key device certificates.
- Cloud account recovery.
- Admin recovery.
- Full transparent encryption of PDFs, Chroma, SQLite, or graph databases.
- Passphrase change.
- Recovery phrase.

The full data encryption layer can be added in a later phase after the key lifecycle and unlock gate are stable.

## 3. Threat Model

The first version primarily protects against casual or opportunistic access when a user's removable workspace is lost or borrowed.

It does not claim to protect against a fully compromised running machine, malware with process memory access, or forensic recovery from an already-unlocked session.

## 4. Key Model

### User Passphrase

The user provides a passphrase during initialization and unlock. The passphrase is never stored.

### Password Key

SoulDrive derives `password_key` from:

- user passphrase
- random salt from `keystore.json`
- PBKDF2-HMAC-SHA256 parameters

`password_key` exists only in memory while handling init or unlock. It wraps and unwraps the workspace data key.

Argon2id remains the preferred future KDF, but the current project already depends on `cryptography` and does not depend on `argon2-cffi`. This first version uses PBKDF2-HMAC-SHA256 from `cryptography` to avoid adding a new dependency during the first pass.

### Workspace Data Key

`workspace_data_key` is a random 32-byte value generated when the workspace keystore is initialized. It is the root secret for this workspace and is never stored in plaintext.

On disk, the workspace stores only `encrypted_workspace_data_key`, encrypted with `password_key` using AES-256-GCM.

### Purpose Keys

After unlock, SoulDrive derives purpose keys using HKDF-SHA256:

- `document_key`
- `index_key`
- `graph_key`
- `audit_key`

This version derives them to prove the lifecycle and keep interfaces ready, but it does not yet encrypt the underlying document, index, or graph files.

## 5. Keystore Format

`SoulDrive/config/keystore.json`:

```json
{
  "version": 1,
  "kdf": "pbkdf2-sha256",
  "kdf_params": {
    "iterations": 600000,
    "salt": "base64"
  },
  "key_wrap": "aes-256-gcm",
  "nonce": "base64",
  "encrypted_workspace_data_key": "base64",
  "created_at": 0
}
```

The file contains no passphrase and no plaintext data key.

## 6. Runtime State

Runtime state adds:

- `crypto_initialized`: whether the current workspace has a keystore.
- `software_unlocked`: whether passphrase unlock has succeeded.
- `security_reason`: current security-specific reason.

When a workspace is mounted but not software-unlocked, runtime remains effectively locked for sensitive workflows. Public status can show that the device exists, but chat/import/index remain blocked.

## 7. API

Add:

- `GET /security/status`
- `POST /security/init`
- `POST /security/unlock`
- `POST /security/lock`

`/security/init` requires `acknowledge_no_recovery: true`. If it is false, the server rejects the request.

`/security/unlock` validates the passphrase and marks the runtime as software-unlocked when successful.

`/security/lock` clears software-unlock state and unloads in-process runtime objects.

## 8. Gatekeeping

The following actions require software unlock:

- `/chat`
- `/documents/import`
- `/papers/import`
- `/index/run`

If blocked, the server returns `423` with status `locked` and reason `workspace passphrase required`.

## 9. Audit Events

Add audit events:

- `security.keystore_initialized`
- `security.unlock_succeeded`
- `security.unlock_failed`
- `security.software_locked`
- `security.password_required`

No audit payload records the passphrase or derived keys.

## 10. UI Behavior

The UI distinguishes:

- storage not available
- workspace available but keystore not initialized
- workspace available and initialized but passphrase required
- workspace unlocked

During initialization, the UI must show this warning:

```text
该口令用于保护本地工作区主密钥。SoulDrive 不会保存口令，也不提供找回能力。忘记口令后，该 U 盘中的加密知识库将无法解锁。
```

The user must acknowledge this warning before calling `/security/init`.

## 11. Testing Strategy

Add tests for:

- keystore creation does not store plaintext passphrase or plaintext workspace data key
- wrong passphrase fails unlock
- correct passphrase unlocks
- runtime state records crypto and software-unlock status
- chat/import/index reject when hardware is mounted but software unlock is missing
- security API emits expected status codes
- audit events are written for init, success, failure, and software lock

## 12. Implementation Order

1. Add the crypto helper module and unit tests.
2. Extend runtime state and gatekeeping helpers.
3. Add security API routes and tests.
4. Add UI initialization/unlock panel.
5. Run full regression.
