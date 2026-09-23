# keyvault-ledger

Local key vault with an append-only version manifest, for services that must reload key material from disk without ever exposing a partially loaded keyring.

## Requirements

Python 3.11 or newer. Standard library only.

## Install

    python3 -m pip install -e .

## Run

    python3 -m keyvault_ledger --root ./vault versions
    python3 -m keyvault_ledger --root ./vault seal <key-id> --material-file <path>
    python3 -m keyvault_ledger --root ./vault reload

## Public interface

`keyvault_ledger.Vault(root)` opens the vault directory `root`.
- `seal(key_id, material) -> int` stores material and returns the new version.
- `derive_seal(key_id, password, salt, iterations, length) -> int` derives
  material with PBKDF2-HMAC-SHA256 (standard library) and stores it through
  the same append path as `seal`, returning the new version. The salt,
  iteration count and derived length are persisted with the version; the
  password is never written to disk.
- `derivation(key_id, version=None) -> dict` returns
  `{"salt": bytes, "iterations": int, "length": int}` for a derived version
  (active version by default), or `{}` for a directly sealed version.
- `load(key_id, version=None) -> bytes` returns stored material.
- `versions(key_id) -> list[int]` ascending.
- `active(key_id) -> int` returns the current version.
- `revoke(key_id, version) -> None` marks a sealed version revoked.
- `is_revoked(key_id, version) -> bool` reports a version's revocation status.
- `revoked_versions(key_id) -> list[int]` lists revoked versions, ascending
  (empty for an unknown or never-revoked key).
- `reload() -> None` re-reads and validates the whole keyring before replacing it.
- `manifest() -> dict` returns the persisted manifest.
- `close() -> None` releases the lock file handle. Calling it more than once
  is harmless; a later operation reopens the lock file and reacquires the
  lock, so observable behaviour is unchanged.

### Password derivation

`derive_seal` takes a password and salt that must be byte strings and an
iteration count and derived length that must be positive integers (bools
and floats count as non-integers). It raises `TypeError` for a non-bytes
password or salt or non-integer parameters, and `ValueError` for an empty
key id or salt or a parameter below 1. Material derived from the same
password, salt and parameters compares byte for byte with what `load`
returns. `derivation` follows the read conventions of the revocation
queries: empty key id → `ValueError`, non-integer version → `TypeError`,
unknown key or version → `KeyError`; a directly sealed version simply
yields `{}`. A full `reload()` rechecks every derivation record against
its material — a missing salt, illegal parameters or a derived length that
does not match the material all fail validation with `ValueError` while
leaving the disk records and the in-memory snapshot untouched.

### Revocation

Revocation is an explicit marker only. Revoking a version appends one
record to the append-only journal `revocations.jsonl`; it never deletes or
alters historical material. A revoked version stays readable via `load`,
stays in the `versions` list, and does not move the active version. The
marker survives a full `reload()`, which validates the manifest, every
material, and the journal together before swapping its in-memory snapshot.

Each version of a key can be revoked at most once. `revoke` raises
`ValueError` for an empty key id or a repeated revocation, `TypeError` for
a non-integer version (floats, bools, …), and `KeyError` for a key that
was never sealed or a version that does not exist. `is_revoked` follows
the same rule (empty key id → `ValueError`, non-integer version →
`TypeError`, unknown key or version → `KeyError`); `revoked_versions`
returns `[]` for an unknown key and raises `ValueError` only for an empty
key id.

The command line keeps its three entry points (`versions`, `seal`,
`reload`); revocation and its queries are library-only.

### Concurrency

Multiple processes may share one vault directory. Opening, sealing,
deriving, revoking and reloading all run under an exclusive, blocking file
lock held on `vault.lock` inside the vault directory (standard-library
`fcntl.flock`, or `msvcrt.locking` on Windows). A writer waits until it
holds the lock and completes its whole record before releasing it, so
interleaved seals, derives and revokes from different processes never
duplicate or skip a version number and never overwrite historical
material. A reload concurrent with a seal reads either the complete old
records or the complete newly persisted ones. The lock file is only a
mutual-exclusion device: it carries no key data and plays no part in
validation.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

No network service. Derivation is PBKDF2-HMAC-SHA256 only; callers may
still supply raw material directly via `seal`. The command line keeps its
three entry points (`versions`, `seal`, `reload`) — derivation and all
queries are library-only.
