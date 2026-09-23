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
- `load(key_id, version=None) -> bytes` returns stored material.
- `versions(key_id) -> list[int]` ascending.
- `active(key_id) -> int` returns the current version.
- `revoke(key_id, version) -> None` marks a sealed version revoked.
- `is_revoked(key_id, version) -> bool` reports a version's revocation status.
- `revoked_versions(key_id) -> list[int]` lists revoked versions, ascending
  (empty for an unknown or never-revoked key).
- `reload() -> None` re-reads and validates the whole keyring before replacing it.
- `manifest() -> dict` returns the persisted manifest.

### Revocation

Revocation is an explicit marker only. Revoking a version appends one
record to the append-only journal `revocations.jsonl`; it never deletes or
alters historical material. A revoked version stays readable via `load`,
stays in the `versions` list, and does not move the active version. The
marker survives a full `reload()`, which validates the manifest, every
material, and the journal together before swapping its in-memory snapshot.

Each version of a key can be revoked at most once. `revoke` raises
`ValueError` for an empty key id or a repeated revocation, and `KeyError`
for a key that was never sealed or a version that does not exist.
`is_revoked` follows the same rule (empty key id → `ValueError`, unknown
key or version → `KeyError`); `revoked_versions` returns `[]` for an
unknown key and raises `ValueError` only for an empty key id.

The command line keeps its three entry points (`versions`, `seal`,
`reload`); revocation and its queries are library-only.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

Single-process use; no cross-process locking.
No key derivation: callers supply the material.
No network service.
