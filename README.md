# keyvault-ledger

Local key vault with an append-only version manifest and an append-only revocation log, for services that must reload key material from disk without ever exposing a partially loaded keyring.

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
- `reload() -> None` re-reads and validates the whole keyring before replacing it.
- `manifest() -> dict` returns the persisted manifest.
- `revoke(key_id, version) -> None` marks a sealed version revoked.
- `is_revoked(key_id, version) -> bool` reports a version's revocation status.
- `revoked_versions(key_id) -> list[int]` returns the revoked versions, ascending.

Revocation is a marker only: it appends one JSON line per revocation to
`revocations.log` and never deletes or alters sealed material, the version
list, or the active version. Revoked material remains readable via `load`,
and revoked markers survive `reload()` (which validates the whole log along
with the manifest and every material, and swaps the in-memory snapshot only
when validation passes).

`revoke` raises `ValueError` for an empty key id or a repeated revocation
and `KeyError` for an unknown key or version. `is_revoked` uses the same
error policy; `revoked_versions` returns an empty list for an unknown key
and raises `ValueError` only for an empty key id.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

Single-process use; no cross-process locking.
No key derivation: callers supply the material.
No network service.
