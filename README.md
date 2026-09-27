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
- `seal(key_id, material) -> int` stores material (a bytes-like object:
  `bytes`, `bytearray` or `memoryview`; any other type raises `TypeError`)
  and returns the new version.
- `derive_seal(key_id, password, salt, iterations, length) -> int` derives the
  material with PBKDF2-HMAC-SHA256 (standard library), seals it through the
  ordinary append-only path, and returns the new version. The passphrase is
  never persisted.
- `derivation(key_id, version=None) -> dict` returns
  `{"salt", "iterations", "length"}` for a derived version (the active one by
  default) or an empty record `{}` for a version sealed directly.
- `load(key_id, version=None) -> bytes` returns stored material.
- `versions(key_id) -> list[int]` ascending.
- `active(key_id) -> int` returns the current version.
- `set_active(key_id, version) -> None` repoints the active version at a
  historical version (library-only; see below).
- `revoke(key_id, version) -> None` marks a sealed version revoked.
- `is_revoked(key_id, version) -> bool` reports a version's revocation status.
- `revoked_versions(key_id) -> list[int]` lists revoked versions, ascending
  (empty for an unknown or never-revoked key).
- `reload() -> None` re-reads and validates the whole keyring before replacing it.
- `manifest() -> dict` returns the persisted manifest.
- `close() -> None` releases the lock file handle; the next operation reopens
  it and reacquires the same lock. Repeated calls are harmless.

### Passphrase-derived sealing

`derive_seal` derives the key material from a passphrase with
PBKDF2-HMAC-SHA256 from the standard library and then runs it through exactly
the same append-only seal path as `seal`: derived and direct seals share one
file lock and one non-repeating version sequence, and historical material is
never overwritten. The salt, iteration count and derived length are stored
alongside that version (the salt as base64 text); the passphrase itself never
touches disk. The bytes read back are byte-for-byte identical to re-running
PBKDF2 with the same passphrase, salt and parameters. `derivation` queries
those parameters, defaulting to the active version; a directly sealed version
yields an empty record.

`derive_seal` raises `ValueError` for an empty key id, an empty salt, or a
non-positive iteration count/length, and `TypeError` when the passphrase or
salt is not `bytes` or when an iteration count/length is not a genuine
integer (bools and floats do not count). `derivation` follows the read
semantics of the revocation queries: empty key id → `ValueError`, non-integer
version → `TypeError`, unknown key or version → `KeyError`. On `reload`, every
derivation record is re-checked: a length that disagrees with the stored
material, a missing/corrupt salt, or an illegal parameter makes the reload
fail with `ValueError` while the existing snapshot and disk records stay
untouched.

### Active version

Normally the active version is the most recently sealed one, and `load` and
`derivation` without a version number resolve to it. `active` follows the
read semantics of the other queries: an empty key id raises `ValueError`
and an unknown key raises `KeyError`. `set_active(key_id,
version)` repoints the active version at any existing historical version:
afterwards `active` reports that version and unversioned `load` /
`derivation` resolve to it, while versioned reads are unchanged. Each call
appends one record to the append-only journal `activations.jsonl` (created
on the first repoint; a vault that was never repointed simply has no such
file); the manifest, the materials and their version numbers are never
rewritten. Repeated repoints of one key append one record each and the last
one wins. The repoint survives a full `reload()` and reopening the vault.

A repoint is bound to the key's newest sealed version at the time of the
call: sealing (or deriving) another version afterwards makes that new
version active again, exactly as if the key had never been repointed, and
the historical records in the journal stay untouched. Repointing at a
revoked version raises `ValueError`; a version that is revoked *after* it
was pointed at is not an error — revocation never moves the active pointer.
`set_active` raises `ValueError` for an empty key id, for the version that
is already active, and for a revoked target, `TypeError` for a non-integer
version (floats, bools, …), and `KeyError` for a key that was never sealed
or a version that does not exist. A failed call appends no record. On
`reload`, the journal is validated together with the manifest, every
material and the revocation journal: a corrupt record or one pointing at a
key/version that never existed makes the reload fail with `ValueError` while
the existing snapshot and disk records stay untouched.

### The activation journal

Every `set_active` call appends exactly one line to `activations.jsonl` in
the vault root: one JSON object per line, in write order, so the file order
is the order the repoints happened. A vault that was never repointed has no
such file; the first repoint creates it. Each line carries exactly three
fields and lands on disk byte-for-byte like this:

    {"key_id": "plain", "latest": 2, "version": 1}

- `key_id` (string): the key whose active version was repointed.
- `version` (integer): the historical version the key was repointed at.
- `latest` (integer): the binding value — the key's newest sealed version
  at the moment the record was written.

The binding value scopes the record: a repoint is honoured only while its
`latest` still equals the key's newest sealed version. Sealing (or
deriving) a higher version afterwards makes that new version active again,
exactly as if the key had never been repointed; the older records stay in
the journal untouched but no longer apply. Among the records still bound,
the last one written for a key wins; earlier ones are kept as history and
are never rewritten or removed.

On `reload`, every line is validated against this shape before the snapshot
is swapped: the line must be one JSON object with exactly these three
fields, `version` and `latest` must be genuine integers, the key and both
versions must exist, and `latest` must itself be a sealed version of the
key with `1 <= version <= latest`. A missing or extra field, a wrongly
typed value, a broken JSON line, a record pointing at a key or version
that never existed, or a binding value above the key's newest sealed
version makes the whole reload fail with `ValueError`; the in-memory
snapshot and every record on disk stay untouched. Removing or correcting
the offending record and reloading again restores the vault, with the
snapshot matching the disk records one to one.

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
`reload`); revocation and its queries, like repointing the active version,
are library-only.

### Concurrency

Multiple processes (and the threads inside them) may share one vault
directory. Opening, sealing, deriving, repointing, revoking and reloading
all run under an exclusive, blocking file lock held on `vault.lock` inside
the vault directory (standard-library `fcntl.flock`, or `msvcrt.locking`
on Windows). A writer waits until it holds the lock and completes its
whole record — material file, manifest replacement, or journal append —
before releasing it; every other process waits until it holds the lock
itself before it begins its own record. So interleaved seals, derives,
repoints and revokes from different processes and threads allocate each
key's version numbers strictly ascending, never duplicating or skipping
one, and never overwrite or truncate historical material.

A reload interleaved with a writer crosses only one visibility boundary:
it holds the same lock for the whole read-and-validate pass — manifest,
every material and both journals — and the validated snapshot is then
swapped in as one object, so a reader resolves every query of an
observation against either the complete old records or the complete newly
persisted ones. A half-old, half-new intermediate state is never
observable, and a reload can never resurrect an older snapshot or reuse a
version number it carried.

When that whole-vault validation fails, `reload()` raises `ValueError`
and the swap never happens: the in-memory snapshot stays exactly as it
was and keys already in hand remain readable, while the records on disk
neither gain nor lose a byte. Repeating any query while the failure
persists returns the same answer again, and a fresh opener on the same
directory raises the same `ValueError`. Once the offending bytes are
removed or corrected, another full `reload()` succeeds and the snapshot
matches the disk records one to one.

The lock file is only a mutual-exclusion device: it carries no key data
and plays no part in validation — its bytes may be empty or arbitrary
without affecting any read, write or reload. `close()` releases its
handle, and another process can take the same lock immediately; repeated
`close()` calls are harmless no-ops, and the next operation reopens the
file and reacquires the lock transparently, with no observable change in
behaviour.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

Material sealed with `seal` must be supplied as a bytes-like object
(`bytes`, `bytearray` or `memoryview`); supplying any other type raises
`TypeError`. The only other way to produce material is `derive_seal`, which
derives it from a passphrase with PBKDF2-HMAC-SHA256; there is no other
derivation. The two entries differ in what they accept: `seal` takes any
bytes-like object, whereas the passphrase and salt of `derive_seal` must be
genuine `bytes` (`bytearray` and `memoryview` are rejected with
`TypeError`). No network service.
