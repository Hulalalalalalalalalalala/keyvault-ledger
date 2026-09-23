"""Append-only local key vault, safe for several processes at once.

The vault persists an append-only version manifest (``manifest.json``) plus
one material file per sealed version under ``materials/``.  Revocations live
in their own append-only journal (``revocations.jsonl``): revoking a version
only appends a record, never deleting or altering historical material.
``reload()`` re-reads and validates the whole keyring, journal included, and
swaps the in-memory snapshot atomically; readers only ever see a complete
old or complete new snapshot.

Multiple processes may work the same vault directory concurrently.  They
serialise on a lock file (``.vault.lock``) living inside the vault, using
the standard library's ``fcntl`` advisory locks.  Every transaction — a
seal, a revoke, or a reload's whole read-and-validate pass — takes the same
exclusive lock and holds it until the transaction is complete; a caller
blocked on the lock waits until it gets it.  Because reload reads disk only
while it alone holds the lock, it sees either the complete pre-write records
or the complete post-write records, never a torn state.  The lock file is
only a mutex: it is never validated as vault data and never read as manifest
or material.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import threading
from pathlib import Path

MANIFEST_NAME = "manifest.json"
MATERIALS_DIR = "materials"
REVOCATIONS_NAME = "revocations.jsonl"
LOCK_NAME = ".vault.lock"
FORMAT_VERSION = 1

_INITIAL_MANIFEST = {"format": FORMAT_VERSION, "keys": {}}


def _encode_key_id(key_id: str) -> str:
    """Filesystem-safe encoding for a key id (hashed when too long)."""
    raw = base64.urlsafe_b64encode(key_id.encode("utf-8"))
    encoded = raw.decode("ascii").rstrip("=")
    if len(encoded) > 100:
        return "h_" + hashlib.sha256(key_id.encode("utf-8")).hexdigest()
    return encoded


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + rename)."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _dump_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _validate_version(version: object) -> int:
    """The single version-number check shared by every entry point.

    Only a plain ``int`` is a legal version.  Booleans are rejected even
    though ``bool`` is a subclass of ``int``, and floats or other non-integer
    values raise ``TypeError`` at the entry point, so an invalid version can
    never reach the disk records.
    """
    if type(version) is not int:
        raise TypeError("version must be an int")
    return version


class _FileLock:
    """Blocking advisory exclusive lock on a file in the vault directory.

    The lock coordinates processes; an in-process :class:`threading.Lock`
    serialises threads within one process.  Every transaction waits (as long
    as needed) for the exclusive lock and holds it until the transaction is
    finished, so readers and writers across processes never overlap.  A
    fresh file descriptor is opened per transaction and closed as soon as
    the lock is released, so no descriptor is left dangling.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._thread_lock = threading.Lock()

    @contextlib.contextmanager
    def hold(self):
        with self._thread_lock:
            fh = open(self._path, "a+b")
            try:
                # Block until the lock is held; the holder finishes the
                # whole transaction before releasing it.
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()


class Vault:
    """Local key vault rooted at a directory on disk."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._lock = threading.RLock()
        fresh = not self._root.exists()
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / MATERIALS_DIR).mkdir(exist_ok=True)
        # The lock file lives inside the vault and is only a mutex; it is
        # never treated as key data and takes no part in validation.
        self._file_lock = _FileLock(self._root / LOCK_NAME)
        with self._file_lock.hold():
            manifest_path = self._root / MANIFEST_NAME
            if not manifest_path.exists():
                if not fresh and not self._is_fresh_dir():
                    # The directory already holds vault content but the
                    # manifest is gone: that is corruption, not a new vault.
                    raise ValueError(f"manifest missing: {manifest_path}")
                _atomic_write(manifest_path, _dump_manifest(_INITIAL_MANIFEST))
            journal_path = self._root / REVOCATIONS_NAME
            if not journal_path.exists():
                # A vault created before revocations existed simply has an
                # empty journal; the journal is append-only and never rewritten.
                _atomic_write(journal_path, b"")
            self._snapshot, self._manifest = self._load_validated()

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def seal(self, key_id: str, material: bytes) -> int:
        """Store ``material`` as a new version of ``key_id``; return the version."""
        if not isinstance(key_id, str):
            raise TypeError("key_id must be a string")
        if key_id == "":
            raise ValueError("key_id must not be empty")
        if not isinstance(material, (bytes, bytearray, memoryview)):
            raise TypeError("material must be a bytes-like object")
        material = bytes(material)

        # Hold the cross-process lock for the whole transaction.  Another
        # process may have sealed versions since this handle last read the
        # disk, so the next version number is computed from a freshly
        # validated disk state, never from this process's possibly stale
        # snapshot: versions can then neither repeat nor skip.
        with self._file_lock.hold(), self._lock:
            fresh_snapshot, fresh_manifest = self._load_validated()
            disk_entry = fresh_manifest["keys"].get(key_id)
            disk_records = disk_entry["versions"] if disk_entry else []
            new_version = (
                disk_records[-1]["version"] + 1 if disk_records else 1
            )

            rel_file = (
                f"{MATERIALS_DIR}/{_encode_key_id(key_id)}/{new_version}.bin"
            )
            digest = hashlib.sha256(material).hexdigest()

            # Write the material first, then flip the manifest.  The manifest
            # is the source of truth and is replaced atomically, so a crash
            # in between can never expose a half-recorded version; if the
            # manifest write itself fails, remove the material just written
            # so a failed seal leaves no orphan behind.
            material_path = self._root / rel_file
            material_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(material_path, material)

            new_manifest = copy.deepcopy(fresh_manifest)
            try:
                entry = new_manifest["keys"].setdefault(
                    key_id, {"active": 0, "versions": []}
                )
                entry["versions"].append(
                    {"version": new_version, "sha256": digest, "file": rel_file}
                )
                entry["active"] = new_version
                _atomic_write(self._root / MANIFEST_NAME, _dump_manifest(new_manifest))
            except BaseException:
                # ``new_manifest`` is a private deep copy and is simply
                # discarded; only the material written above needs cleanup.
                try:
                    material_path.unlink()
                except OSError:
                    pass
                raise

            # The validated snapshot mirrors the disk exactly; reflect the
            # version just appended and publish it with a single assignment.
            slot = fresh_snapshot.setdefault(
                key_id, {"active": 0, "versions": {}, "revoked": set()}
            )
            slot["versions"][new_version] = material
            slot["active"] = new_version
            self._snapshot = fresh_snapshot
            self._manifest = new_manifest
            return new_version

    def load(self, key_id: str, version: int | None = None) -> bytes:
        """Return the stored material for ``key_id`` (active version by default)."""
        snapshot = self._snapshot
        entry = snapshot.get(key_id)
        if entry is None:
            raise KeyError(key_id)
        wanted = entry["active"] if version is None else version
        try:
            return entry["versions"][wanted]
        except KeyError:
            raise KeyError(f"{key_id!r} version {wanted!r}") from None

    def versions(self, key_id: str) -> list[int]:
        """Return the recorded versions of ``key_id``, ascending."""
        entry = self._snapshot.get(key_id)
        if entry is None:
            return []
        return sorted(entry["versions"])

    def active(self, key_id: str) -> int:
        """Return the active (most recently sealed) version of ``key_id``."""
        entry = self._snapshot.get(key_id)
        if entry is None:
            raise KeyError(key_id)
        return entry["active"]

    def revoke(self, key_id: str, version: int) -> None:
        """Mark ``version`` of ``key_id`` as revoked.

        Revocation is an explicit marker only: the version's material stays
        readable, it stays in the version list, and the active version is
        unchanged.  The marker is appended to an append-only journal, so it
        survives a full ``reload()``.
        """
        if not isinstance(key_id, str):
            raise TypeError("key_id must be a string")
        if key_id == "":
            raise ValueError("key_id must not be empty")
        _validate_version(version)

        # Take the exclusive lock for the whole append, and decide against a
        # freshly validated disk state: a concurrent process may have sealed
        # the version in the meantime, or already revoked it.
        with self._file_lock.hold(), self._lock:
            fresh_snapshot, fresh_manifest = self._load_validated()
            entry = fresh_snapshot.get(key_id)
            if entry is None:
                raise KeyError(key_id)
            if version not in entry["versions"]:
                raise KeyError(f"{key_id!r} version {version!r}")
            if version in entry["revoked"]:
                raise ValueError(
                    f"{key_id!r} version {version!r} is already revoked"
                )

            record = {"key_id": key_id, "version": version}
            line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
            journal_path = self._root / REVOCATIONS_NAME
            if not journal_path.exists():
                # The journal is created when the vault opens; its absence
                # means it was removed out of band.  Recreating it here
                # would silently drop earlier revocation records.
                raise ValueError(f"revocation journal missing: {journal_path}")
            # Append-only: a single write in append mode only adds a record,
            # never rewriting history.
            with open(journal_path, "ab") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())

            # The validated snapshot already mirrors the journal; add the
            # marker just appended and publish with a single assignment.
            entry["revoked"].add(version)
            self._snapshot = fresh_snapshot
            self._manifest = fresh_manifest

    def is_revoked(self, key_id: str, version: int) -> bool:
        """Return whether ``version`` of ``key_id`` has been revoked."""
        if not isinstance(key_id, str):
            raise TypeError("key_id must be a string")
        if key_id == "":
            raise ValueError("key_id must not be empty")
        # Same gate as revoke: a non-integer version is a TypeError at the
        # entry point, before any key lookup.
        _validate_version(version)
        entry = self._snapshot.get(key_id)
        if entry is None:
            raise KeyError(key_id)
        if version not in entry["versions"]:
            raise KeyError(f"{key_id!r} version {version!r}")
        return version in entry["revoked"]

    def revoked_versions(self, key_id: str) -> list[int]:
        """Return the revoked versions of ``key_id``, ascending.

        A key that has never been revoked — including an unknown key —
        yields an empty list.
        """
        if not isinstance(key_id, str):
            raise TypeError("key_id must be a string")
        if key_id == "":
            raise ValueError("key_id must not be empty")
        entry = self._snapshot.get(key_id)
        if entry is None:
            return []
        return sorted(entry["revoked"])

    def reload(self) -> None:
        """Re-read and validate the whole keyring, then swap the snapshot.

        Never writes to disk.  The whole read-and-validate pass happens
        under the same exclusive cross-process lock (and in-process lock)
        that every seal and revoke takes, so it cannot interleave with a
        writer in any process: it observes either the complete pre-write
        records or the complete post-write records.  On any validation
        failure the current in-memory snapshot is kept untouched.
        """
        with self._file_lock.hold(), self._lock:
            snapshot, manifest = self._load_validated()
            self._snapshot = snapshot
            self._manifest = manifest

    def manifest(self) -> dict:
        """Return a deep copy of the persisted manifest."""
        return copy.deepcopy(self._manifest)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _is_fresh_dir(self) -> bool:
        for child in self._root.iterdir():
            if child.name == LOCK_NAME:
                # The mutex file is coordination state, not key data.
                continue
            if (
                child.name == MATERIALS_DIR
                and child.is_dir()
                and not any(child.iterdir())
            ):
                continue
            return False
        return True

    def _load_validated(self) -> tuple[dict, dict]:
        """Read and validate the whole keyring from disk.

        Returns ``(snapshot, manifest)``.  The snapshot carries each key's
        materials plus its revoked-version set.  Raises ``ValueError`` if
        the manifest or the revocation journal is corrupt or malformed, a
        stored material is missing or mismatches its record, or the
        journal revokes an unknown/duplicate version.
        """
        manifest_path = self._root / MANIFEST_NAME
        try:
            raw = manifest_path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(f"manifest missing: {manifest_path}") from exc
        except OSError as exc:
            raise ValueError(f"manifest unreadable: {exc}") from exc

        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"manifest corrupt: {exc}") from exc

        if not isinstance(manifest, dict):
            raise ValueError("manifest format invalid: not a JSON object")
        if manifest.get("format") != FORMAT_VERSION:
            raise ValueError("manifest format invalid: unsupported format")
        keys = manifest.get("keys")
        if not isinstance(keys, dict):
            raise ValueError("manifest format invalid: 'keys' must be an object")

        root_resolved = self._root.resolve()
        snapshot: dict[str, dict] = {}
        for key_id, entry in keys.items():
            if not isinstance(key_id, str) or not isinstance(entry, dict):
                raise ValueError("manifest format invalid: bad key entry")
            records = entry.get("versions")
            active = entry.get("active")
            if not isinstance(records, list) or not records:
                raise ValueError(
                    f"manifest format invalid: key {key_id!r} has no versions"
                )
            if type(active) is not int:
                raise ValueError(
                    f"manifest format invalid: key {key_id!r} has bad active"
                )

            versions: dict[int, bytes] = {}
            previous = 0
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(
                        f"manifest format invalid: key {key_id!r} bad record"
                    )
                version = record.get("version")
                digest = record.get("sha256")
                rel_file = record.get("file")
                if type(version) is not int or version <= previous:
                    raise ValueError(
                        f"manifest format invalid: key {key_id!r} versions "
                        "must be strictly increasing"
                    )
                previous = version
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError(
                        f"manifest format invalid: key {key_id!r} version "
                        f"{version} has bad sha256"
                    )
                if not isinstance(rel_file, str):
                    raise ValueError(
                        f"manifest format invalid: key {key_id!r} version "
                        f"{version} has bad file"
                    )
                resolved = (self._root / rel_file).resolve()
                if not resolved.is_relative_to(root_resolved):
                    raise ValueError(
                        f"manifest format invalid: key {key_id!r} version "
                        f"{version} file escapes the vault"
                    )
                try:
                    material = resolved.read_bytes()
                except OSError as exc:
                    raise ValueError(
                        f"material missing for key {key_id!r} version {version}"
                    ) from exc
                if hashlib.sha256(material).hexdigest() != digest:
                    raise ValueError(
                        f"material mismatch for key {key_id!r} version {version}"
                    )
                versions[version] = material

            if active != previous:
                raise ValueError(
                    f"manifest format invalid: key {key_id!r} active version "
                    "does not match the latest sealed version"
                )
            snapshot[key_id] = {
                "active": active,
                "versions": versions,
                "revoked": set(),
            }

        self._load_revocations(snapshot)
        return snapshot, manifest

    def _load_revocations(self, snapshot: dict[str, dict]) -> None:
        """Validate the append-only revocation journal against ``snapshot``.

        Populates each snapshot entry's ``revoked`` set in place.  Raises
        ``ValueError`` if the journal is missing, holds a malformed record,
        revokes an unknown key/version, or revokes the same version twice.
        """
        journal_path = self._root / REVOCATIONS_NAME
        try:
            raw = journal_path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(f"revocation journal missing: {journal_path}") from exc
        except OSError as exc:
            raise ValueError(f"revocation journal unreadable: {exc}") from exc

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"revocation journal corrupt: {exc}") from exc

        seen: set[tuple[str, int]] = set()
        for line in text.splitlines():
            if not line:
                raise ValueError("revocation journal corrupt: empty record")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"revocation journal corrupt: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError("revocation journal corrupt: record not an object")
            key_id = record.get("key_id")
            version = record.get("version")
            if not isinstance(key_id, str) or key_id == "":
                raise ValueError(
                    "revocation journal corrupt: record has bad key_id"
                )
            if type(version) is not int:
                raise ValueError(
                    f"revocation journal corrupt: key {key_id!r} record has "
                    "bad version"
                )
            entry = snapshot.get(key_id)
            if entry is None or version not in entry["versions"]:
                raise ValueError(
                    f"revocation journal corrupt: key {key_id!r} version "
                    f"{version} was never sealed"
                )
            marker = (key_id, version)
            if marker in seen:
                raise ValueError(
                    f"revocation journal corrupt: key {key_id!r} version "
                    f"{version} revoked more than once"
                )
            seen.add(marker)
            entry["revoked"].add(version)
