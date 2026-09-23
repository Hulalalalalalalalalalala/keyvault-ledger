"""Append-only local key vault.

The vault persists an append-only version manifest (``manifest.json``) plus
one material file per sealed version under ``materials/``.  Revocations live
in their own append-only journal (``revocations.jsonl``): revoking a version
only appends a record, never deleting or altering historical material.
``reload()`` re-reads and validates the whole keyring, journal included, and
swaps the in-memory snapshot atomically; readers only ever see a complete
old or complete new snapshot.

Multiple processes may share one vault directory.  Every operation that
reads from or writes to the disk state (open, seal, revoke, reload) runs
under an exclusive, blocking file lock held on ``vault.lock`` inside the
vault directory (``fcntl.flock`` where available, ``msvcrt.locking`` on
Windows — standard library only).  A writer waits until it holds the lock
and completes its whole record before releasing it, so interleaved seals
and revokes from different processes never duplicate or skip a version
number and never overwrite or truncate historical material.  The lock
file is only a mutual-exclusion device: it carries no key data and plays
no part in manifest or material validation.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import json
import os
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: fall back to msvcrt in _file_lock.
    fcntl = None

MANIFEST_NAME = "manifest.json"
MATERIALS_DIR = "materials"
REVOCATIONS_NAME = "revocations.jsonl"
LOCK_NAME = "vault.lock"
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


def _check_key_id(key_id: str) -> None:
    """Entry-point validation shared by every method taking a key id."""
    if not isinstance(key_id, str):
        raise TypeError("key_id must be a string")
    if key_id == "":
        raise ValueError("key_id must not be empty")


def _check_version(version: int) -> None:
    """Entry-point validation for version numbers.

    Only genuine ``int`` values pass: bools, floats and other non-integer
    values are rejected with ``TypeError`` before anything touches the
    in-memory snapshot or the append-only records on disk.
    """
    if type(version) is not int:
        raise TypeError("version must be an int")


class Vault:
    """Local key vault rooted at a directory on disk."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._lock = threading.RLock()
        fresh = not self._root.exists()
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / MATERIALS_DIR).mkdir(exist_ok=True)
        # The lock file coordinates every process using this directory; it
        # is created up front so initialisation itself is serialised.
        self._lock_fh = open(self._root / LOCK_NAME, "a+b")
        try:
            with self._file_lock():
                manifest_path = self._root / MANIFEST_NAME
                if not manifest_path.exists():
                    if not fresh and not self._is_fresh_dir():
                        # The directory already holds vault content but the
                        # manifest is gone: that is corruption, not a new
                        # vault.
                        raise ValueError(f"manifest missing: {manifest_path}")
                    _atomic_write(manifest_path, _dump_manifest(_INITIAL_MANIFEST))
                journal_path = self._root / REVOCATIONS_NAME
                if not journal_path.exists():
                    # A vault created before revocations existed simply has
                    # an empty journal; the journal is append-only and never
                    # rewritten.
                    _atomic_write(journal_path, b"")
                self._snapshot, self._manifest = self._load_validated()
        except BaseException:
            self._lock_fh.close()
            raise

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def seal(self, key_id: str, material: bytes) -> int:
        """Store ``material`` as a new version of ``key_id``; return the version."""
        _check_key_id(key_id)
        if not isinstance(material, (bytes, bytearray, memoryview)):
            raise TypeError("material must be a bytes-like object")
        material = bytes(material)

        with self._locked():
            # Re-read and validate the persisted keyring under the
            # inter-process lock: another process may have sealed since
            # this handle last looked, and the next version number must be
            # allocated from the latest state on disk so concurrent seals
            # never duplicate or skip a version and never overwrite
            # historical material.
            snapshot, disk_manifest = self._load_validated()
            new_manifest = copy.deepcopy(disk_manifest)
            entry = new_manifest["keys"].setdefault(
                key_id, {"active": 0, "versions": []}
            )
            records = entry["versions"]
            new_version = records[-1]["version"] + 1 if records else 1

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

            try:
                records.append(
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

            # ``snapshot`` was just built by ``_load_validated`` and is not
            # shared yet, so it can be extended in place before publishing.
            slot = snapshot.setdefault(
                key_id, {"active": 0, "versions": {}, "revoked": set()}
            )
            slot["versions"][new_version] = material
            slot["active"] = new_version
            self._snapshot = snapshot
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
        _check_key_id(key_id)
        _check_version(version)

        with self._locked():
            # Re-read and validate the persisted keyring under the
            # inter-process lock so the duplicate/unknown checks and the
            # appended record reflect the latest journal on disk, including
            # revocations appended by other processes.
            snapshot, disk_manifest = self._load_validated()
            entry = snapshot.get(key_id)
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

            entry["revoked"].add(version)
            self._snapshot = snapshot
            self._manifest = disk_manifest

    def is_revoked(self, key_id: str, version: int) -> bool:
        """Return whether ``version`` of ``key_id`` has been revoked."""
        _check_key_id(key_id)
        _check_version(version)
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
        _check_key_id(key_id)
        entry = self._snapshot.get(key_id)
        if entry is None:
            return []
        return sorted(entry["revoked"])

    def reload(self) -> None:
        """Re-read and validate the whole keyring, then swap the snapshot.

        Never writes to disk.  The whole read-and-validate pass happens
        under both locks so it cannot interleave with a seal or revoke —
        in this process or any other: the reload reads either the complete
        old records or the complete newly persisted ones, and swapping in a
        snapshot read before a concurrent seal completed would resurrect an
        old snapshot and could later reuse its version numbers.  On any
        validation failure the current in-memory snapshot is kept
        untouched.
        """
        with self._locked():
            snapshot, manifest = self._load_validated()
            self._snapshot = snapshot
            self._manifest = manifest

    def manifest(self) -> dict:
        """Return a deep copy of the persisted manifest."""
        return copy.deepcopy(self._manifest)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _file_lock(self):
        """Exclusive, blocking inter-process lock scoped to the vault dir.

        A caller that cannot take the lock waits until it can; the lock is
        held for the whole disk operation and released only once the record
        is fully done.  Uses ``fcntl.flock`` where available and falls back
        to ``msvcrt.locking`` on Windows — standard library only, no
        network coordination.
        """
        if fcntl is not None:
            fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
        else:
            import msvcrt

            self._lock_fh.seek(0)
            while True:
                try:
                    msvcrt.locking(self._lock_fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:  # timed out waiting; keep waiting
                    time.sleep(0.05)
            try:
                yield
            finally:
                self._lock_fh.seek(0)
                msvcrt.locking(self._lock_fh.fileno(), msvcrt.LK_UNLCK, 1)

    @contextlib.contextmanager
    def _locked(self):
        """Hold the in-process lock and the inter-process file lock.

        The threading lock is always taken first, so lock ordering is
        consistent and deadlock-free.
        """
        with self._lock:
            with self._file_lock():
                yield

    def _is_fresh_dir(self) -> bool:
        for child in self._root.iterdir():
            if child.name == LOCK_NAME:
                # The lock file is a mutual-exclusion device, not vault
                # content; it says nothing about freshness.
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
