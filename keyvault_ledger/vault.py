"""Append-only local key vault.

The vault persists an append-only version manifest (``manifest.json``) plus
one material file per sealed version under ``materials/``.  ``reload()``
re-reads and validates the whole keyring and swaps the in-memory snapshot
atomically; readers only ever see a complete old or complete new snapshot.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import threading
from pathlib import Path

MANIFEST_NAME = "manifest.json"
MATERIALS_DIR = "materials"
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


class Vault:
    """Local key vault rooted at a directory on disk."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._lock = threading.RLock()
        fresh = not self._root.exists()
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / MATERIALS_DIR).mkdir(exist_ok=True)
        manifest_path = self._root / MANIFEST_NAME
        if not manifest_path.exists():
            if not fresh and not self._is_fresh_dir():
                # The directory already holds vault content but the manifest
                # is gone: that is corruption, not a new vault.
                raise ValueError(f"manifest missing: {manifest_path}")
            _atomic_write(manifest_path, _dump_manifest(_INITIAL_MANIFEST))
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

        with self._lock:
            new_manifest = copy.deepcopy(self._manifest)
            entry = new_manifest["keys"].setdefault(
                key_id, {"active": 0, "versions": []}
            )
            records = entry["versions"]
            new_version = records[-1]["version"] + 1 if records else 1

            rel_file = (
                f"{MATERIALS_DIR}/{_encode_key_id(key_id)}/{new_version}.bin"
            )
            digest = hashlib.sha256(material).hexdigest()

            # Write the material first, then flip the manifest.  A crash in
            # between leaves at most an orphan material file — never a
            # half-recorded version, because the manifest is the source of
            # truth and is replaced atomically.
            material_path = self._root / rel_file
            material_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(material_path, material)

            records.append(
                {"version": new_version, "sha256": digest, "file": rel_file}
            )
            entry["active"] = new_version
            _atomic_write(self._root / MANIFEST_NAME, _dump_manifest(new_manifest))

            new_snapshot = {
                k: {"active": e["active"], "versions": dict(e["versions"])}
                for k, e in self._snapshot.items()
            }
            slot = new_snapshot.setdefault(key_id, {"active": 0, "versions": {}})
            slot["versions"][new_version] = material
            slot["active"] = new_version
            self._snapshot = new_snapshot
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

    def reload(self) -> None:
        """Re-read and validate the whole keyring, then swap the snapshot.

        Never writes to disk.  On any validation failure the current
        in-memory snapshot is kept untouched.
        """
        snapshot, manifest = self._load_validated()
        with self._lock:
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

        Returns ``(snapshot, manifest)``.  Raises ``ValueError`` if the
        manifest is missing, corrupt, malformed, or disagrees with the
        stored material.
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
            snapshot[key_id] = {"active": active, "versions": versions}

        return snapshot, manifest
