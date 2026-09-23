"""Local key vault with an append-only version manifest.

The vault persists every sealed version of every key under ``root`` and
keeps a JSON manifest describing the keyring.  ``reload()`` re-reads and
validates the whole keyring from disk and only swaps the in-memory
snapshot once everything checks out, so concurrent readers always see
either the complete old snapshot or the complete new one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path, PurePosixPath
from urllib.parse import quote

__all__ = ["Vault"]

_FORMAT = 1
_MANIFEST_NAME = "manifest.json"


class _Snapshot:
    """Immutable view of the keyring: manifest plus material bytes."""

    __slots__ = ("manifest", "materials")

    def __init__(self, manifest: dict, materials: dict[str, dict[int, bytes]]):
        self.manifest = manifest
        self.materials = materials


class Vault:
    """Append-only, versioned key vault rooted at a local directory."""

    def __init__(self, root):
        self._root = Path(root)
        self._lock = threading.RLock()
        self._root.mkdir(parents=True, exist_ok=True)
        manifest_path = self._root / _MANIFEST_NAME
        if not manifest_path.exists():
            self._write_json_atomic(manifest_path, {"format": _FORMAT, "keys": {}})
        self._snapshot = self._load_snapshot()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def seal(self, key_id, material) -> int:
        """Store ``material`` as a new version of ``key_id``; return the version."""
        if not isinstance(key_id, str) or not key_id:
            raise ValueError("key_id must be a non-empty string")
        if isinstance(material, (bytes, bytearray, memoryview)):
            material = bytes(material)
        else:
            raise TypeError("material must be a bytes-like object")

        with self._lock:
            snapshot = self._snapshot
            entry = snapshot.manifest["keys"].get(key_id)
            version = (entry["active"] if entry is not None else 0) + 1

            rel = PurePosixPath("keys") / quote(key_id, safe="") / f"v{version}.bin"
            dest = self._root.joinpath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._write_bytes_atomic(dest, material)

            record = {
                "version": version,
                "sha256": hashlib.sha256(material).hexdigest(),
                "size": len(material),
                "file": rel.as_posix(),
            }
            new_manifest = copy.deepcopy(snapshot.manifest)
            key_entry = new_manifest["keys"].setdefault(
                key_id, {"active": 0, "versions": []}
            )
            key_entry["versions"].append(record)
            key_entry["active"] = version

            try:
                self._write_json_atomic(self._root / _MANIFEST_NAME, new_manifest)
            except BaseException:
                # The manifest is the commit point: if it was not updated the
                # version never happened, so drop the orphaned material file.
                try:
                    dest.unlink()
                except OSError:
                    pass
                raise

            new_materials = {k: dict(v) for k, v in snapshot.materials.items()}
            new_materials.setdefault(key_id, {})[version] = material
            self._snapshot = _Snapshot(new_manifest, new_materials)
            return version

    def load(self, key_id, version=None) -> bytes:
        """Return the stored material for ``key_id`` (active version by default)."""
        snapshot = self._snapshot
        materials = snapshot.materials.get(key_id)
        if materials is None:
            raise KeyError(key_id)
        if version is None:
            version = snapshot.manifest["keys"][key_id]["active"]
        try:
            return materials[version]
        except KeyError:
            raise KeyError((key_id, version)) from None

    def versions(self, key_id) -> list[int]:
        """Return the stored versions of ``key_id``, ascending ([] if none)."""
        snapshot = self._snapshot
        entry = snapshot.manifest["keys"].get(key_id)
        if entry is None:
            return []
        return [record["version"] for record in entry["versions"]]

    def active(self, key_id) -> int:
        """Return the active (most recently sealed) version of ``key_id``."""
        entry = self._snapshot.manifest["keys"].get(key_id)
        if entry is None:
            raise KeyError(key_id)
        return entry["active"]

    def reload(self) -> None:
        """Re-read and validate the whole keyring, then swap the snapshot.

        Never writes to disk.  On any validation failure the current
        snapshot is left untouched.
        """
        snapshot = self._load_snapshot()
        with self._lock:
            self._snapshot = snapshot

    def manifest(self) -> dict:
        """Return a deep copy of the persisted manifest."""
        return copy.deepcopy(self._snapshot.manifest)

    # ------------------------------------------------------------------
    # Loading and validation
    # ------------------------------------------------------------------

    def _load_snapshot(self) -> _Snapshot:
        manifest_path = self._root / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError(f"manifest missing: {manifest_path}")
        try:
            data = json.loads(manifest_path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"manifest unreadable or corrupt: {exc}") from exc

        if not isinstance(data, dict) or data.get("format") != _FORMAT:
            raise ValueError("manifest has an unsupported or missing format")
        keys = data.get("keys")
        if not isinstance(keys, dict):
            raise ValueError("manifest 'keys' must be an object")

        materials: dict[str, dict[int, bytes]] = {}
        for key_id, entry in keys.items():
            materials[key_id] = self._validate_entry(key_id, entry)
        return _Snapshot(data, materials)

    def _validate_entry(self, key_id, entry) -> dict[int, bytes]:
        if not isinstance(key_id, str) or not key_id:
            raise ValueError("manifest contains an invalid key id")
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry for {key_id!r} must be an object")
        records = entry.get("versions")
        active = entry.get("active")
        if not isinstance(records, list):
            raise ValueError(f"manifest entry for {key_id!r} has no version list")
        if not records:
            raise ValueError(f"manifest entry for {key_id!r} has no versions")

        materials: dict[int, bytes] = {}
        previous = 0
        for record in records:
            version = self._validate_record(key_id, record)
            if version <= previous:
                raise ValueError(
                    f"versions for {key_id!r} must be strictly increasing"
                )
            previous = version
            path = self._resolve_material(record["file"])
            try:
                blob = path.read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"material for {key_id!r} version {version} unreadable: {exc}"
                ) from exc
            if len(blob) != record["size"]:
                raise ValueError(
                    f"material for {key_id!r} version {version} has wrong size"
                )
            if hashlib.sha256(blob).hexdigest() != record["sha256"]:
                raise ValueError(
                    f"material for {key_id!r} version {version} fails checksum"
                )
            materials[version] = blob

        if not isinstance(active, int) or isinstance(active, bool) or active != previous:
            raise ValueError(
                f"active version for {key_id!r} does not match its latest version"
            )
        return materials

    @staticmethod
    def _validate_record(key_id, record) -> int:
        if not isinstance(record, dict):
            raise ValueError(f"version record for {key_id!r} must be an object")
        version = record.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError(f"invalid version number for {key_id!r}")
        sha256 = record.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(c not in "0123456789abcdef" for c in sha256)
        ):
            raise ValueError(f"invalid checksum for {key_id!r} version {version}")
        size = record.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid size for {key_id!r} version {version}")
        if not isinstance(record.get("file"), str) or not record["file"]:
            raise ValueError(f"invalid file for {key_id!r} version {version}")
        return version

    def _resolve_material(self, rel: str) -> Path:
        rel_path = PurePosixPath(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"material path escapes the vault: {rel!r}")
        return self._root.joinpath(*rel_path.parts)

    # ------------------------------------------------------------------
    # Atomic writes
    # ------------------------------------------------------------------

    def _write_json_atomic(self, path: Path, data: dict) -> None:
        payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self._write_bytes_atomic(path, payload)

    @staticmethod
    def _write_bytes_atomic(path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as fh:
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
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
