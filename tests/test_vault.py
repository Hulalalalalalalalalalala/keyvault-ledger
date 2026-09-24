"""Tests for keyvault_ledger, runnable with:

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: lock-holder/inspector tests are skipped.
    fcntl = None

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    ACTIVATIONS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
    _atomic_write,
    _derive_material,
    _dump_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _seal_worker(root: str, key_id: str, payloads: list[bytes]) -> None:
    vault = Vault(root)
    for payload in payloads:
        vault.seal(key_id, payload)


def _reload_worker(root: str, count: int) -> None:
    vault = Vault(root)
    for _ in range(count):
        vault.reload()


def _revoke_worker(root: str, key_id: str, versions: list[int]) -> None:
    vault = Vault(root)
    for version in versions:
        vault.revoke(key_id, version)
        vault.reload()


def _set_active_worker(root: str, key_id: str, versions: list[int]) -> None:
    vault = Vault(root)
    for version in versions:
        vault.set_active(key_id, version)
        vault.reload()


# ---------------------------------------------------------------------------
# Workers for the concurrency regression contract tests below.  Every worker
# is module-level so it stays picklable under any multiprocessing start
# method, and every worker reports through a multiprocessing.Queue instead of
# relying on exit codes alone, so the parent can assert on *what* happened
# (returned versions, raised exception types, observed snapshots), not just
# on success.
# ---------------------------------------------------------------------------


def _seal_one_worker(root: str, key_id: str, payload: bytes, results) -> None:
    """Seal one payload and report the version number it was assigned."""
    vault = Vault(root)
    version = vault.seal(key_id, payload)
    results.put(version)


def _revoke_contest_worker(
    root: str, key_id: str, version: int, results
) -> None:
    """Revoke one version and report the outcome as ("ok" | "error", ...)."""
    vault = Vault(root)
    try:
        vault.revoke(key_id, version)
    except (KeyError, ValueError) as exc:
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok",))


def _repoint_contest_worker(
    root: str, key_id: str, version: int, results
) -> None:
    """Repoint at one version and report the outcome."""
    vault = Vault(root)
    try:
        vault.set_active(key_id, version)
    except (KeyError, ValueError) as exc:
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok",))


def _snapshot_worker(
    root: str, key_id: str, rounds: int, results
) -> None:
    """Reload repeatedly and report every observed snapshot.

    Each observation is ``(versions_tuple, active)``; a torn read would
    either raise inside ``reload`` (validation) or surface here as a
    snapshot that is not a contiguous 1..N prefix or whose active version
    is not the largest one.
    """
    vault = Vault(root)
    for _ in range(rounds):
        vault.reload()
        # The key may not have been sealed yet; an absent key is itself a
        # coherent state (empty version list, no active version).
        versions = vault.versions(key_id)
        active = vault.active(key_id) if versions else None
        results.put((tuple(versions), active))


def _locked_disk_inspector_worker(
    root: str, rounds: int, results
) -> None:
    """Inspect the on-disk records under the exclusive lock, repeatedly.

    Acquiring the same exclusive file lock the writers use means every
    inspection observes a state no writer was in the middle of.  Each round
    reports ``("ok", versions_seen)`` or ``("bad", detail)``; the parent
    asserts nothing bad was ever reported.
    """
    import fcntl as _fcntl

    lock_path = Path(root) / LOCK_NAME

    def parse_journal(name: str) -> list[dict]:
        path = Path(root) / name
        if not path.exists():
            return []
        records = []
        for line in path.read_text("utf-8").splitlines():
            if not line:
                raise ValueError(f"{name}: empty record")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{name}: record not an object")
            records.append(record)
        return records

    for _ in range(rounds):
        problems: list[str] = []
        with open(lock_path, "a+b") as fh:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
            try:
                manifest = json.loads(
                    (Path(root) / MANIFEST_NAME).read_bytes().decode("utf-8")
                )
                for key_id, entry in manifest["keys"].items():
                    numbers = [r["version"] for r in entry["versions"]]
                    if numbers != list(range(1, len(numbers) + 1)):
                        problems.append(f"{key_id}: versions {numbers}")
                    if entry["active"] != numbers[-1]:
                        problems.append(
                            f"{key_id}: active {entry['active']} "
                            f"!= latest {numbers[-1]}"
                        )
                    for record in entry["versions"]:
                        material = (Path(root) / record["file"]).read_bytes()
                        if hashlib.sha256(material).hexdigest() != record["sha256"]:
                            problems.append(
                                f"{key_id} v{record['version']}: digest mismatch"
                            )
                # Every journal line must be a complete, parseable record
                # referencing a real sealed version — never a half line.
                try:
                    revocations = parse_journal(REVOCATIONS_NAME)
                    activations = parse_journal(ACTIVATIONS_NAME)
                except (ValueError, json.JSONDecodeError) as exc:
                    problems.append(str(exc))
                    revocations, activations = [], []
                known = {
                    key_id: {r["version"] for r in entry["versions"]}
                    for key_id, entry in manifest["keys"].items()
                }
                seen_revoked: set[tuple] = set()
                for record in revocations:
                    marker = (record.get("key_id"), record.get("version"))
                    if record.get("version") not in known.get(
                        record.get("key_id"), set()
                    ):
                        problems.append(f"revocation dangles: {record}")
                    if marker in seen_revoked:
                        problems.append(f"duplicate revocation: {record}")
                    seen_revoked.add(marker)
                for record in activations:
                    if record.get("version") not in known.get(
                        record.get("key_id"), set()
                    ):
                        problems.append(f"activation dangles: {record}")
            finally:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
        results.put(("bad", tuple(problems)) if problems else ("ok",))


def _hold_lock_worker(root: str, ready, release) -> None:
    """Hold the vault's exclusive file lock until told to release it.

    Uses the same ``fcntl.flock`` mechanism as the vault itself so the
    inter-process blocking behaviour is exercised through the real lock.
    """
    import fcntl as _fcntl

    fh = open(Path(root) / LOCK_NAME, "a+b")
    _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
    try:
        ready.set()
        release.wait(timeout=30)
    finally:
        _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
        fh.close()


class VaultTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "vault"

    def open_vault(self) -> Vault:
        return Vault(self.root)

    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def disk_manifest(self) -> dict:
        return json.loads(self.manifest_path().read_bytes().decode("utf-8"))

    def journal_path(self) -> Path:
        return self.root / REVOCATIONS_NAME

    def journal_records(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.journal_path().read_text("utf-8").splitlines()
        ]

    def activations_path(self) -> Path:
        return self.root / ACTIVATIONS_NAME

    def activation_records(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.activations_path().read_text("utf-8").splitlines()
        ]


class TestFreshVault(VaultTestCase):
    def test_missing_directory_is_created_on_open(self):
        self.assertFalse(self.root.exists())
        vault = self.open_vault()
        self.assertTrue(self.root.is_dir())
        self.assertEqual(vault.versions("anything"), [])

    def test_fresh_vault_has_no_keys(self):
        vault = self.open_vault()
        self.assertEqual(vault.manifest()["keys"], {})
        self.assertEqual(vault.versions("nope"), [])

    def test_reload_on_fresh_vault_succeeds(self):
        self.open_vault().reload()

    def test_existing_empty_directory_is_initialised(self):
        self.root.mkdir(parents=True)
        vault = self.open_vault()
        self.assertEqual(vault.versions("k"), [])


class TestSealAndLoad(VaultTestCase):
    def test_first_version_is_one_and_versions_increase(self):
        vault = self.open_vault()
        self.assertEqual(vault.seal("signing", b"v1"), 1)
        self.assertEqual(vault.seal("signing", b"v2"), 2)
        self.assertEqual(vault.seal("signing", b"v3"), 3)
        self.assertEqual(vault.versions("signing"), [1, 2, 3])

    def test_active_points_at_most_recent_seal(self):
        vault = self.open_vault()
        vault.seal("k", b"a")
        self.assertEqual(vault.active("k"), 1)
        vault.seal("k", b"b")
        self.assertEqual(vault.active("k"), 2)

    def test_load_defaults_to_active_version(self):
        vault = self.open_vault()
        vault.seal("k", b"old")
        vault.seal("k", b"new")
        self.assertEqual(vault.load("k"), b"new")

    def test_load_specific_version_returns_old_material(self):
        vault = self.open_vault()
        vault.seal("k", b"old")
        vault.seal("k", b"new")
        self.assertEqual(vault.load("k", 1), b"old")
        self.assertEqual(vault.load("k", version=2), b"new")

    def test_material_is_byte_exact(self):
        material = bytes(range(256)) + b"\x00\xff\n\r"
        vault = self.open_vault()
        vault.seal("bin", material)
        self.assertEqual(vault.load("bin"), material)

    def test_keys_are_independent(self):
        vault = self.open_vault()
        vault.seal("a", b"1")
        vault.seal("b", b"1")
        vault.seal("b", b"2")
        self.assertEqual(vault.versions("a"), [1])
        self.assertEqual(vault.versions("b"), [1, 2])
        self.assertEqual(vault.active("a"), 1)
        self.assertEqual(vault.active("b"), 2)

    def test_unusual_key_ids_round_trip(self):
        vault = self.open_vault()
        for key_id in ("a/b", "..", "sp ace", "ünïcode", "x" * 200):
            vault.seal(key_id, b"m")
            self.assertEqual(vault.load(key_id), b"m")
            self.assertEqual(vault.versions(key_id), [1])


class TestErrors(VaultTestCase):
    def test_seal_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.seal("", b"m")

    def test_seal_non_bytes_material_raises_type_error(self):
        vault = self.open_vault()
        for bad in ("text", 123, None, [b"x"], {"k": b"v"}):
            with self.assertRaises(TypeError):
                vault.seal("k", bad)

    def test_load_unknown_key_raises_key_error(self):
        vault = self.open_vault()
        with self.assertRaises(KeyError):
            vault.load("never-sealed")

    def test_load_unknown_version_raises_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.load("k", 2)
        with self.assertRaises(KeyError):
            vault.load("k", 0)

    def test_active_unknown_key_raises_key_error(self):
        vault = self.open_vault()
        with self.assertRaises(KeyError):
            vault.active("never-sealed")

    def test_failed_seal_leaves_no_partial_version(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        before_disk = self.disk_manifest()
        with self.assertRaises(TypeError):
            vault.seal("k", "not-bytes")
        with self.assertRaises(ValueError):
            vault.seal("", b"x")
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.load("k"), b"v1")
        self.assertEqual(self.disk_manifest(), before_disk)


class TestManifest(VaultTestCase):
    def test_manifest_matches_persisted_record(self):
        vault = self.open_vault()
        vault.seal("a", b"one")
        vault.seal("a", b"two")
        vault.seal("b", b"three")
        self.assertEqual(vault.manifest(), self.disk_manifest())

    def test_manifest_records_every_version_and_active(self):
        vault = self.open_vault()
        vault.seal("a", b"one")
        vault.seal("a", b"two")
        entry = vault.manifest()["keys"]["a"]
        self.assertEqual([r["version"] for r in entry["versions"]], [1, 2])
        self.assertEqual(entry["active"], 2)

    def test_manifest_returns_a_copy(self):
        vault = self.open_vault()
        vault.seal("a", b"one")
        snapshot = vault.manifest()
        snapshot["keys"]["a"]["active"] = 999
        self.assertEqual(vault.manifest()["keys"]["a"]["active"], 1)


class TestReload(VaultTestCase):
    def test_reload_picks_up_changes_written_by_another_handle(self):
        first = self.open_vault()
        second = self.open_vault()
        first.seal("k", b"m")
        self.assertEqual(second.versions("k"), [])
        second.reload()
        self.assertEqual(second.load("k"), b"m")
        self.assertEqual(second.active("k"), 1)

    def test_reload_does_not_write_to_disk(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        before = self.manifest_path().read_bytes()
        before_mtime = self.manifest_path().stat().st_mtime_ns
        vault.reload()
        self.assertEqual(self.manifest_path().read_bytes(), before)
        self.assertEqual(self.manifest_path().stat().st_mtime_ns, before_mtime)

    def test_reload_missing_manifest_raises_and_keeps_snapshot(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        before = self.disk_manifest()
        self.manifest_path().unlink()
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"m")
        self.assertEqual(vault.versions("k"), [1])

    def test_reload_corrupt_manifest_raises_and_keeps_snapshot(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        self.manifest_path().write_bytes(b"{not json")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"m")

    def test_reload_invalid_format_raises(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        for bad in (
            [],
            {"format": 999, "keys": {}},
            {"format": 1, "keys": []},
            {"format": 1, "keys": {"k": {"active": 1, "versions": []}}},
            {
                "format": 1,
                "keys": {
                    "k": {
                        "active": 2,
                        "versions": [
                            {"version": 1, "sha256": "0" * 64, "file": "x"}
                        ],
                    }
                },
            },
        ):
            self.manifest_path().write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                vault.reload()
            self.assertEqual(vault.load("k"), b"m")

    def test_reload_material_mismatch_raises_and_disk_is_untouched(self):
        vault = self.open_vault()
        vault.seal("k", b"original")
        before = self.manifest_path().read_bytes()
        entry = self.disk_manifest()["keys"]["k"]["versions"][0]
        material_file = self.root / entry["file"]
        material_file.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"original")
        self.assertEqual(self.manifest_path().read_bytes(), before)

    def test_reload_missing_material_raises(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        entry = self.disk_manifest()["keys"]["k"]["versions"][0]
        (self.root / entry["file"]).unlink()
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"m")

    def test_failed_reload_adds_or_removes_no_versions(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        before_keys = set(vault.manifest()["keys"])
        self.manifest_path().write_bytes(b"garbage")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(set(vault.manifest()["keys"]), before_keys)
        self.assertEqual(vault.versions("k"), [1])


class TestRevocation(VaultTestCase):
    def test_revoke_is_an_explicit_marker_only(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("other", b"m")
        vault.revoke("k", 1)

        # Material of the revoked version is still readable byte for byte.
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.load("k", 2), b"v2")
        # The version list still lists every version, oldest first.
        self.assertEqual(vault.versions("k"), [1, 2])
        # Active version is unaffected.
        self.assertEqual(vault.active("k"), 2)

    def test_is_revoked_reports_status(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        self.assertFalse(vault.is_revoked("k", 1))
        vault.revoke("k", 1)
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertFalse(vault.is_revoked("k", 2))

    def test_revoked_versions_is_ascending_and_complete(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        vault.revoke("k", 3)
        vault.revoke("k", 1)
        self.assertEqual(vault.revoked_versions("k"), [1, 3])

    def test_revoked_versions_empty_when_never_revoked_or_unknown(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        self.assertEqual(vault.revoked_versions("k"), [])
        # Unknown key yields an empty list rather than raising.
        self.assertEqual(vault.revoked_versions("never-sealed"), [])

    def test_revocation_persists_through_full_reload(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)

        reloaded = self.open_vault()
        self.assertTrue(reloaded.is_revoked("k", 1))
        self.assertFalse(reloaded.is_revoked("k", 2))
        self.assertEqual(reloaded.revoked_versions("k"), [1])
        reloaded.reload()
        self.assertEqual(reloaded.revoked_versions("k"), [1])

    def test_revocation_appends_to_journal_without_touching_manifest(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        manifest_before = self.manifest_path().read_bytes()
        vault.revoke("k", 1)
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)
        self.assertEqual(
            self.journal_records(), [{"key_id": "k", "version": 1}]
        )
        # A second, distinct revocation only adds a line.
        vault.seal("k", b"v2")
        vault.revoke("k", 2)
        self.assertEqual(
            self.journal_records(),
            [{"key_id": "k", "version": 1}, {"key_id": "k", "version": 2}],
        )
        self.assertEqual(
            len(self.journal_path().read_bytes().splitlines()), 2
        )

    def test_revoked_material_unchanged_on_disk(self):
        vault = self.open_vault()
        vault.seal("k", b"keep-me")
        entry = self.disk_manifest()["keys"]["k"]["versions"][0]
        material_file = self.root / entry["file"]
        before = material_file.read_bytes()
        vault.revoke("k", 1)
        self.assertEqual(material_file.read_bytes(), before)
        self.assertEqual(before, b"keep-me")

    def test_revocation_does_not_change_active_pointer(self):
        vault = self.open_vault()
        vault.seal("k", b"a")
        vault.seal("k", b"b")
        vault.revoke("k", 2)
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"b")
        vault.reload()
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"b")


class TestRevocationErrors(VaultTestCase):
    def test_revoke_unknown_key_raises_key_error(self):
        vault = self.open_vault()
        with self.assertRaises(KeyError):
            vault.revoke("never-sealed", 1)

    def test_revoke_never_sealed_version_raises_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        for bad_version in (0, 2, -1, 99):
            with self.assertRaises(KeyError):
                vault.revoke("k", bad_version)

    def test_revoke_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(ValueError):
            vault.revoke("", 1)

    def test_double_revoke_same_version_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        # A different version can still be revoked.
        vault.revoke("k", 2)
        self.assertEqual(vault.revoked_versions("k"), [1, 2])

    def test_double_revoke_after_reload_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        reloaded = self.open_vault()
        with self.assertRaises(ValueError):
            reloaded.revoke("k", 1)

    def test_failed_revoke_appends_nothing(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        journal_before = self.journal_path().read_bytes()
        for call in (
            lambda: vault.revoke("never-sealed", 1),
            lambda: vault.revoke("k", 2),
            lambda: vault.revoke("", 1),
        ):
            with self.assertRaises((KeyError, ValueError)):
                call()
        self.assertEqual(self.journal_path().read_bytes(), journal_before)
        self.assertEqual(vault.revoked_versions("k"), [])

    def test_is_revoked_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.is_revoked("", 1)

    def test_is_revoked_unknown_key_or_version_raises_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.is_revoked("never-sealed", 1)
        for bad_version in (0, 2):
            with self.assertRaises(KeyError):
                vault.is_revoked("k", bad_version)

    def test_revoked_versions_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.revoked_versions("")


class TestJournalValidation(VaultTestCase):
    def _write_journal(self, lines: bytes | list[bytes]) -> None:
        if isinstance(lines, bytes):
            self.journal_path().write_bytes(lines)
        else:
            self.journal_path().write_bytes(b"".join(lines))

    def test_corrupt_journal_line_makes_reload_fail_but_keeps_snapshot(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        self._write_journal(b"{not json\n")
        with self.assertRaises(ValueError):
            vault.reload()
        # Existing handle still serves its keys.
        self.assertEqual(vault.load("k"), b"m")
        self.assertTrue(vault.is_revoked("k", 1))

    def test_journal_revoking_unknown_version_is_rejected(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        record = json.dumps({"key_id": "k", "version": 5}) + "\n"
        self._write_journal([record.encode("utf-8")])
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"m")

    def test_journal_revoking_unknown_key_is_rejected(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        record = json.dumps({"key_id": "ghost", "version": 1}) + "\n"
        self._write_journal([record.encode("utf-8")])
        with self.assertRaises(ValueError):
            vault.reload()

    def test_duplicate_revocation_record_is_rejected(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        record = (json.dumps({"key_id": "k", "version": 1}) + "\n").encode()
        self._write_journal([record, record])
        with self.assertRaises(ValueError):
            vault.reload()

    def test_malformed_journal_records_are_rejected(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        bad_payloads = [
            b"[1, 2]\n",
            b'{"key_id": "", "version": 1}\n',
            b'{"key_id": "k", "version": "1"}\n',
            b'{"key_id": "k"}\n',
            b'{"version": 1}\n',
            b"\n",
            b"\xff\xfe\n",
        ]
        for payload in bad_payloads:
            self._write_journal([payload])
            with self.assertRaises(ValueError, msg=payload):
                vault.reload()
            self.assertEqual(vault.load("k"), b"m")

    def test_missing_journal_rejected_on_reload_but_migrated_on_open(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        self.journal_path().unlink()

        # A live handle cannot validate without the journal, so reload
        # fails and keeps its snapshot and markers.
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertTrue(vault.is_revoked("k", 1))

        # Opening a directory with no journal migrates it (the layout also
        # covers vaults created before revocations existed): an empty
        # journal is created and existing material stays readable.
        migrated = self.open_vault()
        self.assertTrue(self.journal_path().exists())
        self.assertEqual(migrated.load("k"), b"m")
        self.assertEqual(migrated.versions("k"), [1])
        self.assertEqual(migrated.revoked_versions("k"), [])

    def test_failed_reload_changes_no_revocation_markers(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        self._write_journal(b"{garbage\n")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.revoked_versions("k"), [1])
        self.assertTrue(vault.is_revoked("k", 1))


class TestSetActive(VaultTestCase):
    def test_set_active_moves_active_and_unversioned_reads(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        self.assertIsNone(vault.set_active("k", 1))
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"v1")

    def test_versioned_reads_stay_pinned_to_the_version(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.set_active("k", 1)
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.load("k", 2), b"v2")
        # The version list is untouched.
        self.assertEqual(vault.versions("k"), [1, 2])

    def test_derivation_without_version_follows_repoint(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("k", b"pw", b"salt", 100, 16)
        vault.set_active("k", 1)
        self.assertEqual(vault.derivation("k"), {})
        self.assertEqual(vault.derivation("k", 2)["iterations"], 100)

    def test_repeated_repoints_last_one_wins(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        vault.set_active("k", 1)
        vault.set_active("k", 3)
        vault.set_active("k", 2)
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"v2")
        self.assertEqual(len(self.activation_records()), 3)

    def test_repoint_survives_reload_and_reopen(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.set_active("k", 1)
        vault.reload()
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"v1")
        reopened = self.open_vault()
        self.assertEqual(reopened.active("k"), 1)
        self.assertEqual(reopened.load("k"), b"v1")
        reopened.reload()
        self.assertEqual(reopened.active("k"), 1)

    def test_new_seal_resets_active_and_supersedes_the_repoint(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.set_active("k", 1)
        self.assertEqual(vault.seal("k", b"v3"), 3)
        self.assertEqual(vault.active("k"), 3)
        self.assertEqual(vault.load("k"), b"v3")
        # The old repoint stays on disk but no longer applies, even after a
        # full reload or reopening.
        vault.reload()
        self.assertEqual(vault.active("k"), 3)
        self.assertEqual(self.open_vault().active("k"), 3)
        self.assertEqual(len(self.activation_records()), 1)

    def test_derived_seal_also_resets_active(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.set_active("k", 1)
        vault.derive_seal("k", b"pw", b"salt", 100, 16)
        self.assertEqual(vault.active("k"), 3)
        self.assertEqual(self.open_vault().active("k"), 3)

    def test_journal_is_created_on_first_repoint_and_only_grows(self):
        vault = self.open_vault()
        vault.seal("k", b"m1")
        vault.seal("k", b"m2")
        # Opening and sealing never creates the activations journal.
        self.assertFalse(self.activations_path().exists())
        vault.set_active("k", 1)
        self.assertTrue(self.activations_path().exists())
        records = self.activation_records()
        self.assertEqual(
            records, [{"key_id": "k", "version": 1, "latest": 2}]
        )
        vault.seal("k", b"m3")
        # A later seal neither rewrites nor appends activation records.
        self.assertEqual(self.activation_records(), records)
        vault.set_active("k", 1)
        self.assertEqual(
            self.activation_records(),
            [
                {"key_id": "k", "version": 1, "latest": 2},
                {"key_id": "k", "version": 1, "latest": 3},
            ],
        )

    def test_repoint_does_not_touch_manifest_or_materials(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        manifest_before = self.manifest_path().read_bytes()
        vault.set_active("k", 1)
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)
        # The persisted manifest keeps recording the newest sealed version.
        self.assertEqual(self.disk_manifest()["keys"]["k"]["active"], 2)
        self.assertEqual(vault.load("k", 2), b"v2")

    def test_repoints_of_different_keys_are_independent(self):
        vault = self.open_vault()
        vault.seal("a", b"a1")
        vault.seal("a", b"a2")
        vault.seal("b", b"b1")
        vault.set_active("a", 1)
        self.assertEqual(vault.active("a"), 1)
        self.assertEqual(vault.active("b"), 1)
        self.assertEqual(self.open_vault().active("a"), 1)
        self.assertEqual(self.open_vault().active("b"), 1)

    def test_set_active_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(ValueError):
            vault.set_active("", 1)

    def test_set_active_non_integer_version_raises_type_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        for bad in (1.0, 2.5, True, False, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.set_active("k", bad)

    def test_set_active_unknown_key_or_version_raises_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.set_active("never-sealed", 1)
        for bad_version in (0, 2, -1, 99):
            with self.assertRaises(KeyError, msg=repr(bad_version)):
                vault.set_active("k", bad_version)

    def test_set_active_already_active_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)
        vault.set_active("k", 1)
        with self.assertRaises(ValueError):
            vault.set_active("k", 1)
        # Repointing away and back is allowed: the middle repoint moved it.
        vault.set_active("k", 2)
        vault.set_active("k", 1)
        self.assertEqual(vault.active("k"), 1)

    def test_set_active_revoked_version_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.set_active("k", 1)

    def test_revoking_after_repoint_does_not_move_pointer(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.set_active("k", 1)
        # Existing semantics: revoking the active version leaves it active.
        vault.revoke("k", 1)
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"v1")
        self.assertTrue(vault.is_revoked("k", 1))
        # The repoint still applies after reload/reopen even though its
        # target is now revoked: being revoked later is not an error.
        vault.reload()
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(self.open_vault().active("k"), 1)

    def test_failed_repoint_appends_nothing_and_keeps_vault_usable(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.set_active("k", 1)
        journal_before = self.activations_path().read_bytes()
        for call in (
            lambda: vault.set_active("never-sealed", 1),
            lambda: vault.set_active("k", 99),
            lambda: vault.set_active("k", 1),
            lambda: vault.set_active("", 1),
        ):
            with self.assertRaises((KeyError, ValueError)):
                call()
        self.assertEqual(self.activations_path().read_bytes(), journal_before)
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"v1")


class TestActivationJournalValidation(VaultTestCase):
    def _write_activations(self, data: bytes) -> None:
        self.activations_path().write_bytes(data)

    def _record(self, key_id="k", version=1, latest=2) -> bytes:
        return (
            json.dumps({"key_id": key_id, "version": version, "latest": latest})
            + "\n"
        ).encode("utf-8")

    def test_corrupt_journal_line_makes_reload_fail_but_keeps_snapshot(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.seal("k", b"m2")
        vault.set_active("k", 1)
        self._write_activations(b"{not json\n")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"m")
        # A failed reload never repairs or removes the journal.
        self.assertEqual(self.activations_path().read_bytes(), b"{not json\n")

    def test_record_pointing_at_missing_version_is_rejected(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        self._write_activations(self._record(version=5, latest=5))
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"m")

    def test_record_for_unknown_key_is_rejected(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        self._write_activations(self._record(key_id="ghost"))
        with self.assertRaises(ValueError):
            Vault(self.root)

    def test_malformed_records_are_rejected(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        bad_payloads = [
            b"[1, 2]\n",
            b'{"key_id": "", "version": 1, "latest": 1}\n',
            b'{"key_id": "k", "version": "1", "latest": 1}\n',
            b'{"key_id": "k", "version": 1, "latest": "1"}\n',
            b'{"key_id": "k", "version": 1}\n',
            b'{"key_id": "k", "latest": 1}\n',
            b'{"version": 1, "latest": 1}\n',
            # version above the bound newest sealed version is impossible.
            b'{"key_id": "k", "version": 2, "latest": 1}\n',
            # latest itself never sealed.
            b'{"key_id": "k", "version": 1, "latest": 9}\n',
            b"\n",
            b"\xff\xfe\n",
        ]
        for payload in bad_payloads:
            self._write_activations(payload)
            with self.assertRaises(ValueError, msg=payload):
                vault.reload()
            self.assertEqual(vault.load("k"), b"m")

    def test_superseded_well_formed_record_is_accepted_but_inert(self):
        # A repoint bound to an older newest-sealed version cannot be told
        # apart from a genuine historical repoint: it validates fine and
        # simply stops applying once a newer version was sealed.
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        self._write_activations(self._record(version=1, latest=2))
        vault.reload()
        self.assertEqual(vault.active("k"), 3)
        self.assertEqual(Vault(self.root).active("k"), 3)

    def test_only_last_record_for_a_key_applies(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        self._write_activations(
            self._record(version=1, latest=3)
            + self._record(version=2, latest=3)
        )
        vault.reload()
        self.assertEqual(vault.active("k"), 2)
        # Every line must validate even if an earlier one cannot win.
        self._write_activations(
            self._record(version=1, latest=3)
            + self._record(version=99, latest=3)
        )
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.active("k"), 2)

    def test_failed_reload_touches_no_disk_records(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.set_active("k", 1)
        manifest_before = self.manifest_path().read_bytes()
        journal_before = self.activations_path().read_bytes()
        self._write_activations(b"{garbage\n")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)
        self.assertEqual(self.activations_path().read_bytes(), b"{garbage\n")
        # The in-memory snapshot survives intact and keys stay readable.
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"v1")
        self.assertEqual(vault.load("k", 2), b"v2")
        # Restoring the journal makes the vault healthy again.
        self._write_activations(journal_before)
        vault.reload()
        self.assertEqual(vault.active("k"), 1)


class TestSealCleanup(VaultTestCase):
    def test_failed_manifest_write_removes_orphan_material(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        materials_dir = self.root / MATERIALS_DIR
        originals = {p for p in materials_dir.rglob("*.bin")}

        real_atomic_write = _atomic_write
        calls = {"n": 0}

        def flaky(path, data):
            # Fail only the manifest write of the next seal; materials
            # still land on disk first, exercising the cleanup branch.
            if path.name == MANIFEST_NAME:
                calls["n"] += 1
                raise OSError("simulated manifest failure")
            return real_atomic_write(path, data)

        with mock.patch(
            "keyvault_ledger.vault._atomic_write", side_effect=flaky
        ):
            with self.assertRaises(OSError):
                vault.seal("k", b"v2")

        self.assertEqual(calls["n"], 1)
        leftovers = {p for p in materials_dir.rglob("*.bin")}
        self.assertEqual(leftovers, originals)
        # The failed seal neither added a version nor moved active.
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"v1")
        # The vault is still usable and the version number is not reused.
        self.assertEqual(vault.seal("k", b"v2-real"), 2)
        self.assertEqual(vault.load("k", 2), b"v2-real")

    def test_reload_read_inside_lock_keeps_seal_from_being_stomped(self):
        # Regression test: reload used to read disk outside the lock.  A
        # seal completing in the window between that read and the locked
        # snapshot swap got silently reverted in memory; the next seal then
        # reused the version number and overwrote historical material.
        vault = self.open_vault()
        vault.seal("k", b"v1")
        manifest_with_v1 = self.manifest_path().read_bytes()

        read_started = threading.Event()
        release_read = threading.Event()
        orig_load = Vault._load_validated

        def slow_load(self):
            read_started.set()
            release_read.wait(timeout=5)
            return orig_load(self)

        seal_error: list[BaseException] = []

        def do_seal():
            try:
                vault.seal("k", b"v2")
            except BaseException as exc:  # captured below
                seal_error.append(exc)

        with mock.patch.object(
            Vault, "_load_validated", autospec=True, side_effect=slow_load
        ):
            reloader = threading.Thread(target=vault.reload)
            reloader.start()
            self.assertTrue(read_started.wait(timeout=5))

            sealer = threading.Thread(target=do_seal)
            sealer.start()
            # The reload holds the lock for the whole read, so the seal
            # cannot complete until the read finishes: it is blocked and
            # the on-disk manifest is still the v1 one.
            sealer.join(timeout=0.2)
            self.assertTrue(sealer.is_alive())
            self.assertEqual(
                self.manifest_path().read_bytes(), manifest_with_v1
            )

            release_read.set()
            reloader.join(timeout=5)
            sealer.join(timeout=5)

        self.assertFalse(reloader.is_alive())
        self.assertFalse(sealer.is_alive())
        self.assertEqual(seal_error, [])

        # The seal was not reverted in memory...
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.active("k"), 2)
        # ...and historical material was not overwritten.
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.load("k", 2), b"v2")
        # A subsequent seal takes version 3, not a reused 2.
        self.assertEqual(vault.seal("k", b"v3"), 3)
        self.assertEqual(vault.load("k", 2), b"v2")
        self.assertEqual(vault.load("k", 3), b"v3")


class TestVersionEntryValidation(VaultTestCase):
    def test_non_int_version_raises_type_error_on_revoke_and_query(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        journal_before = self.journal_path().read_bytes()
        for bad in (1.0, 2.5, True, False, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.revoke("k", bad)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.is_revoked("k", bad)
        # Rejected versions never reach the append-only journal and the
        # in-memory markers are untouched.
        self.assertEqual(self.journal_path().read_bytes(), journal_before)
        self.assertEqual(vault.revoked_versions("k"), [])
        self.assertFalse(vault.is_revoked("k", 1))

    def test_non_int_version_rejected_even_for_unknown_key(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        # Entry validation fires before any key lookup.
        with self.assertRaises(TypeError):
            vault.is_revoked("never-sealed", 1.5)
        with self.assertRaises(TypeError):
            vault.revoke("never-sealed", True)


class TestMultiProcess(VaultTestCase):
    def _run_workers(self, workers: list[multiprocessing.Process]) -> None:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=120)
            self.assertEqual(worker.exitcode, 0)

    def test_concurrent_seals_allocate_unique_contiguous_versions(self):
        procs, seals_each = 4, 5
        payloads = {
            (p, i): f"proc-{p}-seal-{i}".encode("utf-8")
            for p in range(procs)
            for i in range(seals_each)
        }
        workers = [
            multiprocessing.Process(
                target=_seal_worker,
                args=(
                    str(self.root),
                    "shared",
                    [payloads[(p, i)] for i in range(seals_each)],
                ),
            )
            for p in range(procs)
        ]
        # Reloaders run alongside: every reload must see either the
        # complete old records or the complete newly persisted ones —
        # a torn read would fail validation and exit non-zero.
        workers += [
            multiprocessing.Process(target=_reload_worker, args=(str(self.root), 20))
            for _ in range(2)
        ]
        self._run_workers(workers)

        vault = self.open_vault()
        total = procs * seals_each
        # No duplicated, skipped or reused version numbers.
        self.assertEqual(vault.versions("shared"), list(range(1, total + 1)))
        self.assertEqual(vault.active("shared"), total)
        # Every sealed payload landed exactly once, byte for byte.
        self.assertEqual(
            {vault.load("shared", v) for v in range(1, total + 1)},
            set(payloads.values()),
        )
        vault.reload()
        self.assertEqual(vault.versions("shared"), list(range(1, total + 1)))

    def test_concurrent_revokes_and_reloads_keep_journal_consistent(self):
        vault = self.open_vault()
        total = 12
        for i in range(total):
            vault.seal("k", f"m-{i}".encode("utf-8"))

        procs = 3
        workers = [
            multiprocessing.Process(
                target=_revoke_worker,
                args=(
                    str(self.root),
                    "k",
                    [v for v in range(1, total + 1) if v % procs == p],
                ),
            )
            for p in range(procs)
        ]
        self._run_workers(workers)

        # The journal only grew: one well-formed record per version.
        records = self.journal_records()
        self.assertEqual(len(records), total)
        self.assertEqual(
            sorted(record["version"] for record in records),
            list(range(1, total + 1)),
        )
        self.assertTrue(all(record["key_id"] == "k" for record in records))

        reloaded = self.open_vault()
        self.assertEqual(reloaded.revoked_versions("k"), list(range(1, total + 1)))
        # Historical material was never overwritten or truncated.
        for i in range(1, total + 1):
            self.assertEqual(reloaded.load("k", i), f"m-{i - 1}".encode("utf-8"))

    def test_concurrent_repoints_seals_revokes_and_reloads_stay_consistent(self):
        vault = self.open_vault()
        total = 12
        for i in range(total):
            vault.seal("k", f"m-{i}".encode("utf-8"))

        # One process repoints a subset, another seals fresh versions and a
        # third revokes; reloaders hammer the directory concurrently.  Every
        # reload must see a complete, self-consistent snapshot (exit 0).
        repoints = [1, 3, 5, 7, 9, 11]
        seals = [b"fresh-1", b"fresh-2", b"fresh-3"]
        workers = [
            multiprocessing.Process(
                target=_set_active_worker, args=(str(self.root), "k", repoints)
            ),
            multiprocessing.Process(
                target=_seal_worker, args=(str(self.root), "k", seals)
            ),
            multiprocessing.Process(
                target=_revoke_worker,
                args=(str(self.root), "k", [2, 4, 6]),
            ),
        ]
        workers += [
            multiprocessing.Process(target=_reload_worker, args=(str(self.root), 20))
            for _ in range(2)
        ]
        self._run_workers(workers)

        # Every repoint attempt produced one well-formed record...
        records = [
            json.loads(line)
            for line in (self.root / ACTIVATIONS_NAME)
            .read_text("utf-8")
            .splitlines()
        ]
        self.assertEqual(
            [r["version"] for r in records], repoints
        )
        for record in records:
            self.assertEqual(record["key_id"], "k")
            self.assertIn(record["version"], range(1, total + 1))
            self.assertIn(record["latest"], range(1, total + len(seals) + 1))

        final = self.open_vault()
        # The three fresh seals extended the one shared sequence.
        self.assertEqual(
            final.versions("k"), list(range(1, total + len(seals) + 1))
        )
        # Seals happened after every repoint attempt began; whether the
        # last seal landed before or after the last repoint, the active
        # version must be a real, non-revoked sealed version.
        active = final.active("k")
        self.assertIn(active, final.versions("k"))
        self.assertNotIn(active, [2, 4, 6])
        self.assertEqual(final.load("k"), final.load("k", active))
        self.assertEqual(final.revoked_versions("k"), [2, 4, 6])

    def test_concurrent_seals_and_derive_seals_share_one_sequence(self):
        seal_procs, each = 3, 4
        workers = [
            multiprocessing.Process(
                target=_seal_worker,
                args=(
                    str(self.root),
                    "shared",
                    [f"seal-{p}-{i}".encode() for i in range(each)],
                ),
            )
            for p in range(seal_procs)
        ]
        workers += [
            multiprocessing.Process(
                target=_derive_worker,
                args=(
                    str(self.root),
                    "shared",
                    [f"pw-{p}-{i}".encode() for i in range(each)],
                ),
            )
            for p in range(2)
        ]
        self._run_workers(workers)

        total = (seal_procs + 2) * each
        vault = self.open_vault()
        # Derived and plain seals draw from one non-repeating sequence.
        self.assertEqual(vault.versions("shared"), list(range(1, total + 1)))
        self.assertEqual(vault.active("shared"), total)
        seal_payloads = {
            f"seal-{p}-{i}".encode()
            for p in range(seal_procs)
            for i in range(each)
        }
        derived_payloads = {
            hashlib.pbkdf2_hmac(
                "sha256", f"pw-{p}-{i}".encode(), b"worker-salt", 100, dklen=16
            )
            for p in range(2)
            for i in range(each)
        }
        self.assertEqual(
            {vault.load("shared", v) for v in range(1, total + 1)},
            seal_payloads | derived_payloads,
        )

    def test_lock_file_is_not_vault_content(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        self.assertTrue((self.root / LOCK_NAME).exists())
        # The lock file shows up neither as a key nor in validation.
        self.assertEqual(set(vault.manifest()["keys"]), {"k"})
        self.assertEqual(vault.versions("k"), [1])
        vault.reload()
        self.assertEqual(vault.load("k"), b"m")


def _derive_worker(root: str, key_id: str, payloads: list[bytes]) -> None:
    vault = Vault(root)
    salt = b"worker-salt"
    for i, payload in enumerate(payloads):
        vault.derive_seal(key_id, payload, salt, 100, 16)


class TestMultiProcessConcurrencyContract(VaultTestCase):
    """Regression tests pinning the multi-process concurrency contract.

    Real processes interleave real operations in a temporary vault; every
    case observes what landed on disk and what each process got back.
    """

    def _start(self, target, *args) -> multiprocessing.Process:
        proc = multiprocessing.Process(target=target, args=args)
        proc.start()
        self.addCleanup(self._terminate, proc)
        return proc

    @staticmethod
    def _terminate(proc: multiprocessing.Process) -> None:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)

    def _join(self, proc: multiprocessing.Process) -> None:
        proc.join(timeout=120)
        self.assertFalse(proc.is_alive(), f"worker {proc.name} hung")
        self.assertEqual(proc.exitcode, 0)

    def _drain(self, results, count: int, timeout: float = 60.0) -> list:
        items = []
        deadline = time.monotonic() + timeout
        while len(items) < count and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            items.append(results.get(timeout=max(0.1, remaining)))
        self.assertEqual(
            len(items), count, f"expected {count} results, got {len(items)}"
        )
        return items

    def _seed_versions(self, key_id: str, total: int) -> None:
        vault = self.open_vault()
        for i in range(total):
            vault.seal(key_id, f"seed-{i}".encode("utf-8"))

    def test_concurrent_revokes_of_same_version_exactly_one_wins(self):
        self._seed_versions("k", 1)
        results = multiprocessing.Queue()
        contenders = 6
        workers = [
            self._start(_revoke_contest_worker, str(self.root), "k", 1, results)
            for _ in range(contenders)
        ]
        outcomes = self._drain(results, contenders)
        for proc in workers:
            self._join(proc)

        # Exactly one revocation lands; every other process observes it was
        # already revoked, with ValueError (never a silent success or crash).
        self.assertEqual(sum(o[0] == "ok" for o in outcomes), 1)
        self.assertEqual(
            sorted(o[1] for o in outcomes if o[0] == "error"),
            ["ValueError"] * (contenders - 1),
        )
        # The append-only journal gained exactly one well-formed record.
        self.assertEqual(self.journal_records(), [{"key_id": "k", "version": 1}])
        reloaded = self.open_vault()
        self.assertEqual(reloaded.revoked_versions("k"), [1])
        self.assertTrue(reloaded.is_revoked("k", 1))
        # History is untouched and still readable.
        self.assertEqual(reloaded.load("k", 1), b"seed-0")

    def test_concurrent_repoints_at_same_version_exactly_one_wins(self):
        self._seed_versions("k", 2)  # v2 active; v1 is the repoint target
        results = multiprocessing.Queue()
        contenders = 6
        workers = [
            self._start(_repoint_contest_worker, str(self.root), "k", 1, results)
            for _ in range(contenders)
        ]
        outcomes = self._drain(results, contenders)
        for proc in workers:
            self._join(proc)

        # Exactly one repoint moves the pointer; the rest see it already
        # active and get ValueError.
        self.assertEqual(sum(o[0] == "ok" for o in outcomes), 1)
        self.assertEqual(
            sorted(o[1] for o in outcomes if o[0] == "error"),
            ["ValueError"] * (contenders - 1),
        )
        self.assertEqual(len(self.activation_records()), 1)
        self.assertEqual(self.open_vault().active("k"), 1)

    def test_observed_snapshot_is_always_a_contiguous_prefix(self):
        # Observers reload in a tight loop while writers seal; every observed
        # version list must be a 1..N prefix and active must equal N.
        results = multiprocessing.Queue()
        rounds = 40
        writers = 4
        per_writer = 6
        workers = [
            self._start(_snapshot_worker, str(self.root), "shared", rounds, results)
            for _ in range(2)
        ]
        workers += [
            self._start(
                _seal_worker,
                str(self.root),
                "shared",
                [f"w-{w}-{i}".encode() for i in range(per_writer)],
            )
            for w in range(writers)
        ]
        observations = self._drain(results, 2 * rounds)
        for proc in workers:
            self._join(proc)

        for versions, active in observations:
            self.assertEqual(
                list(versions), list(range(1, len(versions) + 1))
            )
            if versions:
                self.assertEqual(active, versions[-1])
            else:
                self.assertIsNone(active)
        # Every writer's records landed exactly once; the sequence is whole.
        final = self.open_vault()
        total = writers * per_writer
        self.assertEqual(final.versions("shared"), list(range(1, total + 1)))
        self.assertEqual(final.active("shared"), total)

    @unittest.skipUnless(fcntl is not None, "fcntl file locks are required")
    def test_another_process_holding_the_lock_blocks_whole_transaction(self):
        self._seed_versions("k", 1)
        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = self._start(_hold_lock_worker, str(self.root), ready, release)
        self.assertTrue(ready.wait(timeout=10))

        results = multiprocessing.Queue()
        sealer = self._start(
            _seal_one_worker, str(self.root), "k", b"while-locked", results
        )
        # While the lock is held the seal cannot begin: no version appears
        # on disk and the worker stays blocked.
        sealer.join(timeout=0.5)
        self.assertTrue(sealer.is_alive())
        self.assertEqual(self.disk_manifest()["keys"]["k"]["versions"][-1]["version"], 1)

        release.set()
        self.assertEqual(self._drain(results, 1), [2])
        self._join(sealer)
        self._join(holder)
        # The transaction completed cleanly once the lock was free.
        self.assertEqual(self.open_vault().load("k", 2), b"while-locked")

    @unittest.skipUnless(fcntl is not None, "fcntl file locks are required")
    def test_other_processes_wait_until_a_full_record_is_persisted(self):
        # A slow seal (material write delayed) holds the lock for the whole
        # record; an inspector reading on that same lock never sees a state
        # with a half-written material file or a manifest that names a
        # version whose material is absent.
        self._seed_versions("k", 1)
        results = multiprocessing.Queue()
        inspector = self._start(
            _locked_disk_inspector_worker, str(self.root), 30, results
        )
        # Concurrent writers extend the version list while the inspector
        # repeatedly validates everything under the lock.
        writers = [
            self._start(
                _seal_worker,
                str(self.root),
                "k",
                [f"c-{w}-{i}".encode() for i in range(4)],
            )
            for w in range(3)
        ]
        reports = self._drain(results, 30)
        self._join(inspector)
        for proc in writers:
            self._join(proc)

        self.assertTrue(reports)
        self.assertTrue(
            all(report[0] == "ok" for report in reports),
            [r for r in reports if r[0] != "ok"],
        )
        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, 1 + 1 + 3 * 4)))
        for version in range(1, 1 + 1 + 3 * 4):
            final.load("k", version)  # every material is present and intact

    def test_reload_interleaved_with_seals_never_sees_half_old_half_new(self):
        # The locked reload either reads the complete old manifest or the
        # complete new one; here we assert every observer snapshot is one of
        # those coherent states by re-validating it against disk after.
        results = multiprocessing.Queue()
        rounds = 30
        workers = [
            self._start(_snapshot_worker, str(self.root), "mix", rounds, results)
            for _ in range(3)
        ]
        workers += [
            self._start(
                _seal_worker,
                str(self.root),
                "mix",
                [f"m-{i}".encode() for i in range(8)],
            )
            for _ in range(2)
        ]
        observations = self._drain(results, 3 * rounds)
        for proc in workers:
            self._join(proc)

        # Coherence check: every observed prefix is a valid 1..N prefix and
        # the active pointer, when the key exists, is the newest material.
        for versions, active in observations:
            self.assertEqual(versions, tuple(range(1, len(versions) + 1)))
            self.assertEqual(active, versions[-1] if versions else None)
        final = self.open_vault()
        self.assertEqual(final.versions("mix"), list(range(1, 17)))
        for version in range(1, 17):
            final.load("mix", version)

    def test_concurrent_derived_seals_keep_one_strict_sequence(self):
        # Multiple processes deriving with PBKDF2 must draw from one shared
        # strictly increasing sequence with no repeat or gap.
        procs, each = 4, 5
        workers = [
            self._start(
                _derive_worker,
                str(self.root),
                "derived",
                [f"pw-{p}-{i}".encode() for i in range(each)],
            )
            for p in range(procs)
        ]
        for proc in workers:
            self._join(proc)

        total = procs * each
        vault = self.open_vault()
        self.assertEqual(vault.versions("derived"), list(range(1, total + 1)))
        self.assertEqual(vault.active("derived"), total)
        expected = {
            hashlib.pbkdf2_hmac(
                "sha256",
                f"pw-{p}-{i}".encode(),
                b"worker-salt",
                100,
                dklen=16,
            )
            for p in range(procs)
            for i in range(each)
        }
        self.assertEqual(
            {vault.load("derived", v) for v in range(1, total + 1)}, expected
        )


class TestFailedReloadInvariants(VaultTestCase):
    """After a failed whole-vault reload: snapshot and disk both stay put."""

    def _snapshot_state(self, vault: Vault) -> tuple:
        return (
            vault.manifest(),
            [(k, vault.versions(k), vault.active(k), vault.revoked_versions(k))
             for k in sorted(vault.manifest()["keys"])],
        )

    def test_value_error_leaves_memory_snapshot_and_disk_unchanged(self):
        vault = self.open_vault()
        vault.seal("a", b"a1")
        vault.seal("a", b"a2")
        vault.seal("a", b"a3")
        vault.seal("b", b"b1")
        vault.revoke("a", 3)
        vault.set_active("a", 1)  # repoint at a non-revoked historical version

        files_before = {
            p.relative_to(self.root): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        memory_before = self._snapshot_state(vault)

        # Corrupt the revocation journal so whole-vault validation fails.
        self.journal_path().write_bytes(b"{broken json\n")
        # Snapshot the on-disk state as it stands when reload is attempted:
        # the corruption is the precondition, and the failed reload itself
        # must neither add nor remove nor repair anything.
        files_at_reload = {
            p.relative_to(self.root): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        with self.assertRaises(ValueError):
            vault.reload()

        # In-memory snapshot is untouched: keys already in hand read fine.
        self.assertEqual(vault.load("a", 1), b"a1")
        self.assertEqual(vault.load("a", 2), b"a2")
        self.assertEqual(vault.load("a", 3), b"a3")
        self.assertEqual(vault.load("b"), b"b1")
        self.assertEqual(vault.active("a"), 1)
        self.assertFalse(vault.is_revoked("a", 1))
        self.assertTrue(vault.is_revoked("a", 3))
        self.assertEqual(vault.revoked_versions("a"), [3])
        self.assertEqual(self._snapshot_state(vault), memory_before)

        # Disk gained and lost nothing across the failed reload: every file
        # is byte-for-byte what it was the instant reload was attempted.
        files_after = {
            p.relative_to(self.root): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        self.assertEqual(files_after, files_at_reload)
        # The corrupt journal is still exactly the corruption we wrote (the
        # failed reload did not repair or truncate it), and healthy files
        # are untouched relative to the pre-corruption state apart from it.
        self.assertEqual(self.journal_path().read_bytes(), b"{broken json\n")
        self.assertNotEqual(files_before, files_at_reload)  # precondition check

        # A second read/query after the failure matches verbatim.
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(self._snapshot_state(vault), memory_before)
        self.assertEqual(vault.load("a"), b"a1")
        self.assertEqual(vault.revoked_versions("a"), [3])

    def test_query_results_after_failure_match_results_before(self):
        vault = self.open_vault()
        for i in range(3):
            vault.seal("k", f"m{i}".encode())
        vault.revoke("k", 2)
        queries_before = {
            "versions": vault.versions("k"),
            "active": vault.active("k"),
            "revoked": vault.revoked_versions("k"),
            "load_active": vault.load("k"),
            "load_each": [vault.load("k", v) for v in (1, 2, 3)],
            "is_revoked": {v: vault.is_revoked("k", v) for v in (1, 2, 3)},
            "manifest": vault.manifest(),
        }

        self.manifest_path().write_bytes(b"not json at all")
        with self.assertRaises(ValueError):
            vault.reload()

        self.assertEqual(vault.versions("k"), queries_before["versions"])
        self.assertEqual(vault.active("k"), queries_before["active"])
        self.assertEqual(vault.revoked_versions("k"), queries_before["revoked"])
        self.assertEqual(vault.load("k"), queries_before["load_active"])
        self.assertEqual(
            [vault.load("k", v) for v in (1, 2, 3)],
            queries_before["load_each"],
        )
        self.assertEqual(
            {v: vault.is_revoked("k", v) for v in (1, 2, 3)},
            queries_before["is_revoked"],
        )
        self.assertEqual(vault.manifest(), queries_before["manifest"])


class TestLockFileSemantics(VaultTestCase):
    """The lock file is only a mutex: no key data, no role in validation."""

    @unittest.skipUnless(fcntl is not None, "fcntl file locks are required")
    def test_clearing_lock_file_does_not_break_reload_or_seal(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        lock_path = self.root / LOCK_NAME

        # Empty it out: it carries no data, so reload validates normally.
        lock_path.write_bytes(b"")
        vault.reload()
        self.assertEqual(vault.load("k"), b"one")

        # Even junk bytes in it are harmless; it is never parsed.
        lock_path.write_bytes(b"\x00\xffnot-a-lock-just-gibberish\n")
        vault.reload()
        self.assertEqual(vault.seal("k", b"two"), 2)
        self.assertEqual(vault.load("k", 2), b"two")
        # A brand-new handle opening the same vault validates fine too.
        reopened = Vault(self.root)
        reopened.reload()
        self.assertEqual(reopened.versions("k"), [1, 2])

    @unittest.skipUnless(fcntl is not None, "fcntl file locks are required")
    def test_other_process_acquires_lock_immediately_after_close(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.close()
        # No residual handle: another process takes the lock right away and
        # completes a full write without blocking.
        results = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_seal_one_worker,
            args=(str(self.root), "k", b"after-close", results),
        )
        proc.start()
        try:
            self.assertEqual(results.get(timeout=30), 2)
            proc.join(timeout=30)
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        self.assertEqual(proc.exitcode, 0)
        self.assertEqual(self.open_vault().load("k", 2), b"after-close")

    @unittest.skipUnless(fcntl is not None, "fcntl file locks are required")
    def test_lock_release_allows_queued_writer_then_local_write(self):
        # Hold the lock in another process, queue a writer behind it, then
        # release: the queued writer completes and a subsequent local
        # operation observes its record — close/reopen re-locks cleanly.
        local = self.open_vault()
        local.seal("k", b"base")
        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_hold_lock_worker, args=(str(self.root), ready, release)
        )
        holder.start()
        self.assertTrue(ready.wait(timeout=10))

        results = multiprocessing.Queue()
        queued = multiprocessing.Process(
            target=_seal_one_worker,
            args=(str(self.root), "k", b"queued", results),
        )
        queued.start()
        queued.join(timeout=0.5)
        self.assertTrue(queued.is_alive())  # blocked on the holder

        release.set()
        self.assertEqual(results.get(timeout=30), 2)
        queued.join(timeout=30)
        holder.join(timeout=30)
        self.assertEqual(queued.exitcode, 0)
        self.assertEqual(holder.exitcode, 0)

        # The local handle reacquires its lock after the holder is gone and
        # sees the queued writer's record.
        local.close()
        self.assertEqual(local.seal("k", b"local"), 3)
        self.assertEqual(local.load("k", 2), b"queued")
        self.assertEqual(local.load("k", 3), b"local")


class TestEntryValidationContract(VaultTestCase):
    """Pin the exact exception types at every entry point, with no new version."""

    def test_empty_key_id_rejected_at_every_validating_entry_point(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.derive_seal("k", b"pw", b"salt", 1, 8)
        # Seal, derive, revoke, repoint and every key-id-validating query
        # reject an empty id with ValueError.
        for call in (
            lambda: vault.seal("", b"m"),
            lambda: vault.derive_seal("", b"pw", b"salt", 1, 8),
            lambda: vault.revoke("", 1),
            lambda: vault.set_active("", 1),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
            lambda: vault.derivation(""),
        ):
            with self.assertRaises(ValueError, msg=call):
                call()
        # No new version was created by any rejected call.
        self.assertEqual(vault.versions("k"), [1, 2])

    def test_unknown_key_and_missing_version_raise_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.load("ghost")
        with self.assertRaises(KeyError):
            vault.active("ghost")
        with self.assertRaises(KeyError):
            vault.derivation("ghost")
        with self.assertRaises(KeyError):
            vault.is_revoked("ghost", 1)
        with self.assertRaises(KeyError):
            vault.load("k", 2)
        with self.assertRaises(KeyError):
            vault.derivation("k", 2)
        with self.assertRaises(KeyError):
            vault.is_revoked("k", 2)
        # Querying revoked versions of an unknown key is the one read that
        # returns an empty list rather than raising.
        self.assertEqual(vault.revoked_versions("ghost"), [])

    def test_double_revoke_and_repoint_at_active_raise_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)  # 2 is already active

    def test_non_bytes_inputs_and_non_integer_derivation_numbers(self):
        vault = self.open_vault()
        for bad in ("text", 1, None, object()):
            with self.assertRaises(TypeError, msg=f"material {bad!r}"):
                vault.seal("k", bad)
            with self.assertRaises(TypeError, msg=f"password {bad!r}"):
                vault.derive_seal("k", bad, b"salt", 1, 8)
            with self.assertRaises(TypeError, msg=f"salt {bad!r}"):
                vault.derive_seal("k", b"pw", bad, 1, 8)
        for bad in (1.0, True, "2", None):
            with self.assertRaises(TypeError, msg=f"iterations {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 8)
            with self.assertRaises(TypeError, msg=f"length {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)
        for bad in (0, -5):
            with self.assertRaises(ValueError, msg=f"iterations {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 8)
            with self.assertRaises(ValueError, msg=f"length {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 8)
        # Nothing rejected above created a key or a version.
        self.assertEqual(vault.versions("k"), [])


class TestDeriveSeal(VaultTestCase):
    PASSWORD = b"correct horse battery staple"
    SALT = b"\x00\x01\x02 salty bytes \xff"
    ITERATIONS = 2000
    LENGTH = 48

    def test_derive_seal_returns_next_version_alongside_plain_seals(self):
        vault = self.open_vault()
        self.assertEqual(vault.seal("k", b"plain-one"), 1)
        self.assertEqual(
            vault.derive_seal("k", self.PASSWORD, self.SALT, self.ITERATIONS, 16),
            2,
        )
        self.assertEqual(vault.seal("k", b"plain-two"), 3)
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        self.assertEqual(vault.active("k"), 3)

    def test_stored_material_is_pbkdf2_output_byte_for_byte(self):
        vault = self.open_vault()
        version = vault.derive_seal(
            "k", self.PASSWORD, self.SALT, self.ITERATIONS, self.LENGTH
        )
        expected = hashlib.pbkdf2_hmac(
            "sha256", self.PASSWORD, self.SALT, self.ITERATIONS, dklen=self.LENGTH
        )
        self.assertEqual(vault.load("k", version), expected)
        self.assertEqual(len(vault.load("k", version)), self.LENGTH)

    def test_readback_matches_fresh_rederivation_with_same_parameters(self):
        vault = self.open_vault()
        vault.derive_seal(
            "k", self.PASSWORD, self.SALT, self.ITERATIONS, self.LENGTH
        )
        # No version: the active version is queried and re-derived.
        record = vault.derivation("k")
        rederived = hashlib.pbkdf2_hmac(
            "sha256",
            self.PASSWORD,
            record["salt"],
            record["iterations"],
            dklen=record["length"],
        )
        self.assertEqual(vault.load("k"), rederived)

    def test_derivation_returns_salt_iterations_and_length(self):
        vault = self.open_vault()
        version = vault.derive_seal(
            "k", self.PASSWORD, self.SALT, self.ITERATIONS, self.LENGTH
        )
        self.assertEqual(
            vault.derivation("k", version),
            {"salt": self.SALT, "iterations": self.ITERATIONS, "length": self.LENGTH},
        )
        # Defaults to the active version.
        self.assertEqual(vault.derivation("k"), vault.derivation("k", version))

    def test_directly_sealed_version_has_empty_derivation_record(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        self.assertEqual(vault.derivation("k"), {})
        vault.derive_seal("k", self.PASSWORD, self.SALT, 100, 16)
        # The derived version keeps its record; the plain one stays empty.
        self.assertEqual(vault.derivation("k", 1), {})
        self.assertTrue(vault.derivation("k", 2))
        # Active is the derived version now.
        self.assertEqual(vault.derivation("k")["iterations"], 100)

    def test_derivation_persists_through_reload_and_reopen(self):
        vault = self.open_vault()
        version = vault.derive_seal(
            "k", self.PASSWORD, self.SALT, self.ITERATIONS, self.LENGTH
        )
        vault.reload()
        self.assertEqual(
            vault.derivation("k", version)["salt"], self.SALT
        )
        reopened = self.open_vault()
        record = reopened.derivation("k", version)
        self.assertEqual(
            record,
            {"salt": self.SALT, "iterations": self.ITERATIONS, "length": self.LENGTH},
        )
        expected = hashlib.pbkdf2_hmac(
            "sha256", self.PASSWORD, self.SALT, self.ITERATIONS, dklen=self.LENGTH
        )
        self.assertEqual(reopened.load("k", version), expected)

    def test_derivation_returns_a_copy(self):
        vault = self.open_vault()
        vault.derive_seal("k", self.PASSWORD, self.SALT, 100, 16)
        record = vault.derivation("k")
        record["salt"] = b"tampered"
        record["iterations"] = 1
        self.assertEqual(vault.derivation("k")["salt"], self.SALT)
        self.assertEqual(vault.derivation("k")["iterations"], 100)

    def test_passphrase_never_reaches_disk(self):
        vault = self.open_vault()
        secret = b"a-passphrase-nobody-should-ever-see"
        vault.derive_seal("k", secret, self.SALT, 100, 32)
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != LOCK_NAME:
                self.assertNotIn(secret, path.read_bytes())

    def test_derivation_parameters_travel_with_manifest_record(self):
        vault = self.open_vault()
        vault.derive_seal("k", self.PASSWORD, self.SALT, 1234, 24)
        import base64

        record = self.disk_manifest()["keys"]["k"]["versions"][0]["derivation"]
        self.assertEqual(
            record,
            {
                "salt": base64.b64encode(self.SALT).decode("ascii"),
                "iterations": 1234,
                "length": 24,
            },
        )


class TestDerivationErrors(VaultTestCase):
    SALT = b"salty"

    def test_derive_seal_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.derive_seal("", b"pw", self.SALT, 1, 1)

    def test_password_or_salt_non_bytes_raises_type_error(self):
        vault = self.open_vault()
        for bad_password in ("text", 123, None, [b"x"], bytearray(b"x"), memoryview(b"x")):
            with self.assertRaises(TypeError, msg=repr(bad_password)):
                vault.derive_seal("k", bad_password, self.SALT, 1, 1)
        for bad_salt in ("text", 123, None, ["s"], bytearray(b"s"), memoryview(b"s")):
            with self.assertRaises(TypeError, msg=repr(bad_salt)):
                vault.derive_seal("k", b"pw", bad_salt, 1, 1)

    def test_empty_salt_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 1)

    def test_iterations_or_length_non_integer_raises_type_error(self):
        vault = self.open_vault()
        for bad in (True, False, 1.0, 2.5, "1", None):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derive_seal("k", b"pw", self.SALT, bad, 1)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derive_seal("k", b"pw", self.SALT, 1, bad)

    def test_iterations_or_length_below_one_raises_value_error(self):
        vault = self.open_vault()
        for bad in (0, -1, -1000):
            with self.assertRaises(ValueError, msg=repr(bad)):
                vault.derive_seal("k", b"pw", self.SALT, bad, 1)
            with self.assertRaises(ValueError, msg=repr(bad)):
                vault.derive_seal("k", b"pw", self.SALT, 1, bad)

    def test_rejected_derive_seal_writes_nothing(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        before = self.disk_manifest()
        bad_calls = (
            lambda: vault.derive_seal("", b"pw", self.SALT, 1, 1),
            lambda: vault.derive_seal("k", "pw", self.SALT, 1, 1),
            lambda: vault.derive_seal("k", b"pw", "salt", 1, 1),
            lambda: vault.derive_seal("k", b"pw", b"", 1, 1),
            lambda: vault.derive_seal("k", b"pw", self.SALT, True, 1),
            lambda: vault.derive_seal("k", b"pw", self.SALT, 1, 0),
        )
        for call in bad_calls:
            with self.assertRaises((TypeError, ValueError)):
                call()
        self.assertEqual(self.disk_manifest(), before)
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.load("k"), b"v1")

    def test_derivation_query_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.derivation("")

    def test_derivation_query_non_integer_version_raises_type_error(self):
        vault = self.open_vault()
        vault.derive_seal("k", b"pw", self.SALT, 1, 1)
        # None is the documented "active version" sentinel, not an error.
        self.assertIsNotNone(vault.derivation("k", None))
        for bad in (1.0, True, False, "1", 2.5, (1,)):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derivation("k", bad)

    def test_derivation_query_unknown_key_or_version_raises_key_error(self):
        vault = self.open_vault()
        vault.derive_seal("k", b"pw", self.SALT, 1, 8)
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed")
        for bad_version in (0, 2, 99):
            with self.assertRaises(KeyError, msg=repr(bad_version)):
                vault.derivation("k", bad_version)


class TestDerivationReloadValidation(VaultTestCase):
    SALT = b"salty-salt"

    def _fresh_vault(self) -> tuple[Vault, Path]:
        root = Path(self._tmp.name) / f"case-{next(self._counter)}"
        return Vault(root), root

    def setUp(self) -> None:
        super().setUp()
        self._counter = iter(range(1000))

    def _derive_one(self) -> tuple[Vault, int, Path]:
        vault, root = self._fresh_vault()
        version = vault.derive_seal("k", b"pw", self.SALT, 1000, 32)
        return vault, version, root

    def _tamper(self, root: Path, mutate) -> None:
        path = root / MANIFEST_NAME
        manifest = json.loads(path.read_bytes().decode("utf-8"))
        record = manifest["keys"]["k"]["versions"][0]
        mutate(record.setdefault("derivation", {}))
        path.write_bytes(_dump_manifest(manifest))

    def test_declared_length_smaller_than_material_is_rejected(self):
        vault, version, root = self._derive_one()
        self._tamper(root, lambda d: d.update(length=16))
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.versions("k"), [version])
        self.assertEqual(vault.derivation("k")["length"], 32)

    def test_declared_length_larger_than_material_is_rejected(self):
        vault, version, root = self._derive_one()
        self._tamper(root, lambda d: d.update(length=48))
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.versions("k"), [version])

    def test_missing_or_empty_salt_is_rejected(self):
        for mutate in (
            lambda d: d.pop("salt"),
            lambda d: d.update(salt=""),
            lambda d: d.update(salt=123),
            lambda d: d.update(salt="!!!not-base64!!!"),
        ):
            vault, _, root = self._derive_one()
            self._tamper(root, mutate)
            with self.assertRaises(ValueError):
                vault.reload()
            self.assertEqual(
                vault.load("k"), _derive_material(b"pw", self.SALT, 1000, 32)
            )

    def test_illegal_parameters_are_rejected(self):
        bad_values = (0, -1, 1.5, "1000", True, False, None)
        for bad in bad_values:
            vault, _, root = self._derive_one()
            self._tamper(root, lambda d, b=bad: d.update(iterations=b))
            with self.assertRaises(ValueError, msg=f"iterations={bad!r}"):
                vault.reload()
            self.assertEqual(vault.derivation("k")["iterations"], 1000)
        for bad in bad_values:
            vault, _, root = self._derive_one()
            self._tamper(root, lambda d, b=bad: d.update(length=b))
            with self.assertRaises(ValueError, msg=f"length={bad!r}"):
                vault.reload()
            self.assertEqual(vault.derivation("k")["length"], 32)

    def test_non_object_derivation_is_rejected(self):
        for bad in ("x", 123, ["salt"]):
            vault, _, root = self._derive_one()
            path = root / MANIFEST_NAME
            manifest = json.loads(path.read_bytes().decode("utf-8"))
            manifest["keys"]["k"]["versions"][0]["derivation"] = bad
            path.write_bytes(_dump_manifest(manifest))
            with self.assertRaises(ValueError, msg=repr(bad)):
                vault.reload()

    def test_failed_reload_touches_no_disk_records(self):
        vault, version, root = self._derive_one()
        vault.seal("k", b"plain")
        materials = sorted(
            p.read_bytes() for p in (root / MATERIALS_DIR).rglob("*.bin")
        )
        self._tamper(root, lambda d: d.update(length=1))
        # Snapshot the on-disk state as it stands at reload time: a failed
        # reload must neither repair nor remove anything on disk.
        manifest_path = root / MANIFEST_NAME
        manifest_before = manifest_path.read_bytes()
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            sorted(p.read_bytes() for p in (root / MATERIALS_DIR).rglob("*.bin")),
            materials,
        )
        # Keys already in hand stay readable and the snapshot is untouched.
        self.assertEqual(vault.versions("k"), [version, 2])
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.derivation("k", 2), {})


class TestClose(VaultTestCase):
    def test_close_releases_lock_handle_and_is_idempotent(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        self.assertIsNotNone(vault._lock_fh)
        vault.close()
        self.assertIsNone(vault._lock_fh)
        vault.close()  # repeated release is not an error
        self.assertIsNone(vault._lock_fh)

    def test_operation_after_close_reopens_lock_with_same_behaviour(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.close()
        self.assertEqual(vault.seal("k", b"two"), 2)
        self.assertIsNotNone(vault._lock_fh)
        self.assertEqual(vault.load("k", 1), b"one")
        self.assertEqual(vault.load("k"), b"two")
        vault.close()
        self.assertEqual(
            vault.derive_seal("k2", b"pw", b"salt", 100, 16), 1
        )
        vault.reload()
        self.assertEqual(vault.active("k"), 2)
        vault.close()

    def test_close_and_concurrent_handle_still_serialised(self):
        first = self.open_vault()
        second = self.open_vault()
        first.close()
        first.seal("k", b"from-first")
        second.reload()
        self.assertEqual(second.load("k"), b"from-first")


class TestWindowsLockFallback(VaultTestCase):
    """Force the ``msvcrt.locking`` branch used on Windows.

    ``fcntl`` is patched away and a tiny in-process fake ``msvcrt`` stands
    in for the C runtime byte-range locks, so the Windows fallback branch
    (acquire, timed-out retry and release) is exercised on every platform.
    """

    def _fake_msvcrt(self) -> tuple[types.ModuleType, dict]:
        module = types.ModuleType("msvcrt")
        module.LK_LOCK = 1
        module.LK_NBLCK = 2
        module.LK_UNLCK = 3
        state = {"held": {}, "guard": threading.Lock(), "contended": 0}

        def fake_locking(fd, mode, nbytes):
            st = os.fstat(fd)
            key = (st.st_dev, st.st_ino)
            with state["guard"]:
                if mode == module.LK_UNLCK:
                    state["held"].pop(key, None)
                else:
                    owner = state["held"].get(key)
                    if owner is not None and owner != threading.get_ident():
                        state["contended"] += 1
                        raise OSError("locking delay")
                    state["held"][key] = threading.get_ident()

        module.locking = fake_locking
        return module, state

    def test_fallback_acquire_and_release_path(self):
        import keyvault_ledger.vault as vault_mod

        fake, state = self._fake_msvcrt()
        with mock.patch.object(vault_mod, "fcntl", None):
            with mock.patch.dict(sys.modules, {"msvcrt": fake}):
                vault = Vault(self.root)
                self.assertEqual(vault.seal("k", b"m"), 1)
                vault.derive_seal("k", b"pw", b"salt", 10, 8)
                vault.seal("k", b"m2")
                vault.set_active("k", 1)
                vault.revoke("k", 1)
                vault.reload()
                self.assertEqual(vault.active("k"), 1)
                self.assertEqual(state["held"], {})
                vault.close()
        self.assertEqual(Vault(self.root).load("k", 1), b"m")

    def test_fallback_retries_until_lock_is_released(self):
        import keyvault_ledger.vault as vault_mod

        fake, state = self._fake_msvcrt()
        with mock.patch.object(vault_mod, "fcntl", None):
            with mock.patch.dict(sys.modules, {"msvcrt": fake}):
                # Both handles exist before anyone holds the lock, so their
                # construction never contends.
                holder_vault = Vault(self.root)
                waiter = Vault(self.root)
                holder_vault.seal("k", b"seed")

                holder_ready = threading.Event()
                release_holder = threading.Event()
                holder_error: list[BaseException] = []

                def hold() -> None:
                    try:
                        # The file-lock helper's contract requires the
                        # caller to already hold the in-process RLock, so
                        # take it explicitly before entering _file_lock().
                        # Only the inter-process file lock is simulated by
                        # the fake msvcrt; the sealer uses a different
                        # vault instance and thus a different RLock.
                        with holder_vault._lock:
                            with holder_vault._file_lock():
                                holder_ready.set()
                                release_holder.wait(timeout=5)
                    except BaseException as exc:  # captured below
                        holder_error.append(exc)

                holder = threading.Thread(target=hold)
                holder.start()
                self.assertTrue(holder_ready.wait(timeout=5))

                seal_done = threading.Event()

                def do_seal() -> None:
                    # The Windows fallback must keep retrying (LK_LOCK raises
                    # while the byte range is locked) rather than crashing or
                    # proceeding unlocked.
                    waiter.seal("k", b"waited")
                    seal_done.set()

                sealer = threading.Thread(target=do_seal)
                sealer.start()
                # Holder keeps the lock well past the first 0.05s retry, so
                # the sealer is guaranteed blocked, having seen contention.
                sealer.join(timeout=1.0)
                self.assertTrue(sealer.is_alive())
                self.assertGreaterEqual(state["contended"], 1)

                release_holder.set()
                self.assertTrue(seal_done.wait(timeout=5))
                holder.join(timeout=5)

        self.assertFalse(holder.is_alive())
        self.assertEqual(holder_error, [])
        self.assertEqual(waiter.load("k"), b"waited")
        self.assertEqual(state["held"], {})


class CliTestCase(VaultTestCase):
    def run_cli(self, *args: str, root: Path | str | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "keyvault_ledger",
                "--root",
                str(self.root if root is None else root),
                *args,
            ],
            capture_output=True,
            text=True,
            env=env,
        )

    def material_file(self, name: str, data: bytes) -> Path:
        path = Path(self._tmp.name) / name
        path.write_bytes(data)
        return path


class TestCli(CliTestCase):

    def test_seal_versions_reload_round_trip(self):
        material = Path(self._tmp.name) / "material.bin"
        material.write_bytes(b"cli-material")

        result = self.run_cli("seal", "cli-key", "--material-file", str(material))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1")

        result = self.run_cli("seal", "cli-key", "--material-file", str(material))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2")

        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cli-key", result.stdout)
        self.assertIn("active=2", result.stdout)
        self.assertIn("versions=1,2", result.stdout)

        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_versions_on_fresh_vault_is_empty(self):
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_cli_reports_errors(self):
        material = Path(self._tmp.name) / "material.bin"
        material.write_bytes(b"x")
        result = self.run_cli("seal", "", "--material-file", str(material))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error", result.stderr.lower())


def _material_rel_path(root: Path, key_id: str, version: int) -> str:
    manifest = json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))
    for record in manifest["keys"][key_id]["versions"]:
        if record["version"] == version:
            return record["file"]
    raise KeyError((key_id, version))


class TestCliFrozenOutputs(CliTestCase):
    """Byte-for-byte frozen baselines for the three README entry points.

    Every case spawns the real ``python -m keyvault_ledger`` process in a
    temporary vault and pins stdout, stderr and the exit code exactly,
    including whitespace and the trailing newline.
    """

    def test_versions_on_a_brand_new_empty_root(self):
        # A genuinely empty/absent vault directory is initialised on open,
        # so ``versions`` succeeds and prints nothing at all.
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def test_reload_on_a_brand_new_empty_root(self):
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "reloaded\n")
        self.assertEqual(result.stderr, "")

    def test_seal_prints_the_version_number_with_a_single_newline(self):
        material = self.material_file("m.bin", b"cli-material")
        first = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stdout, "1\n")
        self.assertEqual(first.stderr, "")
        second = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(second.returncode, 0)
        self.assertEqual(second.stdout, "2\n")
        self.assertEqual(second.stderr, "")

    def test_versions_starts_at_one_and_increases_strictly(self):
        material = self.material_file("m.bin", b"m")
        for expected in ("1", "2", "3"):
            result = self.run_cli("seal", "k", "--material-file", str(material))
            self.assertEqual(result.stdout, f"{expected}\n")
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "k\tactive=3\tversions=1,2,3\n")
        self.assertEqual(result.stderr, "")

    def test_versions_lines_are_sorted_and_tab_separated(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "alpha", "--material-file", str(material))
        self.run_cli("seal", "alpha", "--material-file", str(material))
        self.run_cli("seal", "beta", "--material-file", str(material))
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            result.stdout,
            "alpha\tactive=2\tversions=1,2\n"
            "beta\tactive=1\tversions=1\n",
        )
        self.assertEqual(result.stderr, "")

    def test_reload_success_output_is_frozen(self):
        self.run_cli("reload")
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "reloaded\n")
        self.assertEqual(result.stderr, "")


class TestCliSealErrors(CliTestCase):
    def test_empty_key_id_is_rejected_byte_for_byte(self):
        material = self.material_file("m.bin", b"m")
        result = self.run_cli("seal", "", "--material-file", str(material))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "error: key_id must not be empty\n")

    def test_empty_key_id_creates_no_new_version(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        before = self.run_cli("versions")
        manifest_before = self.manifest_path().read_bytes()

        result = self.run_cli("seal", "", "--material-file", str(material))
        self.assertEqual(result.returncode, 1)

        after = self.run_cli("versions")
        self.assertEqual(after.stdout, before.stdout)
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)

    def test_missing_material_file_is_rejected_with_its_path(self):
        missing = Path(self._tmp.name) / "absent.bin"
        result = self.run_cli("seal", "k", "--material-file", str(missing))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            f"error: [Errno 2] No such file or directory: '{missing}'\n",
        )

    def test_material_file_that_is_a_directory_is_rejected(self):
        directory = Path(self._tmp.name) / "a-dir"
        directory.mkdir()
        result = self.run_cli("seal", "k", "--material-file", str(directory))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr, f"error: [Errno 21] Is a directory: '{directory}'\n"
        )

    @unittest.skipUnless(
        hasattr(os, "geteuid") and os.geteuid() != 0,
        "permission bits are not enforced for root",
    )
    def test_unreadable_material_file_is_rejected(self):
        locked = self.material_file("locked.bin", b"secret")
        locked.chmod(0o000)
        self.addCleanup(lambda: locked.chmod(0o644))
        result = self.run_cli("seal", "k", "--material-file", str(locked))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            f"error: [Errno 13] Permission denied: '{locked}'\n",
        )

    def test_failed_seal_leaves_no_half_version_and_manifest_only_grew(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        listing_before = self.run_cli("versions")
        manifest_before = self.manifest_path().read_bytes()
        materials_before = sorted(
            path.read_bytes()
            for path in (self.root / MATERIALS_DIR).rglob("*.bin")
        )

        missing = Path(self._tmp.name) / "absent.bin"
        failed_missing = self.run_cli(
            "seal", "k", "--material-file", str(missing)
        )
        failed_empty = self.run_cli(
            "seal", "", "--material-file", str(material)
        )
        self.assertEqual((failed_missing.returncode, failed_empty.returncode), (1, 1))

        listing_after = self.run_cli("versions")
        self.assertEqual(listing_after.returncode, 0)
        self.assertEqual(listing_after.stdout, listing_before.stdout)
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)
        materials_after = sorted(
            path.read_bytes()
            for path in (self.root / MATERIALS_DIR).rglob("*.bin")
        )
        self.assertEqual(materials_after, materials_before)

        # The next successful seal takes version 2, not a reused number.
        again = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(again.stdout, "2\n")


class TestCliManifestMissing(CliTestCase):
    def _assert_manifest_missing(self, root: Path, material: Path) -> None:
        expected = f"error: manifest missing: {root / MANIFEST_NAME}\n"
        for argv in (
            ("versions",),
            ("reload",),
            ("seal", "k", "--material-file", str(material)),
        ):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)

    def test_populated_directory_without_manifest_errors_for_all_entries(self):
        # Baseline: opening a directory that already holds unrelated content
        # but no manifest is treated as corruption, not as a new vault.  The
        # missing-manifest error is frozen as-is; the CLI does not silently
        # create a manifest to repair it.
        root = Path(self._tmp.name) / "stray"
        root.mkdir()
        (root / "notes.txt").write_bytes(b"not a vault")
        material = self.material_file("m.bin", b"m")

        self._assert_manifest_missing(root, material)
        self.assertFalse((root / MANIFEST_NAME).exists())

    def test_initialized_vault_with_manifest_deleted_errors_for_all_entries(
        self,
    ):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        self.manifest_path().unlink()

        self._assert_manifest_missing(self.root, material)
        self.assertFalse(self.manifest_path().exists())


class TestCliReloadFailure(CliTestCase):
    def test_missing_material_fails_and_does_not_touch_disk(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        rel = _material_rel_path(self.root, "k", 1)
        material_path = self.root / rel
        material_path.unlink()
        manifest_before = self.manifest_path().read_bytes()

        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr, "error: material missing for key 'k' version 1\n"
        )
        # Reload never writes: the manifest is byte-identical and stays.
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)

    def test_failed_reload_is_deterministic_and_versions_match_after_repair(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        healthy = self.run_cli("versions")
        self.assertEqual(healthy.stdout, "k\tactive=1\tversions=1\n")

        rel = _material_rel_path(self.root, "k", 1)
        material_path = self.root / rel
        material_path.unlink()
        manifest_before = self.manifest_path().read_bytes()

        first = self.run_cli("reload")
        second = self.run_cli("reload")
        # Repeating the same failing input gives identical output and code.
        self.assertEqual(first.returncode, 1)
        self.assertEqual(second.returncode, 1)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stderr, second.stderr)

        # A fresh process re-validates on open, so versions reports the same
        # failure while the on-disk manifest stays exactly where it was.
        listing = self.run_cli("versions")
        self.assertEqual(listing.returncode, 1)
        self.assertEqual(listing.stdout, "")
        self.assertEqual(listing.stderr, first.stderr)
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)

        # Once the material is restored, versions is byte-identical to what
        # it showed before the failed reloads.
        material_path.write_bytes(b"m")
        recovered = self.run_cli("versions")
        self.assertEqual(recovered.returncode, 0)
        self.assertEqual(recovered.stdout, healthy.stdout)
        self.assertEqual(recovered.stderr, "")
        reloaded = self.run_cli("reload")
        self.assertEqual(reloaded.returncode, 0)
        self.assertEqual(reloaded.stdout, "reloaded\n")

    def test_material_mismatch_fails_without_touching_disk(self):
        material = self.material_file("m.bin", b"original")
        self.run_cli("seal", "k", "--material-file", str(material))
        rel = _material_rel_path(self.root, "k", 1)
        material_path = self.root / rel
        material_path.write_bytes(b"tampered")
        manifest_before = self.manifest_path().read_bytes()

        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr, "error: material mismatch for key 'k' version 1\n"
        )
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)

    def test_unsupported_manifest_format_is_rejected_for_reload_and_versions(
        self,
    ):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        manifest = self.disk_manifest()
        manifest["format"] = 999
        self.manifest_path().write_text(json.dumps(manifest))
        frozen = "error: manifest format invalid: unsupported format\n"

        reload_result = self.run_cli("reload")
        self.assertEqual(reload_result.returncode, 1)
        self.assertEqual(reload_result.stdout, "")
        self.assertEqual(reload_result.stderr, frozen)

        versions_result = self.run_cli("versions")
        self.assertEqual(versions_result.returncode, 1)
        self.assertEqual(versions_result.stdout, "")
        self.assertEqual(versions_result.stderr, frozen)


class TestCliVaultDirectoryProblems(CliTestCase):
    def test_root_that_is_a_file_is_rejected_by_every_entry(self):
        material = self.material_file("m.bin", b"m")
        root = Path(self._tmp.name) / "a-file"
        root.write_bytes(b"not a directory")
        expected = f"error: [Errno 17] File exists: '{root}'\n"
        for argv in (
            ("versions",),
            ("reload",),
            ("seal", "k", "--material-file", str(material)),
        ):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)

    @unittest.skipUnless(
        hasattr(os, "geteuid") and os.geteuid() != 0,
        "permission bits are not enforced for root",
    )
    def test_root_under_a_non_writable_parent_is_rejected(self):
        parent = Path(self._tmp.name) / "ro-parent"
        parent.mkdir()
        parent.chmod(0o555)
        self.addCleanup(lambda: parent.chmod(0o755))
        root = parent / "vault"
        material = self.material_file("m.bin", b"m")

        result = self.run_cli(
            "seal", "k", "--material-file", str(material), root=root
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr, f"error: [Errno 13] Permission denied: '{root}'\n"
        )

    @unittest.skipUnless(
        hasattr(os, "geteuid") and os.geteuid() != 0,
        "permission bits are not enforced for root",
    )
    def test_read_only_initialized_vault_reads_work_but_seal_fails(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        self.root.chmod(0o555)
        self.addCleanup(lambda: self.root.chmod(0o755))

        versions = self.run_cli("versions")
        self.assertEqual(versions.returncode, 0)
        self.assertEqual(versions.stdout, "k\tactive=1\tversions=1\n")
        self.assertEqual(versions.stderr, "")

        reloaded = self.run_cli("reload")
        self.assertEqual(reloaded.returncode, 0)
        self.assertEqual(reloaded.stdout, "reloaded\n")
        self.assertEqual(reloaded.stderr, "")

        sealed = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(sealed.returncode, 1)
        self.assertEqual(sealed.stdout, "")
        # The temp file name carries the process id, so pin the fixed text
        # around it rather than the pid.
        self.assertRegex(
            sealed.stderr,
            r"^error: \[Errno 13\] Permission denied: '"
            + re.escape(str(self.root / "manifest.json.tmp."))
            + r"\d+'\n$",
        )
        # The failed write added no version.
        still = self.run_cli("versions")
        self.assertEqual(still.stdout, "k\tactive=1\tversions=1\n")


class TestCliDeterminism(CliTestCase):
    def _scripted_run(self, root: Path, missing: Path) -> list[tuple[int, str, str]]:
        material = Path(self._tmp.name) / "shared-material.bin"
        material.write_bytes(b"m")
        captured: list[tuple[int, str, str]] = []
        for argv in (
            ("versions",),
            ("reload",),
            ("seal", "alpha", "--material-file", str(material)),
            ("seal", "alpha", "--material-file", str(material)),
            ("seal", "beta", "--material-file", str(material)),
            ("versions",),
            ("reload",),
            ("seal", "alpha", "--material-file", str(missing)),
            ("seal", "", "--material-file", str(material)),
        ):
            result = self.run_cli(*argv, root=root)
            captured.append((result.returncode, result.stdout, result.stderr))
        return captured

    def test_same_inputs_in_two_fresh_vaults_yield_identical_outputs(self):
        missing = Path(self._tmp.name) / "absent.bin"
        root_a = Path(self._tmp.name) / "a"
        root_b = Path(self._tmp.name) / "b"
        self.assertEqual(
            self._scripted_run(root_a, missing),
            self._scripted_run(root_b, missing),
        )


class TestCliInvocationShape(CliTestCase):
    """The README entry-point names, arguments and call shape are frozen."""

    def test_missing_root_exits_two_with_frozen_usage(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "keyvault_ledger", "versions"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: keyvault_ledger [-h] --root ROOT "
            "{versions,seal,reload} ...\n"
            "keyvault_ledger: error: the following arguments are "
            "required: --root\n",
        )

    def test_unknown_subcommand_exits_two_with_frozen_usage(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "keyvault_ledger",
                "--root",
                str(self.root),
                "bogus",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: keyvault_ledger [-h] --root ROOT "
            "{versions,seal,reload} ...\n"
            "keyvault_ledger: error: argument command: invalid choice: "
            "'bogus' (choose from versions, seal, reload)\n",
        )

    def test_seal_without_material_file_exits_two_with_frozen_usage(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "keyvault_ledger",
                "--root",
                str(self.root),
                "seal",
                "k",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: keyvault_ledger seal [-h] --material-file MATERIAL_FILE "
            "key_id\n"
            "keyvault_ledger seal: error: the following arguments are "
            "required: --material-file\n",
        )

    def test_seal_without_key_id_or_material_exits_two_with_frozen_usage(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "keyvault_ledger",
                "--root",
                str(self.root),
                "seal",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: keyvault_ledger seal [-h] --material-file MATERIAL_FILE "
            "key_id\n"
            "keyvault_ledger seal: error: the following arguments are "
            "required: key_id, --material-file\n",
        )


if __name__ == "__main__":
    unittest.main()
