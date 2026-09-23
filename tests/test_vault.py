"""Tests for keyvault_ledger, runnable with:

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger import vault as vault_module
from keyvault_ledger.vault import (
    LOCK_NAME,
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
    _atomic_write,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _seal_worker(root: str, key_id: str, payloads: list[bytes]) -> None:
    vault = Vault(root)
    for payload in payloads:
        vault.seal(key_id, payload)


def _derive_worker(
    root: str, key_id: str, specs: list[tuple[bytes, bytes, int, int]]
) -> None:
    vault = Vault(root)
    for password, salt, iterations, length in specs:
        vault.derive_seal(key_id, password, salt, iterations, length)


def _reload_worker(root: str, count: int) -> None:
    vault = Vault(root)
    for _ in range(count):
        vault.reload()


def _revoke_worker(root: str, key_id: str, versions: list[int]) -> None:
    vault = Vault(root)
    for version in versions:
        vault.revoke(key_id, version)
        vault.reload()


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

    def test_concurrent_seal_and_derive_share_the_version_sequence(self):
        procs, each = 3, 4
        raw_payloads = {
            (p, i): f"seal-{p}-{i}".encode("utf-8")
            for p in range(procs)
            for i in range(each)
        }
        derive_specs = {
            (p, i): (
                f"pw-{p}-{i}".encode("utf-8"),
                f"salt-{p}-{i}".encode("utf-8"),
                100,
                16,
            )
            for p in range(procs)
            for i in range(each)
        }
        workers = [
            multiprocessing.Process(
                target=_seal_worker,
                args=(
                    str(self.root),
                    "shared",
                    [raw_payloads[(p, i)] for i in range(each)],
                ),
            )
            for p in range(procs)
        ]
        workers += [
            multiprocessing.Process(
                target=_derive_worker,
                args=(
                    str(self.root),
                    "shared",
                    [derive_specs[(p, i)] for i in range(each)],
                ),
            )
            for p in range(procs)
        ]
        workers += [
            multiprocessing.Process(target=_reload_worker, args=(str(self.root), 20))
        ]
        self._run_workers(workers)

        vault = self.open_vault()
        total = 2 * procs * each
        # Direct seals and derivations draw from the same locked version
        # sequence: nothing duplicated, skipped or reused.
        self.assertEqual(vault.versions("shared"), list(range(1, total + 1)))
        self.assertEqual(vault.active("shared"), total)

        seen_raw: set[bytes] = set()
        seen_derived: set[tuple[bytes, int, int, bytes]] = set()
        for version in vault.versions("shared"):
            material = vault.load("shared", version)
            record = vault.derivation("shared", version)
            if record:
                seen_derived.add(
                    (
                        record["salt"],
                        record["iterations"],
                        record["length"],
                        material,
                    )
                )
            else:
                seen_raw.add(material)
        self.assertEqual(seen_raw, set(raw_payloads.values()))
        self.assertEqual(
            seen_derived,
            {
                (
                    salt,
                    iterations,
                    length,
                    hashlib.pbkdf2_hmac(
                        "sha256", password, salt, iterations, length
                    ),
                )
                for password, salt, iterations, length in derive_specs.values()
            },
        )

        # A fresh handle revalidates every derivation record on disk.
        reloaded = self.open_vault()
        reloaded.reload()
        self.assertEqual(reloaded.versions("shared"), list(range(1, total + 1)))
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

    def test_lock_file_is_not_vault_content(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        self.assertTrue((self.root / LOCK_NAME).exists())
        # The lock file shows up neither as a key nor in validation.
        self.assertEqual(set(vault.manifest()["keys"]), {"k"})
        self.assertEqual(vault.versions("k"), [1])
        vault.reload()
        self.assertEqual(vault.load("k"), b"m")


class TestCli(VaultTestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "keyvault_ledger", "--root", str(self.root), *args],
            capture_output=True,
            text=True,
            env=env,
        )

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


class TestDeriveSeal(VaultTestCase):
    PASSWORD = b"correct horse battery staple"
    SALT = b"\x00\x01\x02 salty \xff\xfe"
    ITERATIONS = 2000
    LENGTH = 48

    def expected_material(self) -> bytes:
        return hashlib.pbkdf2_hmac(
            "sha256", self.PASSWORD, self.SALT, self.ITERATIONS, dklen=self.LENGTH
        )

    def test_derive_seal_returns_version_and_persists_parameters(self):
        vault = self.open_vault()
        self.assertEqual(
            vault.derive_seal(
                "k", self.PASSWORD, self.SALT, self.ITERATIONS, self.LENGTH
            ),
            1,
        )
        record = vault.derivation("k")
        self.assertEqual(
            record,
            {
                "salt": self.SALT,
                "iterations": self.ITERATIONS,
                "length": self.LENGTH,
            },
        )
        # The stored material is the standard-library derivation, byte exact.
        self.assertEqual(vault.load("k"), self.expected_material())

    def test_rederive_with_returned_parameters_matches_load(self):
        vault = self.open_vault()
        vault.derive_seal("k", self.PASSWORD, self.SALT, self.ITERATIONS, 32)
        record = vault.derivation("k")
        rederived = hashlib.pbkdf2_hmac(
            "sha256",
            self.PASSWORD,
            record["salt"],
            record["iterations"],
            dklen=record["length"],
        )
        self.assertEqual(rederived, vault.load("k"))

    def test_derivation_parameters_survive_reopen_and_reload(self):
        vault = self.open_vault()
        vault.derive_seal("k", self.PASSWORD, self.SALT, self.ITERATIONS, 16)

        reloaded = self.open_vault()
        self.assertEqual(
            reloaded.derivation("k", 1),
            {
                "salt": self.SALT,
                "iterations": self.ITERATIONS,
                "length": 16,
            },
        )
        self.assertEqual(reloaded.load("k"), self.expected_material()[:16])
        reloaded.reload()
        self.assertEqual(reloaded.derivation("k")["salt"], self.SALT)

    def test_directly_sealed_version_has_empty_derivation_record(self):
        vault = self.open_vault()
        vault.seal("k", b"raw")
        self.assertEqual(vault.derivation("k"), {})
        self.assertEqual(vault.derivation("k", 1), {})

    def test_mixed_history_shares_version_numbers(self):
        vault = self.open_vault()
        self.assertEqual(vault.seal("k", b"raw-1"), 1)
        self.assertEqual(
            vault.derive_seal("k", self.PASSWORD, self.SALT, 100, 20), 2
        )
        self.assertEqual(vault.seal("k", b"raw-3"), 3)
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        # The active version is the direct seal, so the default query is empty.
        self.assertEqual(vault.derivation("k"), {})
        self.assertEqual(vault.derivation("k", 1), {})
        self.assertEqual(vault.derivation("k", 3), {})
        middle = vault.derivation("k", 2)
        self.assertEqual(middle["salt"], self.SALT)
        self.assertEqual(middle["iterations"], 100)
        self.assertEqual(middle["length"], 20)
        # Historical material of each kind is intact.
        self.assertEqual(vault.load("k", 1), b"raw-1")
        self.assertEqual(
            vault.load("k", 2),
            hashlib.pbkdf2_hmac("sha256", self.PASSWORD, self.SALT, 100, 20),
        )
        self.assertEqual(vault.load("k", 3), b"raw-3")

    def test_manifest_carries_derivation_parameters_without_password(self):
        vault = self.open_vault()
        vault.derive_seal("k", self.PASSWORD, self.SALT, 1234, 32)
        record = self.disk_manifest()["keys"]["k"]["versions"][0]["derivation"]
        self.assertEqual(
            set(record), {"salt", "iterations", "length"}
        )
        self.assertEqual(record["iterations"], 1234)
        self.assertEqual(record["length"], 32)
        import base64

        self.assertEqual(base64.b64decode(record["salt"]), self.SALT)
        # The password never lands anywhere on disk.
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(self.PASSWORD, path.read_bytes())

    def test_reload_picks_up_derived_version_from_another_handle(self):
        first = self.open_vault()
        second = self.open_vault()
        first.derive_seal(
            "k", self.PASSWORD, self.SALT, self.ITERATIONS, self.LENGTH
        )
        with self.assertRaises(KeyError):
            second.load("k")
        second.reload()
        self.assertEqual(second.derivation("k")["length"], self.LENGTH)
        self.assertEqual(second.load("k"), self.expected_material())


class TestDerivationErrors(VaultTestCase):
    PASSWORD = b"pw"
    SALT = b"salt"

    def test_derive_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.derive_seal("", self.PASSWORD, self.SALT, 1, 1)

    def test_derive_non_bytes_password_or_salt_raises_type_error(self):
        vault = self.open_vault()
        for bad_password in ("pw", bytearray(b"pw"), memoryview(b"pw"), 1, None):
            with self.assertRaises(TypeError, msg=repr(bad_password)):
                vault.derive_seal("k", bad_password, self.SALT, 1, 1)
        for bad_salt in ("salt", bytearray(b"salt"), memoryview(b"salt"), 1, None):
            with self.assertRaises(TypeError, msg=repr(bad_salt)):
                vault.derive_seal("k", self.PASSWORD, bad_salt, 1, 1)

    def test_derive_empty_salt_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.derive_seal("k", self.PASSWORD, b"", 1, 1)

    def test_derive_non_integer_parameters_raise_type_error(self):
        vault = self.open_vault()
        for bad in (True, False, 1.0, 2.5, "1", None):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derive_seal("k", self.PASSWORD, self.SALT, bad, 1)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derive_seal("k", self.PASSWORD, self.SALT, 1, bad)

    def test_derive_non_positive_parameters_raise_value_error(self):
        vault = self.open_vault()
        for bad in (0, -1, -1000):
            with self.assertRaises(ValueError, msg=repr(bad)):
                vault.derive_seal("k", self.PASSWORD, self.SALT, bad, 1)
            with self.assertRaises(ValueError, msg=repr(bad)):
                vault.derive_seal("k", self.PASSWORD, self.SALT, 1, bad)

    def test_derivation_query_validation_matches_revocation_queries(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(ValueError):
            vault.derivation("", 1)
        with self.assertRaises(TypeError):
            vault.derivation("k", True)
        for bad in (1.0, "1", [1]):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derivation("k", bad)
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed")
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed", 1)
        with self.assertRaises(KeyError):
            vault.derivation("k", 2)
        with self.assertRaises(KeyError):
            vault.derivation("k", 0)

    def test_failed_derive_leaves_no_partial_version(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        before_disk = self.disk_manifest()
        materials_before = {
            p for p in (self.root / MATERIALS_DIR).rglob("*.bin")
        }
        for call in (
            lambda: vault.derive_seal("", self.PASSWORD, self.SALT, 1, 1),
            lambda: vault.derive_seal("k", "pw", self.SALT, 1, 1),
            lambda: vault.derive_seal("k", self.PASSWORD, "salt", 1, 1),
            lambda: vault.derive_seal("k", self.PASSWORD, b"", 1, 1),
            lambda: vault.derive_seal("k", self.PASSWORD, self.SALT, 0, 1),
            lambda: vault.derive_seal("k", self.PASSWORD, self.SALT, 1, 1.5),
        ):
            with self.assertRaises((ValueError, TypeError)):
                call()
        self.assertEqual(self.disk_manifest(), before_disk)
        self.assertEqual(
            {p for p in (self.root / MATERIALS_DIR).rglob("*.bin")},
            materials_before,
        )
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.load("k"), b"v1")
        # The rejected attempt neither consumed a version number nor moved
        # active.
        self.assertEqual(vault.derive_seal("k", self.PASSWORD, self.SALT, 1, 8), 2)
        self.assertEqual(vault.active("k"), 2)


class TestDerivationRecordValidation(VaultTestCase):
    PASSWORD = b"pw"
    SALT = b"salt"

    def test_reload_checks_derived_material_against_parameters(self):
        vault = self.open_vault()
        vault.derive_seal("k", self.PASSWORD, self.SALT, 100, 32)
        expected = vault.load("k")
        params = vault.derivation("k")
        valid_manifest = self.manifest_path().read_bytes()

        tamperings = [
            lambda d: d.update(length=31),
            lambda d: d.update(length=33),
            lambda d: d.pop("salt"),
            lambda d: d.update(salt=""),
            lambda d: d.update(salt="not base64!"),
            lambda d: d.update(salt="AA==ZZ"),
            lambda d: d.pop("iterations"),
            lambda d: d.update(iterations=0),
            lambda d: d.update(iterations="100"),
            lambda d: d.update(iterations=True),
            lambda d: d.pop("length"),
            lambda d: d.update(length=0),
            lambda d: d.update(length="32"),
        ]
        for tamper in tamperings:
            manifest = json.loads(valid_manifest.decode("utf-8"))
            tamper(manifest["keys"]["k"]["versions"][0]["derivation"])
            self.manifest_path().write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                vault.reload()
            # Validation failure is read-only: the handle still serves the
            # keys it held and the in-memory derivation snapshot is intact.
            self.assertEqual(vault.load("k"), expected)
            self.assertEqual(vault.derivation("k"), params)
            # Restore a valid record before the next tampering.
            self.manifest_path().write_bytes(valid_manifest)

    def test_reload_rejects_non_object_derivation_record(self):
        vault = self.open_vault()
        vault.derive_seal("k", self.PASSWORD, self.SALT, 100, 32)
        manifest = self.disk_manifest()
        manifest["keys"]["k"]["versions"][0]["derivation"] = ["salt"]
        self.manifest_path().write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(
            vault.derivation("k"),
            {"salt": self.SALT, "iterations": 100, "length": 32},
        )

    def test_reload_rejects_tampered_derived_material(self):
        vault = self.open_vault()
        vault.derive_seal("k", self.PASSWORD, self.SALT, 100, 32)
        before = self.manifest_path().read_bytes()
        entry = self.disk_manifest()["keys"]["k"]["versions"][0]
        (self.root / entry["file"]).write_bytes(b"tampered material")
        with self.assertRaises(ValueError):
            vault.reload()
        # Disk records are not added or removed and existing keys stay
        # readable from the in-memory snapshot.
        self.assertEqual(self.manifest_path().read_bytes(), before)
        self.assertEqual(
            vault.load("k"),
            hashlib.pbkdf2_hmac("sha256", self.PASSWORD, self.SALT, 100, 32),
        )


class TestClose(VaultTestCase):
    def test_close_is_idempotent(self):
        vault = self.open_vault()
        vault.close()
        vault.close()

    def test_operations_after_close_reopen_and_relock(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.close()
        # A write after close reopens the lock file and takes the lock.
        self.assertEqual(vault.derive_seal("k", b"pw", b"salt", 100, 16), 2)
        vault.close()
        # A read-only call reopens too; observable behaviour is unchanged.
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.versions("k"), [1, 2])
        vault.close()
        vault.reload()
        self.assertEqual(vault.derivation("k", 2)["length"], 16)
        vault.close()
        vault.revoke("k", 1)
        self.assertTrue(vault.is_revoked("k", 1))

    def test_close_releases_lock_handle(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        fh = vault._lock_fh
        self.assertFalse(fh.closed)
        vault.close()
        self.assertTrue(fh.closed)
        self.assertIsNone(vault._lock_fh)
        vault.seal("k", b"m2")
        self.assertFalse(vault._lock_fh.closed)


class TestWindowsLockFallback(VaultTestCase):
    """Regression coverage for the ``msvcrt.locking`` branch on non-Windows.

    The real branch only runs when ``fcntl`` is unavailable (Windows); here
    a fake ``msvcrt`` module is injected and ``fcntl`` temporarily hidden
    so the fallback's lock, retry-on-timeout, unlock and reopen paths all
    execute on this platform.
    """

    def _install_fake_msvcrt(self):
        fake = types.ModuleType("msvcrt")
        fake.LK_LOCK = 1
        fake.LK_UNLCK = 2
        state = {
            "locked": False,
            "calls": [],
            "forced_failures": 0,
        }

        def locking(fileno, mode, nbytes):
            if mode == fake.LK_LOCK:
                if state["forced_failures"] > 0:
                    # msvcrt signals a lock timeout as OSError; _file_lock
                    # sleeps and retries.
                    state["forced_failures"] -= 1
                    raise OSError("lock timed out")
                state["locked"] = True
                state["calls"].append(("lock", fileno))
            elif mode == fake.LK_UNLCK:
                if state["locked"]:
                    state["locked"] = False
                    state["calls"].append(("unlock", fileno))

        fake.locking = locking
        fake.state = state
        original_fcntl = vault_module.fcntl
        original_msvcrt = sys.modules.get("msvcrt")
        vault_module.fcntl = None
        sys.modules["msvcrt"] = fake
        slept = []
        sleep_patch = mock.patch.object(
            vault_module.time, "sleep", side_effect=lambda s: slept.append(s)
        )
        sleep_patch.start()

        def cleanup():
            vault_module.fcntl = original_fcntl
            if original_msvcrt is None:
                sys.modules.pop("msvcrt", None)
            else:
                sys.modules["msvcrt"] = original_msvcrt
            sleep_patch.stop()

        self.addCleanup(cleanup)
        return fake, state, slept

    def test_fallback_lock_guards_normal_operations(self):
        fake, state, slept = self._install_fake_msvcrt()
        vault = Vault(self.root)
        self.addCleanup(vault.close)
        vault.seal("k", b"v1")
        vault.derive_seal("k", b"pw", b"salt", 50, 8)
        vault.reload()
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.derivation("k", 2)["length"], 8)
        # Every acquisition was released, all via the msvcrt branch.
        self.assertIn(("lock", vault._lock_fh.fileno()), state["calls"])
        self.assertEqual(
            [kind for kind, _ in state["calls"]].count("lock"),
            [kind for kind, _ in state["calls"]].count("unlock"),
        )
        self.assertTrue(all(kind in ("lock", "unlock") for kind, _ in state["calls"]))
        # close() releases the handle; the next operation reopens and locks.
        vault.close()
        vault.seal("k", b"v3")
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        vault.close()

    def test_fallback_retries_after_lock_timeout(self):
        fake, state, slept = self._install_fake_msvcrt()
        holder = Vault(self.root)
        other = Vault(self.root)
        self.addCleanup(holder.close)
        self.addCleanup(other.close)
        state["forced_failures"] = 3

        release = threading.Event()
        acquired = threading.Event()

        def take_lock():
            with holder._file_lock():
                acquired.set()
                release.wait(timeout=5)

        holder_thread = threading.Thread(target=take_lock)
        holder_thread.start()
        self.assertTrue(acquired.wait(timeout=5))

        with holder._file_lock():
            # The first three attempts timed out (OSError); the context
            # manager retried after sleeping and the fourth succeeded.
            self.assertEqual(state["forced_failures"], 0)

        release.set()
        holder_thread.join(timeout=5)
        self.assertFalse(holder_thread.is_alive())
        self.assertEqual(slept, [0.05, 0.05, 0.05])


if __name__ == "__main__":
    unittest.main()
