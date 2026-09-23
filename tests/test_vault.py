"""Tests for keyvault_ledger, runnable with:

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
    _atomic_write,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


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


if __name__ == "__main__":
    unittest.main()
