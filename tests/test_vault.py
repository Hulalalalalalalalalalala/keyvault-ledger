"""Tests for keyvault_ledger, runnable with:

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from keyvault_ledger import Vault
from keyvault_ledger import vault as vault_module
from keyvault_ledger.vault import (
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
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

    def revocations_path(self) -> Path:
        return self.root / REVOCATIONS_NAME

    def disk_revocation_records(self) -> list[dict]:
        raw = self.revocations_path().read_bytes()
        return [
            json.loads(line)
            for line in raw.decode("utf-8").splitlines()
            if line
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
    def test_revoke_is_only_a_marker(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        manifest_before = self.manifest_path().read_bytes()

        vault.revoke("k", 1)

        self.assertTrue(vault.is_revoked("k", 1))
        self.assertFalse(vault.is_revoked("k", 2))
        # Material still readable, including the revoked version itself.
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.load("k", 2), b"v2")
        # Version list and active version are untouched.
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        self.assertEqual(vault.active("k"), 3)
        self.assertEqual(vault.load("k"), b"v3")
        # Neither the manifest nor historical material changed on disk.
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)

    def test_revoking_active_version_does_not_move_active(self):
        vault = self.open_vault()
        vault.seal("k", b"a")
        vault.seal("k", b"b")
        vault.revoke("k", 2)
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"b")

    def test_revoked_versions_are_sorted_ascending(self):
        vault = self.open_vault()
        for _ in range(4):
            vault.seal("k", b"m")
        vault.revoke("k", 3)
        vault.revoke("k", 1)
        vault.revoke("k", 4)
        self.assertEqual(vault.revoked_versions("k"), [1, 3, 4])
        self.assertTrue(vault.is_revoked("k", 3))
        self.assertFalse(vault.is_revoked("k", 2))

    def test_revoked_versions_empty_when_nothing_revoked(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        self.assertEqual(vault.revoked_versions("k"), [])

    def test_revoked_versions_unknown_key_returns_empty_list(self):
        vault = self.open_vault()
        self.assertEqual(vault.revoked_versions("never-sealed"), [])

    def test_revoke_writes_one_append_only_log_line(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        self.assertEqual(
            self.disk_revocation_records(),
            [{"key_id": "k", "version": 1}],
        )
        vault.seal("k", b"m2")
        vault.revoke("k", 2)
        # The log only ever grows: the first record is still there.
        self.assertEqual(
            self.disk_revocation_records(),
            [{"key_id": "k", "version": 1}, {"key_id": "k", "version": 2}],
        )

    def test_sealing_after_revoke_is_unaffected(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.revoke("k", 1)
        self.assertEqual(vault.seal("k", b"v2"), 2)
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.revoked_versions("k"), [1])
        self.assertFalse(vault.is_revoked("k", 2))


class TestRevocationErrors(VaultTestCase):
    def test_revoke_unknown_key_raises_key_error(self):
        vault = self.open_vault()
        with self.assertRaises(KeyError):
            vault.revoke("never-sealed", 1)

    def test_revoke_never_sealed_version_raises_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.revoke("k", 2)
        with self.assertRaises(KeyError):
            vault.revoke("k", 0)

    def test_revoke_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.revoke("", 1)

    def test_repeat_revoke_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        self.assertEqual(vault.revoked_versions("k"), [1])

    def test_failed_revoke_appends_nothing(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        for bad_call in (
            lambda: vault.revoke("", 1),
            lambda: vault.revoke("k", 9),
            lambda: vault.revoke("nope", 1),
        ):
            with self.assertRaises((ValueError, KeyError)):
                bad_call()
        self.assertEqual(self.revocations_path().read_bytes(), b"")
        self.assertEqual(vault.revoked_versions("k"), [])

    def test_is_revoked_unknown_key_raises_key_error(self):
        vault = self.open_vault()
        with self.assertRaises(KeyError):
            vault.is_revoked("never-sealed", 1)

    def test_is_revoked_unknown_version_raises_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.is_revoked("k", 2)

    def test_status_queries_empty_key_id_raise_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.is_revoked("", 1)
        with self.assertRaises(ValueError):
            vault.revoked_versions("")


class TestRevocationPersistence(VaultTestCase):
    def test_revocation_survives_reload_on_new_handle(self):
        first = self.open_vault()
        first.seal("k", b"v1")
        first.seal("k", b"v2")
        first.revoke("k", 1)

        second = self.open_vault()
        self.assertTrue(second.is_revoked("k", 1))
        self.assertFalse(second.is_revoked("k", 2))
        self.assertEqual(second.revoked_versions("k"), [1])
        # The marker hides nothing: full history still reads through.
        self.assertEqual(second.load("k", 1), b"v1")
        self.assertEqual(second.versions("k"), [1, 2])
        self.assertEqual(second.active("k"), 2)

    def test_reload_picks_up_revocation_from_another_handle(self):
        first = self.open_vault()
        second = self.open_vault()
        first.seal("k", b"m")
        first.revoke("k", 1)
        self.assertEqual(second.revoked_versions("k"), [])
        second.reload()
        self.assertTrue(second.is_revoked("k", 1))
        # The freshly loaded history also enforces at-most-once.
        with self.assertRaises(ValueError):
            second.revoke("k", 1)

    def test_reload_does_not_rewrite_revocations_log(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        before = self.revocations_path().read_bytes()
        before_mtime = self.revocations_path().stat().st_mtime_ns
        vault.reload()
        self.assertEqual(self.revocations_path().read_bytes(), before)
        self.assertEqual(self.revocations_path().stat().st_mtime_ns, before_mtime)

    def test_preexisting_vault_without_log_is_upgraded(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        # A vault directory written before revocations existed.
        self.revocations_path().unlink()

        reopened = self.open_vault()
        self.assertEqual(reopened.revoked_versions("k"), [])
        self.assertEqual(reopened.load("k"), b"m")
        reopened.revoke("k", 1)
        self.assertTrue(self.open_vault().is_revoked("k", 1))

    def test_corrupt_revocation_log_fails_reload_but_keeps_state(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        bad_logs = (
            b"{not json\n",
            b"123\n",
            json.dumps({"key_id": 7, "version": 1}).encode() + b"\n",
            json.dumps({"key_id": "k", "version": "1"}).encode() + b"\n",
            json.dumps({"key_id": "nope", "version": 1}).encode() + b"\n",
            json.dumps({"key_id": "k", "version": 2}).encode() + b"\n",
            (
                json.dumps({"key_id": "k", "version": 1}).encode()
                + b"\n"
                + json.dumps({"key_id": "k", "version": 1}).encode()
                + b"\n"
            ),
        )
        original = self.revocations_path().read_bytes()
        for bad in bad_logs:
            self.revocations_path().write_bytes(bad)
            with self.assertRaises(ValueError):
                vault.reload()
            # Failed reload changes nothing in memory, versions or markers.
            self.assertEqual(vault.load("k"), b"m")
            self.assertEqual(vault.versions("k"), [1])
            self.assertTrue(vault.is_revoked("k", 1))
            self.assertEqual(vault.revoked_versions("k"), [1])
            with self.assertRaises(ValueError):
                self.open_vault()
        self.revocations_path().write_bytes(original)
        vault.reload()
        self.assertTrue(vault.is_revoked("k", 1))

    def test_missing_revocation_log_fails_reload_but_keeps_state(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        self.revocations_path().unlink()
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"m")
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertEqual(vault.revoked_versions("k"), [1])
        # Once the log exists again, a successful reload reflects disk:
        # reopening a vault with a missing log seeds an empty history.
        self.open_vault()
        vault.reload()
        self.assertEqual(vault.revoked_versions("k"), [])

    def test_revocation_markers_independent_across_keys(self):
        vault = self.open_vault()
        vault.seal("a", b"1")
        vault.seal("b", b"1")
        vault.revoke("a", 1)
        self.assertEqual(vault.revoked_versions("a"), [1])
        self.assertEqual(vault.revoked_versions("b"), [])
        self.assertFalse(vault.is_revoked("b", 1))


class TestSealFailureCleanup(VaultTestCase):
    def test_failed_manifest_write_removes_fresh_material(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        manifest_before = self.disk_manifest()
        material_dir = (
            self.root
            / MATERIALS_DIR
            / vault_module._encode_key_id("k")
        )
        orphan = material_dir / "2.bin"

        real_atomic_write = vault_module._atomic_write

        def failing_write(path, data):
            if Path(path).name == MANIFEST_NAME:
                raise OSError("simulated disk full")
            return real_atomic_write(path, data)

        with mock.patch.object(
            vault_module, "_atomic_write", side_effect=failing_write
        ):
            with self.assertRaises(OSError):
                vault.seal("k", b"v2")

        # No orphan material, disk and memory both still show only version 1.
        self.assertFalse(orphan.exists())
        self.assertEqual(self.disk_manifest(), manifest_before)
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.load("k"), b"v1")

        # The failed attempt consumed no version number.
        self.assertEqual(vault.seal("k", b"v2-retry"), 2)
        self.assertEqual(vault.load("k", 2), b"v2-retry")
        self.assertEqual(sorted(p.name for p in material_dir.iterdir()), ["1.bin", "2.bin"])


class TestReloadSealConcurrency(VaultTestCase):
    def test_reload_reading_under_lock_cannot_restore_old_snapshot(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")

        entered_write = threading.Event()
        release_write = threading.Event()
        real_atomic_write = vault_module._atomic_write

        def blocking_write(path, data):
            if Path(path).name == MANIFEST_NAME:
                entered_write.set()
                self.assertTrue(release_write.wait(timeout=5))
            return real_atomic_write(path, data)

        errors: queue.Queue = queue.Queue()

        def seal() -> None:
            try:
                vault.seal("k", b"v2")
            except BaseException as exc:  # pragma: no cover - failure reporting
                errors.put(exc)

        with mock.patch.object(
            vault_module, "_atomic_write", side_effect=blocking_write
        ):
            seal_thread = threading.Thread(target=seal)
            seal_thread.start()
            self.assertTrue(entered_write.wait(timeout=5))

            # The seal holds the lock with its material written but the
            # manifest flip blocked; reload must wait instead of reading the
            # old manifest and swapping it back in afterwards.
            reload_thread = threading.Thread(
                target=self._reload, args=(vault, errors)
            )
            reload_thread.start()
            release_write.set()
            seal_thread.join(timeout=5)
            reload_thread.join(timeout=5)
            self.assertFalse(seal_thread.is_alive())
            self.assertFalse(reload_thread.is_alive())

        while not errors.empty():
            raise errors.get_nowait()
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.load("k", 2), b"v2")
        # A further seal must reuse no version number or overwrite material.
        self.assertEqual(vault.seal("k", b"v3"), 3)
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.load("k", 2), b"v2")
        self.assertEqual(vault.load("k", 3), b"v3")

    @staticmethod
    def _reload(vault, errors):
        try:
            vault.reload()
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.put(exc)


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
