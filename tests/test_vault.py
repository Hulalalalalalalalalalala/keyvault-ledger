"""Tests for keyvault_ledger, runnable with:

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import MANIFEST_NAME, MATERIALS_DIR

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
