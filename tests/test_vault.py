import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from keyvault_ledger import Vault


class VaultTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="vault-test-")) / "vault"
        self.addCleanup(shutil.rmtree, self.root.parent, ignore_errors=True)

    def open_vault(self):
        return Vault(self.root)

    def manifest_path(self):
        return self.root / "manifest.json"

    def write_manifest(self, data):
        if isinstance(data, (dict, list)):
            data = json.dumps(data).encode("utf-8")
        self.manifest_path().write_bytes(data)


class TestFreshVault(VaultTestCase):
    def test_missing_directory_is_created_on_open(self):
        self.assertFalse(self.root.exists())
        self.open_vault()
        self.assertTrue(self.root.is_dir())
        self.assertTrue(self.manifest_path().is_file())

    def test_fresh_vault_has_no_keys(self):
        vault = self.open_vault()
        self.assertEqual(vault.manifest()["keys"], {})
        self.assertEqual(vault.versions("anything"), [])


class TestSeal(VaultTestCase):
    def test_first_version_is_one(self):
        vault = self.open_vault()
        self.assertEqual(vault.seal("signing", b"v1"), 1)

    def test_versions_increase_strictly(self):
        vault = self.open_vault()
        versions = [vault.seal("k", bytes([i])) for i in range(5)]
        self.assertEqual(versions, [1, 2, 3, 4, 5])
        self.assertEqual(vault.versions("k"), [1, 2, 3, 4, 5])

    def test_history_is_kept(self):
        vault = self.open_vault()
        vault.seal("k", b"old")
        vault.seal("k", b"new")
        self.assertEqual(vault.load("k", 1), b"old")
        self.assertEqual(vault.load("k", 2), b"new")
        self.assertEqual(vault.load("k"), b"new")

    def test_active_points_at_latest_seal(self):
        vault = self.open_vault()
        vault.seal("k", b"a")
        self.assertEqual(vault.active("k"), 1)
        vault.seal("k", b"b")
        self.assertEqual(vault.active("k"), 2)

    def test_material_roundtrips_byte_for_byte(self):
        vault = self.open_vault()
        material = bytes(range(256)) + b"\x00\xff\n\r"
        vault.seal("blob", material)
        self.assertEqual(vault.load("blob"), material)

    def test_empty_material_is_allowed(self):
        vault = self.open_vault()
        vault.seal("empty", b"")
        self.assertEqual(vault.load("empty"), b"")

    def test_bytearray_material_is_stored_as_bytes(self):
        vault = self.open_vault()
        vault.seal("k", bytearray(b"abc"))
        self.assertEqual(vault.load("k"), b"abc")
        self.assertIsInstance(vault.load("k"), bytes)

    def test_empty_key_id_raises_value_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.seal("", b"x")

    def test_non_bytes_material_raises_type_error(self):
        vault = self.open_vault()
        for bad in ("text", 123, None, [b"x"], {"k": b"v"}):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.seal("k", bad)

    def test_failed_seal_leaves_no_trace(self):
        vault = self.open_vault()
        before = vault.manifest()
        with self.assertRaises(TypeError):
            vault.seal("k", "not-bytes")
        self.assertEqual(vault.manifest(), before)
        self.assertEqual(vault.versions("k"), [])

    def test_key_ids_with_special_characters(self):
        vault = self.open_vault()
        key_id = "svc/../weird key ünïcode"
        vault.seal(key_id, b"safe")
        self.assertEqual(vault.load(key_id), b"safe")
        # Material must stay inside the vault root.
        for path in self.root.rglob("*.bin"):
            self.assertTrue(path.is_relative_to(self.root))


class TestLookupErrors(VaultTestCase):
    def test_load_unknown_key_raises_key_error(self):
        vault = self.open_vault()
        with self.assertRaises(KeyError):
            vault.load("nope")

    def test_load_unknown_version_raises_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        with self.assertRaises(KeyError):
            vault.load("k", 2)
        with self.assertRaises(KeyError):
            vault.load("k", 0)

    def test_active_unknown_key_raises_key_error(self):
        vault = self.open_vault()
        with self.assertRaises(KeyError):
            vault.active("nope")

    def test_versions_unknown_key_is_empty(self):
        vault = self.open_vault()
        self.assertEqual(vault.versions("nope"), [])


class TestManifest(VaultTestCase):
    def test_manifest_matches_disk(self):
        vault = self.open_vault()
        vault.seal("a", b"1")
        vault.seal("a", b"2")
        vault.seal("b", b"x")
        on_disk = json.loads(self.manifest_path().read_text("utf-8"))
        self.assertEqual(vault.manifest(), on_disk)
        self.assertEqual(on_disk["keys"]["a"]["active"], 2)
        self.assertEqual(
            [r["version"] for r in on_disk["keys"]["a"]["versions"]], [1, 2]
        )
        self.assertEqual(on_disk["keys"]["b"]["active"], 1)

    def test_manifest_returns_a_copy(self):
        vault = self.open_vault()
        vault.seal("a", b"1")
        snapshot = vault.manifest()
        snapshot["keys"]["a"]["active"] = 999
        self.assertEqual(vault.active("a"), 1)


class TestReload(VaultTestCase):
    def test_reload_picks_up_disk_state(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        other = self.open_vault()
        other.seal("k", b"two")
        other.seal("fresh", b"new")
        self.assertEqual(vault.versions("k"), [1])
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.load("k"), b"two")
        self.assertEqual(vault.load("fresh"), b"new")

    def test_reload_does_not_write_disk(self):
        vault = self.open_vault()
        vault.seal("k", b"data")
        before = {
            p: p.read_bytes()
            for p in sorted(self.root.rglob("*"))
            if p.is_file()
        }
        vault.reload()
        after = {
            p: p.read_bytes()
            for p in sorted(self.root.rglob("*"))
            if p.is_file()
        }
        self.assertEqual(before, after)

    def test_reload_matches_disk_exactly(self):
        vault = self.open_vault()
        vault.seal("a", b"1")
        vault.seal("a", b"2")
        vault.reload()
        self.assertEqual(
            vault.manifest(), json.loads(self.manifest_path().read_text("utf-8"))
        )

    def test_missing_manifest_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        self.manifest_path().unlink()
        with self.assertRaises(ValueError):
            vault.reload()
        # Existing material stays readable.
        self.assertEqual(vault.load("k"), b"v1")

    def test_corrupt_manifest_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        self.write_manifest(b"{not json")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"v1")

    def test_invalid_manifest_format_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        for bad in (
            {"format": 999, "keys": {}},
            {"keys": []},
            ["not", "a", "dict"],
            {"format": 1, "keys": {"k": {"active": 1, "versions": []}}},
        ):
            self.write_manifest(bad)
            with self.assertRaises(ValueError, msg=repr(bad)):
                vault.reload()
        self.assertEqual(vault.load("k"), b"v1")

    def test_tampered_material_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"original")
        (self.root / "keys" / "k" / "v1.bin").write_bytes(b"tampered!")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"original")

    def test_missing_material_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        (self.root / "keys" / "k" / "v1.bin").unlink()
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("k"), b"v1")

    def test_manifest_material_mismatch_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        data = json.loads(self.manifest_path().read_text("utf-8"))
        data["keys"]["k"]["versions"][0]["size"] = 999
        self.write_manifest(data)
        with self.assertRaises(ValueError):
            vault.reload()

    def test_non_increasing_versions_raise_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        data = json.loads(self.manifest_path().read_text("utf-8"))
        data["keys"]["k"]["versions"][1]["version"] = 1
        self.write_manifest(data)
        with self.assertRaises(ValueError):
            vault.reload()

    def test_active_mismatch_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        data = json.loads(self.manifest_path().read_text("utf-8"))
        data["keys"]["k"]["active"] = 7
        self.write_manifest(data)
        with self.assertRaises(ValueError):
            vault.reload()

    def test_failed_reload_changes_nothing_on_disk(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        self.write_manifest(b"garbage")
        before = {
            p: p.read_bytes()
            for p in sorted(self.root.rglob("*"))
            if p.is_file()
        }
        with self.assertRaises(ValueError):
            vault.reload()
        after = {
            p: p.read_bytes()
            for p in sorted(self.root.rglob("*"))
            if p.is_file()
        }
        self.assertEqual(before, after)

    def test_failed_reload_keeps_key_set(self):
        vault = self.open_vault()
        vault.seal("a", b"1")
        vault.seal("b", b"2")
        keys_before = set(vault.manifest()["keys"])
        self.write_manifest(b"\x00\x01")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(set(vault.manifest()["keys"]), keys_before)


class TestConcurrency(VaultTestCase):
    def test_readers_see_whole_snapshots(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        errors = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    material = vault.load("k")
                    # Any sealed value is acceptable; a half-loaded snapshot
                    # would raise KeyError or return inconsistent bytes.
                    n = int(material[1:])
                    if vault.load("k", n) != material:
                        errors.append(f"inconsistent read: {material!r}")
                except (KeyError, ValueError) as exc:
                    errors.append(f"read failed mid-swap: {exc!r}")
                time.sleep(0.001)

        def writer():
            for i in range(2, 20):
                vault.seal("k", f"v{i}".encode())
                vault.reload()
            stop.set()

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(vault.active("k"), 19)


class TestCli(VaultTestCase):
    def run_cli(self, *argv):
        from keyvault_ledger.__main__ import main

        return main(["--root", str(self.root), *argv])

    def test_seal_versions_reload_roundtrip(self):
        material = self.root.parent / "material.bin"
        material.write_bytes(b"cli-material")
        self.assertEqual(self.run_cli("seal", "cli-key", "--material-file", str(material)), 0)
        self.assertEqual(self.run_cli("versions"), 0)
        self.assertEqual(self.run_cli("reload"), 0)
        vault = self.open_vault()
        self.assertEqual(vault.load("cli-key"), b"cli-material")
        self.assertEqual(vault.active("cli-key"), 1)

    def test_versions_on_empty_vault(self):
        self.assertEqual(self.run_cli("versions"), 0)


if __name__ == "__main__":
    unittest.main()
