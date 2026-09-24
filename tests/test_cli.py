"""Regression tests pinning the command line interface's exact behaviour.

Each test invokes ``python3 -m keyvault_ledger`` as a real subprocess against
a temporary vault directory and pins the exit code plus the verbatim bytes on
stdout and stderr (text, whitespace and newlines included) for the three
public entry points: ``versions``, ``seal`` and ``reload``.  Nothing here
changes behaviour; it only freezes what the frozen baseline already does.

Runnable with the whole suite:

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from keyvault_ledger.vault import MANIFEST_NAME, MATERIALS_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "vault"

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        """Invoke the CLI in a real process, capturing stdout and stderr."""
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        )
        return subprocess.run(
            [sys.executable, "-m", "keyvault_ledger", "--root", str(self.root), *args],
            capture_output=True,
            text=True,
            env=env,
        )

    def write_material(self, name: str, data: bytes) -> Path:
        path = self.tmp / name
        path.write_bytes(data)
        return path

    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def materials_snapshot(self) -> dict[str, bytes]:
        materials_dir = self.root / MATERIALS_DIR
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in sorted(materials_dir.rglob("*"))
            if path.is_file()
        }

    def vault_files_snapshot(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
        }


class TestCliVersions(CliTestCase):
    def test_versions_on_missing_vault_dir_creates_it_and_prints_nothing(self):
        self.assertFalse(self.root.exists())
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        # The missing directory was initialised as a fresh vault.
        self.assertTrue(self.root.is_dir())
        self.assertTrue(self.manifest_path().is_file())

    def test_versions_lists_keys_sorted_with_tab_separated_fields(self):
        first = self.write_material("m1.bin", b"material-one")
        second = self.write_material("m2.bin", b"material-two")
        for expected_version in ("1", "2", "3"):
            result = self.run_cli(
                "seal", "alpha", "--material-file",
                str(first if expected_version != "3" else second),
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, f"{expected_version}\n")
            self.assertEqual(result.stderr, "")
        result = self.run_cli("seal", "beta", "--material-file", str(first))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "1\n")
        self.assertEqual(result.stderr, "")

        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        # Keys sorted, fields tab-separated, versions comma-separated, one
        # newline per key and a trailing newline at the end.
        self.assertEqual(
            result.stdout,
            "alpha\tactive=3\tversions=1,2,3\n"
            "beta\tactive=1\tversions=1\n",
        )
        self.assertEqual(result.stderr, "")
        # Repeating the same command yields byte-identical output.
        again = self.run_cli("versions")
        self.assertEqual(again.returncode, 0)
        self.assertEqual(again.stdout, result.stdout)
        self.assertEqual(again.stderr, result.stderr)


class TestCliSeal(CliTestCase):
    def test_seal_on_missing_vault_dir_creates_it_and_prints_version(self):
        material = self.write_material("m.bin", b"cli-material")
        self.assertFalse(self.root.exists())
        result = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "1\n")
        self.assertEqual(result.stderr, "")

    def test_seal_in_directory_without_manifest_reports_manifest_missing(self):
        # A directory that already holds content but no manifest is not
        # auto-initialised: seal fails and points at the missing manifest.
        self.root.mkdir(parents=True)
        stray = self.root / "stray.txt"
        stray.write_text("not a vault", encoding="utf-8")
        material = self.write_material("m.bin", b"cli-material")

        expected_stderr = (
            f"error: manifest missing: {self.root / MANIFEST_NAME}\n"
        )
        for _ in range(2):  # identical input, identical output
            result = self.run_cli("seal", "k", "--material-file", str(material))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, expected_stderr)

        # The failed seal created no manifest and no material record.
        self.assertFalse(self.manifest_path().exists())
        self.assertEqual(self.materials_snapshot(), {})
        self.assertEqual(stray.read_text(encoding="utf-8"), "not a vault")

    def test_seal_empty_key_id_fails_and_creates_no_version(self):
        material = self.write_material("m.bin", b"cli-material")
        result = self.run_cli("seal", "", "--material-file", str(material))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "error: key_id must not be empty\n")

        # No new version number was produced anywhere.
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        manifest = json.loads(self.manifest_path().read_bytes().decode("utf-8"))
        self.assertEqual(manifest, {"format": 1, "keys": {}})

    def test_seal_missing_material_file_fails(self):
        missing = self.tmp / "no-such-material.bin"
        result = self.run_cli("seal", "k", "--material-file", str(missing))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            f"error: [Errno {errno.ENOENT}] {os.strerror(errno.ENOENT)}: "
            f"'{missing}'\n",
        )
        # The failed seal left no version behind.
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    @unittest.skipIf(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        "root ignores file permission bits",
    )
    def test_seal_unreadable_material_file_fails(self):
        material = self.write_material("m.bin", b"cli-material")
        material.chmod(0)
        self.addCleanup(
            lambda: material.chmod(stat.S_IRUSR | stat.S_IWUSR)
        )
        result = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            f"error: [Errno {errno.EACCES}] {os.strerror(errno.EACCES)}: "
            f"'{material}'\n",
        )

    def test_failed_seal_leaves_manifest_and_materials_untouched(self):
        material = self.write_material("m.bin", b"cli-material")
        result = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "1\n")

        manifest_before = self.manifest_path().read_bytes()
        materials_before = self.materials_snapshot()
        self.assertTrue(materials_before)

        # Two failing seals: an empty key id and a missing material file.
        result = self.run_cli("seal", "", "--material-file", str(material))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "error: key_id must not be empty\n")
        missing = self.tmp / "missing.bin"
        result = self.run_cli("seal", "k", "--material-file", str(missing))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")

        # The on-disk records only ever grow: the failed seals neither
        # added a half version nor rewrote or truncated what was there.
        self.assertEqual(self.manifest_path().read_bytes(), manifest_before)
        self.assertEqual(self.materials_snapshot(), materials_before)
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "k\tactive=1\tversions=1\n")
        self.assertEqual(result.stderr, "")


class TestCliReload(CliTestCase):
    def test_reload_on_missing_vault_dir_succeeds(self):
        self.assertFalse(self.root.exists())
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "reloaded\n")
        self.assertEqual(result.stderr, "")

    def test_reload_success_prints_reloaded_and_touches_no_disk_record(self):
        material = self.write_material("m.bin", b"cli-material")
        for expected in ("1\n", "2\n"):
            result = self.run_cli("seal", "k", "--material-file", str(material))
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, expected)
            self.assertEqual(result.stderr, "")

        files_before = self.vault_files_snapshot()
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "reloaded\n")
        self.assertEqual(result.stderr, "")
        # A successful reload changes nothing on disk.
        self.assertEqual(self.vault_files_snapshot(), files_before)

    def test_failed_reload_leaves_disk_and_later_versions_output_untouched(self):
        material = self.write_material("m.bin", b"cli-material")
        for expected in ("1\n", "2\n"):
            result = self.run_cli("seal", "k", "--material-file", str(material))
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, expected)

        versions_before = self.run_cli("versions")
        self.assertEqual(versions_before.returncode, 0)
        self.assertEqual(versions_before.stdout, "k\tactive=2\tversions=1,2\n")
        self.assertEqual(versions_before.stderr, "")
        manifest_before = self.manifest_path().read_bytes()

        # Corrupt the manifest out of band: reload must fail non-zero and
        # must not repair, rewrite or truncate the file on its way out.
        corrupt = b"{not json\n"
        self.manifest_path().write_bytes(corrupt)
        try:
            json.loads(corrupt.decode("utf-8"))
        except json.JSONDecodeError as exc:
            detail = str(exc)
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, f"error: manifest corrupt: {detail}\n")
        self.assertEqual(self.manifest_path().read_bytes(), corrupt)

        # Once the manifest is restored, versions reports exactly what it
        # reported before the failed reload.
        self.manifest_path().write_bytes(manifest_before)
        versions_after = self.run_cli("versions")
        self.assertEqual(versions_after.returncode, 0)
        self.assertEqual(versions_after.stdout, versions_before.stdout)
        self.assertEqual(versions_after.stderr, versions_before.stderr)


class TestCliVaultRootErrors(CliTestCase):
    @unittest.skipIf(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        "root ignores file permission bits",
    )
    def test_unwritable_vault_location_fails(self):
        # The vault directory cannot be created inside an unwritable
        # parent: every command fails the same way.
        parent = self.tmp / "read-only-parent"
        parent.mkdir()
        parent.chmod(stat.S_IRUSR | stat.S_IXUSR)
        self.addCleanup(
            lambda: parent.chmod(
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            )
        )
        self.root = parent / "vault"

        expected_stderr = (
            f"error: [Errno {errno.EACCES}] {os.strerror(errno.EACCES)}: "
            f"'{self.root}'\n"
        )
        material = self.write_material("m.bin", b"cli-material")
        for args in (
            ("versions",),
            ("reload",),
            ("seal", "k", "--material-file", str(material)),
        ):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 1, args)
            self.assertEqual(result.stdout, "", args)
            self.assertEqual(result.stderr, expected_stderr, args)
        self.assertFalse(self.root.exists())


class TestCliDeterminism(CliTestCase):
    def test_repeated_identical_inputs_produce_identical_output(self):
        material = self.write_material("m.bin", b"cli-material")
        # Only state-preserving commands belong here: a successful seal
        # legitimately prints a new version number on every repeat.
        commands = [
            ("versions",),
            ("reload",),
            ("seal", "", "--material-file", str(material)),
            ("seal", "k", "--material-file", str(self.tmp / "missing.bin")),
            ("versions",),
        ]
        for args in commands:
            first = self.run_cli(*args)
            second = self.run_cli(*args)
            self.assertEqual(first.returncode, second.returncode, args)
            self.assertEqual(first.stdout, second.stdout, args)
            # The failing seals above keep no state, so their error text
            # is identical on every repeat.
            self.assertEqual(first.stderr, second.stderr, args)


if __name__ == "__main__":
    unittest.main()
