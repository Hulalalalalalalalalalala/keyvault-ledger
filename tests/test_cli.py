"""Regression tests for the three public command line entry points.

These tests drive the CLI the way the README documents it::

    python3 -m keyvault_ledger --root ./vault versions
    python3 -m keyvault_ledger --root ./vault seal <key-id> --material-file <path>
    python3 -m keyvault_ledger --root ./vault reload

Every case pins the three externally observable things only: the exact bytes
written to standard output, the exact bytes written to standard error and the
process exit code.  No internal function or formatting helper is relied upon.

One boundary behaviour is pinned on purpose: sealing into a directory that
already exists but was never initialised as a vault (it holds unrelated
content and has no ``manifest.json``) must fail with a one-line
``manifest missing`` error and a non-zero exit code.  The CLI must not
silently invent a manifest for such a directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import MANIFEST_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent

_MAIN_USAGE = (
    "usage: keyvault_ledger [-h] --root ROOT {versions,seal,reload} ..."
)
_SEAL_USAGE = (
    "usage: keyvault_ledger seal [-h] --material-file MATERIAL_FILE key_id"
)


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "vault"

    def run_cli(self, *args: str, root: Path | None = None) -> subprocess.CompletedProcess:
        """Run a public CLI invocation and return the completed process.

        Output is captured as bytes so trailing newlines and the two output
        channels can be compared exactly.
        """
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "keyvault_ledger",
                "--root",
                str(root if root is not None else self.root),
                *args,
            ],
            capture_output=True,
            env=env,
        )

    def write_material(self, name: str, data: bytes) -> Path:
        path = Path(self._tmp.name) / name
        path.write_bytes(data)
        return path

    def open_vault(self) -> Vault:
        vault = Vault(self.root)
        self.addCleanup(vault.close)
        return vault

    def assert_argparse_error(
        self, result: subprocess.CompletedProcess, expected_message: str
    ) -> None:
        """argparse failures: exit 2, nothing on stdout, one message on stderr."""
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(
            result.stderr,
            (f"{_MAIN_USAGE}\nkeyvault_ledger: error: {expected_message}\n").encode(),
        )


class TestReadmeCallForms(CliTestCase):
    def test_versions_on_a_brand_new_empty_directory_prints_nothing(self):
        # The directory does not even exist yet; the first command creates it.
        self.assertFalse(self.root.exists())
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"")
        # Initialising as a side effect does leave a manifest behind.
        self.assertTrue((self.root / MANIFEST_NAME).is_file())

    def test_versions_on_an_initialised_empty_vault_prints_nothing(self):
        Vault(self.root).close()
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"")

    def test_full_seal_versions_reload_lifecycle(self):
        material = self.write_material("material.bin", b"cli-material")

        first = self.run_cli("seal", "cli-key", "--material-file", str(material))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, b"1\n")
        self.assertEqual(first.stderr, b"")

        second = self.run_cli("seal", "cli-key", "--material-file", str(material))
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(second.stdout, b"2\n")
        self.assertEqual(second.stderr, b"")

        listing = self.run_cli("versions")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertEqual(listing.stderr, b"")
        self.assertEqual(
            listing.stdout, b"cli-key\tactive=2\tversions=1,2\n"
        )

        reloaded = self.run_cli("reload")
        self.assertEqual(reloaded.returncode, 0, reloaded.stderr)
        self.assertEqual(reloaded.stdout, b"reloaded\n")
        self.assertEqual(reloaded.stderr, b"")

        # Reload is a read-only validation pass that reports the same way on
        # every successful run.
        again = self.run_cli("reload")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(again.stdout, b"reloaded\n")
        self.assertEqual(again.stderr, b"")

    def test_versions_lists_keys_sorted_with_one_line_each(self):
        material = self.write_material("m.bin", b"m")
        # Seal each key once, in an order that is not sorted.
        self.assertEqual(
            self.run_cli("seal", "zeta", "--material-file", str(material)).returncode,
            0,
        )
        self.assertEqual(
            self.run_cli("seal", "alpha", "--material-file", str(material)).returncode,
            0,
        )
        self.assertEqual(
            self.run_cli("seal", "mid", "--material-file", str(material)).returncode,
            0,
        )

        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            b"".join(
                [
                    b"alpha\tactive=1\tversions=1\n",
                    b"mid\tactive=1\tversions=1\n",
                    b"zeta\tactive=1\tversions=1\n",
                ]
            ),
        )

    def test_reload_on_a_brand_new_empty_directory_reports_reloaded(self):
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"reloaded\n")
        self.assertEqual(result.stderr, b"")

    def test_seal_into_a_truly_empty_preexisting_directory_initialises_it(self):
        # An existing but completely empty directory is a valid fresh vault.
        self.root.mkdir(parents=True)
        material = self.write_material("m.bin", b"m")
        result = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"1\n")
        self.assertEqual(result.stderr, b"")


class TestSealBoundaryManifestMissing(CliTestCase):
    def prepare_nonempty_uninitialised_dir(self) -> Path:
        """A directory that exists and holds content but is not a vault."""
        self.root.mkdir(parents=True)
        stray = self.root / "notes.txt"
        stray.write_bytes(b"not vault data\n")
        return stray

    def test_seal_without_a_manifest_fails_and_does_not_create_one(self):
        stray = self.prepare_nonempty_uninitialised_dir()
        material = self.write_material("m.bin", b"m")

        result = self.run_cli("seal", "k", "--material-file", str(material))

        # One line naming the missing manifest, on stderr, non-zero exit.
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(
            result.stderr,
            (f"error: manifest missing: {self.root / MANIFEST_NAME}\n").encode(),
        )
        # The current behaviour is to refuse rather than auto-creating the
        # manifest: there must be no manifest afterwards.
        self.assertFalse((self.root / MANIFEST_NAME).exists())
        # The unrelated content that made the directory non-fresh is left
        # exactly as it was.
        self.assertEqual(stray.read_bytes(), b"not vault data\n")

    def test_versions_on_a_nonempty_uninitialised_dir_fails_the_same_way(self):
        self.prepare_nonempty_uninitialised_dir()
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(
            result.stderr,
            (f"error: manifest missing: {self.root / MANIFEST_NAME}\n").encode(),
        )
        self.assertFalse((self.root / MANIFEST_NAME).exists())

    def test_repeated_failed_seal_keeps_failing_without_a_manifest(self):
        self.prepare_nonempty_uninitialised_dir()
        material = self.write_material("m.bin", b"m")
        for _ in range(2):
            result = self.run_cli("seal", "k", "--material-file", str(material))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, b"")
            self.assertTrue(result.stderr.startswith(b"error: manifest missing: "))
            self.assertFalse((self.root / MANIFEST_NAME).exists())


class TestSealMaterialBytes(CliTestCase):
    def test_bytes_read_from_material_file_are_sealed_byte_for_byte(self):
        material = bytes(range(256)) + b"\x00\xff\n\r\t" + b"trailing-bytes"
        material_path = self.write_material("exact.bin", material)

        result = self.run_cli("seal", "bin", "--material-file", str(material_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"1\n")
        self.assertEqual(result.stderr, b"")

        # The library interface hands back exactly the bytes the CLI read
        # from the material file...
        vault = self.open_vault()
        self.assertEqual(vault.load("bin"), material)
        self.assertEqual(vault.load("bin", 1), material)

        # ...and the per-version material file on disk is byte-for-byte the
        # same content, named by the persisted manifest record.
        record = vault.manifest()["keys"]["bin"]["versions"][0]
        on_disk = (self.root / record["file"]).read_bytes()
        self.assertEqual(on_disk, material)
        self.assertEqual(on_disk, material_path.read_bytes())

    def test_empty_material_file_seals_zero_bytes(self):
        material_path = self.write_material("empty.bin", b"")
        result = self.run_cli("seal", "k", "--material-file", str(material_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"1\n")
        self.assertEqual(self.open_vault().load("k"), b"")


class TestCliRuntimeErrors(CliTestCase):
    def test_missing_material_file_is_a_runtime_error_on_stderr_exit_1(self):
        # Initialise the vault first so the failure is purely the missing
        # material file rather than the vault directory.
        self.run_cli("versions")
        missing = Path(self._tmp.name) / "never-existed.bin"
        self.assertFalse(missing.exists())

        result = self.run_cli("seal", "k", "--material-file", str(missing))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        # The error reports the missing path; only the OS error number text
        # and message are pinned, not internal framing.
        self.assertTrue(
            result.stderr.startswith(
                b"error: [Errno 2] No such file or directory: '"
            ),
            result.stderr,
        )
        self.assertTrue(result.stderr.endswith(b"'\n"), result.stderr)
        self.assertIn(os.fsencode(missing.name), result.stderr)
        # A failed seal stored nothing.
        self.assertEqual(self.open_vault().manifest()["keys"], {})

    def test_empty_key_id_is_rejected_exit_1(self):
        material = self.write_material("m.bin", b"m")
        result = self.run_cli("seal", "", "--material-file", str(material))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error: key_id must not be empty\n")
        self.assertEqual(self.open_vault().manifest()["keys"], {})


class TestCliUsageErrors(CliTestCase):
    def test_no_arguments_at_all_reports_missing_root(self):
        result = self.run_cli_raw()
        self.assert_argparse_error(
            result, "the following arguments are required: --root, command"
        )

    def test_root_without_command_reports_missing_command(self):
        result = self.run_cli()  # supplies --root, no subcommand
        self.assert_argparse_error(
            result, "the following arguments are required: command"
        )

    def test_unknown_subcommand_is_rejected(self):
        result = self.run_cli("bogus")
        self.assert_argparse_error(
            result,
            "argument command: invalid choice: 'bogus' "
            "(choose from versions, seal, reload)",
        )

    def test_unrecognized_option_is_rejected(self):
        result = self.run_cli("--bogus", "versions")
        self.assert_argparse_error(result, "unrecognized arguments: --bogus")

    def test_seal_without_key_id_is_a_usage_error(self):
        result = self.run_cli("seal", "--material-file", "whatever")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(
            result.stderr,
            (
                f"{_SEAL_USAGE}\n"
                "keyvault_ledger seal: error: "
                "the following arguments are required: key_id\n"
            ).encode(),
        )

    def test_seal_without_material_file_is_a_usage_error(self):
        result = self.run_cli("seal", "somekey")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(
            result.stderr,
            (
                f"{_SEAL_USAGE}\n"
                "keyvault_ledger seal: error: "
                "the following arguments are required: --material-file\n"
            ).encode(),
        )

    def test_usage_errors_write_nothing_to_stdout_and_never_seal(self):
        for argv in (("bogus",), ("seal", "k"), ("seal", "--material-file", "x")):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 2, argv)
            self.assertEqual(result.stdout, b"", argv)
        # The failed parse never touched the vault: no manifest was created.
        self.assertFalse((self.root / MANIFEST_NAME).exists())

    def run_cli_raw(self, *args: str) -> subprocess.CompletedProcess:
        """Invoke without injecting ``--root`` (used for the no-args case)."""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "keyvault_ledger", *args],
            capture_output=True,
            env=env,
        )


if __name__ == "__main__":
    unittest.main()
