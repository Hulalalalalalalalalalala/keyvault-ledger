"""Regression tests pinning the three command line entry points.

These tests drive the public command line interface exactly as documented in
the README::

    python3 -m keyvault_ledger --root ./vault versions
    python3 -m keyvault_ledger --root ./vault seal <key-id> --material-file <path>
    python3 -m keyvault_ledger --root ./vault reload

For every invocation they assert the observable contract only: the exact
bytes written to standard output, the exact bytes written to standard error
and the process exit code.  The command line implementation is meant to stay
frozen; nothing here imports or relies on a private helper, and a change to
any printed line, error line or exit code fails a test.

Runnable with::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from keyvault_ledger import Vault

REPO_ROOT = Path(__file__).resolve().parent.parent

MODULE = ["-m", "keyvault_ledger"]


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # A fresh, not-yet-existing vault root per test.
        self.root = Path(self._tmp.name) / "vault"

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        """Invoke one public command line call and capture raw byte streams."""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, *MODULE, "--root", str(self.root), *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            env=env,
        )

    def write_material(self, name: str, data: bytes) -> Path:
        path = Path(self._tmp.name) / name
        path.write_bytes(data)
        return path

    def last_stderr_line(self, result: subprocess.CompletedProcess) -> str:
        # argparse prints a ``usage:`` block followed by one ``error:`` line.
        return result.stderr.decode("utf-8").splitlines()[-1]


class TestReadmeInvocationForms(CliTestCase):
    """The three call shapes from the README, in order, on a fresh vault."""

    def test_versions_then_seal_then_reload_walk_through(self):
        # Form 1 on a root that has never existed: no keys, no noise.
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"")

        # Form 2 seals material as version 1 ...
        first = self.write_material("first.bin", b"first-material")
        result = self.run_cli("seal", "cli-key", "--material-file", str(first))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(result.stdout, b"1\n")

        # ... and again, allocating the next version.
        second = self.write_material("second.bin", b"second-material")
        result = self.run_cli("seal", "cli-key", "--material-file", str(second))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(result.stdout, b"2\n")

        # Form 1 again reports the sealed versions, one sorted line per key.
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(
            result.stdout, b"cli-key\tactive=2\tversions=1,2\n"
        )

        # Form 3 re-reads and validates the whole keyring.
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(result.stdout, b"reloaded\n")

        # After reload the listing is byte-for-byte unchanged.
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"cli-key\tactive=2\tversions=1,2\n")

    def test_versions_empty_prints_nothing_at_all(self):
        # An empty version manifest prints nothing at all.
        self.assertEqual(self.run_cli("versions").stdout, b"")

    def test_versions_lists_keys_sorted_one_line_each(self):
        alpha_1 = self.write_material("a1", b"a-one")
        alpha_2 = self.write_material("a2", b"a-two")
        beta = self.write_material("b1", b"b-one")
        for args in (
            ("seal", "alpha", "--material-file", str(alpha_1)),
            ("seal", "beta", "--material-file", str(beta)),
            ("seal", "alpha", "--material-file", str(alpha_2)),
        ):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)

        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        # Keys are sorted; tab-separated; versions comma-joined ascending.
        self.assertEqual(
            result.stdout,
            b"alpha\tactive=2\tversions=1,2\n"
            b"beta\tactive=1\tversions=1\n",
        )

    def test_reload_on_brand_new_directory_prints_reloaded(self):
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(result.stdout, b"reloaded\n")


class TestSealedMaterialIsByteExact(CliTestCase):
    def test_bytes_read_from_file_are_the_bytes_written_to_the_vault(self):
        # Every byte value plus CR/LF edges: nothing may be encoded, translated
        # or trimmed between the file and the stored material.
        payload = bytes(range(256)) + b"\r\n\x00\xff trailing "
        material = self.write_material("exact.bin", payload)

        result = self.run_cli("seal", "bin-key", "--material-file", str(material))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"1\n")

        # The on-disk material the library interface wrote equals the source
        # file byte for byte.
        stored_files = list((self.root / "materials").rglob("*.bin"))
        self.assertEqual(len(stored_files), 1)
        self.assertEqual(stored_files[0].read_bytes(), payload)
        self.assertEqual(stored_files[0].read_bytes(), material.read_bytes())

        # And it reads back identically through the public library interface.
        vault = Vault(self.root)
        self.addCleanup(vault.close)
        self.assertEqual(vault.load("bin-key"), payload)

    def test_two_seals_keep_distinct_bytes(self):
        first = self.write_material("one", b"line-one\n")
        second = self.write_material("two", b"line-two\n")
        self.assertEqual(
            self.run_cli("seal", "k", "--material-file", str(first)).stdout, b"1\n"
        )
        self.assertEqual(
            self.run_cli("seal", "k", "--material-file", str(second)).stdout, b"2\n"
        )
        vault = Vault(self.root)
        self.addCleanup(vault.close)
        self.assertEqual(vault.load("k", 1), b"line-one\n")
        self.assertEqual(vault.load("k", 2), b"line-two\n")


class TestSealWithoutManifestBoundary(CliTestCase):
    """Pinned edge behaviour: sealing on a directory that already holds
    non-manifest content fails loudly instead of silently creating a
    manifest.  This status quo must not be "improved" into auto-creation."""

    def make_content_dir_without_manifest(self) -> Path:
        self.root.mkdir(parents=True)
        marker = self.root / "loose-material.bin"
        marker.write_bytes(b"i was here first")
        return marker

    def test_seal_on_content_dir_without_manifest_fails_with_one_error_line(self):
        marker = self.make_content_dir_without_manifest()
        source = self.write_material("source.bin", b"never stored")

        result = self.run_cli("seal", "k", "--material-file", str(source))
        # Non-zero exit ...
        self.assertEqual(result.returncode, 1)
        # ... nothing on stdout ...
        self.assertEqual(result.stdout, b"")
        # ... exactly one line on stderr naming the missing manifest.
        self.assertEqual(
            result.stderr,
            f"error: manifest missing: {self.root / 'manifest.json'}\n".encode(),
        )
        # The directory is not silently turned into a vault ...
        self.assertFalse((self.root / "manifest.json").exists())
        # ... and its pre-existing content is left untouched.
        self.assertEqual(marker.read_bytes(), b"i was here first")

    def test_boundary_persists_repeated_calls_do_not_create_manifest(self):
        self.make_content_dir_without_manifest()
        source = self.write_material("source.bin", b"never stored")
        for _ in range(2):
            result = self.run_cli("seal", "k", "--material-file", str(source))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, b"")
            self.assertTrue(result.stderr.startswith(b"error: manifest missing: "))
            self.assertFalse((self.root / "manifest.json").exists())

    def test_versions_and_reload_on_such_dir_fail_the_same_way(self):
        self.make_content_dir_without_manifest()
        expected = (
            f"error: manifest missing: {self.root / 'manifest.json'}\n".encode()
        )
        for command in (("versions",), ("reload",)):
            result = self.run_cli(*command)
            self.assertEqual(result.returncode, 1, command)
            self.assertEqual(result.stdout, b"", command)
            self.assertEqual(result.stderr, expected, command)

    def test_genuinely_empty_existing_directory_is_still_initialised(self):
        # The boundary is content-without-manifest only: a truly empty
        # directory (and a not-yet-existing one, covered elsewhere) is a fresh
        # vault and seals normally.
        self.root.mkdir(parents=True)
        material = self.write_material("m.bin", b"ok")
        result = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"1\n")
        self.assertEqual(result.stderr, b"")


class TestRuntimeErrors(CliTestCase):
    """Errors caught by main(): one stderr line, empty stdout, exit code 1."""

    def test_missing_material_file_is_exit_1_with_named_error(self):
        missing = Path(self._tmp.name) / "does-not-exist.bin"
        result = self.run_cli(
            "seal", "k", "--material-file", str(missing)
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        expected = (
            f"error: [Errno {errno.ENOENT}] {os.strerror(errno.ENOENT)}: "
            f"'{missing}'\n"
        ).encode()
        self.assertEqual(result.stderr, expected)

    def test_empty_key_id_is_exit_1_with_named_error(self):
        material = self.write_material("m.bin", b"m")
        result = self.run_cli("seal", "", "--material-file", str(material))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error: key_id must not be empty\n")


class TestUsageErrors(CliTestCase):
    """argparse rejects bad invocation: usage + error on stderr, exit code 2."""

    def assert_usage_error(
        self, result: subprocess.CompletedProcess, last_line: str
    ) -> None:
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertTrue(
            result.stderr.startswith(b"usage:"), result.stderr
        )
        self.assertEqual(self.last_stderr_line(result), last_line)

    def test_unknown_subcommand(self):
        result = self.run_cli("bogus")
        self.assert_usage_error(
            result,
            "keyvault_ledger: error: argument command: invalid choice: 'bogus' "
            "(choose from versions, seal, reload)",
        )

    def test_no_subcommand(self):
        result = self.run_cli()
        self.assert_usage_error(
            result,
            "keyvault_ledger: error: the following arguments are required: command",
        )

    def test_seal_missing_key_id_and_material_file(self):
        result = self.run_cli("seal")
        self.assert_usage_error(
            result,
            "keyvault_ledger seal: error: the following arguments are required: "
            "key_id, --material-file",
        )

    def test_seal_missing_material_file_option(self):
        result = self.run_cli("seal", "k")
        self.assert_usage_error(
            result,
            "keyvault_ledger seal: error: the following arguments are required: "
            "--material-file",
        )

    def test_material_file_option_without_value(self):
        result = self.run_cli("seal", "k", "--material-file")
        self.assert_usage_error(
            result,
            "keyvault_ledger seal: error: argument --material-file: "
            "expected one argument",
        )

    def test_missing_root_option(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, *MODULE, "versions"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            env=env,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertTrue(result.stderr.startswith(b"usage:"))
        self.assertEqual(
            self.last_stderr_line(result),
            "keyvault_ledger: error: the following arguments are required: --root",
        )


if __name__ == "__main__":
    unittest.main()
