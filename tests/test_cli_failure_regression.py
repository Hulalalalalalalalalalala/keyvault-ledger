"""Explicit regression cases for the failure paths of the three CLI entries.

This module only *adds* test cases: the product code, the public interface
and every pre-existing test stay untouched.  Each case drives the real
``python -m keyvault_ledger`` program as a subprocess against a vault inside
its own temporary directory and captures stdout, stderr and the exit code
exactly, pinning as checkable facts:

* a wrongly shaped invocation always ends with exit code 2 and argparse's
  frozen usage/error text on stderr, with the vault directory never opened;

* the filesystem failure paths -- a root that is a regular file, a missing
  manifest, key material that cannot be read -- each keep their frozen
  ``error: ...`` line and exit code 1, and none of them produces half a new
  version: the manifest and the material records on disk neither gain a
  record nor lose a byte;

* a failed seal is invisible to the version sequence: the next successful
  seal continues the existing sequence without reusing or skipping a
  number;

* a whole-vault reload that fails validation leaves the in-memory snapshot
  of a live handle exactly as it was -- keys already in hand stay readable
  -- and the disk records neither grow nor shrink, so repeating any query
  afterwards returns the pre-failure answer verbatim;

* the library exception vocabulary stays exactly as the public contract
  documents it: ``versions`` answers ``[]`` for empty and unknown ids,
  ``revoked_versions`` answers ``[]`` for an unknown key but raises
  ``ValueError`` for an empty id, unknown keys/versions raise ``KeyError``,
  non-genuine-int versions raise ``TypeError``, ``seal`` accepts
  ``bytes``/``bytearray``/``memoryview`` and rejects everything else with
  ``TypeError``, and ``derive_seal`` takes genuine ``bytes`` passphrases and
  salts only (``TypeError``) with positive integer parameters
  (``ValueError``);

* with a warning policy enabled, no entry lets an unclosed-resource warning
  line appear at the tail of its output;

* the same inputs run twice produce byte-identical output once the
  temporary directory names are substituted (the program prints no timing
  data, so paths are the only floating tokens).

The three entry points keep their existing names, arguments and invocation
shapes; nothing here adds an entry point.  Every case reads and writes only
inside its own temporary directory, cases share no state so their execution
order cannot matter, and teardown removes the whole tree without leaving a
warning tail.  The cases join the existing suite and run with the
documented command::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
)
from tests._fixtures import VaultFixture, cli_env

_ROOT_USAGE = (
    "usage: keyvault_ledger [-h] --root ROOT {versions,seal,reload} ...\n"
)
_SEAL_USAGE = (
    "usage: keyvault_ledger seal [-h] --material-file MATERIAL_FILE key_id\n"
)

# Sentinel meaning "prepend the case's own --root".  ``None`` means run the
# argv exactly as given (for shapes that omit --root entirely).
_DEFAULT_ROOT = object()


class CliFailureRegressionCase(unittest.TestCase):
    """Shared subprocess plumbing for the failure-path regressions."""

    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.tmp_path / "vault"

    def run_cli(
        self,
        *args: str,
        root: object = _DEFAULT_ROOT,
        warnings: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run the real CLI as a subprocess, capturing both streams."""
        env = cli_env()
        if warnings is not None:
            env["PYTHONWARNINGS"] = warnings
        cmd = [sys.executable, "-m", "keyvault_ledger"]
        if root is _DEFAULT_ROOT:
            cmd += ["--root", str(self.root)]
        elif root is not None:
            cmd += ["--root", str(root)]
        cmd += list(args)
        return subprocess.run(
            cmd, capture_output=True, text=True, env=env
        )

    def material_file(self, name: str, data: bytes) -> Path:
        path = self.tmp_path / name
        path.write_bytes(data)
        return path

    def manifest_bytes(self, root: Path | None = None) -> bytes:
        target = self.root if root is None else root
        return (target / MANIFEST_NAME).read_bytes()

    def material_listing(self, root: Path | None = None) -> list[str]:
        target = self.root if root is None else root
        materials = target / MATERIALS_DIR
        if not materials.exists():
            return []
        return sorted(
            str(path.relative_to(target))
            for path in materials.rglob("*.bin")
        )

    def material_bytes(self, root: Path | None = None) -> list[bytes]:
        target = self.root if root is None else root
        return sorted(
            path.read_bytes()
            for path in (target / MATERIALS_DIR).rglob("*.bin")
        )


# ---------------------------------------------------------------------------
# wrong invocation shape: argparse rejection, frozen stderr, exit code 2
# ---------------------------------------------------------------------------


class TestInvocationShapeExitTwo(CliFailureRegressionCase):
    def test_missing_root_option_exits_two(self):
        result = self.run_cli("versions", root=None)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE
            + "keyvault_ledger: error: the following arguments are required: "
            "--root\n",
        )

    def test_missing_command_exits_two(self):
        result = self.run_cli(root=self.root)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE
            + "keyvault_ledger: error: the following arguments are required: "
            "command\n",
        )

    def test_unknown_command_exits_two(self):
        result = self.run_cli("frobnicate")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE
            + "keyvault_ledger: error: argument command: invalid choice: "
            "'frobnicate' (choose from versions, seal, reload)\n",
        )

    def test_seal_missing_material_flag_exits_two(self):
        result = self.run_cli("seal", "some-key")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _SEAL_USAGE
            + "keyvault_ledger seal: error: the following arguments are "
            "required: --material-file\n",
        )

    def test_seal_missing_key_id_exits_two(self):
        material = self.material_file("m.bin", b"m")
        result = self.run_cli("seal", "--material-file", str(material))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _SEAL_USAGE
            + "keyvault_ledger seal: error: the following arguments are "
            "required: key_id\n",
        )

    def test_versions_rejects_an_extra_argument(self):
        result = self.run_cli("versions", "stray")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE
            + "keyvault_ledger: error: unrecognized arguments: stray\n",
        )

    def test_reload_rejects_an_extra_flag(self):
        result = self.run_cli("reload", "--force")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE
            + "keyvault_ledger: error: unrecognized arguments: --force\n",
        )

    def test_rejected_shapes_never_create_the_vault(self):
        # argparse fails before main() opens anything: no directory, no
        # lock file, no manifest appears for any rejected shape.
        shapes = (
            ("frobnicate",),
            ("seal", "some-key"),
            ("versions", "stray"),
            ("reload", "--force"),
        )
        for argv in shapes:
            self.assertFalse(self.root.exists(), argv)
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 2, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertFalse(self.root.exists(), argv)


# ---------------------------------------------------------------------------
# filesystem failure paths: exit code 1, frozen error line, append-only disk
# ---------------------------------------------------------------------------


class TestFilesystemFailureExitOne(CliFailureRegressionCase):
    ALL_ENTRIES = (
        ("versions",),
        ("reload",),
        ("seal", "k", "--material-file", "__MATERIAL__"),
    )

    def _all_entries(self, material: Path) -> list[tuple[str, ...]]:
        return [
            tuple(
                str(material) if token == "__MATERIAL__" else token
                for token in argv
            )
            for argv in self.ALL_ENTRIES
        ]

    def test_root_that_is_a_regular_file_fails_all_entries(self):
        material = self.material_file("m.bin", b"m")
        root = self.tmp_path / "plain-file"
        root.write_bytes(b"not a vault directory")
        expected = f"error: [Errno 17] File exists: '{root}'\n"
        for argv in self._all_entries(material):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        # The file is neither replaced by a directory nor modified.
        self.assertTrue(root.is_file())
        self.assertEqual(root.read_bytes(), b"not a vault directory")

    def test_populated_directory_without_manifest_fails_all_entries(self):
        material = self.material_file("m.bin", b"m")
        root = self.tmp_path / "stray-dir"
        root.mkdir()
        stray = root / "notes.txt"
        stray.write_bytes(b"leftover notes")
        expected = f"error: manifest missing: {root / MANIFEST_NAME}\n"
        for argv in self._all_entries(material):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        # The CLI reports the corruption; it does not silently initialise a
        # manifest over someone else's directory.
        self.assertFalse((root / MANIFEST_NAME).exists())
        self.assertEqual(stray.read_bytes(), b"leftover notes")

    def test_manifest_deleted_from_initialised_vault_fails_all_entries(self):
        material = self.material_file("m.bin", b"m")
        sealed = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(sealed.returncode, 0)
        materials_before = self.material_bytes()
        self.assertEqual(materials_before, [b"m"])

        (self.root / MANIFEST_NAME).unlink()
        expected = f"error: manifest missing: {self.root / MANIFEST_NAME}\n"
        for argv in self._all_entries(material):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        # No manifest was rebuilt and the sealed material was not touched.
        self.assertFalse((self.root / MANIFEST_NAME).exists())
        self.assertEqual(self.material_bytes(), materials_before)

    def test_missing_material_file_exits_one_with_frozen_line(self):
        missing = self.tmp_path / "no-such-material.bin"
        result = self.run_cli(
            "seal", "k", "--material-file", str(missing)
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            f"error: [Errno 2] No such file or directory: '{missing}'\n",
        )
        # The failed seal produced no version at all.
        listing = self.run_cli("versions")
        self.assertEqual(listing.returncode, 0)
        self.assertEqual(listing.stdout, "")
        self.assertEqual(self.material_bytes(), [])

    def test_material_path_that_is_a_directory_exits_one(self):
        directory = self.tmp_path / "material-dir"
        directory.mkdir()
        result = self.run_cli(
            "seal", "k", "--material-file", str(directory)
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            f"error: [Errno 21] Is a directory: '{directory}'\n",
        )
        self.assertEqual(self.material_bytes(), [])

    @unittest.skipUnless(
        hasattr(os, "geteuid") and os.geteuid() != 0,
        "permission bits are not enforced for root",
    )
    def test_unreadable_material_file_exits_one(self):
        locked = self.material_file("locked.bin", b"secret")
        locked.chmod(0o000)
        self.addCleanup(lambda: locked.chmod(0o644))
        result = self.run_cli(
            "seal", "k", "--material-file", str(locked)
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            f"error: [Errno 13] Permission denied: '{locked}'\n",
        )
        self.assertEqual(self.material_bytes(), [])

    def test_empty_key_id_over_the_cli_exits_one(self):
        material = self.material_file("m.bin", b"m")
        result = self.run_cli("seal", "", "--material-file", str(material))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "error: key_id must not be empty\n")
        self.assertEqual(self.material_bytes(), [])

    def test_failures_leave_no_partial_version_behind(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        manifest_before = self.manifest_bytes()
        materials_before = self.material_listing()
        listing_before = self.run_cli("versions").stdout

        directory = self.tmp_path / "material-dir"
        directory.mkdir()
        missing = self.tmp_path / "no-such-material.bin"
        failures = (
            ("seal", "k", "--material-file", str(missing)),
            ("seal", "k", "--material-file", str(directory)),
            ("seal", "", "--material-file", str(material)),
        )
        for argv in failures:
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)

        # The manifest is byte-for-byte the pre-failure one, the material
        # records neither gained nor lost a file, and the listing is
        # unchanged: no half-recorded version exists anywhere.
        self.assertEqual(self.manifest_bytes(), manifest_before)
        self.assertEqual(self.material_listing(), materials_before)
        listing_after = self.run_cli("versions")
        self.assertEqual(listing_after.returncode, 0)
        self.assertEqual(listing_after.stdout, listing_before)
        self.assertEqual(listing_after.stderr, "")


# ---------------------------------------------------------------------------
# a failed seal does not disturb the version sequence
# ---------------------------------------------------------------------------


class TestSealSequenceAfterFailure(CliFailureRegressionCase):
    def test_next_seal_continues_the_sequence_without_gap_or_reuse(self):
        material = self.material_file("m.bin", b"m")
        first = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual((first.returncode, first.stdout), (0, "1\n"))

        missing = self.tmp_path / "no-such-material.bin"
        failed = self.run_cli("seal", "k", "--material-file", str(missing))
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(failed.stdout, "")

        # The next successful seal takes version 2: the failure neither
        # consumed a number nor rolled the sequence back.
        second = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual((second.returncode, second.stdout), (0, "2\n"))
        self.assertEqual(second.stderr, "")
        listing = self.run_cli("versions")
        self.assertEqual(listing.stdout, "k\tactive=2\tversions=1,2\n")
        self.assertEqual(self.material_bytes(), [b"m", b"m"])


# ---------------------------------------------------------------------------
# failed whole-vault reload: snapshot and disk records stay exactly as they
# were, and repeated queries answer verbatim
# ---------------------------------------------------------------------------


class TestFailedReloadKeepsState(CliFailureRegressionCase):
    def test_snapshot_untouched_and_disk_records_frozen(self):
        vault = self.fixture.open(self.root)
        vault.seal("alpha", b"one")
        vault.seal("alpha", b"two")
        vault.derive_seal("beta", b"pw", b"salt", 100, 16)
        vault.revoke("alpha", 1)

        # The pre-failure answers, captured to compare verbatim afterwards.
        queries_before = {
            "manifest": vault.manifest(),
            "alpha_versions": vault.versions("alpha"),
            "alpha_active": vault.active("alpha"),
            "alpha_v1": vault.load("alpha", 1),
            "alpha_v2": vault.load("alpha", 2),
            "alpha_active_load": vault.load("alpha"),
            "alpha_revoked": vault.revoked_versions("alpha"),
            "alpha_v1_is_revoked": vault.is_revoked("alpha", 1),
            "beta_derivation": vault.derivation("beta"),
            "beta_load": vault.load("beta"),
        }
        manifest_bytes_before = self.manifest_bytes()
        materials_before = self.material_listing()
        journal_before = (self.root / REVOCATIONS_NAME).read_bytes()

        # Break the whole-vault validation out of band.
        (self.root / MANIFEST_NAME).write_bytes(b"{broken json")

        # Both the CLI reload and the library reload fail the same way.
        cli_result = self.run_cli("reload")
        self.assertEqual(cli_result.returncode, 1)
        self.assertEqual(cli_result.stdout, "")
        self.assertTrue(
            cli_result.stderr.startswith("error: manifest corrupt: ")
        )
        self.assertTrue(cli_result.stderr.endswith("\n"))
        with self.assertRaises(ValueError):
            vault.reload()

        # The in-memory snapshot is exactly as it was: every query repeats
        # its pre-failure answer, and keys already in hand stay readable.
        self.assertEqual(vault.manifest(), queries_before["manifest"])
        self.assertEqual(
            vault.versions("alpha"), queries_before["alpha_versions"]
        )
        self.assertEqual(vault.active("alpha"), queries_before["alpha_active"])
        self.assertEqual(vault.load("alpha", 1), queries_before["alpha_v1"])
        self.assertEqual(vault.load("alpha", 2), queries_before["alpha_v2"])
        self.assertEqual(
            vault.load("alpha"), queries_before["alpha_active_load"]
        )
        self.assertEqual(
            vault.revoked_versions("alpha"), queries_before["alpha_revoked"]
        )
        self.assertEqual(
            vault.is_revoked("alpha", 1),
            queries_before["alpha_v1_is_revoked"],
        )
        self.assertEqual(
            vault.derivation("beta"), queries_before["beta_derivation"]
        )
        self.assertEqual(vault.load("beta"), queries_before["beta_load"])

        # The disk records neither gained nor lost a byte: the deliberately
        # broken manifest stays as written, and nothing else moved.
        self.assertEqual(self.manifest_bytes(), b"{broken json")
        self.assertNotEqual(self.manifest_bytes(), manifest_bytes_before)
        self.assertEqual(self.material_listing(), materials_before)
        self.assertEqual(
            (self.root / REVOCATIONS_NAME).read_bytes(), journal_before
        )

    def test_repeated_failing_reload_reports_identically(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        manifest = json.loads(self.manifest_bytes().decode("utf-8"))
        rel_file = manifest["keys"]["k"]["versions"][0]["file"]
        (self.root / rel_file).unlink()
        manifest_before = self.manifest_bytes()

        first = self.run_cli("reload")
        second = self.run_cli("reload")
        self.assertEqual((first.returncode, first.stdout), (1, ""))
        self.assertEqual(
            first.stderr, "error: material missing for key 'k' version 1\n"
        )
        # The same failure queried again returns the very same bytes.
        self.assertEqual(
            (second.returncode, second.stdout, second.stderr),
            (first.returncode, first.stdout, first.stderr),
        )
        # Two failing reloads wrote nothing and removed nothing.
        self.assertEqual(self.manifest_bytes(), manifest_before)


# ---------------------------------------------------------------------------
# library contract: the documented exception vocabulary, pinned as-is
# ---------------------------------------------------------------------------


class TestLibraryContractUnchanged(CliFailureRegressionCase):
    def open(self) -> Vault:
        return self.fixture.open(self.root)

    def test_versions_answers_empty_for_empty_and_unknown_ids(self):
        vault = self.open()
        vault.seal("k", b"m")
        # versions alone has no id gate: both an empty id and an unknown id
        # answer an empty list instead of raising.
        self.assertEqual(vault.versions(""), [])
        self.assertEqual(vault.versions("never-sealed"), [])
        self.assertEqual(vault.versions("k"), [1])

    def test_revoked_versions_empty_list_for_unknown_value_error_for_empty(self):
        vault = self.open()
        vault.seal("k", b"m")
        self.assertEqual(vault.revoked_versions("never-sealed"), [])
        self.assertEqual(vault.revoked_versions("k"), [])
        with self.assertRaises(ValueError):
            vault.revoked_versions("")

    def test_load_and_derivation_raise_key_error_for_unknown(self):
        vault = self.open()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.load("never-sealed")
        with self.assertRaises(KeyError):
            vault.load("never-sealed", 1)
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed")
        with self.assertRaises(KeyError):
            vault.load("k", 2)
        with self.assertRaises(KeyError):
            vault.derivation("k", 2)

    def test_non_genuine_int_version_raises_type_error(self):
        vault = self.open()
        vault.seal("k", b"m")
        for bad in (1.0, 2.5, True, False, "1"):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derivation("k", bad)
        # A float numerically equal to an existing version still does not
        # count; the genuine int reads back fine.
        self.assertEqual(vault.load("k", 1), b"m")

    def test_seal_accepts_bytes_like_and_rejects_the_rest(self):
        vault = self.open()
        accepted = (b"bytes", bytearray(b"byte-array"), memoryview(b"view"))
        for index, material in enumerate(accepted):
            version = vault.seal(f"key-{index}", material)
            self.assertEqual(vault.load(f"key-{index}", version), bytes(material))
        for bad in ("text", 123, None, [b"x"], {"k": b"v"}):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.seal("k", bad)
        # The rejections produced no versions.
        self.assertEqual(vault.versions("k"), [])

    def test_derive_seal_type_and_value_contract(self):
        vault = self.open()
        # Passphrase and salt must be genuine bytes: str, bytearray and
        # memoryview are all rejected with TypeError.
        with self.assertRaises(TypeError):
            vault.derive_seal("k", "pw", b"salt", 1, 1)
        with self.assertRaises(TypeError):
            vault.derive_seal("k", bytearray(b"pw"), b"salt", 1, 1)
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", "salt", 1, 1)
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", memoryview(b"salt"), 1, 1)
        # Iterations and length must be genuine positive ints.
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", b"salt", 1.5, 1)
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", b"salt", 1, True)
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 1)
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"salt", 0, 1)
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"salt", 1, 0)
        # None of the rejections sealed anything.
        self.assertEqual(vault.versions("k"), [])


# ---------------------------------------------------------------------------
# warning policies: no unclosed-resource warning at the tail of any entry
# ---------------------------------------------------------------------------


class TestNoResourceWarningTail(CliFailureRegressionCase):
    POLICIES = ("always::ResourceWarning", "error::ResourceWarning", "always")
    _WARNING_RE = re.compile(
        r"resourcewarning|unclosed|exception ignored in|unraisablehook|"
        r"still running",
        re.IGNORECASE,
    )

    def assertCleanTail(self, result: subprocess.CompletedProcess) -> None:
        self.assertFalse(
            self._WARNING_RE.search(result.stderr),
            f"warning text leaked into stderr: {result.stderr!r}",
        )

    def test_every_entry_is_clean_under_every_policy(self):
        for policy in self.POLICIES:
            tag = policy.replace(":", "-")
            root = self.tmp_path / f"root-{tag}"
            material = self.tmp_path / f"m-{tag}.bin"
            material.write_bytes(b"warn-material")
            with self.subTest(policy=policy):
                versions = self.run_cli("versions", root=root, warnings=policy)
                self.assertEqual(versions.returncode, 0, versions.stderr)
                self.assertEqual(versions.stderr, "")

                reload_result = self.run_cli(
                    "reload", root=root, warnings=policy
                )
                self.assertEqual(reload_result.returncode, 0, reload_result.stderr)
                self.assertEqual(reload_result.stdout, "reloaded\n")
                self.assertEqual(reload_result.stderr, "")

                seal_result = self.run_cli(
                    "seal", "k", "--material-file", str(material),
                    root=root, warnings=policy,
                )
                self.assertEqual(seal_result.returncode, 0, seal_result.stderr)
                self.assertEqual(seal_result.stdout, "1\n")
                self.assertEqual(seal_result.stderr, "")

    def test_error_paths_emit_only_the_error_line_under_always(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        missing = self.tmp_path / "no-such-material.bin"
        cases = (
            ("seal", "", "--material-file", str(material)),
            ("seal", "k", "--material-file", str(missing)),
        )
        for argv in cases:
            with self.subTest(argv=argv):
                result = self.run_cli(*argv, warnings="always")
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                # Exactly one frozen error line; the lock handle came back
                # on the error path too, so no warning follows it.
                self.assertEqual(result.stderr.count("\n"), 1)
                self.assertTrue(result.stderr.startswith("error: "))
                self.assertCleanTail(result)

    def test_failing_reload_is_clean_under_always(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        manifest = json.loads(self.manifest_bytes().decode("utf-8"))
        rel_file = manifest["keys"]["k"]["versions"][0]["file"]
        (self.root / rel_file).unlink()

        result = self.run_cli("reload", warnings="always")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr, "error: material missing for key 'k' version 1\n"
        )
        self.assertCleanTail(result)


# ---------------------------------------------------------------------------
# same inputs twice: byte-identical output apart from floating path names
# ---------------------------------------------------------------------------


class TestRepeatedRunsDeterminism(CliFailureRegressionCase):
    def test_mixed_script_twice_is_identical_after_normalising_paths(self):
        # A script mixing all three entries, success and failure alike.
        # The program prints no timing data, so the only floating tokens in
        # any output line are the temporary paths embedded by OSError.
        def run_script(root: Path, missing: Path) -> list[tuple[int, str, str]]:
            material = self.tmp_path / "shared-material.bin"
            material.write_bytes(b"m")
            script = (
                ("versions",),
                ("reload",),
                ("seal", "alpha", "--material-file", str(material)),
                ("seal", "alpha", "--material-file", str(missing)),
                ("seal", "alpha", "--material-file", str(material)),
                ("versions",),
                ("reload",),
            )
            captured = []
            for argv in script:
                result = self.run_cli(*argv, root=root)
                captured.append(
                    (result.returncode, result.stdout, result.stderr)
                )
            return captured

        def normalise(
            captured: list[tuple[int, str, str]], root: Path, missing: Path
        ) -> list[tuple[int, str, str]]:
            return [
                (
                    code,
                    out.replace(str(root), "<ROOT>").replace(
                        str(missing), "<MATERIAL>"
                    ),
                    err.replace(str(root), "<ROOT>").replace(
                        str(missing), "<MATERIAL>"
                    ),
                )
                for code, out, err in captured
            ]

        root_a = self.tmp_path / "vault-a"
        root_b = self.tmp_path / "vault-b"
        missing_a = self.tmp_path / "absent-a.bin"
        missing_b = self.tmp_path / "absent-b.bin"

        first = normalise(run_script(root_a, missing_a), root_a, missing_a)
        second = normalise(run_script(root_b, missing_b), root_b, missing_b)
        self.assertEqual(first, second)
        # The failure inside the script is the pinned missing-material line,
        # and the seal after it continues the sequence at version 2.
        self.assertEqual(first[3][0], 1)
        self.assertEqual(
            first[3][2],
            "error: [Errno 2] No such file or directory: '<MATERIAL>'\n",
        )
        self.assertEqual(first[4], (0, "2\n", ""))
        self.assertEqual(
            first[5], (0, "alpha\tactive=2\tversions=1,2\n", "")
        )


if __name__ == "__main__":
    unittest.main()
