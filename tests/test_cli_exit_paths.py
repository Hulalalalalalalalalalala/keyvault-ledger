"""Regression tests pinning the three command line exit paths.

The baseline command line already returns its vault lock handle on every
exit path (success, handled error, argparse rejection); nothing in the
package or the existing suite is changed here.  These cases only freeze
observable behaviour, each one driving the real ``python -m
keyvault_ledger`` program as a subprocess in its own temporary vault and
capturing stdout, stderr and the exit code exactly:

* the three README entry points ``versions``, ``seal`` and ``reload`` keep
  their frozen success output byte for byte, including every space, tab
  and trailing newline;

* the same inputs run twice (either twice against one vault, or once
  against each of two fresh vaults) produce byte-identical output once the
  genuinely floating parts -- the temporary directory name, which only
  ever occurs inside an error line, and anything timing related (the
  program prints no timing data at all) -- are removed;

* a wrong invocation shape is argparse's own rejection: exit code 2, no
  stdout and the frozen two-line usage/error text on stderr, before any
  vault directory is ever opened;

* the filesystem failure paths (a root that is a file, a missing manifest
  in a non-empty directory, material that is missing/unreadable/a
  directory) keep their frozen ``error: ...`` line and exit code 1 and
  never produce half a new version: the on-disk manifest and material
  records only ever grow, never change or get truncated;

* a whole-vault reload that fails validation exits 1 with the frozen
  complaint, writes nothing, and for a live handle the in-memory snapshot
  stays exactly as it was -- keys already in hand remain readable;

* with warning policies enabled (``always``/``error`` for
  ``ResourceWarning`` and the broad ``always`` policy) the tail of every
  subcommand's output carries no unclosed-resource warning line -- on the
  success paths stderr is zero bytes, and on the error paths it holds only
  the single frozen ``error:`` line;

* the library contract is unchanged: empty identifiers raise
  ``ValueError``, non-integer versions raise ``TypeError``, unknown
  keys/versions raise ``KeyError`` and non-bytes material raises
  ``TypeError``; ``close()`` is idempotent and a later operation on the
  same vault reacquires the lock with identical observable behaviour.

Every case reads and writes only inside its own temporary directory; the
subprocess has exited (and therefore released the lock file) before
teardown removes the tree, and cases share no state, so their execution
order cannot change the result::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    LOCK_NAME,
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
# argv exactly as given (used for invocation shapes that omit --root).
_DEFAULT_ROOT = object()


class CliExitPathsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.tmp_path / "vault"

    # ------------------------------------------------------------------
    # subprocess plumbing
    # ------------------------------------------------------------------

    def run_cli(
        self,
        *args: str,
        root: object = _DEFAULT_ROOT,
        warnings: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run the real CLI as a subprocess and capture both streams.

        The environment is rebuilt for every call so an ambient
        ``PYTHONWARNINGS`` can neither add warning lines nor change the
        frozen output; ``warnings`` opts a single call into a warning
        policy explicitly.
        """
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

    def material_bytes(self, root: Path | None = None) -> list[bytes]:
        target = self.root if root is None else root
        return sorted(
            path.read_bytes()
            for path in (target / MATERIALS_DIR).rglob("*.bin")
        )

    def material_rel_path(self, root: Path, key_id: str, version: int) -> Path:
        manifest = json.loads(
            (root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )
        for record in manifest["keys"][key_id]["versions"]:
            if record["version"] == version:
                return root / record["file"]
        raise KeyError((key_id, version))


# ---------------------------------------------------------------------------
# frozen success output for the three README entry points
# ---------------------------------------------------------------------------


class TestFrozenSuccessOutputs(CliExitPathsTestCase):
    def test_versions_on_an_absent_root_prints_nothing_at_all(self):
        # Frozen baseline: opening an absent directory creates and
        # initialises it, so versions succeeds with zero bytes on either
        # stream.  This is the existing behaviour, pinned as-is.
        self.assertFalse(self.root.exists())
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertTrue((self.root / MANIFEST_NAME).exists())

    def test_reload_on_an_absent_root_prints_the_frozen_line(self):
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "reloaded\n")
        self.assertEqual(result.stderr, "")

    def test_seal_prints_each_version_with_one_trailing_newline(self):
        material = self.material_file("m.bin", b"frozen-material")
        first = self.run_cli(
            "seal", "k", "--material-file", str(material)
        )
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stdout, "1\n")
        self.assertEqual(first.stderr, "")
        second = self.run_cli(
            "seal", "k", "--material-file", str(material)
        )
        self.assertEqual(second.returncode, 0)
        self.assertEqual(second.stdout, "2\n")
        self.assertEqual(second.stderr, "")

    def test_versions_line_is_tab_separated_with_a_single_newline(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "only", "--material-file", str(material))
        result = self.run_cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            result.stdout, "only\tactive=1\tversions=1\n"
        )
        self.assertEqual(result.stderr, "")

    def test_versions_lines_are_sorted_and_fully_frozen(self):
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

    def test_reload_success_line_is_frozen_after_seals(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "reloaded\n")
        self.assertEqual(result.stderr, "")

    def test_a_fresh_empty_root_lists_zero_bytes_twice(self):
        # Two independent fresh roots agree byte for byte, including the
        # total absence of any output.
        other = self.tmp_path / "other"
        first = self.run_cli("versions", root=self.root)
        second = self.run_cli("versions", root=other)
        self.assertEqual((first.returncode, first.stdout, first.stderr), (0, "", ""))
        self.assertEqual(
            (second.returncode, second.stdout, second.stderr),
            (first.returncode, first.stdout, first.stderr),
        )


# ---------------------------------------------------------------------------
# same inputs run twice: identical output apart from floating path names
# ---------------------------------------------------------------------------


class TestRepeatedInvocationDeterminism(CliExitPathsTestCase):
    SUCCESS_SCRIPT = (
        ("versions",),
        ("reload",),
        ("seal", "alpha", None),
        ("seal", "alpha", None),
        ("seal", "beta", None),
        ("versions",),
        ("reload",),
    )

    def _run_script(self, root: Path, material: Path) -> list[tuple[int, str, str]]:
        captured: list[tuple[int, str, str]] = []
        for argv in self.SUCCESS_SCRIPT:
            argv = tuple(
                str(material) if token is None else token for token in argv
            )
            result = self.run_cli(*argv, root=root)
            captured.append((result.returncode, result.stdout, result.stderr))
        return captured

    def test_success_script_in_two_fresh_vaults_is_byte_identical(self):
        material = self.material_file("shared-material.bin", b"m")
        root_a = self.tmp_path / "a"
        root_b = self.tmp_path / "b"
        self.assertEqual(
            self._run_script(root_a, material),
            self._run_script(root_b, material),
        )

    def test_failing_reload_run_twice_is_byte_identical(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        missing = self.material_rel_path(self.root, "k", 1)
        missing.unlink()
        manifest_before = self.manifest_bytes()

        first = self.run_cli("reload")
        second = self.run_cli("reload")
        self.assertEqual((first.returncode, first.stdout), (1, ""))
        self.assertEqual(
            first.stderr, "error: material missing for key 'k' version 1\n"
        )
        # The second run repeats the exact code and both streams; the
        # message names the key and version, so nothing floats.
        self.assertEqual(
            (second.returncode, second.stdout, second.stderr),
            (first.returncode, first.stdout, first.stderr),
        )
        # Two failing runs changed no disk record and created none.
        self.assertEqual(self.manifest_bytes(), manifest_before)

    def test_error_lines_agree_after_substituting_floating_paths(self):
        material = self.material_file("m.bin", b"m")
        root_a = self.tmp_path / "a"
        root_b = self.tmp_path / "b"
        missing_a = self.tmp_path / "absent-a.bin"
        missing_b = self.tmp_path / "absent-b.bin"

        def failing_run(root: Path, missing: Path) -> tuple[int, str, str]:
            result = self.run_cli(
                "seal", "k", "--material-file", str(missing), root=root
            )
            return result.returncode, result.stdout, result.stderr

        def normalise(text: str, root: Path, missing: Path) -> str:
            # The only floating tokens are the temporary paths embedded by
            # OSError's repr; replace them with fixed placeholders.
            return text.replace(str(missing), "<MATERIAL>").replace(
                str(root), "<ROOT>"
            )

        first = failing_run(root_a, missing_a)
        second = failing_run(root_b, missing_b)
        self.assertEqual((first[0], first[1]), (1, ""))
        self.assertEqual((second[0], second[1]), (1, ""))
        self.assertEqual(
            normalise(first[2], root_a, missing_a),
            normalise(second[2], root_b, missing_b),
        )
        self.assertEqual(
            normalise(first[2], root_a, missing_a),
            "error: [Errno 2] No such file or directory: '<MATERIAL>'\n",
        )

    def test_same_failure_command_twice_against_one_root_is_identical(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        missing = self.tmp_path / "absent.bin"
        argv = ("seal", "k", "--material-file", str(missing))

        first = self.run_cli(*argv)
        second = self.run_cli(*argv)
        self.assertEqual(first.returncode, 1)
        self.assertEqual(second.returncode, 1)
        self.assertEqual(first.stdout, "")
        self.assertEqual(second.stdout, "")
        self.assertEqual(first.stderr, second.stderr)
        self.assertEqual(
            first.stderr,
            f"error: [Errno 2] No such file or directory: '{missing}'\n",
        )
        # Two failed reads still leave exactly one version.
        listing = self.run_cli("versions")
        self.assertEqual(
            listing.stdout, "k\tactive=1\tversions=1\n"
        )


# ---------------------------------------------------------------------------
# wrong invocation shape: argparse rejection, exit code 2
# ---------------------------------------------------------------------------


class TestInvocationShapeExitTwo(CliExitPathsTestCase):
    def test_missing_root_exits_two_with_frozen_usage(self):
        result = self.run_cli("versions", root=None)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE +
            "keyvault_ledger: error: the following arguments are required: "
            "--root\n",
        )

    def test_missing_subcommand_exits_two_with_frozen_usage(self):
        result = self.run_cli(root=self.root)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE +
            "keyvault_ledger: error: the following arguments are required: "
            "command\n",
        )

    def test_unknown_subcommand_exits_two_with_frozen_usage(self):
        result = self.run_cli("bogus")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE +
            "keyvault_ledger: error: argument command: invalid choice: "
            "'bogus' (choose from versions, seal, reload)\n",
        )

    def test_seal_without_material_file_exits_two_with_frozen_usage(self):
        result = self.run_cli("seal", "k")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _SEAL_USAGE +
            "keyvault_ledger seal: error: the following arguments are "
            "required: --material-file\n",
        )

    def test_seal_without_key_id_or_material_exits_two_with_frozen_usage(self):
        result = self.run_cli("seal")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _SEAL_USAGE +
            "keyvault_ledger seal: error: the following arguments are "
            "required: key_id, --material-file\n",
        )

    def test_seal_without_key_id_but_with_material_exits_two(self):
        material = self.material_file("m.bin", b"m")
        result = self.run_cli(
            "seal", "--material-file", str(material)
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _SEAL_USAGE +
            "keyvault_ledger seal: error: the following arguments are "
            "required: key_id\n",
        )

    def test_unexpected_argument_on_versions_exits_two(self):
        result = self.run_cli("versions", "--extra")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE +
            "keyvault_ledger: error: unrecognized arguments: --extra\n",
        )

    def test_unexpected_positional_on_reload_exits_two(self):
        result = self.run_cli("reload", "extra")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE +
            "keyvault_ledger: error: unrecognized arguments: extra\n",
        )

    def test_exit_two_shapes_never_open_or_create_the_vault(self):
        # argparse rejects the shape before main() opens the vault: no
        # directory, lock file or manifest is left behind.
        for argv in (("bogus",), ("seal", "k"), ("reload", "extra")):
            self.assertFalse(self.root.exists(), argv)
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 2, argv)
            self.assertFalse(self.root.exists(), argv)


# ---------------------------------------------------------------------------
# filesystem failure paths: exit code 1, frozen error line, append-only disk
# ---------------------------------------------------------------------------


class TestFilesystemFailurePaths(CliExitPathsTestCase):
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

    def test_absent_target_directory_is_created_and_succeeds(self):
        # Baseline frozen as it stands: a missing root is not an error, the
        # vault is initialised on open and all three entries exit 0.
        deep = self.tmp_path / "no" / "such" / "vault"
        material = self.material_file("m.bin", b"m")

        versions = self.run_cli("versions", root=deep)
        self.assertEqual((versions.returncode, versions.stdout, versions.stderr),
                         (0, "", ""))
        reload_result = self.run_cli("reload", root=deep)
        self.assertEqual(
            (reload_result.returncode, reload_result.stdout, reload_result.stderr),
            (0, "reloaded\n", ""),
        )
        seal_result = self.run_cli(
            "seal", "k", "--material-file", str(material), root=deep
        )
        self.assertEqual(
            (seal_result.returncode, seal_result.stdout, seal_result.stderr),
            (0, "1\n", ""),
        )
        self.assertTrue((deep / MANIFEST_NAME).is_file())
        self.assertTrue((deep / LOCK_NAME).exists())

    def test_root_that_is_a_file_is_rejected_by_every_entry(self):
        material = self.material_file("m.bin", b"m")
        root = self.tmp_path / "a-file"
        root.write_bytes(b"not a directory")
        expected = f"error: [Errno 17] File exists: '{root}'\n"
        for argv in self._all_entries(material):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        # The file is neither replaced nor augmented by a directory.
        self.assertTrue(root.is_file())
        self.assertEqual(root.read_bytes(), b"not a directory")

    def test_populated_directory_without_manifest_errors_for_all_entries(self):
        material = self.material_file("m.bin", b"m")
        root = self.tmp_path / "stray"
        root.mkdir()
        stray_file = root / "notes.txt"
        stray_file.write_bytes(b"not a vault")
        expected = f"error: manifest missing: {root / MANIFEST_NAME}\n"

        for argv in self._all_entries(material):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        # The CLI reports corruption; it does not silently create one.
        self.assertFalse((root / MANIFEST_NAME).exists())
        self.assertEqual(stray_file.read_bytes(), b"not a vault")

    def test_initialized_vault_with_manifest_deleted_errors_for_all_entries(self):
        material = self.material_file("m.bin", b"m")
        sealed = self.run_cli(
            "seal", "k", "--material-file", str(material)
        )
        self.assertEqual(sealed.returncode, 0)
        sealed_material = self.material_bytes()
        self.assertEqual(sealed_material, [b"m"])

        (self.root / MANIFEST_NAME).unlink()
        expected = f"error: manifest missing: {self.root / MANIFEST_NAME}\n"
        for argv in self._all_entries(material):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        # No manifest was rebuilt and the sealed material was not touched.
        self.assertFalse((self.root / MANIFEST_NAME).exists())
        self.assertEqual(self.material_bytes(), sealed_material)

    def test_missing_material_file_exits_one_with_frozen_path_line(self):
        missing = self.tmp_path / "absent.bin"
        result = self.run_cli(
            "seal", "k", "--material-file", str(missing)
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            f"error: [Errno 2] No such file or directory: '{missing}'\n",
        )
        # The failed seal created no version: the brand new vault lists
        # nothing.
        listing = self.run_cli("versions")
        self.assertEqual(listing.returncode, 0)
        self.assertEqual(listing.stdout, "")
        self.assertEqual(self.material_bytes(), [])

    def test_material_path_that_is_a_directory_exits_one(self):
        directory = self.tmp_path / "a-dir"
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

    def test_failed_material_reads_leave_disk_append_only(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        listing_before = self.run_cli("versions")
        manifest_before = self.manifest_bytes()
        materials_before = self.material_bytes()

        directory = self.tmp_path / "a-dir"
        directory.mkdir()
        missing = self.tmp_path / "absent.bin"
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
        # set neither shrank nor grew, and the listing is unchanged.
        self.assertEqual(self.manifest_bytes(), manifest_before)
        self.assertEqual(self.material_bytes(), materials_before)
        listing_after = self.run_cli("versions")
        self.assertEqual(listing_after.stdout, listing_before.stdout)
        self.assertEqual(listing_after.stderr, "")

        # The next successful seal takes version 2; version 1 is not reused.
        again = self.run_cli(
            "seal", "k", "--material-file", str(material)
        )
        self.assertEqual(again.returncode, 0)
        self.assertEqual(again.stdout, "2\n")
        final = self.run_cli("versions")
        self.assertEqual(
            final.stdout, "k\tactive=2\tversions=1,2\n"
        )

    def test_lock_file_left_by_subprocess_does_not_block_tree_removal(self):
        # Every subprocess is awaited (run()) and therefore has released its
        # lock handle by the time it returns; removing the live tree must
        # not be obstructed by an occupied lock file.
        material = self.material_file("m.bin", b"m")
        for _ in range(3):
            result = self.run_cli(
                "seal", "k", "--material-file", str(material)
            )
            self.assertEqual(result.returncode, 0)
        self.assertTrue((self.root / LOCK_NAME).exists())
        shutil.rmtree(self.root)
        self.assertFalse(self.root.exists())


# ---------------------------------------------------------------------------
# failed whole-vault reload: exit 1, frozen complaint, frozen state/disk
# ---------------------------------------------------------------------------


class TestReloadFailureExitPath(CliExitPathsTestCase):
    # The fixed prefix of the frozen line; the JSON decoder's tail text is
    # supplied by the Python runtime and is deliberately not pinned (the
    # project supports every CPython >= 3.11, whose wording varies).
    _MANIFEST_CORRUPT_PREFIX = "error: manifest corrupt: "
    _REVOCATION_CORRUPT_PREFIX = "error: revocation journal corrupt: "

    def test_missing_material_reload_exits_one_and_writes_nothing(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        material_path = self.material_rel_path(self.root, "k", 1)
        material_path.unlink()
        manifest_before = self.manifest_bytes()
        material_listing = sorted(
            str(path.relative_to(self.root))
            for path in (self.root / MATERIALS_DIR).rglob("*.bin")
        )

        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "error: material missing for key 'k' version 1\n",
        )
        # A read-only validation pass: the manifest bytes are unchanged and
        # no material file appeared or vanished.
        self.assertEqual(self.manifest_bytes(), manifest_before)
        self.assertEqual(
            sorted(
                str(path.relative_to(self.root))
                for path in (self.root / MATERIALS_DIR).rglob("*.bin")
            ),
            material_listing,
        )

        # versions is a read that validates on open, so it reports the very
        # same failure with the very same line while the disk stays put.
        listing = self.run_cli("versions")
        self.assertEqual(listing.returncode, 1)
        self.assertEqual(listing.stdout, "")
        self.assertEqual(listing.stderr, result.stderr)
        self.assertEqual(self.manifest_bytes(), manifest_before)

        # Restoring the material makes both entries healthy again, with the
        # original listing.
        material_path.write_bytes(b"m")
        recovered_versions = self.run_cli("versions")
        self.assertEqual(recovered_versions.returncode, 0)
        self.assertEqual(
            recovered_versions.stdout, "k\tactive=1\tversions=1\n"
        )
        self.assertEqual(recovered_versions.stderr, "")
        recovered_reload = self.run_cli("reload")
        self.assertEqual(recovered_reload.returncode, 0)
        self.assertEqual(recovered_reload.stdout, "reloaded\n")
        self.assertEqual(recovered_reload.stderr, "")

    def test_corrupt_manifest_reload_exits_one_without_touching_disk(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        (self.root / MANIFEST_NAME).write_bytes(b"{not json")
        corrupt_bytes = self.manifest_bytes()
        materials_before = self.material_bytes()

        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.startswith(self._MANIFEST_CORRUPT_PREFIX))
        self.assertTrue(result.stderr.endswith("\n"))
        # The corrupt bytes are neither repaired nor replaced.
        self.assertEqual(self.manifest_bytes(), corrupt_bytes)
        self.assertEqual(self.material_bytes(), materials_before)

    def test_corrupt_revocation_journal_reload_exits_one(self):
        vault = self.fixture.open(self.root)
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        vault.close()
        (self.root / REVOCATIONS_NAME).write_bytes(b"{garbage\n")
        manifest_before = self.manifest_bytes()
        materials_before = self.material_bytes()

        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertTrue(
            result.stderr.startswith(self._REVOCATION_CORRUPT_PREFIX)
        )
        self.assertTrue(result.stderr.endswith("\n"))
        # Reload is strictly read-only even when validation fails.
        self.assertEqual(self.manifest_bytes(), manifest_before)
        self.assertEqual(self.material_bytes(), materials_before)
        self.assertEqual(
            (self.root / REVOCATIONS_NAME).read_bytes(), b"{garbage\n"
        )

    def test_live_handle_snapshot_survives_failing_cli_and_library_reloads(self):
        # The subprocess failing its reload cannot affect this process's
        # open handle; the in-process reload must likewise keep its snapshot
        # exactly, with keys already in hand still readable, while disk
        # records neither grow nor shrink.
        vault = self.fixture.open(self.root)
        vault.seal("k", b"one")
        vault.seal("k", b"two")

        materials_before = self.material_bytes()
        healthy_revocation_journal = (
            self.root / REVOCATIONS_NAME
        ).read_bytes()
        (self.root / MANIFEST_NAME).write_bytes(b"{broken")

        cli_result = self.run_cli("reload")
        self.assertEqual(cli_result.returncode, 1)
        self.assertTrue(
            cli_result.stderr.startswith(self._MANIFEST_CORRUPT_PREFIX)
        )

        # Snapshot frozen: the live handle answers exactly as before.
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k", 1), b"one")
        self.assertEqual(vault.load("k", 2), b"two")
        self.assertEqual(vault.load("k"), b"two")

        # Disk records: the deliberately broken manifest stays exactly as
        # written and the failed reloads add/remove nothing else.
        self.assertEqual(self.manifest_bytes(), b"{broken")
        self.assertEqual(self.material_bytes(), materials_before)
        self.assertEqual(len(materials_before), 2)
        self.assertEqual(
            (self.root / REVOCATIONS_NAME).read_bytes(),
            healthy_revocation_journal,
        )


# ---------------------------------------------------------------------------
# warning policies: no unclosed-resource warning at the tail of any command
# ---------------------------------------------------------------------------


class TestNoResourceWarnings(CliExitPathsTestCase):
    POLICIES = ("always::ResourceWarning", "error::ResourceWarning", "always")
    # Signatures CPython itself appends when a resource (here the lock
    # file handle) reaches shutdown without being closed.
    _WARNING_RE = re.compile(
        r"resourcewarning|unclosed|exception ignored in|unraisablehook|"
        r"still running",
        re.IGNORECASE,
    )

    def assertCleanTail(self, result: subprocess.CompletedProcess) -> None:
        """The output tail carries no resource/warning machinery text."""
        self.assertFalse(
            self._WARNING_RE.search(result.stderr),
            f"warning text leaked into stderr: {result.stderr!r}",
        )

    def test_every_success_entry_is_clean_under_every_policy(self):
        for policy in self.POLICIES:
            with self.subTest(policy=policy):
                # A fresh root per (policy, entry) so the frozen outputs are
                # unaffected by earlier seals.
                root = self.tmp_path / f"root-{policy.replace(':', '-')}"
                material = self.tmp_path / f"m-{policy.replace(':', '-')}.bin"
                material.write_bytes(b"warn-material")

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

                versions_after = self.run_cli(
                    "versions", root=root, warnings=policy
                )
                self.assertEqual(versions_after.returncode, 0)
                self.assertEqual(
                    versions_after.stdout,
                    "k\tactive=1\tversions=1\n",
                )
                self.assertEqual(versions_after.stderr, "")

    def test_error_entries_emit_only_the_error_line_under_always_warnings(self):
        material = self.material_file("m.bin", b"m")
        root = self.tmp_path / "warned"
        self.run_cli("seal", "k", "--material-file", str(material), root=root)
        missing = self.tmp_path / "absent.bin"

        cases = (
            ("seal", "", "--material-file", str(material)),
            ("seal", "k", "--material-file", str(missing)),
        )
        for argv in cases:
            with self.subTest(argv=argv):
                result = self.run_cli(*argv, root=root, warnings="always")
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                # Exactly one frozen line; the handle was returned on the
                # error path too, so no warning follows it.
                self.assertEqual(result.stderr.count("\n"), 1)
                self.assertTrue(result.stderr.startswith("error: "))
                self.assertCleanTail(result)

    def test_failing_reload_is_clean_under_always_warnings(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        self.material_rel_path(self.root, "k", 1).unlink()

        result = self.run_cli("reload", warnings="always")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "error: material missing for key 'k' version 1\n",
        )
        self.assertCleanTail(result)

    def test_warning_policy_does_not_change_success_bytes(self):
        # Enabling warnings changes nothing observable: output under a
        # policy is byte-identical to output with the scrubbed environment.
        material = self.material_file("m.bin", b"m")
        plain_root = self.tmp_path / "plain"
        warned_root = self.tmp_path / "warned"

        plain = self.run_cli(
            "seal", "k", "--material-file", str(material), root=plain_root
        )
        warned = self.run_cli(
            "seal", "k", "--material-file", str(material),
            root=warned_root, warnings="always",
        )
        self.assertEqual(plain.stdout, warned.stdout)
        self.assertEqual(plain.stderr, warned.stderr)
        self.assertEqual(plain.returncode, warned.returncode)

        plain_versions = self.run_cli("versions", root=plain_root)
        warned_versions = self.run_cli(
            "versions", root=warned_root, warnings="always"
        )
        self.assertEqual(plain_versions.stdout, warned_versions.stdout)
        self.assertEqual(warned_versions.stderr, "")


# ---------------------------------------------------------------------------
# library contract: the exception vocabulary stays exactly as documented
# ---------------------------------------------------------------------------


class TestLibraryContractVocabulary(CliExitPathsTestCase):
    def open(self, root: Path | None = None) -> Vault:
        return self.fixture.open(self.root if root is None else root)

    def test_empty_identifier_raises_value_error(self):
        vault = self.open()
        with self.assertRaises(ValueError) as caught:
            vault.seal("", b"m")
        self.assertEqual(str(caught.exception), "key_id must not be empty")
        with self.assertRaises(ValueError):
            vault.derive_seal("", b"pw", b"salt", 1, 1)
        with self.assertRaises(ValueError):
            vault.derivation("")
        with self.assertRaises(ValueError):
            vault.revoke("", 1)
        with self.assertRaises(ValueError):
            vault.is_revoked("", 1)
        with self.assertRaises(ValueError):
            vault.set_active("", 1)
        with self.assertRaises(ValueError):
            vault.revoked_versions("")

    def test_non_integer_version_raises_type_error(self):
        vault = self.open()
        vault.seal("k", b"m")
        # None is deliberately absent: it is the documented "active
        # version" sentinel, not a rejected value.
        for bad in (1.0, 2.5, True, False, "1"):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derivation("k", bad)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.revoke("k", bad)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.is_revoked("k", bad)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.set_active("k", bad)
        with self.assertRaises(TypeError) as caught:
            vault.derivation("k", "1")
        self.assertEqual(str(caught.exception), "version must be an int")

    def test_unknown_key_or_version_raises_key_error(self):
        vault = self.open()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.load("never-sealed")
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed")
        with self.assertRaises(KeyError):
            vault.active("never-sealed")
        with self.assertRaises(KeyError):
            vault.load("k", 2)
        with self.assertRaises(KeyError):
            vault.derivation("k", 2)
        with self.assertRaises(KeyError):
            vault.revoke("never-sealed", 1)
        with self.assertRaises(KeyError):
            vault.revoke("k", 9)
        with self.assertRaises(KeyError):
            vault.set_active("never-sealed", 1)
        # An unknown key simply has no revoked versions (documented read
        # semantic), while an empty id still raises ValueError.
        self.assertEqual(vault.revoked_versions("never-sealed"), [])

    def test_non_bytes_material_raises_type_error(self):
        vault = self.open()
        for bad in ("text", 123, None, [b"x"], {"k": b"v"}):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.seal("k", bad)
        with self.assertRaises(TypeError) as caught:
            vault.seal("k", "text")
        self.assertEqual(
            str(caught.exception), "material must be a bytes-like object"
        )
        # The accepted bytes-like types really are accepted and round-trip.
        for material in (b"bytes", bytearray(b"byte-array"), memoryview(b"view")):
            version = vault.seal("bin", material)
            self.assertEqual(vault.load("bin", version), bytes(material))

    def test_derive_seal_type_and_value_contract_is_unchanged(self):
        vault = self.open()
        with self.assertRaises(TypeError):
            vault.derive_seal("k", "pw", b"s", 1, 1)
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", "s", 1, 1)
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", bytearray(b"s"), 1, 1)
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", b"s", 1.5, 1)
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", b"s", 1, True)
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 1)
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"s", 0, 1)
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"s", 1, 0)

    def test_failed_library_calls_produce_no_version(self):
        vault = self.open()
        vault.seal("k", b"v1")
        for call in (
            lambda: vault.seal("", b"x"),
            lambda: vault.seal("k", "not-bytes"),
            lambda: vault.revoke("never-sealed", 1),
            lambda: vault.set_active("k", 99),
        ):
            with self.assertRaises((ValueError, TypeError, KeyError)):
                call()
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.load("k"), b"v1")
        self.assertEqual(vault.seal("k", b"v2"), 2)


# ---------------------------------------------------------------------------
# close(): idempotent release, then reacquire with unchanged behaviour
# ---------------------------------------------------------------------------


class TestCloseReleaseAndReacquire(CliExitPathsTestCase):
    def open(self) -> Vault:
        return self.fixture.open(self.root)

    def test_repeated_close_is_not_an_error(self):
        vault = self.open()
        vault.seal("k", b"m")
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())

    def test_external_process_takes_the_lock_once_handle_is_returned(self):
        vault = self.open()
        vault.seal("k", b"one")
        vault.close()
        vault.close()  # repeated release stays harmless

        # While this process holds no handle, a real CLI subprocess acquires
        # the very same lock immediately and seals successfully.
        material = self.material_file("external.bin", b"external")
        result = self.run_cli(
            "seal", "ext", "--material-file", str(material)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "1\n")
        self.assertEqual(result.stderr, "")

    def test_operation_after_close_reopens_lock_with_same_behaviour(self):
        vault = self.open()
        self.assertEqual(vault.seal("k", b"one"), 1)
        vault.close()

        # Operating again reacquires the lock: versioning, reads, reload and
        # derived seals all behave exactly as without the intervening close.
        self.assertEqual(vault.seal("k", b"two"), 2)
        self.assertEqual(vault.load("k", 1), b"one")
        self.assertEqual(vault.load("k"), b"two")
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(
            vault.derive_seal("k2", b"pw", b"salt", 100, 16), 1
        )
        vault.reload()
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k", 2), b"two")

        # The external seal from the sibling case shape is visible after a
        # reload on this same handle, proving real cross-process lock
        # handoff rather than a stale cached snapshot.
        material = self.material_file("external.bin", b"external")
        self.run_cli("seal", "ext", "--material-file", str(material))
        vault.reload()
        self.assertEqual(vault.load("ext"), b"external")
        self.assertEqual(vault.active("ext"), 1)
        vault.close()
        vault.close()

    def test_close_release_survives_warning_policy_via_subprocesses(self):
        # Two independent CLI processes use the same root in sequence under
        # the strictest warning policy; the second reacquires the lock and
        # no warning leaks, matching the handle-return contract end to end.
        material = self.material_file("m.bin", b"m")
        first = self.run_cli(
            "seal", "k", "--material-file", str(material),
            warnings="error::ResourceWarning",
        )
        second = self.run_cli(
            "seal", "k", "--material-file", str(material),
            warnings="error::ResourceWarning",
        )
        self.assertEqual(first.stdout, "1\n")
        self.assertEqual(second.stdout, "2\n")
        self.assertEqual(first.stderr, "")
        self.assertEqual(second.stderr, "")


if __name__ == "__main__":
    unittest.main()
