"""Explicit regression tests for the three command line failure paths.

This module only *adds* tests: no product code, public interface or
existing case is touched.  Every case drives the real ``python -m
keyvault_ledger`` program as a subprocess inside its own temporary vault
directory and captures stdout, stderr and the exit code exactly, pinning
the observable failure contract:

* a wrong invocation shape is argparse's own rejection -- exit code 2,
  empty stdout, the frozen two-line usage/error text on stderr -- and it
  happens before any vault directory is opened, so nothing is created;

* the filesystem failure paths (root that is a file, a root whose path
  runs through a file, the materials entry occupied by a file, a missing
  manifest, seal material that is missing/a directory/unreadable, an
  empty key id, stored material that is missing/a directory/mismatched)
  keep their frozen ``error: ...`` line and exit code 1; none of them
  mints half a new version -- the manifest and material records neither
  grow nor shrink, and a failed seal is simply followed, on the next
  successful call, by the next version in the existing sequence, never
  reusing or skipping a number;

* a whole-vault reload that fails validation exits 1 with the frozen
  complaint for ``versions``, ``reload`` and ``seal`` alike (every entry
  validates on open), writes nothing, and for a live handle the
  in-memory snapshot stays exactly as it was -- keys already in hand
  keep reading byte for byte; repeating a query while the failure
  persists returns the same answer, and disk records stay byte-identical;

* the library read contract pinned through the same fixture: the
  version listing answers an empty or unknown id with an empty list,
  the revoked-version listing answers an unknown key with an empty list
  while an empty id still raises ``ValueError``, reads and per-version
  queries raise ``KeyError`` for unknown keys/versions and ``TypeError``
  for non-genuine-integer versions, ``seal`` accepts ``bytes``,
  ``bytearray`` and ``memoryview`` and rejects every other type, and the
  derivation entry takes only genuine ``bytes`` passphrase/salt
  (``TypeError`` otherwise) and positive genuine-integer parameters
  (``ValueError`` otherwise);

* with warning policies turned on, the tail of every entry's output --
  success and failure, including the argparse rejection -- carries no
  unclosed-resource warning line;

* the same inputs run twice (against one vault or two fresh vaults) give
  a byte-identical transcript once the genuinely floating token, the
  temporary directory name, is substituted away; the program prints no
  timing data.

Cases read and write only inside their temporary directories, share no
state and therefore do not depend on execution order; the shared fixture
returns every lock handle and deletes every tree at teardown::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    ACTIVATIONS_NAME,
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


class CliFailureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.root

    # ------------------------------------------------------------------
    # subprocess plumbing
    # ------------------------------------------------------------------

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
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

    def material_file(self, name: str, data: bytes) -> Path:
        path = self.tmp_path / name
        path.write_bytes(data)
        return path

    def all_entries(self, material: Path) -> list[tuple[str, ...]]:
        """The three entry points invoked the README way."""
        return [
            ("versions",),
            ("reload",),
            ("seal", "k", "--material-file", str(material)),
        ]

    def disk_snapshot(self, root: Path | None = None) -> dict[str, bytes]:
        """Every non-lock file under the vault, keyed by its relative path."""
        target = self.root if root is None else root
        files: dict[str, bytes] = {}
        for path in sorted(target.rglob("*")):
            if path.is_file() and path.name != LOCK_NAME:
                files[str(path.relative_to(target))] = path.read_bytes()
        return files

    def material_records(self, root: Path | None = None) -> list[bytes]:
        target = self.root if root is None else root
        return sorted(
            path.read_bytes()
            for path in (target / MATERIALS_DIR).rglob("*.bin")
        )

    def stored_material_path(self, key_id: str, version: int) -> Path:
        manifest = json.loads(self.manifest_bytes().decode("utf-8"))
        for record in manifest["keys"][key_id]["versions"]:
            if record["version"] == version:
                return self.root / record["file"]
        raise KeyError((key_id, version))

    def manifest_bytes(self) -> bytes:
        return (self.root / MANIFEST_NAME).read_bytes()


# ---------------------------------------------------------------------------
# wrong invocation shape: argparse rejection, exit code 2, frozen stderr
# ---------------------------------------------------------------------------


class TestMalformedInvocationExitTwo(CliFailureTestCase):
    def assertRejectedBeforeVault(
        self, argv: tuple[str, ...], expected_stderr: str
    ) -> None:
        """A malformed shape exits 2, prints nothing to stdout and never
        opens (or creates) the vault directory."""
        target = self.tmp_path / "never-opened"
        self.assertFalse(target.exists())
        result = self.run_cli(*argv, root=target)
        self.assertEqual(result.returncode, 2, argv)
        self.assertEqual(result.stdout, "", argv)
        self.assertEqual(result.stderr, expected_stderr, argv)
        self.assertFalse(target.exists(), argv)

    def test_root_without_value_exits_two(self):
        # "exact argv": argparse rejects before a root is ever named.
        result = self.run_cli("--root", root=None)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            _ROOT_USAGE
            + "keyvault_ledger: error: argument --root: expected one "
            "argument\n",
        )

    def test_material_file_without_value_exits_two(self):
        self.assertRejectedBeforeVault(
            ("seal", "k", "--material-file"),
            _SEAL_USAGE
            + "keyvault_ledger seal: error: argument --material-file: "
            "expected one argument\n",
        )

    def test_unknown_subcommand_exits_two(self):
        self.assertRejectedBeforeVault(
            ("bogus",),
            _ROOT_USAGE
            + "keyvault_ledger: error: argument command: invalid choice: "
            "'bogus' (choose from versions, seal, reload)\n",
        )

    def test_unknown_option_on_seal_exits_two(self):
        self.assertRejectedBeforeVault(
            ("seal", "k", "--material-file", "f", "--bogus"),
            _ROOT_USAGE
            + "keyvault_ledger: error: unrecognized arguments: --bogus\n",
        )

    def test_extra_positional_on_seal_exits_two(self):
        self.assertRejectedBeforeVault(
            ("seal", "k", "--material-file", "f", "extra"),
            _ROOT_USAGE
            + "keyvault_ledger: error: unrecognized arguments: extra\n",
        )

    def test_extra_positional_on_reload_exits_two(self):
        self.assertRejectedBeforeVault(
            ("reload", "extra"),
            _ROOT_USAGE
            + "keyvault_ledger: error: unrecognized arguments: extra\n",
        )

    def test_short_unknown_option_on_versions_exits_two(self):
        self.assertRejectedBeforeVault(
            ("versions", "-x"),
            _ROOT_USAGE
            + "keyvault_ledger: error: unrecognized arguments: -x\n",
        )

    def test_root_repeated_after_subcommand_exits_two(self):
        self.assertRejectedBeforeVault(
            ("versions", "--root", "x"),
            _ROOT_USAGE
            + "keyvault_ledger: error: unrecognized arguments: --root x\n",
        )

    def test_option_unknown_to_reload_exits_two(self):
        self.assertRejectedBeforeVault(
            ("reload", "--material-file", "x"),
            _ROOT_USAGE
            + "keyvault_ledger: error: unrecognized arguments: "
            "--material-file x\n",
        )

    def test_every_bad_shape_is_rejected_consistently_twice(self):
        # The same bad shape run twice in a row is rejected identically;
        # the first rejection leaves no state behind to change the second.
        shapes = (
            ("bogus",),
            ("seal", "k"),
            ("reload", "extra"),
            ("versions", "-x"),
        )
        for argv in shapes:
            first = self.run_cli(*argv)
            second = self.run_cli(*argv)
            self.assertEqual((first.returncode, first.stdout), (2, ""), argv)
            self.assertEqual(
                (second.returncode, second.stdout, second.stderr),
                (first.returncode, first.stdout, first.stderr),
                argv,
            )


# ---------------------------------------------------------------------------
# filesystem failures: exit code 1, frozen error line, append-only disk
# ---------------------------------------------------------------------------


class TestFilesystemFailurePaths(CliFailureTestCase):
    def test_root_that_is_a_file_is_rejected_by_every_entry(self):
        material = self.material_file("m.bin", b"m")
        root = self.tmp_path / "a-file"
        root.write_bytes(b"not a directory")
        expected = f"error: [Errno 17] File exists: '{root}'\n"
        for argv in self.all_entries(material):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        # The plain file is neither replaced by a directory nor altered.
        self.assertTrue(root.is_file())
        self.assertEqual(root.read_bytes(), b"not a directory")

    def test_root_through_a_file_component_is_rejected_by_every_entry(self):
        material = self.material_file("m.bin", b"m")
        blocker = self.tmp_path / "blocker"
        blocker.write_bytes(b"not a directory")
        root = blocker / "sub" / "vault"
        expected = f"error: [Errno 20] Not a directory: '{root}'\n"
        for argv in self.all_entries(material):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        self.assertEqual(blocker.read_bytes(), b"not a directory")

    def test_materials_entry_occupied_by_a_file_is_rejected(self):
        # A vault-looking directory whose materials entry is a plain file
        # fails while the root is being prepared, for all three entries.
        material = self.material_file("m.bin", b"m")
        root = self.tmp_path / "blocked-materials"
        root.mkdir()
        occupied = root / MATERIALS_DIR
        occupied.write_bytes(b"not a directory")
        expected = f"error: [Errno 17] File exists: '{occupied}'\n"
        for argv in self.all_entries(material):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        # No manifest is silently created and the file is untouched.
        self.assertFalse((root / MANIFEST_NAME).exists())
        self.assertEqual(occupied.read_bytes(), b"not a directory")

    def test_populated_directory_without_manifest_errors_for_all_entries(self):
        material = self.material_file("m.bin", b"m")
        root = self.tmp_path / "stray"
        root.mkdir()
        stray = root / "notes.txt"
        stray.write_bytes(b"not a vault")
        expected = f"error: manifest missing: {root / MANIFEST_NAME}\n"
        for argv in self.all_entries(material):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        self.assertFalse((root / MANIFEST_NAME).exists())
        self.assertEqual(stray.read_bytes(), b"not a vault")

    def test_deleting_manifest_after_init_errors_for_all_entries(self):
        material = self.material_file("m.bin", b"m")
        sealed = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(sealed.returncode, 0)
        sealed_materials = self.material_records()

        manifest = self.root / MANIFEST_NAME
        manifest.unlink()
        expected = f"error: manifest missing: {manifest}\n"
        for argv in self.all_entries(material):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        self.assertFalse(manifest.exists())
        self.assertEqual(self.material_records(), sealed_materials)

    def test_missing_seal_material_exits_one_and_mints_no_version(self):
        missing = self.tmp_path / "absent.bin"
        result = self.run_cli("seal", "k", "--material-file", str(missing))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            f"error: [Errno 2] No such file or directory: '{missing}'\n",
        )
        # The brand-new vault lists nothing and holds no material record.
        listing = self.run_cli("versions")
        self.assertEqual(listing.returncode, 0)
        self.assertEqual(listing.stdout, "")
        self.assertEqual(self.material_records(), [])

    def test_directory_as_seal_material_exits_one(self):
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
    def test_unreadable_seal_material_exits_one(self):
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

    def test_empty_key_id_seal_exits_one_without_a_record(self):
        material = self.material_file("m.bin", b"m")
        result = self.run_cli("seal", "", "--material-file", str(material))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "error: key_id must not be empty\n")
        # The ValueError crosses the CLI's handled-error path, not argparse.
        self.assertEqual(self.run_cli("versions").stdout, "")
        self.assertEqual(self.material_records(), [])

    def test_missing_stored_material_fails_every_entry_with_frozen_line(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        stored = self.stored_material_path("k", 1)
        stored.unlink()
        manifest_before = self.manifest_bytes()
        expected = "error: material missing for key 'k' version 1\n"
        # seal re-validates on open too, so it fails before sealing anything.
        for argv in self.all_entries(material):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
            self.assertEqual(self.manifest_bytes(), manifest_before, argv)

    def test_stored_material_replaced_by_a_directory_fails_every_entry(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        stored = self.stored_material_path("k", 1)
        stored.unlink()
        stored.mkdir()
        expected = "error: material missing for key 'k' version 1\n"
        for argv in self.all_entries(material):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
        self.assertTrue(stored.is_dir())

    def test_mismatched_stored_material_fails_with_frozen_line(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        stored = self.stored_material_path("k", 1)
        stored.write_bytes(b"tampered")
        manifest_before = self.manifest_bytes()
        expected = "error: material mismatch for key 'k' version 1\n"
        for argv in (("versions",), ("reload",)):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(result.stderr, expected, argv)
            self.assertEqual(self.manifest_bytes(), manifest_before, argv)

    def test_failed_seals_leave_disk_append_only_then_versioning_continues(self):
        material = self.material_file("m.bin", b"m")
        first = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(first.stdout, "1\n")
        listing_before = self.run_cli("versions").stdout
        before = self.disk_snapshot()

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

        # Manifest/material records neither gained nor lost a byte.
        self.assertEqual(self.disk_snapshot(), before)
        self.assertEqual(self.run_cli("versions").stdout, listing_before)

        # The next successful seal resumes the existing sequence at 2:
        # version 1 is neither reused nor skipped.
        again = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual((again.returncode, again.stdout), (0, "2\n"))
        self.assertEqual(
            self.run_cli("versions").stdout,
            "k\tactive=2\tversions=1,2\n",
        )


# ---------------------------------------------------------------------------
# failed whole-vault reload: frozen complaint, snapshot and disk frozen
# ---------------------------------------------------------------------------


class TestReloadFailureStateAndDisk(CliFailureTestCase):
    def test_corrupt_manifest_fails_every_entry_and_touches_no_bytes(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        (self.root / MANIFEST_NAME).write_bytes(b"{not json")
        corrupt = self.manifest_bytes()
        materials_before = self.material_records()
        for argv in self.all_entries(material):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertTrue(
                result.stderr.startswith("error: manifest corrupt: "), argv
            )
            self.assertTrue(result.stderr.endswith("\n"), argv)
            self.assertEqual(self.manifest_bytes(), corrupt, argv)
            self.assertEqual(self.material_records(), materials_before, argv)

    def test_unsupported_format_line_is_frozen(self):
        material = self.material_file("m.bin", b"m")
        root = self.tmp_path / "bad-format"
        root.mkdir()
        (root / MATERIALS_DIR).mkdir(exist_ok=True)
        (root / REVOCATIONS_NAME).write_bytes(b"")
        (root / MANIFEST_NAME).write_bytes(
            json.dumps({"format": 999, "keys": {}}).encode()
        )
        for argv in self.all_entries(material):
            result = self.run_cli(*argv, root=root)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(
                result.stderr,
                "error: manifest format invalid: unsupported format\n",
                argv,
            )

    def test_derivation_length_mismatch_line_is_frozen(self):
        vault = self.fixture.open(self.root)
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        vault.close()
        manifest_path = self.root / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_bytes())
        record = manifest["keys"]["d"]["versions"][0]
        record["derivation"]["length"] = 8
        manifest_path.write_text(json.dumps(manifest))
        before = self.disk_snapshot()

        for argv in (("versions",), ("reload",)):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertEqual(
                result.stderr,
                "error: derivation length mismatch for key 'd' version 1\n",
            )
        self.assertEqual(self.disk_snapshot(), before)

    def test_corrupt_journal_lines_fail_and_leave_disk_byte_identical(self):
        material = self.material_file("m.bin", b"m")
        vault = self.fixture.open(self.root)
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        vault.seal("a", b"a1")
        vault.seal("a", b"a2")
        vault.set_active("a", 1)
        vault.close()

        revocations = self.root / REVOCATIONS_NAME
        activations = self.root / ACTIVATIONS_NAME
        healthy_revocations = revocations.read_bytes()
        healthy_activations = activations.read_bytes()
        materials_before = self.material_records()

        revocations.write_bytes(b"{garbage\n")
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertTrue(
            result.stderr.startswith("error: revocation journal corrupt: ")
        )
        self.assertEqual(revocations.read_bytes(), b"{garbage\n")
        self.assertEqual(activations.read_bytes(), healthy_activations)

        revocations.write_bytes(healthy_revocations)
        activations.write_bytes(b"{bad\n")
        result = self.run_cli("reload")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertTrue(
            result.stderr.startswith("error: activation journal corrupt: ")
        )
        # A failed reload is strictly read-only: nothing gained or lost.
        self.assertEqual(revocations.read_bytes(), healthy_revocations)
        self.assertEqual(activations.read_bytes(), b"{bad\n")
        self.assertEqual(self.material_records(), materials_before)
        # A seal attempted while validation fails fails the open and so
        # cannot append a version.
        blocked = self.run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(blocked.returncode, 1)
        self.assertTrue(
            blocked.stderr.startswith("error: activation journal corrupt: ")
        )
        self.assertEqual(self.material_records(), materials_before)

        # Restoring the journal makes reload healthy again.
        activations.write_bytes(healthy_activations)
        recovered = self.run_cli("reload")
        self.assertEqual((recovered.returncode, recovered.stdout), (0, "reloaded\n"))

    def test_live_handle_snapshot_and_disk_survive_failing_reloads(self):
        vault = self.fixture.open(self.root)
        vault.seal("k", b"one")
        vault.seal("k", b"two")
        vault.derive_seal("d", b"pw", b"salty", 100, 16)

        answers_before = {
            "versions-k": vault.versions("k"),
            "active-k": vault.active("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "load-active": vault.load("k"),
            "versions-d": vault.versions("d"),
            "derivation-d": vault.derivation("d", 1),
        }
        healthy_manifest = self.manifest_bytes()

        (self.root / MANIFEST_NAME).write_bytes(b"{broken")
        # The deliberately corrupt bytes are now the on-disk state; the
        # failed reloads must not change *that* either (no repair, no
        # further growth or truncation).
        disk_during_failure = self.disk_snapshot()

        # The subprocess entry fails ...
        for argv in (("versions",), ("reload",)):
            result = self.run_cli(*argv)
            self.assertEqual(result.returncode, 1, argv)
            self.assertEqual(result.stdout, "", argv)
            self.assertTrue(
                result.stderr.startswith("error: manifest corrupt: "), argv
            )
        # ... and repeating the failing query gives the verbatim same line.
        self.assertEqual(
            self.run_cli("reload").stderr, result.stderr
        )

        # ... while the live in-process handle keeps its old snapshot:
        # the failed swap leaves keys already in hand exactly readable.
        with self.assertRaises(ValueError):
            vault.reload()
        answers_after = {
            "versions-k": vault.versions("k"),
            "active-k": vault.active("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "load-active": vault.load("k"),
            "versions-d": vault.versions("d"),
            "derivation-d": vault.derivation("d", 1),
        }
        self.assertEqual(answers_after, answers_before)

        # Disk records did not gain or lose a byte during the failures.
        self.assertEqual(self.disk_snapshot(), disk_during_failure)

        # Correcting the bytes restores both the entries and the handle.
        (self.root / MANIFEST_NAME).write_bytes(healthy_manifest)
        recovered = self.run_cli("versions")
        self.assertEqual(recovered.returncode, 0)
        self.assertEqual(
            recovered.stdout,
            "d\tactive=1\tversions=1\n"
            "k\tactive=2\tversions=1,2\n",
        )
        vault.reload()
        self.assertEqual(vault.load("k", 2), b"two")


# ---------------------------------------------------------------------------
# a failed seal after a failed reload still resumes the version sequence
# ---------------------------------------------------------------------------


class TestVersionSequenceSurvivesFailures(CliFailureTestCase):
    def test_failed_reload_then_failed_seal_then_seal_is_version_two(self):
        material = self.material_file("m.bin", b"m")
        self.assertEqual(
            self.run_cli("seal", "k", "--material-file", str(material)).stdout,
            "1\n",
        )
        stored = self.stored_material_path("k", 1)
        stored.unlink()

        # A failing reload (twice) must not allocate or retire a version.
        for _ in range(2):
            broken = self.run_cli("reload")
            self.assertEqual(broken.returncode, 1)
        # A failed seal attempt during the broken window fails the same way.
        broken_seal = self.run_cli(
            "seal", "k", "--material-file", str(material)
        )
        self.assertEqual(broken_seal.returncode, 1)

        stored.write_bytes(b"m")
        resumed = self.run_cli(
            "seal", "k", "--material-file", str(material)
        )
        self.assertEqual((resumed.returncode, resumed.stdout), (0, "2\n"))
        self.assertEqual(
            self.run_cli("versions").stdout,
            "k\tactive=2\tversions=1,2\n",
        )

    def test_failures_on_one_key_do_not_move_another_key_sequence(self):
        material = self.material_file("m.bin", b"m")
        missing = self.tmp_path / "absent.bin"
        self.assertEqual(
            self.run_cli("seal", "a", "--material-file", str(material)).stdout,
            "1\n",
        )
        for argv in (
            ("seal", "b", "--material-file", str(missing)),
            ("seal", "", "--material-file", str(material)),
        ):
            self.assertEqual(self.run_cli(*argv).returncode, 1, argv)
        # Key b's first real version is still 1; key a continues at 2.
        self.assertEqual(
            self.run_cli("seal", "b", "--material-file", str(material)).stdout,
            "1\n",
        )
        self.assertEqual(
            self.run_cli("seal", "a", "--material-file", str(material)).stdout,
            "2\n",
        )
        listing = self.run_cli("versions")
        self.assertEqual(
            listing.stdout,
            "a\tactive=2\tversions=1,2\n"
            "b\tactive=1\tversions=1\n",
        )


# ---------------------------------------------------------------------------
# read-only identifier queries: empty/unknown listings, exception vocabulary
# ---------------------------------------------------------------------------


class TestIdentifierQueryContract(CliFailureTestCase):
    def open_vault(self) -> Vault:
        return self.fixture.open(self.root)

    def test_version_listing_is_empty_for_empty_and_unknown_ids(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        # Empty and unknown both answer an empty list, never an error.
        self.assertEqual(vault.versions(""), [])
        self.assertEqual(vault.versions("never-sealed"), [])
        self.assertEqual(vault.versions("other"), [])

    def test_revoked_listing_empty_for_unknown_but_value_error_for_empty(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        self.assertEqual(vault.revoked_versions("never-sealed"), [])
        self.assertEqual(vault.revoked_versions("ghost"), [])
        self.assertEqual(vault.revoked_versions("k"), [1])
        with self.assertRaises(ValueError):
            vault.revoked_versions("")

    def test_reads_unknown_key_is_key_error_and_bad_version_type_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.load("never-sealed")
        with self.assertRaises(KeyError):
            vault.load("never-sealed", 1)
        with self.assertRaises(KeyError):
            vault.load("k", 2)
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed")
        with self.assertRaises(KeyError):
            vault.derivation("k", 2)
        for bad in (1.0, True, "1"):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derivation("k", bad)
        # The genuine int still reaches the record.
        self.assertEqual(vault.load("k", 1), b"m")
        self.assertEqual(vault.derivation("k", 1), {})


# ---------------------------------------------------------------------------
# material and derivation entry-point type/value contracts
# ---------------------------------------------------------------------------


class TestMaterialAndDerivationContracts(CliFailureTestCase):
    def open_vault(self) -> Vault:
        return self.fixture.open(self.root)

    def test_seal_accepts_bytes_bytearray_memoryview_only(self):
        vault = self.open_vault()
        for material in (b"bytes", bytearray(b"byte-array"), memoryview(b"view")):
            version = vault.seal("k", material)
            recovered = vault.load("k", version)
            self.assertIs(type(recovered), bytes)
            self.assertEqual(recovered, bytes(material))
        for bad in ("text", 1, 1.5, None, [b"x"], {"k": b"v"}, object()):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.seal("k", bad)

    def test_derivation_password_and_salt_must_be_genuine_bytes(self):
        vault = self.open_vault()
        for bad in ("text", 1, None, [b"x"], bytearray(b"s"), memoryview(b"s")):
            with self.assertRaises(TypeError, msg=f"password {bad!r}"):
                vault.derive_seal("k", bad, b"salt", 1, 1)
            with self.assertRaises(TypeError, msg=f"salt {bad!r}"):
                vault.derive_seal("k", b"pw", bad, 1, 1)
        # Genuine bytes derive, seal and round-trip; a CLI reload agrees.
        password, salt, iterations, length = b"pw", b"salty", 500, 32
        version = vault.derive_seal("k", password, salt, iterations, length)
        self.assertEqual(version, 1)
        self.assertEqual(
            vault.derivation("k", version),
            {"salt": salt, "iterations": iterations, "length": length},
        )
        expected = hashlib.pbkdf2_hmac(
            "sha256", password, salt, iterations, dklen=length
        )
        self.assertEqual(vault.load("k", version), expected)
        result = self.run_cli("reload")
        self.assertEqual((result.returncode, result.stdout), (0, "reloaded\n"))

    def test_derivation_parameters_type_then_value_contract(self):
        vault = self.open_vault()
        # Type gate first: bools and floats are TypeErrors even at value 1.
        for bad in (True, False, 1.0, 2.5, "1", None):
            with self.assertRaises(TypeError, msg=f"iterations {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(TypeError, msg=f"length {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)
        # A genuine non-positive int is the ValueError path instead.
        for bad in (0, -1, -1000):
            with self.assertRaises(ValueError, msg=f"iterations {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(ValueError, msg=f"length {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 1)
        # None of the rejected calls sealed anything.
        self.assertEqual(vault.versions("k"), [])


# ---------------------------------------------------------------------------
# warning policies: no unclosed-resource warning at any entry's tail
# ---------------------------------------------------------------------------


class TestNoResourceWarningAtTail(CliFailureTestCase):
    POLICIES = ("always::ResourceWarning", "error::ResourceWarning", "always")
    _WARNING_RE = re.compile(
        r"resourcewarning|unclosed|exception ignored in|unraisablehook|"
        r"still running",
        re.IGNORECASE,
    )

    def assertCleanTail(self, result: subprocess.CompletedProcess) -> None:
        self.assertIsNone(
            self._WARNING_RE.search(result.stderr),
            f"warning text leaked into stderr: {result.stderr!r}",
        )
        self.assertIsNone(
            self._WARNING_RE.search(result.stdout),
            f"warning text leaked into stdout: {result.stdout!r}",
        )

    def test_every_entry_is_clean_under_every_policy(self):
        material = self.material_file("m.bin", b"m")
        for policy in self.POLICIES:
            root = self.tmp_path / f"root-{policy.replace(':', '-')}"
            with self.subTest(policy=policy):
                versions = self.run_cli("versions", root=root, warnings=policy)
                self.assertEqual(versions.returncode, 0, versions.stderr)
                self.assertEqual(versions.stderr, "")
                reload_result = self.run_cli(
                    "reload", root=root, warnings=policy
                )
                self.assertEqual(reload_result.returncode, 0)
                self.assertEqual(reload_result.stderr, "")
                seal_result = self.run_cli(
                    "seal", "k", "--material-file", str(material),
                    root=root, warnings=policy,
                )
                self.assertEqual(seal_result.returncode, 0, seal_result.stderr)
                self.assertEqual(seal_result.stderr, "")
                self.assertCleanTail(seal_result)

    def test_failure_paths_are_clean_under_always_warnings(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        self.stored_material_path("k", 1).unlink()
        missing = self.tmp_path / "absent.bin"

        cases = (
            ("seal", "", "--material-file", str(material)),
            ("seal", "k", "--material-file", str(missing)),
            ("reload",),
            ("versions",),
            ("bogus",),
        )
        for argv in cases:
            with self.subTest(argv=argv):
                result = self.run_cli(*argv, warnings="always")
                self.assertIn(result.returncode, (1, 2), argv)
                self.assertCleanTail(result)
                # The handled-error lines stay a single frozen line; only
                # the argparse rejection legitimately has its usage prefix.
                if result.returncode == 1:
                    self.assertEqual(result.stderr.count("\n"), 1)
                    self.assertTrue(result.stderr.startswith("error: "))


# ---------------------------------------------------------------------------
# same inputs twice: byte-identical transcripts after floating-path removal
# ---------------------------------------------------------------------------


class TestRepeatedInvocationDeterminism(CliFailureTestCase):
    # Commands run, in order, against each identically prepared root.  The
    # shared material and its absent sibling live outside the roots, so
    # their paths are identical across transcripts; only the root name is
    # a floating token.
    SCRIPT = (
        ("versions",),
        ("reload",),
        ("seal", "k", "--material-file", "__MATERIAL__"),
        ("seal", "k", "--material-file", "__MISSING__"),
        ("seal", "", "--material-file", "__MATERIAL__"),
        ("bogus",),
        ("versions",),
    )

    def _run_script(self, root: Path, material: Path, missing: Path):
        transcript = []
        for argv in self.SCRIPT:
            argv = tuple(
                str(missing)
                if token == "__MISSING__"
                else str(material)
                if token == "__MATERIAL__"
                else token
                for token in argv
            )
            result = self.run_cli(*argv, root=root)
            transcript.append((result.returncode, result.stdout, result.stderr))
        return transcript

    def test_script_in_two_fresh_vaults_matches_after_root_substitution(self):
        material = self.material_file("shared-material.bin", b"m")
        missing = self.tmp_path / "shared-absent.bin"
        root_a = self.tmp_path / "a"
        root_b = self.tmp_path / "b"

        def normalise(transcript, root):
            return [
                (
                    code,
                    out,
                    err.replace(str(root), "<ROOT>"),
                )
                for code, out, err in transcript
            ]

        first = normalise(self._run_script(root_a, material, missing), root_a)
        second = normalise(self._run_script(root_b, material, missing), root_b)
        self.assertEqual(second, first)
        # Spot-check the frozen anchor lines inside the shared transcript.
        self.assertEqual(first[0], (0, "", ""))
        self.assertEqual(first[1], (0, "reloaded\n", ""))
        self.assertEqual(first[2], (0, "1\n", ""))
        self.assertEqual(first[3][0], 1)
        self.assertEqual(first[4], (1, "", "error: key_id must not be empty\n"))
        self.assertEqual(first[5][0], 2)
        self.assertEqual(
            first[6],
            (0, "k\tactive=1\tversions=1\n", ""),
        )

    def test_manifest_missing_failure_matches_in_two_vaults(self):
        material = self.material_file("m.bin", b"m")
        root_a = self.tmp_path / "miss-a"
        root_b = self.tmp_path / "miss-b"
        for root in (root_a, root_b):
            root.mkdir()
            (root / "notes.txt").write_bytes(b"not a vault")

        def run(root):
            return self.run_cli("reload", root=root)

        first = run(root_a)
        second = run(root_b)
        self.assertEqual((first.returncode, first.stdout), (1, ""))
        self.assertEqual((second.returncode, second.stdout), (1, ""))
        self.assertEqual(
            first.stderr.replace(str(root_a), "<ROOT>"),
            second.stderr.replace(str(root_b), "<ROOT>"),
        )
        self.assertEqual(
            first.stderr.replace(str(root_a), "<ROOT>"),
            "error: manifest missing: <ROOT>/manifest.json\n",
        )

    def test_failing_reload_run_twice_against_one_root_matches(self):
        material = self.material_file("m.bin", b"m")
        self.run_cli("seal", "k", "--material-file", str(material))
        self.stored_material_path("k", 1).unlink()
        manifest_before = self.manifest_bytes()

        first = self.run_cli("reload")
        second = self.run_cli("reload")
        self.assertEqual(
            (second.returncode, second.stdout, second.stderr),
            (first.returncode, first.stdout, first.stderr),
        )
        self.assertEqual(
            first.stderr, "error: material missing for key 'k' version 1\n"
        )
        # Repeating the failed query writes nothing either time.
        self.assertEqual(self.manifest_bytes(), manifest_before)


if __name__ == "__main__":
    unittest.main()
