"""Regression tests pinning the command line's three exit paths verbatim.

The baseline CLI already returns its vault lock handle on every exit path
(success, handled error, argparse rejection); these cases only freeze the
observable behaviour of the three entry points published in the README
(``versions``, ``seal``, ``reload``).  No product code and no existing test
is touched.

Pinned here, each observed through a real ``python3 -m keyvault_ledger``
subprocess running against a temporary vault, capturing stdout, stderr and
the exit code:

* success paths: the exact stdout text (whitespace, tabs and newlines
  included), an empty stderr and exit code 0 for all three subcommands;
* invocation forms written wrong: argparse exits with code 2, writes nothing
  to stdout, writes its frozen usage/error text to stderr and never even
  creates the vault directory;
* failure paths: a missing manifest, an unreadable/missing material file
  (including one inside a directory that does not exist) and a vault root
  that is a regular file each exit with code 1 and their frozen
  ``error: ...`` line; no failure produces half a new version — the manifest
  and the material records on disk neither grow nor shrink and are never
  truncated;
* a whole-vault reload that fails validation leaves the in-memory snapshot
  of a handle opened on the same directory untouched (keys already in hand
  stay readable) and the disk records byte-for-byte unchanged;
* with the warning policy turned up (``-X dev -W always::ResourceWarning``)
  every subcommand's output stays clean: no "unclosed resource" style
  warning tail on stdout or stderr, because the handle is always returned;
* determinism: the same scripted sequence of inputs run twice yields
  byte-for-byte identical transcripts once floating content (the temporary
  directory names) is normalised away.

The library contract is not re-specified here and stays exactly as the
existing suite pins it (empty id -> ``ValueError``, non-integer version ->
``TypeError``, unknown key/version -> ``KeyError``, non-bytes material ->
``TypeError``; ``close()`` is idempotent and later operations transparently
reacquire the same lock with no observable change).

Every case works inside its own temporary directory, uses the standard
library only and is independent of execution order — running the suite
forwards, backwards or selectively gives the same result::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from keyvault_ledger.vault import LOCK_NAME, MANIFEST_NAME
from tests._fixtures import VaultFixture, cli_env

# Turn the warning policy all the way up: any unclosed-resource warning the
# CLI process emitted would show up as a tail line on its stderr.
WARNING_FLAGS = ("-X", "dev", "-W", "always::ResourceWarning")

USAGE_MAIN = "usage: keyvault_ledger [-h] --root ROOT {versions,seal,reload} ...\n"
USAGE_SEAL = "usage: keyvault_ledger seal [-h] --material-file MATERIAL_FILE key_id\n"

# Invocation forms written wrong, with the stderr argparse produces today,
# frozen verbatim.  Every one of them exits with code 2, writes nothing to
# stdout and never reaches the point of creating the vault directory.
ARGPARSE_CASES = [
    (
        "no_arguments",
        [],
        USAGE_MAIN
        + "keyvault_ledger: error: the following arguments are required: "
        "--root, command\n",
    ),
    (
        "root_without_command",
        ["--root", "{root}"],
        USAGE_MAIN
        + "keyvault_ledger: error: the following arguments are required: command\n",
    ),
    (
        "command_without_root",
        ["versions"],
        USAGE_MAIN
        + "keyvault_ledger: error: the following arguments are required: --root\n",
    ),
    (
        "unknown_command",
        ["--root", "{root}", "frobnicate"],
        USAGE_MAIN
        + "keyvault_ledger: error: argument command: invalid choice: "
        "'frobnicate' (choose from versions, seal, reload)\n",
    ),
    (
        "seal_missing_material_option",
        ["--root", "{root}", "seal", "k"],
        USAGE_SEAL
        + "keyvault_ledger seal: error: the following arguments are required: "
        "--material-file\n",
    ),
    (
        "seal_missing_key_id",
        ["--root", "{root}", "seal"],
        USAGE_SEAL
        + "keyvault_ledger seal: error: the following arguments are required: "
        "key_id, --material-file\n",
    ),
    (
        "versions_unexpected_argument",
        ["--root", "{root}", "versions", "extra"],
        USAGE_MAIN + "keyvault_ledger: error: unrecognized arguments: extra\n",
    ),
    (
        "reload_unexpected_argument",
        ["--root", "{root}", "reload", "extra"],
        USAGE_MAIN + "keyvault_ledger: error: unrecognized arguments: extra\n",
    ),
    (
        "unknown_option",
        ["--root", "{root}", "--bogus", "versions"],
        USAGE_MAIN + "keyvault_ledger: error: unrecognized arguments: --bogus\n",
    ),
    (
        "root_missing_value",
        ["--root"],
        USAGE_MAIN + "keyvault_ledger: error: argument --root: expected one argument\n",
    ),
]


class CliExitPathTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.root = self.fixture.root

    # ------------------------------------------------------------------
    # subprocess helpers
    # ------------------------------------------------------------------

    def run_cli(self, argv: list[str], python_flags: tuple[str, ...] = ()):
        """Run the CLI as a real subprocess and capture all three channels."""
        return subprocess.run(
            [sys.executable, *python_flags, "-m", "keyvault_ledger", *argv],
            capture_output=True,
            text=True,
            env=cli_env(),
        )

    def normalize(self, text: str, base: Path) -> str:
        """Remove floating content (this case's temporary directory names)."""
        return text.replace(str(base), "<TMP>")

    def disk_bytes(self, root: Path) -> dict[str, bytes]:
        """All vault records on disk keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, so it is deliberately excluded.
        """
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }


# ---------------------------------------------------------------------------
# success paths: frozen stdout, clean stderr, exit code 0
# ---------------------------------------------------------------------------


class TestSuccessPathsFrozen(CliExitPathTestCase):
    def test_versions_seal_reload_outputs_are_frozen_verbatim(self):
        material = self.fixture.path("material.bin")
        material.write_bytes(b"frozen-material")

        def cli(*argv: str):
            return self.run_cli(["--root", str(self.root), *argv])

        # A fresh vault directory is created on first use; an empty vault
        # lists nothing at all.
        result = cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

        # seal prints exactly the new version number and one newline.
        result = cli("seal", "alpha", "--material-file", str(material))
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "1\n", ""))
        result = cli("seal", "alpha", "--material-file", str(material))
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "2\n", ""))
        result = cli("seal", "beta", "--material-file", str(material))
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "1\n", ""))

        # versions prints one tab-separated line per key, keys sorted.
        result = cli("versions")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            result.stdout,
            "alpha\tactive=2\tversions=1,2\nbeta\tactive=1\tversions=1\n",
        )
        self.assertEqual(result.stderr, "")

        # reload prints exactly "reloaded" and one newline.
        result = cli("reload")
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "reloaded\n", ""))

    def test_same_inputs_twice_give_byte_identical_transcripts(self):
        # One scripted sequence of inputs, run against two independent fresh
        # vaults: once the temporary directory names are normalised away the
        # two transcripts must be verbatim identical, and both must equal the
        # frozen expectation.
        expected = [
            ("versions-empty", 0, "", ""),
            ("seal-first", 0, "1\n", ""),
            ("seal-second", 0, "2\n", ""),
            ("seal-other-key", 0, "1\n", ""),
            (
                "versions-two-keys",
                0,
                "alpha\tactive=2\tversions=1,2\nbeta\tactive=1\tversions=1\n",
                "",
            ),
            ("reload", 0, "reloaded\n", ""),
            (
                "seal-missing-material",
                1,
                "",
                "error: [Errno 2] No such file or directory: '<TMP>/gone.bin'\n",
            ),
            (
                "versions-after-failed-seal",
                0,
                "alpha\tactive=2\tversions=1,2\nbeta\tactive=1\tversions=1\n",
                "",
            ),
            (
                "reload-manifest-missing",
                1,
                "",
                "error: manifest missing: <TMP>/vault/manifest.json\n",
            ),
            (
                "versions-manifest-missing",
                1,
                "",
                "error: manifest missing: <TMP>/vault/manifest.json\n",
            ),
        ]

        transcripts = [self._scenario("a"), self._scenario("b")]
        self.assertEqual(transcripts[0], expected)
        self.assertEqual(transcripts[1], expected)
        self.assertEqual(transcripts[0], transcripts[1])

    def _scenario(self, tag: str) -> list[tuple[str, int, str, str]]:
        """Run one fixed input sequence against a fresh vault; return the
        normalised ``(label, exit code, stdout, stderr)`` transcript."""
        base = self.fixture.path(f"scenario-{tag}")
        root = base / "vault"
        material = base / "material.bin"
        base.mkdir(parents=True)
        material.write_bytes(b"scenario-material")
        missing = base / "gone.bin"

        transcript = []

        def step(label: str, argv: list[str]) -> None:
            result = self.run_cli(argv)
            transcript.append(
                (
                    label,
                    result.returncode,
                    self.normalize(result.stdout, base),
                    self.normalize(result.stderr, base),
                )
            )

        step("versions-empty", ["--root", str(root), "versions"])
        step("seal-first", ["--root", str(root), "seal", "alpha", "--material-file", str(material)])
        step("seal-second", ["--root", str(root), "seal", "alpha", "--material-file", str(material)])
        step("seal-other-key", ["--root", str(root), "seal", "beta", "--material-file", str(material)])
        step("versions-two-keys", ["--root", str(root), "versions"])
        step("reload", ["--root", str(root), "reload"])
        step("seal-missing-material", ["--root", str(root), "seal", "gamma", "--material-file", str(missing)])
        step("versions-after-failed-seal", ["--root", str(root), "versions"])
        (root / MANIFEST_NAME).unlink()
        step("reload-manifest-missing", ["--root", str(root), "reload"])
        step("versions-manifest-missing", ["--root", str(root), "versions"])
        return transcript


# ---------------------------------------------------------------------------
# invocation forms written wrong: argparse exits 2 with frozen stderr
# ---------------------------------------------------------------------------


class TestInvocationErrorsExit2(CliExitPathTestCase):
    def test_bad_invocation_forms_exit_2_with_frozen_stderr(self):
        for name, argv_template, expected_stderr in ARGPARSE_CASES:
            with self.subTest(form=name):
                root = self.fixture.path(f"argparse-{name}")
                argv = [item.replace("{root}", str(root)) for item in argv_template]
                # The same wrong input twice must give the same outcome.
                for _ in range(2):
                    result = self.run_cli(argv)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(result.stderr, expected_stderr)
                # argparse rejects the call before any vault is opened: the
                # directory is never even created.
                self.assertFalse(root.exists())


# ---------------------------------------------------------------------------
# failure paths: exit code 1, frozen error line, no half-written records
# ---------------------------------------------------------------------------


class TestFailurePathsExit1(CliExitPathTestCase):
    def _healthy_vault(self) -> Path:
        material = self.fixture.path("material.bin")
        material.write_bytes(b"failure-path-material")
        for key_id in ("alpha", "beta"):
            result = self.run_cli(
                ["--root", str(self.root), "seal", key_id, "--material-file", str(material)]
            )
            self.assertEqual(result.returncode, 0)
        return material

    def test_seal_with_unreadable_material_fails_frozen_and_writes_nothing(self):
        material = self._healthy_vault()
        disk_before = self.disk_bytes(self.root)

        tmp = self.fixture.tmp_path
        cases = [
            (
                "material_file_missing",
                self.fixture.path("no-such.bin"),
                f"error: [Errno 2] No such file or directory: '{tmp / 'no-such.bin'}'\n",
            ),
            (
                "material_file_directory_missing",
                self.fixture.path("no-such-dir") / "material.bin",
                "error: [Errno 2] No such file or directory: "
                f"'{tmp / 'no-such-dir' / 'material.bin'}'\n",
            ),
            (
                "material_file_is_a_directory",
                tmp,
                f"error: [Errno 21] Is a directory: '{tmp}'\n",
            ),
        ]
        for name, bad_path, expected_stderr in cases:
            with self.subTest(case=name):
                argv = [
                    "--root",
                    str(self.root),
                    "seal",
                    "gamma",
                    "--material-file",
                    str(bad_path),
                ]
                for _ in range(2):
                    result = self.run_cli(argv)
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(result.stderr, expected_stderr)
                # No half of a new version: the manifest and every material
                # record on disk are byte-for-byte what they were.
                self.assertEqual(self.disk_bytes(self.root), disk_before)

        # The vault is still fully usable afterwards; the next seal continues
        # the sequence untouched by the failures.
        result = self.run_cli(
            ["--root", str(self.root), "seal", "alpha", "--material-file", str(material)]
        )
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "2\n", ""))

    def test_root_that_is_a_regular_file_fails_frozen(self):
        file_root = self.fixture.path("a-file")
        file_root.write_bytes(b"not a directory")
        for command in ("versions", "reload"):
            with self.subTest(command=command):
                result = self.run_cli(["--root", str(file_root), command])
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertEqual(
                    result.stderr,
                    f"error: [Errno 17] File exists: '{file_root}'\n",
                )

    def test_missing_manifest_fails_frozen_and_writes_nothing(self):
        self._healthy_vault()
        (self.root / MANIFEST_NAME).unlink()
        disk_before = self.disk_bytes(self.root)
        expected_stderr = f"error: manifest missing: {self.root / MANIFEST_NAME}\n"

        for command in ("versions", "reload"):
            with self.subTest(command=command):
                for _ in range(2):
                    result = self.run_cli(["--root", str(self.root), command])
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(result.stderr, expected_stderr)
        # The failed runs neither repaired nor removed anything on disk.
        self.assertEqual(self.disk_bytes(self.root), disk_before)
        self.assertFalse((self.root / MANIFEST_NAME).exists())

    def test_failed_reload_keeps_snapshot_and_disk_verbatim(self):
        # A handle opened in this process holds a validated snapshot; a CLI
        # reload failing whole-vault validation must not disturb either that
        # snapshot or a single byte on disk.
        vault = self.fixture.open()
        vault.seal("alpha", b"one")
        vault.seal("alpha", b"two")
        vault.seal("beta", b"three")
        answers_before = (
            vault.versions("alpha"),
            vault.active("alpha"),
            vault.load("alpha"),
            vault.load("alpha", 1),
            vault.load("beta"),
            vault.manifest(),
        )
        healthy_manifest = (self.root / MANIFEST_NAME).read_bytes()

        (self.root / MANIFEST_NAME).write_bytes(b"{not json")
        disk_corrupt = self.disk_bytes(self.root)
        expected_stderr = (
            "error: manifest corrupt: Expecting property name enclosed in "
            "double quotes: line 1 column 2 (char 1)\n"
        )

        # The same failing input twice gives the identical frozen outcome.
        for _ in range(2):
            result = self.run_cli(["--root", str(self.root), "reload"])
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, expected_stderr)

        # The in-memory snapshot is exactly as it was: keys already in hand
        # stay readable, byte for byte.
        self.assertEqual(
            answers_before,
            (
                vault.versions("alpha"),
                vault.active("alpha"),
                vault.load("alpha"),
                vault.load("alpha", 1),
                vault.load("beta"),
                vault.manifest(),
            ),
        )
        # The failed reloads are strictly read-only: the disk records neither
        # grew nor shrank, byte for byte.
        self.assertEqual(self.disk_bytes(self.root), disk_corrupt)

        # Undo the corruption: the CLI reloads cleanly again and the handle's
        # own reload lands on the same healthy state.
        (self.root / MANIFEST_NAME).write_bytes(healthy_manifest)
        result = self.run_cli(["--root", str(self.root), "reload"])
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "reloaded\n", ""))
        vault.reload()
        self.assertEqual(vault.load("alpha"), b"two")
        self.assertEqual(vault.versions("alpha"), [1, 2])


# ---------------------------------------------------------------------------
# warning policy turned up: no unclosed-resource tail on any subcommand
# ---------------------------------------------------------------------------


class TestWarningPolicyCleanOutput(CliExitPathTestCase):
    def test_every_subcommand_stays_clean_under_warning_policy(self):
        material = self.fixture.path("material.bin")
        material.write_bytes(b"warning-policy-material")

        steps = [
            (["seal", "alpha", "--material-file", str(material)], "1\n"),
            (["versions"], "alpha\tactive=1\tversions=1\n"),
            (["reload"], "reloaded\n"),
            (["seal", "alpha", "--material-file", str(material)], "2\n"),
            (["versions"], "alpha\tactive=2\tversions=1,2\n"),
        ]
        for argv, expected_stdout in steps:
            with self.subTest(argv=argv[0]):
                result = self.run_cli(
                    ["--root", str(self.root), *argv], python_flags=WARNING_FLAGS
                )
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout, expected_stdout)
                # No "ResourceWarning: unclosed ..." tail, nothing at all.
                self.assertEqual(result.stderr, "")

    def test_error_path_stays_clean_under_warning_policy(self):
        vault = self.fixture.open()
        vault.seal("alpha", b"one")
        vault.close()
        (self.root / MANIFEST_NAME).write_bytes(b"{not json")

        result = self.run_cli(
            ["--root", str(self.root), "reload"], python_flags=WARNING_FLAGS
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            result.stderr,
            "error: manifest corrupt: Expecting property name enclosed in "
            "double quotes: line 1 column 2 (char 1)\n",
        )
        self.assertNotIn("ResourceWarning", result.stderr)
        self.assertNotIn("unclosed", result.stderr)


if __name__ == "__main__":
    unittest.main()
