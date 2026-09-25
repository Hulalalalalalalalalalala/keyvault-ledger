"""Whole-suite warning-hygiene and teardown regressions.

These cases pin the observable shape of a *whole* test run, not individual
vault behaviours (those live in the other modules):

* with the warning policy at its strictest — every warning an error and
  ``ResourceWarning`` additionally forced to be displayed — the complete
  suite run via the README command ends on a clean ``OK`` line, leaves no
  warning tail line (in particular no ``ResourceWarning`` for the vault lock
  handle the three CLI entry points now return before exit), and exits zero;

* the very same strict run, launched twice, is byte-for-byte identical once
  the only floating value (the ``Ran ... in <elapsed>s`` timing) is stripped;

* execution order cannot change the result: a forward discover run, an
  explicit reverse-order run and an arbitrary selected-subset run are green
  with the same clean ending line;

* failing cases are still cleaned up (gates released, parked contention
  threads drained, lock handles returned, temporary directory removed) and the
  following case runs unaffected; a teardown that cannot drain a thread,
  return a handle or delete its temporary directory is reported truthfully
  (never silenced) without replacing the case's own assertion failure, and
  repeating the teardown stays error-free.

The whole-suite cases below spawn fresh ``python -m unittest discover``
subprocesses and therefore *are* a test run themselves.  When this module is
imported inside such a spawned discover run the sentinel environment variable
is set and :func:`load_tests` contributes no tests, so the child cannot
recursively re-spawn discover runs (and records no skip, keeping the child's
ending a plain ``OK``).

Run with the standard library only::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._fixtures import REPO_ROOT, VaultFixture, cli_env

# Set in the environment of every spawned discover child; load_tests() below
# reads it to keep this module from re-spawning child runs.
_CHILD_SENTINEL = "KEYVAULT_LEDGER_META_CHILD"

# The strictest policy: every warning is an error, and ResourceWarning is
# additionally forced on so an unclosed lock handle would print a tail line
# even where it is normally hidden.  Expressed both as -W flags and via
# PYTHONWARNINGS so the policy also reaches the CLI subprocesses the suite
# spawns.
_STRICT_PYTHONWARNINGS = "error,always::ResourceWarning"
_STRICT_WARNING_ARGS = ["-W", "error", "-W", "always::ResourceWarning"]

_DISCOVER_ARGS = ["-m", "unittest", "discover", "-s", "tests", "-t", "."]

# Whole-suite runs are process-heavy (real multi-process contention cases).
_RUN_TIMEOUT = 600

_TIMING_RE = re.compile(r"(Ran \d+ tests in )[0-9]+(?:\.[0-9]+)?s")


def load_tests(loader, tests, pattern):  # noqa: ANN001 - unittest protocol
    """Contribute nothing when imported inside a spawned child discover run."""
    if os.environ.get(_CHILD_SENTINEL):
        return unittest.TestSuite()
    return tests


def _normalize(text: str) -> str:
    """Strip the only floating value in a green unittest transcript."""
    return _TIMING_RE.sub(r"\1<elapsed>s", text)


def _run_suite(args: list[str], *, strict: bool) -> subprocess.CompletedProcess:
    """Run ``python [args]`` in the repo root with a clean child environment."""
    cmd = [sys.executable, *(_STRICT_WARNING_ARGS if strict else []), *args]
    env = dict(os.environ)
    env[_CHILD_SENTINEL] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if strict:
        env["PYTHONWARNINGS"] = _STRICT_PYTHONWARNINGS
    else:
        # Run the child under the interpreter default policy no matter what
        # the ambient environment was, so order/selection results do not
        # depend on the launcher's warning settings.
        env.pop("PYTHONWARNINGS", None)
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT,
    )


def _assert_green_and_clean(
    case: unittest.TestCase, result: subprocess.CompletedProcess
) -> str:
    """Assert a green, warning-free transcript; return normalized stderr."""
    case.assertEqual(
        result.returncode,
        0,
        "suite failed\n--- stdout ---\n%s\n--- stderr ---\n%s"
        % (result.stdout, result.stderr),
    )
    # unittest writes progress dots and the summary to stderr.  A green run
    # ends on the OK line with nothing after it, and no warning of any kind
    # (nor an ignored-exception finalizer message) may appear anywhere.
    transcript = result.stderr
    case.assertNotIn("Warning", transcript, transcript)
    case.assertNotIn("Exception ignored", transcript, transcript)
    case.assertNotIn("Traceback", transcript, transcript)
    case.assertEqual(result.stdout, "", result.stdout)
    normalized = _normalize(transcript)
    # unittest prints a blank line between the timing line and the OK line.
    case.assertRegex(
        normalized, r"\nRan \d+ tests in <elapsed>s\n\nOK\n$", normalized
    )
    return normalized


def _test_module_names() -> list[str]:
    """All real test modules (this meta module excluded), sorted."""
    return sorted(
        f"tests.{p.stem}"
        for p in (REPO_ROOT / "tests").glob("test_*.py")
        if p.stem != "test_warning_clean_exit"
    )


def _parked_thread_worker(started, release) -> None:
    """Thread that blocks on an event until the fixture releases the gate."""
    started.set()
    release.wait(timeout=60)


class _AlwaysAliveThread:
    """Stub thread that never finishes: forces the drain step to report."""

    name = "stub-never-drains"

    def is_alive(self) -> bool:
        return True

    def join(self, timeout) -> None:  # noqa: ANN001
        return None


class _FailingCloseVault:
    """Stub vault whose handle return fails."""

    def close(self) -> None:
        raise OSError("simulated close failure")


def _run_synthetic_case(configure, body) -> tuple[unittest.TestResult, VaultFixture]:
    """Build one case around a VaultFixture, run it directly; return result."""
    holder: dict[str, VaultFixture] = {}

    class T(unittest.TestCase):
        def setUp(self_inner) -> None:  # noqa: N805
            fixture = VaultFixture(self_inner)
            configure(fixture)
            holder["fixture"] = fixture

        def test_body(self_inner) -> None:  # noqa: N805
            body(holder["fixture"])

    suite = unittest.TestLoader().loadTestsFromTestCase(T)
    result = unittest.TestResult()
    suite.run(result)
    return result, holder["fixture"]


class TestCliEntriesReleaseLockUnderStrictWarnings(unittest.TestCase):
    """Directly pin the three README entries under forced warnings."""

    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.root

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        env = cli_env()
        env["PYTHONWARNINGS"] = _STRICT_PYTHONWARNINGS
        return subprocess.run(
            [
                sys.executable,
                *_STRICT_WARNING_ARGS,
                "-m",
                "keyvault_ledger",
                "--root",
                str(self.root),
                *args,
            ],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_all_three_entries_are_clean_with_frozen_output(self):
        # versions on a brand-new empty root succeeds silently.
        versions_empty = self._run_cli("versions")
        self.assertEqual(versions_empty.returncode, 0, versions_empty.stderr)
        self.assertEqual(versions_empty.stdout, "")
        self.assertEqual(versions_empty.stderr, "")

        # reload on the fresh vault.
        reloaded = self._run_cli("reload")
        self.assertEqual(reloaded.returncode, 0, reloaded.stderr)
        self.assertEqual(reloaded.stdout, "reloaded\n")
        self.assertEqual(reloaded.stderr, "")

        # seal twice; prints and exit codes frozen, no warning tail lines.
        material = self.tmp_path / "m.bin"
        material.write_bytes(b"cli-material")
        first = self._run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, "1\n")
        self.assertEqual(first.stderr, "")
        second = self._run_cli("seal", "k", "--material-file", str(material))
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(second.stdout, "2\n")
        self.assertEqual(second.stderr, "")

        # versions lists the sealed versions, still warning-free.
        listing = self._run_cli("versions")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertEqual(listing.stdout, "k\tactive=2\tversions=1,2\n")
        self.assertEqual(listing.stderr, "")

    def test_error_path_is_also_clean_with_frozen_output(self):
        material = self.tmp_path / "m.bin"
        material.write_bytes(b"m")
        bad = self._run_cli("seal", "", "--material-file", str(material))
        self.assertEqual(bad.returncode, 1)
        self.assertEqual(bad.stdout, "")
        self.assertEqual(bad.stderr, "error: key_id must not be empty\n")


class TestWholeSuiteStrictWarnings(unittest.TestCase):
    def test_strict_suite_twice_is_clean_and_identical(self):
        # The whole README suite under the strictest policy, run twice back
        # to back: each run is green, ends on a clean OK line with no warning
        # tail, and (after stripping the timing) the two transcripts are
        # byte-for-byte identical — which also pins the test count.
        first = _run_suite(_DISCOVER_ARGS, strict=True)
        second = _run_suite(_DISCOVER_ARGS, strict=True)
        first_text = _assert_green_and_clean(self, first)
        second_text = _assert_green_and_clean(self, second)
        self.assertEqual(first_text, second_text)


class TestExecutionOrderIndependence(unittest.TestCase):
    def test_forward_reverse_and_selected_runs_agree(self):
        # Forward discover and an explicit reverse run cover exactly the same
        # real test modules (this meta module contributes nothing in a child),
        # so after stripping the timing their green transcripts are
        # byte-for-byte identical.  An arbitrary selected subset is green and
        # clean too: picking cases changes neither result nor ending line.
        names = _test_module_names()
        forward = _run_suite(_DISCOVER_ARGS, strict=False)
        reverse = _run_suite(
            ["-m", "unittest", *reversed(names)], strict=False
        )
        selected = _run_suite(
            [
                "-m",
                "unittest",
                "tests.test_vault.TestFreshVault",
                "tests.test_vault.TestClose",
            ],
            strict=False,
        )
        forward_text = _assert_green_and_clean(self, forward)
        reverse_text = _assert_green_and_clean(self, reverse)
        _assert_green_and_clean(self, selected)
        self.assertEqual(
            re.search(r"Ran (\d+) tests", forward_text).group(1),
            re.search(r"Ran (\d+) tests", reverse_text).group(1),
        )
        self.assertEqual(forward_text, reverse_text)


class TestFailingCaseIsStillCleanedUp(unittest.TestCase):
    def test_failed_case_drains_parked_thread_and_next_case_runs(self):
        with tempfile.TemporaryDirectory() as scratch:
            module_path = Path(scratch) / "case_module.py"
            module_path.write_text(
                "import threading\n"
                "import unittest\n"
                "from tests._fixtures import VaultFixture\n"
                "from tests.test_warning_clean_exit import _parked_thread_worker\n"
                "\n"
                "class T(unittest.TestCase):\n"
                "    def setUp(self):\n"
                "        self.fixture = VaultFixture(self)\n"
                "    def test_a_fails(self):\n"
                "        started, release = threading.Event(), threading.Event()\n"
                "        worker = threading.Thread(\n"
                "            target=_parked_thread_worker, args=(started, release))\n"
                "        self.fixture.track_gate(release)\n"
                "        self.fixture.track_thread(worker)\n"
                "        worker.start()\n"
                "        self.assertTrue(started.wait(5))\n"
                "        self.fail('deliberate failure')\n"
                "    def test_b_runs_after_the_failure(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = (
                str(REPO_ROOT)
                + os.pathsep
                + str(Path(scratch))
                + os.pathsep
                + env.get("PYTHONPATH", "")
            )
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "case_module"],
                cwd=str(scratch),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        # Reported as a FAILURE (the assertion), with no teardown ERROR: the
        # gate was released, the parked thread drained and the temp dir went.
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Ran 2 tests", result.stderr)
        self.assertIn("FAILED (failures=1", result.stderr)
        self.assertIn("deliberate failure", result.stderr)
        self.assertNotIn("did not drain", result.stderr)
        self.assertNotIn("still present", result.stderr)


class TestTeardownReportsTruthfullyWithoutMasking(unittest.TestCase):
    def test_thread_that_never_drains_is_reported(self):
        result, fixture = _run_synthetic_case(
            lambda fx: fx.track_thread(_AlwaysAliveThread()),
            lambda fx: None,
        )
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.errors), 1)
        self.assertIn("did not drain", result.errors[0][1])
        # Everything else still ran: the temp dir is gone.
        self.assertFalse(fixture.tmp_path.exists())
        # Repeating teardown is an error-free no-op (does not re-report).
        fixture._cleanup()
        fixture._cleanup()

    def test_failed_handle_return_is_reported(self):
        result, fixture = _run_synthetic_case(
            lambda fx: fx._vaults.append(_FailingCloseVault()),
            lambda fx: None,
        )
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.errors), 1)
        self.assertIn("returning a vault lock handle failed", result.errors[0][1])
        self.assertFalse(fixture.tmp_path.exists())
        fixture._cleanup()

    def test_persistent_removal_failure_reported_and_assertion_kept(self):
        # Force the directory removal to fail on every attempt: the teardown
        # must surface that as an ERROR while the body's own assertion stays
        # the FAILURE, neither swallowed nor rewritten.  (The failure is
        # simulated, so the temporary object is finally detached and really
        # removed below — the test itself must not leak a ResourceWarning.)
        def configure(fx) -> None:
            return None

        def body(fx) -> None:
            raise AssertionError("THE ORIGINAL ASSERTION")

        fixture: VaultFixture | None = None
        try:
            with mock.patch.object(
                tempfile.TemporaryDirectory,
                "cleanup",
                side_effect=OSError("simulated cleanup failure"),
            ):
                result, fixture = _run_synthetic_case(configure, body)

            self.assertEqual(result.testsRun, 1)
            # The case's own failure is preserved verbatim...
            self.assertEqual(len(result.failures), 1)
            self.assertIn("THE ORIGINAL ASSERTION", result.failures[0][1])
            # ...and the un-removable directory is additionally reported.
            self.assertEqual(len(result.errors), 1)
            self.assertIn(
                "could not remove temporary directory", result.errors[0][1]
            )
            # Repeating teardown raises nothing; it is a no-op rather than a
            # second failure.
            fixture._cleanup()
            fixture._cleanup()
        finally:
            # Detach the TemporaryDirectory's implicit finalizer and really
            # remove the tree (removal was only simulated, never physical), so
            # no implicit-cleanup ResourceWarning tails the process.
            if fixture is not None:
                tempfile.TemporaryDirectory.cleanup(fixture._tmp)
                shutil.rmtree(fixture.tmp_path, ignore_errors=True)

    def test_one_failed_removal_attempt_recovers_without_error(self):
        # The first removal attempt fails but the chmod+rmtree retry succeeds:
        # the case is green and the directory is gone (no silenced error, no
        # spurious one).
        original_cleanup = tempfile.TemporaryDirectory.cleanup
        calls = {"n": 0}

        def flaky_cleanup(self) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("simulated first-attempt failure")
            return original_cleanup(self)

        with mock.patch.object(
            tempfile.TemporaryDirectory, "cleanup", flaky_cleanup
        ):
            result, fixture = _run_synthetic_case(lambda fx: None, lambda fx: None)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.errors, [])
        self.assertFalse(fixture.tmp_path.exists())


if __name__ == "__main__":
    unittest.main()
