"""Tests for the single teardown exit in ``tests._fixtures``.

The product code, public interface and CLI semantics are already final;
this module pins the *test-side* cleanup contract only:

* teardown runs in one fixed order -- gates released, threads drained,
  processes drained, vault handles returned, temporary tree deleted last;

* a successful teardown leaves nothing behind, and a ``vault.lock`` file
  present on disk (including one left by an exited subprocess) does not
  obstruct removing the tree;

* a temporary directory that cannot be deleted is never ignored: the case
  fails with an error naming the exact directory and what remains in it;

* when the case itself already failed an assertion (or raised anything),
  that original exception is reported verbatim -- the cleanup failure is
  surfaced too, but it neither swallows nor rewrites the original;

* one case's teardown failure never stops later cases from running;

* returning a lock handle repeatedly stays an error-free no-op observable
  through a second process taking the very same lock and the same vault
  continuing its sequence afterwards;

* results do not depend on execution order: the cases of this module run
  forward, run again, run in reverse or run as a picked subset give, per
  case, the same normalized outcome;

* the same fixed subset driven as a real ``python -m unittest`` subprocess
  twice prints byte-identical output once the genuinely floating tokens
  (the timing line, temporary paths) are normalized away, and its tail
  carries no resource/warning machinery text.

Everything reads and writes inside temporary directories only; cases that
deliberately force a removal failure restore the real remover and delete
their directory themselves, so no failed simulation leaves debris::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from keyvault_ledger import Vault
from tests._fixtures import VaultFixture, cli_env


# ---------------------------------------------------------------------------
# recording result (plain TestResult does not keep the successful cases)
# ---------------------------------------------------------------------------


class _RecordingResult(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.successes: list[unittest.TestCase] = []

    def addSuccess(self, test: unittest.TestCase) -> None:  # noqa: D401
        super().addSuccess(test)
        self.successes.append(test)


def _run(suite: unittest.TestSuite) -> _RecordingResult:
    result = _RecordingResult()
    suite.run(result)
    return result


def _flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []

    def visit(node) -> None:
        for item in node:
            if isinstance(item, unittest.TestSuite):
                visit(item)
            else:
                tests.append(item)

    visit(suite)
    return tests


def _tracebacks(result: unittest.TestResult) -> list[str]:
    return [tb for _, tb in result.failures + result.errors]


# ---------------------------------------------------------------------------
# small simulated cases (kept out of discovery via load_tests below; they are
# only ever run through the explicit suites in this module)
# ---------------------------------------------------------------------------


# Every temporary path touched by a simulated case, so the driving tests can
# prove none of them survives (including the forced-failure ones).
_SIMULATED_PATHS: list[Path] = []


def _force_removal_failure(test_case: unittest.TestCase) -> VaultFixture:
    """Give ``test_case`` a fixture whose tree removal is forced to fail.

    The recovery cleanup is registered *before* the fixture cleanup, so
    LIFO ordering runs the (failing) fixture cleanup first -- while the
    remover is still patched -- and only then restores the real remover
    and deletes the directory for real.  A simulated failure therefore
    surfaces through unittest exactly as a real one would, yet leaves no
    directory behind.
    """
    patcher = mock.patch.object(
        shutil, "rmtree", side_effect=OSError("simulated removal failure")
    )
    fixture_holder: list[VaultFixture] = []

    def recover() -> None:
        patcher.stop()
        if fixture_holder:
            shutil.rmtree(fixture_holder[0].tmp_path, ignore_errors=True)

    test_case.addCleanup(recover)
    fixture = VaultFixture(test_case)
    fixture_holder.append(fixture)
    _SIMULATED_PATHS.append(fixture.tmp_path)
    patcher.start()
    return fixture


class _SimAssertThenCleanupFail(unittest.TestCase):
    """The case fails an assertion and its directory removal fails too."""

    def test_body(self) -> None:
        fixture = _force_removal_failure(self)
        fixture.open().seal("k", b"m")
        self.assertTrue(False, "ORIGINAL-ASSERTION-MARKER")


class _SimRaiseThenCleanupFail(unittest.TestCase):
    """The case raises a non-assertion error and removal fails too."""

    def test_body(self) -> None:
        fixture = _force_removal_failure(self)
        fixture.open().seal("k", b"m")
        raise ValueError("ORIGINAL-VALUE-ERROR-MARKER")


class _SimHealthyCleanup(unittest.TestCase):
    """An ordinary passing case whose lock file must not block removal."""

    def test_body(self) -> None:
        fixture = VaultFixture(self)
        _SIMULATED_PATHS.append(fixture.tmp_path)
        vault = fixture.open()
        vault.seal("k", b"m")
        self.assertTrue((fixture.root / "vault.lock").exists())
        # An exited CLI subprocess leaves a lock file on disk as well; the
        # single teardown must still remove the whole tree.
        material = fixture.tmp_path / "material.bin"
        material.write_bytes(b"cli-material")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "keyvault_ledger",
                "--root",
                str(fixture.root),
                "seal",
                "cli",
                "--material-file",
                str(material),
            ],
            capture_output=True,
            text=True,
            env=cli_env(),
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((fixture.root / "vault.lock").exists())


class _SimRaiseWithHealthyCleanup(unittest.TestCase):
    """The case raises and teardown itself succeeds."""

    def test_body(self) -> None:
        fixture = VaultFixture(self)
        _SIMULATED_PATHS.append(fixture.tmp_path)
        fixture.open().seal("k", b"m")
        raise ValueError("HEALTHY-CLEANUP-ERROR-MARKER")


def load_tests(loader, tests, pattern):  # noqa: ANN001 - unittest protocol
    """Keep the ``_Sim*`` helper cases out of ordinary discovery.

    They are driven explicitly through the suites in this module only.
    """
    suite = unittest.TestSuite()
    for test in _flatten(tests):
        if not type(test).__name__.startswith("_Sim"):
            suite.addTest(test)
    return suite


# ---------------------------------------------------------------------------
# the fixture cleanup itself, driven directly
# ---------------------------------------------------------------------------


class TestTeardownRemoval(unittest.TestCase):
    def _fixture(self) -> VaultFixture:
        # Attached to a throwaway case that is never run, so its registered
        # cleanup never fires; the tests below invoke the one teardown path
        # explicitly and assert on the observable aftermath.
        fixture = VaultFixture(unittest.TestCase())
        self.addCleanup(
            lambda: shutil.rmtree(fixture.tmp_path, ignore_errors=True)
        )
        return fixture

    def test_successful_cleanup_leaves_no_directory_at_all(self):
        fixture = self._fixture()
        vault = fixture.open()
        vault.seal("k", b"m")
        self.assertTrue(fixture.tmp_path.exists())

        fixture._cleanup()

        self.assertFalse(fixture.tmp_path.exists())

    def test_open_handle_and_lock_file_do_not_block_removal(self):
        fixture = self._fixture()
        vault = fixture.open()
        vault.seal("k", b"m")
        lock_path = fixture.root / "vault.lock"
        self.assertTrue(lock_path.exists())
        # Deliberately do not close: the teardown returns this tracked
        # handle itself and must then delete the lock file with everything
        # else.
        fixture._cleanup()
        self.assertFalse(lock_path.exists())
        self.assertFalse(fixture.root.exists())

    def test_cleanup_drains_gate_thread_before_removing(self):
        fixture = self._fixture()
        vault = fixture.open()
        vault.seal("k", b"m")

        gate = threading.Event()

        def worker() -> None:
            gate.wait(timeout=30)
            vault.reload()

        thread = threading.Thread(target=worker)
        fixture.track_gate(gate)
        fixture.track_thread(thread)
        thread.start()
        self.assertTrue(_wait_until(thread.is_alive, timeout=5))

        fixture._cleanup()  # releases the gate and drains the thread

        self.assertFalse(thread.is_alive())
        self.assertFalse(fixture.tmp_path.exists())

    def test_cleanup_drains_worker_process_before_removing(self):
        fixture = self._fixture()
        vault = fixture.open()
        vault.seal("k", b"m")

        gate = multiprocessing.Event()
        process = multiprocessing.Process(
            target=_process_then_seal, args=(str(fixture.root), gate)
        )
        fixture.track_gate(gate)
        fixture.track_process(process)
        process.start()
        self.assertTrue(_wait_until(process.is_alive, timeout=5))

        fixture._cleanup()  # sets the gate, waits, closes, removes

        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
        self.assertFalse(fixture.tmp_path.exists())


def _wait_until(predicate, timeout: float, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _process_then_seal(root: str, gate: "multiprocessing.synchronize.Event") -> None:
    if not gate.wait(timeout=30):
        raise RuntimeError("gate was never released")
    vault = Vault(root)
    try:
        vault.seal("k", b"from-worker")
    finally:
        vault.close()


# ---------------------------------------------------------------------------
# removal failure: surfaced, named, never silent, and never masking the body
# ---------------------------------------------------------------------------


class TestTeardownFailureSurfaced(unittest.TestCase):
    def test_failed_removal_fails_the_case_and_names_the_directory(self):
        fixture = VaultFixture(unittest.TestCase())
        self.addCleanup(
            lambda: shutil.rmtree(fixture.tmp_path, ignore_errors=True)
        )
        fixture.open().seal("k", b"m")
        (fixture.tmp_path / "extra.txt").write_bytes(b"leftover")

        with mock.patch.object(
            shutil, "rmtree", side_effect=OSError("simulated removal failure")
        ):
            with self.assertRaises(AssertionError) as caught:
                fixture._cleanup()

        message = str(caught.exception)
        self.assertIn("teardown could not delete temporary directory", message)
        self.assertIn(str(fixture.tmp_path), message)
        # It says what could not be removed.
        self.assertIn("extra.txt", message)
        # The failure was not swallowed into nothingness.
        self.assertTrue(fixture.tmp_path.exists())

    def test_close_handle_error_is_surfaced_not_swallowed(self):
        fixture = VaultFixture(unittest.TestCase())
        self.addCleanup(
            lambda: shutil.rmtree(fixture.tmp_path, ignore_errors=True)
        )
        vault = fixture.open()
        vault.seal("k", b"m")
        with mock.patch.object(
            vault, "close", side_effect=RuntimeError("simulated close failure")
        ):
            with self.assertRaises(RuntimeError) as caught:
                fixture._cleanup()
        self.assertEqual(str(caught.exception), "simulated close failure")
        # The simulated failure skipped the real handle return; the mock is
        # gone now, so close for real and leave no unclosed lock handle
        # (which would surface as a ResourceWarning at interpreter shutdown).
        vault.close()
        # The directory step still ran before the handle error was raised.
        self.assertFalse(fixture.tmp_path.exists())

    def test_original_assertion_is_reported_verbatim_when_cleanup_fails(self):
        _SIMULATED_PATHS.clear()
        result = _run(
            unittest.TestSuite([_SimAssertThenCleanupFail("test_body")])
        )

        tracebacks = _tracebacks(result)
        original = [
            tb for tb in tracebacks if "ORIGINAL-ASSERTION-MARKER" in tb
        ]
        # The case's own assertion is still there, as its own failure ...
        self.assertEqual(len(original), 1)
        # ... verbatim (message and the assertion line itself) and not
        # polluted by the cleanup complaint.
        self.assertIn("AssertionError", original[0])
        self.assertIn(
            'self.assertTrue(False, "ORIGINAL-ASSERTION-MARKER")', original[0]
        )
        self.assertIn("False is not true : ORIGINAL-ASSERTION-MARKER", original[0])
        self.assertNotIn("teardown could not delete", original[0])
        # The cleanup failure is surfaced separately instead of being hidden.
        self.assertTrue(
            any("teardown could not delete" in tb for tb in tracebacks)
        )
        for path in list(_SIMULATED_PATHS):
            self.assertFalse(path.exists())

    def test_original_non_assertion_error_is_reported_verbatim(self):
        _SIMULATED_PATHS.clear()
        result = _run(unittest.TestSuite([_SimRaiseThenCleanupFail("test_body")]))

        tracebacks = _tracebacks(result)
        original = [tb for tb in tracebacks if "ORIGINAL-VALUE-ERROR-MARKER" in tb]
        self.assertEqual(len(original), 1)
        self.assertIn("ValueError: ORIGINAL-VALUE-ERROR-MARKER", original[0])
        self.assertNotIn("teardown could not delete", original[0])
        self.assertTrue(
            any("teardown could not delete" in tb for tb in tracebacks)
        )
        for path in list(_SIMULATED_PATHS):
            self.assertFalse(path.exists())

    def test_healthy_cleanup_reports_only_the_case_error(self):
        _SIMULATED_PATHS.clear()
        result = _run(
            unittest.TestSuite([_SimRaiseWithHealthyCleanup("test_body")])
        )
        tracebacks = _tracebacks(result)
        self.assertEqual(len(tracebacks), 1)
        self.assertIn("ValueError: HEALTHY-CLEANUP-ERROR-MARKER", tracebacks[0])
        self.assertNotIn("teardown could not delete", tracebacks[0])
        for path in list(_SIMULATED_PATHS):
            self.assertFalse(path.exists())

    def test_one_case_cleanup_failure_does_not_stop_later_cases(self):
        _SIMULATED_PATHS.clear()
        suite = unittest.TestSuite(
            [
                _SimAssertThenCleanupFail("test_body"),
                _SimHealthyCleanup("test_body"),
                _SimRaiseWithHealthyCleanup("test_body"),
            ]
        )
        result = _run(suite)

        # The passing middle case really ran and really passed despite the
        # failing teardown of the case before it.
        succeeded = {test.id() for test in result.successes}
        self.assertIn(
            "tests.test_teardown_cleanup._SimHealthyCleanup.test_body",
            succeeded,
        )
        # The trailing case still ran and still surfaced its own error.
        self.assertTrue(
            any("HEALTHY-CLEANUP-ERROR-MARKER" in tb for tb in _tracebacks(result))
        )
        # Every simulated temporary directory is gone, failures included.
        for path in list(_SIMULATED_PATHS):
            self.assertFalse(path.exists())

    def test_passing_case_with_lock_file_cleans_up_completely(self):
        _SIMULATED_PATHS.clear()
        result = _run(unittest.TestSuite([_SimHealthyCleanup("test_body")]))
        self.assertTrue(result.wasSuccessful())
        for path in list(_SIMULATED_PATHS):
            self.assertFalse(path.exists())


# ---------------------------------------------------------------------------
# repeated handle return: observable outcomes only
# ---------------------------------------------------------------------------


class TestRepeatedCloseObservable(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.root = self.fixture.root

    def test_repeated_close_is_none_and_lock_is_retaken(self):
        vault = self.fixture.open()
        self.assertEqual(vault.seal("k", b"one"), 1)
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())

        # While no handle is held, a real CLI subprocess takes the same lock.
        material = self.fixture.tmp_path / "material.bin"
        material.write_bytes(b"external")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "keyvault_ledger",
                "--root",
                str(self.root),
                "seal",
                "ext",
                "--material-file",
                str(material),
            ],
            capture_output=True,
            text=True,
            env=cli_env(),
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "1\n")

        # The same vault reacquires the lock and sees the external write and
        # continues its own sequence, with identical observable behaviour.
        vault.reload()
        self.assertEqual(vault.load("ext"), b"external")
        self.assertEqual(vault.seal("k", b"two"), 2)
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())


# ---------------------------------------------------------------------------
# runner exit codes: zero when green, non-zero when a case fails
# ---------------------------------------------------------------------------


_RUNNER_SOURCE = """
import unittest


class Probe(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(True)

    def test_bad(self):
        self.assertTrue(False, "runner-failure-marker")


if __name__ == "__main__":
    unittest.main()
"""


class TestRunnerExitCodes(unittest.TestCase):
    def test_unittest_runner_exit_code_matches_the_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe_tests.py"
            path.write_text(_RUNNER_SOURCE)
            env = dict(cli_env())
            env["PYTHONPATH"] = (
                str(directory) + os.pathsep + env.get("PYTHONPATH", "")
            )
            common = [sys.executable, str(path)]

            green = subprocess.run(
                common + ["Probe.test_ok"],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(green.returncode, 0, green.stderr)
            self.assertTrue(green.stderr.rstrip().endswith("OK"))

            failing = subprocess.run(
                common, capture_output=True, text=True, env=env, timeout=30
            )
            self.assertNotEqual(failing.returncode, 0)
            self.assertIn("FAILED", failing.stderr)
            self.assertIn("runner-failure-marker", failing.stderr)


# ---------------------------------------------------------------------------
# execution-order independence and repeated-run consistency
# ---------------------------------------------------------------------------


_TIMING_RE = re.compile(r"^Ran \d+ tests? in [0-9.]+s\s*$", re.MULTILINE)
_TMPNAME_RE = re.compile(r"tmp[A-Za-z0-9_-]{4,}")
_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
_WARNING_RE = re.compile(
    r"resourcewarning|unclosed|exception ignored in|unraisablehook|"
    r"still running",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    """Strip the genuinely floating tokens from captured runner output."""
    text = _TIMING_RE.sub("Ran <N> tests in <T>s", text)
    text = text.replace(str(Path(tempfile.gettempdir())), "<TMPROOT>")
    text = _TMPNAME_RE.sub("tmpXXXX", text)
    text = _ADDR_RE.sub("0xADDR", text)
    return text


def _stable_classes() -> list[type[unittest.TestCase]]:
    """The concrete, self-contained classes of this module.

    ``TestRepeatedAndOrderIndependent`` itself is deliberately excluded:
    those cases *drive* this fixed suite, so including the driver would
    re-enter the run recursively.  ``load_tests`` additionally keeps the
    simulated ``_Sim*`` helper cases out of every loader-built suite.
    """
    return [
        TestTeardownRemoval,
        TestTeardownFailureSurfaced,
        TestRepeatedCloseObservable,
        TestRunnerExitCodes,
    ]


def _load_stable_suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in _stable_classes():
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


def _outcome_map(result: _RecordingResult) -> dict[str, tuple[str, str]]:
    outcomes: dict[str, tuple[str, str]] = {}
    for test in result.successes:
        outcomes[test.id()] = ("success", "")
    for test, tb in result.failures:
        outcomes[test.id()] = ("failure", _normalise(tb))
    for test, tb in result.errors:
        outcomes[test.id()] = ("error", _normalise(tb))
    for test, _ in result.skipped:
        outcomes[test.id()] = ("skipped", "")
    return outcomes


class TestRepeatedAndOrderIndependent(unittest.TestCase):
    def test_two_forward_runs_agree_case_by_case(self):
        first = _run(_load_stable_suite())
        second = _run(_load_stable_suite())
        self.assertTrue(first.wasSuccessful(), _tracebacks(first))
        self.assertTrue(second.wasSuccessful(), _tracebacks(second))
        self.assertEqual(_outcome_map(first), _outcome_map(second))

    def test_reverse_order_gives_the_same_outcomes(self):
        forward = _run(_load_stable_suite())
        reversed_suite = unittest.TestSuite()
        reversed_suite.addTests(reversed(_flatten(_load_stable_suite())))
        reverse = _run(reversed_suite)
        self.assertTrue(reverse.wasSuccessful(), _tracebacks(reverse))
        self.assertEqual(_outcome_map(forward), _outcome_map(reverse))

    def test_picked_subset_matches_those_cases_in_a_full_run(self):
        full = _outcome_map(_run(_load_stable_suite()))
        all_tests = _flatten(_load_stable_suite())
        picked = [all_tests[3], all_tests[0], all_tests[-1]]
        subset = _run(unittest.TestSuite(picked))
        self.assertTrue(subset.wasSuccessful(), _tracebacks(subset))
        self.assertEqual(
            _outcome_map(subset),
            {test.id(): full[test.id()] for test in picked},
        )

    def test_every_case_works_only_inside_the_temporary_root(self):
        tmp_root = Path(tempfile.gettempdir()).resolve()
        result = _run(_load_stable_suite())
        self.assertTrue(result.wasSuccessful(), _tracebacks(result))
        for path in list(_SIMULATED_PATHS):
            self.assertEqual(Path(path).resolve().parent, tmp_root)
            self.assertFalse(path.exists())

    def test_same_subset_twice_as_subprocess_is_byte_identical_and_clean(self):
        # A fixed, fast subset of this module driven through the real
        # unittest command line twice: apart from the timing line and
        # temporary names nothing floats, and the tail is clean.
        target = "tests.test_teardown_cleanup.TestRunnerExitCodes"
        env = cli_env()
        command = [sys.executable, "-m", "unittest", target]
        first = subprocess.run(
            command, capture_output=True, text=True, env=env, timeout=120
        )
        second = subprocess.run(
            command, capture_output=True, text=True, env=env, timeout=120
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(
            _normalise(first.stdout), _normalise(second.stdout)
        )
        self.assertEqual(
            _normalise(first.stderr), _normalise(second.stderr)
        )
        for captured in (first, second):
            self.assertFalse(_WARNING_RE.search(captured.stderr), captured.stderr)
            self.assertTrue(captured.stderr.rstrip().endswith("OK"))


if __name__ == "__main__":
    unittest.main()
