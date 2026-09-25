"""Warning-cleanliness and teardown-contract regressions.

These cases pin the end-to-end promise of this change:

* with warnings dialed to their strictest, forced-visible setting
  (``always`` for everything, ``error`` for ``ResourceWarning``), the whole
  README suite run as its own process finishes on its ordinary ending line —
  ``... OK`` — with return code zero and not a single warning tail line
  (``ResourceWarning``, ``Exception ignored while finalizing``,
  ``resource_tracker`` chatter, ...) on either stream, and that strict
  policy really reaches the ``python -m keyvault_ledger`` grandchildren, so
  all three CLI entry points are proven to return their vault lock handle
  before exit;
* the identical strict suite run twice in a row prints byte-for-byte the
  same thing once floating figures (the timing number) are normalized away;
* the single fixture teardown reports truthfully: a gate it could not
  release, a thread or process it could not drain, a handle whose return
  raised, or a temporary tree it could not delete is surfaced instead of
  silently skipped — and such a report is chained onto, never substituted
  for, the case's own assertion failure;
* calling the teardown repeatedly is an error-free no-op;
* a failing case still releases its gates, drains its contention threads,
  returns every lock handle and deletes its temporary directory, and the
  cases after it still run;
* none of the above depends on execution order: the same cases run forward,
  backward or picked individually with the same all-green result.

Everything here runs in temporary directories with the standard library
only.  The full-suite checks spawn ``python3 -m unittest discover`` in
subprocesses; an environment marker keeps those nested runs from spawning
further nested full runs::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from tests._fixtures import REPO_ROOT, VaultFixture, cli_env

# Strictest forced-visible warning policy.  Filters are parsed left to right
# and later entries win, so this forces every warning visible while turning
# ResourceWarning itself into an error.  Delivered through the environment
# (not just -W flags) so the policy propagates to CLI subprocesses.
STRICT_WARNINGS = "always,error::ResourceWarning"

# Marker set inside the strict full-suite child so the meta-runner test
# skips itself there instead of recursing without bound.
_INNER_MARKER = "_KEYVAULT_LEDGER_STRICT_SUITE"

# Bound for one whole-suite child run; generous on purpose for a loaded
# machine spawning many short-lived processes.
_SUITE_TIMEOUT = 600

_RAN_LINE = re.compile(r"Ran (\d+) tests in [0-9.]+s")
_ENDING = re.compile(
    r"\n-{70}\nRan \d+ tests in [0-9.]+s\n\nOK\n\Z"
)
_FORBIDDEN_TAIL_TOKENS = (
    "warning",
    "exception ignored",
    "resource_tracker",
    "unclosed",
    "traceback",
)


def _normalize_floating(text: str) -> str:
    """Collapse the one floating figure unittest prints: the run timing."""
    return _RAN_LINE.sub("Ran \\1 tests in <timing>", text)


class WarningCleanlinessTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path

    def _strict_env(self) -> dict[str, str]:
        # Set the policy explicitly rather than inheriting it, so the
        # assertion means the same thing regardless of ambient settings.
        env = cli_env()
        env["PYTHONWARNINGS"] = STRICT_WARNINGS
        return env

    def _run_two_strict_suites(
        self,
    ) -> tuple[subprocess.CompletedProcess, subprocess.CompletedProcess]:
        """Two whole-suite strict runs, launched together.

        Independent runs use independent temporary vaults, so running them
        concurrently changes no output while halving the wall time added to
        the outer discover.
        """
        env = self._strict_env()
        env[_INNER_MARKER] = "1"
        argv = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-t",
            ".",
        ]
        first = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        second = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=dict(env), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        try:
            first_out, first_err = first.communicate(timeout=_SUITE_TIMEOUT)
            second_out, second_err = second.communicate(timeout=_SUITE_TIMEOUT)
        except subprocess.TimeoutExpired:
            for process in (first, second):
                process.kill()
            self.fail(f"strict suite hung past {_SUITE_TIMEOUT}s")
        return (
            subprocess.CompletedProcess(argv, first.returncode, first_out, first_err),
            subprocess.CompletedProcess(
                argv, second.returncode, second_out, second_err
            ),
        )

    # ------------------------------------------------------------------
    # the three CLI entries directly, strict policy inherited
    # ------------------------------------------------------------------

    def _run_cli_strict(self, *argv, root=None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "keyvault_ledger",
                "--root",
                str(self.tmp_path / (root or "vault")),
                *argv,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=self._strict_env(),
        )

    def test_every_cli_entry_exits_clean_with_any_warning_tail(self):
        material = self.tmp_path / "m.bin"
        material.write_bytes(b"cli-material")

        cases = [
            (("versions",), 0, "", ""),
            (("seal", "k", "--material-file", str(material)), 0, "1\n", ""),
            (("versions",), 0, "k\tactive=1\tversions=1\n", ""),
            (("reload",), 0, "reloaded\n", ""),
        ]
        for argv, code, out, err in cases:
            result = self._run_cli_strict(*argv)
            self.assertEqual(result.returncode, code, argv)
            self.assertEqual(result.stdout, out, argv)
            self.assertEqual(result.stderr, err, argv)

    def test_cli_error_paths_stay_frozen_and_tail_free_while_strict(self):
        material = self.tmp_path / "m.bin"
        material.write_bytes(b"m")
        missing = self.tmp_path / "absent.bin"

        empty_id = self._run_cli_strict(
            "seal", "", "--material-file", str(material)
        )
        self.assertEqual(empty_id.returncode, 1)
        self.assertEqual(empty_id.stdout, "")
        self.assertEqual(
            empty_id.stderr, "error: key_id must not be empty\n"
        )

        missing_file = self._run_cli_strict(
            "seal", "k", "--material-file", str(missing)
        )
        self.assertEqual(missing_file.returncode, 1)
        self.assertEqual(missing_file.stdout, "")
        self.assertEqual(
            missing_file.stderr,
            f"error: [Errno 2] No such file or directory: '{missing}'\n",
        )

    # ------------------------------------------------------------------
    # the whole suite under forced strict warnings, run twice
    # ------------------------------------------------------------------

    def test_strict_full_suite_ends_clean_and_two_runs_match_verbatim(self):
        # Inside a strict full-suite child this case is inert: it must not
        # spawn nested discovers.  Returning early (rather than skipping)
        # keeps the child ending the same plain ``OK`` the parent asserts.
        if os.environ.get(_INNER_MARKER):
            return

        first, second = self._run_two_strict_suites()

        for label, result in (("first", first), ("second", second)):
            # Green exit code, nothing on stdout, and the clean ending line.
            self.assertEqual(
                result.returncode, 0, f"{label} run failed:\n{result.stderr}"
            )
            self.assertEqual(result.stdout, "", f"{label} run wrote stdout")
            self.assertRegex(
                result.stderr,
                _ENDING,
                f"{label} run did not finish on the clean ending line",
            )
            combined = (result.stdout + result.stderr).lower()
            for token in _FORBIDDEN_TAIL_TOKENS:
                self.assertNotIn(
                    token,
                    combined,
                    f"{label} run left a warning tail ({token!r})",
                )

        # Same suite, same inputs, twice in a row: identical once the timing
        # figure is normalized away.
        self.assertEqual(
            _normalize_floating(first.stderr),
            _normalize_floating(second.stderr),
        )


# ---------------------------------------------------------------------------
# the single teardown: truthful reporting, idempotence, failure resilience
# ---------------------------------------------------------------------------


class TeardownContractTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)

    def test_teardown_is_idempotent_and_reports_via_registered_cleanup(self):
        case = unittest.TestCase()
        fixture = VaultFixture(case)
        vault = fixture.open()
        vault.seal("k", b"m")
        tmp = fixture.tmp_path

        # Explicit teardown returns the handle and removes the tree...
        fixture._cleanup()
        self.assertIsNone(vault._lock_fh)
        self.assertFalse(tmp.exists())
        # ...a second explicit call is an error-free no-op...
        fixture._cleanup()
        # ...and the registered cleanup (a third invocation) is just as
        # harmless: it must not raise or resurrect anything.
        case.doCleanups()
        self.assertFalse(tmp.exists())

    def _probe(self, name: str, source: str) -> tuple[subprocess.CompletedProcess, Path]:
        obs = self.fixture.path(name)
        obs.mkdir()
        probe_dir = self.fixture.path(f"{name}-probe")
        probe_dir.mkdir()
        (probe_dir / "test_probe.py").write_text(textwrap.dedent(source))
        env = cli_env()
        env["PROBE_OBS"] = str(obs)
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", "test_probe"],
            cwd=probe_dir,
            capture_output=True,
            text=True,
            env=env,
        )
        return result, obs

    def test_failed_case_keeps_its_assertion_and_truthful_handle_report(self):
        # A failing case whose tracked handle refuses to close: the original
        # assertion must survive verbatim AND the failed handle return must
        # be reported, and the run must still be non-zero.
        source = '''
            import unittest

            from tests._fixtures import VaultFixture

            class _BadHandle:
                def close(self):
                    raise RuntimeError("forced handle return failure")

            class TestProbe(unittest.TestCase):
                def setUp(self):
                    self.fixture = VaultFixture(self)

                def test_failure_and_report_both_surface(self):
                    vault = self.fixture.open()
                    vault.seal("k", b"m")
                    self.fixture._vaults.append(_BadHandle())
                    import os
                    with open(os.path.join(os.environ["PROBE_OBS"], "tmp"), "w") as fh:
                        fh.write(str(self.fixture.tmp_path))
                    self.fail("the original assertion text")
        '''
        result, obs = self._probe("handle-report", source)
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("the original assertion text", output)
        self.assertIn("fixture teardown problems:", output)
        self.assertIn("lock handle return failed", output)
        self.assertIn("forced handle return failure", output)
        # Every other step still ran: the temporary tree is gone.
        tmp = Path((obs / "tmp").read_text())
        self.assertFalse(tmp.exists())

    def test_failed_case_releases_gate_drains_thread_and_cleans_tmp(self):
        # The case fails while a worker thread is parked on a tracked gate:
        # teardown must let it through, drain it (no "still alive" report),
        # return the handles and delete the temporary directory.
        source = '''
            import os
            import threading
            import unittest

            from tests._fixtures import VaultFixture

            class TestProbe(unittest.TestCase):
                def setUp(self):
                    self.fixture = VaultFixture(self)

                def test_failing_case_still_drains_its_worker(self):
                    obs = os.environ["PROBE_OBS"]
                    gate = threading.Event()

                    def worker():
                        gate.wait(timeout=30)
                        with open(os.path.join(obs, "released"), "w") as fh:
                            fh.write("yes")

                    thread = threading.Thread(target=worker, name="probe-worker")
                    self.fixture.track_gate(gate)
                    self.fixture.track_thread(thread)
                    thread.start()
                    vault = self.fixture.open()
                    vault.seal("k", b"m")
                    with open(os.path.join(obs, "tmp"), "w") as fh:
                        fh.write(str(self.fixture.tmp_path))
                    self.fail("the original assertion text")
        '''
        result, obs = self._probe("drain-thread", source)
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("the original assertion text", output)
        # The gate was released and the worker actually ran to completion...
        self.assertEqual((obs / "released").read_text(), "yes")
        # ...it was drained inside teardown rather than left stuck...
        self.assertNotIn("threads still alive after drain", output)
        # ...and the temporary space was removed afterwards.
        tmp = Path((obs / "tmp").read_text())
        self.assertFalse(tmp.exists())

    def test_thread_that_cannot_be_drained_is_reported_truthfully(self):
        # A worker that ignores its release cannot be drained inside the
        # bound: teardown must say so rather than pretend it is gone.  The
        # join bound is shortened in this child process only, and the worker
        # is a daemon so the probe process can still exit promptly.
        source = '''
            import threading
            import time
            import unittest

            import tests._fixtures as fx
            from tests._fixtures import VaultFixture

            class TestProbe(unittest.TestCase):
                def setUp(self):
                    self.fixture = VaultFixture(self)

                def test_undrainable_worker_is_reported(self):
                    fx._THREAD_JOIN_TIMEOUT = 0.05

                    def worker():
                        time.sleep(30)

                    thread = threading.Thread(
                        target=worker, name="probe-stuck", daemon=True
                    )
                    self.fixture.track_thread(thread)
                    thread.start()
                    self.fixture.open().seal("k", b"m")
                    import os
                    with open(os.path.join(os.environ["PROBE_OBS"], "tmp"), "w") as fh:
                        fh.write(str(self.fixture.tmp_path))
                    self.fail("the original assertion text")
        '''
        result, obs = self._probe("stuck-thread", source)
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("the original assertion text", output)
        self.assertIn("threads still alive after drain", output)
        self.assertIn("probe-stuck", output)
        # The report never blocks the remaining steps: the tree is removed.
        tmp = Path((obs / "tmp").read_text())
        self.assertFalse(tmp.exists())

    def test_case_after_a_failure_still_runs(self):
        # Alphabetical class order puts the failure first and the passing
        # case second; the second marker proves the suite kept going.
        source = '''
            import os
            import unittest

            from tests._fixtures import VaultFixture

            class TestAaaFail(unittest.TestCase):
                def setUp(self):
                    self.fixture = VaultFixture(self)

                def test_fails_on_purpose(self):
                    self.fail("the original assertion text")

            class TestZzzPass(unittest.TestCase):
                def test_runs_after_the_failure(self):
                    obs = os.environ["PROBE_OBS"]
                    with open(os.path.join(obs, "second-ran"), "w") as fh:
                        fh.write("yes")
        '''
        result, obs = self._probe("subsequent-case", source)
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("the original assertion text", output)
        self.assertIn("Ran 2 tests", output)
        self.assertIn("FAILED (failures=1)", output)
        self.assertEqual((obs / "second-ran").read_text(), "yes")


# ---------------------------------------------------------------------------
# execution order cannot change results: forward, backward, picked
# ---------------------------------------------------------------------------


_ORDER_INDEPENDENT_IDS = [
    "tests.test_vault.TestFreshVault",
    "tests.test_vault.TestSealAndLoad",
    "tests.test_vault.TestClose",
    "tests.test_concurrency.TestEntryPointContract",
    "tests.test_reload_regression.TestSuccessfulReloadCorrespondence",
]


class ExecutionOrderTestCase(unittest.TestCase):
    def _run_ids(self, ids: list[str]) -> tuple[int, int, str, str]:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", *ids],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=cli_env(),
        )
        match = _RAN_LINE.search(result.stderr)
        self.assertIsNotNone(match, result.stderr)
        return result.returncode, int(match.group(1)), result.stdout, result.stderr

    def test_forward_reverse_and_selective_runs_are_all_green(self):
        chosen = [_ORDER_INDEPENDENT_IDS[0], _ORDER_INDEPENDENT_IDS[-1]]
        runs = {
            "forward": self._run_ids(_ORDER_INDEPENDENT_IDS),
            "reverse": self._run_ids(list(reversed(_ORDER_INDEPENDENT_IDS))),
            "picked": self._run_ids(chosen),
        }

        for label, (rc, _n, out, err) in runs.items():
            self.assertEqual(rc, 0, f"{label}: {err}")
            # unittest output goes to stderr; nothing is written to stdout.
            self.assertEqual(out, "", f"{label} run wrote stdout")
            self.assertRegex(err, _ENDING, f"{label} run had no clean ending")

        # Forward and reverse run the same set, the same number of cases.
        self.assertEqual(runs["forward"][1], runs["reverse"][1])
        # The selective run executes exactly the two picked classes, proving
        # neither depends on cases that run before or after it.
        self.assertEqual(
            runs["picked"][1],
            sum(self._count_cases(dotted) for dotted in chosen),
        )

    @staticmethod
    def _count_cases(dotted: str) -> int:
        suite = unittest.defaultTestLoader.loadTestsFromName(dotted)

        def count(node) -> int:
            if isinstance(node, unittest.TestSuite):
                return sum(count(item) for item in node)
            return 1

        return count(suite)


if __name__ == "__main__":
    unittest.main()
