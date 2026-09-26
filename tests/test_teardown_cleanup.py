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

* when returning a lock handle and deleting the tree both fail, the handle
  failure is surfaced together with the removal failure -- named in the
  message and chained as its cause -- instead of being swallowed by it;
  this also holds when the temporary tree has already been cleared before
  the exit reports: through a controllable, platform-independent simulation
  (the tree emptied for real, both steps forced to fail, the exit's two
  existence decisions staged -- no racing thread and no reliance on the
  platform's open-file deletion semantics) the removal error is reported
  verbatim, byte-for-byte against a stable text that names a fixed
  directory, and the handle error is chained as its direct cause carrying
  its own original traceback, the two errors intact, neither swallowing nor
  rewriting the other;
  a handle failure with a healthy removal is reported on its own, and a
  removal failure with healthy handles names only the directory;

* a read-only leftover that blocks the first removal attempt is made
  writable and the removal retried; a successful retry ends the case
  cleanly, leaving neither the tree nor any read-only attribute behind;

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
import traceback
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


def _force_handle_and_removal_failure(
    test_case: unittest.TestCase,
) -> tuple[VaultFixture, Vault]:
    """Give ``test_case`` a fixture whose handle return *and* tree removal
    are both forced to fail.

    Same recovery contract as :func:`_force_removal_failure`: the recovery
    cleanup is registered before the fixture cleanup, so LIFO ordering runs
    the (doubly failing) fixture cleanup first -- while both patches are
    still in place -- and only then restores the real close and remover,
    returns the handle for real and deletes the directory.  The simulated
    double failure therefore surfaces through unittest exactly as a real
    one would, yet leaves neither a directory nor an unclosed lock handle
    behind.
    """
    rmtree_patcher = mock.patch.object(
        shutil, "rmtree", side_effect=OSError("simulated removal failure")
    )
    fixture_holder: list[VaultFixture] = []
    vault_holder: list[Vault] = []
    close_patcher_holder: list = []

    def recover() -> None:
        rmtree_patcher.stop()
        for close_patcher in close_patcher_holder:
            close_patcher.stop()
        # The simulated failures skipped the real handle return and the
        # real removal; do both for real now, leaving nothing behind.
        for vault in vault_holder:
            vault.close()
        if fixture_holder:
            shutil.rmtree(fixture_holder[0].tmp_path, ignore_errors=True)

    test_case.addCleanup(recover)
    fixture = VaultFixture(test_case)
    fixture_holder.append(fixture)
    _SIMULATED_PATHS.append(fixture.tmp_path)
    vault = fixture.open()
    vault_holder.append(vault)
    close_patcher = mock.patch.object(
        vault, "close", side_effect=RuntimeError("SIMULATED-CLOSE-MARKER")
    )
    close_patcher_holder.append(close_patcher)
    close_patcher.start()
    rmtree_patcher.start()
    return fixture, vault


# Stable, platform-independent simulation of the single exit's last branch:
# the temporary tree has already been cleared, and returning the lock handle
# and deleting the tree both fail.  The texts are fixed literals (the
# directory they name is a fixed leaf inside a real system-temporary parent,
# never a randomly minted path), so repeated runs produce byte-identical
# output; nothing depends on a platform's open-file deletion semantics.
_DOUBLE_FAILURE_TREE_NAME = "teardown-double-failure-tree"
_SIMULATED_CLOSE_TEXT = (
    "SIMULATED-CLOSE-FAILURE-MARKER: lock handle was not returned"
)
_SIMULATED_REMOVAL_TEXT = (
    "SIMULATED-REMOVAL-FAILURE-MARKER: could not delete temporary "
    f"directory {_DOUBLE_FAILURE_TREE_NAME!r}"
)


def _double_failure_tree_path() -> Path:
    """Return the fixed-named leaf used for the double-failure simulation.

    It is a direct child of the resolved system temporary root, so every
    simulated path still lives inside that root (and its parent is the
    temporary root exactly, as the order-independence cases require), yet
    the leaf always carries the same name.  The simulated removal error
    names only this fixed leaf, never the machine-specific temporary root,
    so the error text is byte-stable run after run on every platform; the
    case's own recovery always removes the leaf, so a fixed name leaves no
    debris.
    """
    return Path(tempfile.gettempdir()).resolve() / _DOUBLE_FAILURE_TREE_NAME


def _double_failure_branch_patches(
    fixture: VaultFixture, vault: Vault
) -> tuple[list, dict]:
    """Build (unstarted) patchers that drive the one exit to its last
    branch -- the lock handle is not returned and both removal attempts
    fail while the tree has already been cleared before the exit reports --
    together with the call-state the tests assert on.

    The state is prepared deterministically and independently of any
    platform deletion semantics; the caller is expected to have really
    returned the vault handle and really emptied ``fixture.tmp_path`` first.
    This helper then stages only two things:

    * ``Vault.close`` fails with one fixed error for the whole exit, so
      returning the handle during teardown raises the original exception
      complete with its genuine traceback -- never a string pasted into the
      removal message;

    * the remover fails with one fixed, verbatim text (naming the fixed
      directory) on every call against this tree, while every other path
      keeps using the real remover untouched;

    * ``Path.exists`` is staged only for this tree: its first call (the
      inner check right after the recovery retry) reports the tree present
      so the removal error is retained, and its second (the outer reporting
      check) reports it already cleared, taking the exit to its last
      branch.  Nothing is deleted here -- the tree was emptied by the
      caller -- so there is no racing thread, timing window or platform
      behaviour involved.  Every other ``exists`` call is left real.
    """
    real_rmtree = shutil.rmtree
    real_exists = Path.exists
    state = {"rmtree_calls": 0, "exists_calls": 0}

    def staged_rmtree(path, *args, **kwargs):
        if Path(path) != fixture.tmp_path:
            # Never intercept anything but this case's own tree: safety nets
            # and unrelated cleanup always use the real remover.
            return real_rmtree(path, *args, **kwargs)
        state["rmtree_calls"] += 1
        # The tree was already emptied in setup; every removal the exit
        # attempts against it fails with one fixed, verbatim text.
        raise OSError(_SIMULATED_REMOVAL_TEXT)

    def staged_exists(self_):
        if self_ != fixture.tmp_path:
            return real_exists(self_)
        state["exists_calls"] += 1
        if state["exists_calls"] == 1:
            # Inner check right after the failed recovery retry: report the
            # tree present so the removal error is retained rather than
            # discarded as a vanished-between-attempts success.
            return True
        if state["exists_calls"] == 2:
            # Outer reporting check: the tree was already cleared in setup;
            # report it gone so the exit takes its both-failed/cleared branch.
            return False
        return real_exists(self_)

    patchers = [
        mock.patch.object(
            vault, "close", side_effect=RuntimeError(_SIMULATED_CLOSE_TEXT)
        ),
        mock.patch.object(shutil, "rmtree", side_effect=staged_rmtree),
        # A plain function (not a MagicMock) replaces the unbound method, so
        # descriptor binding hands the real ``Path`` instance to
        # ``staged_exists`` as its first argument; ``stop`` restores it.
        mock.patch.object(Path, "exists", new=staged_exists),
    ]
    return patchers, state


def _really_close_and_clear(fixture: VaultFixture, vault: Vault) -> None:
    """Real work finished: really return the handle and really clear the tree.

    Only ordinary platform calls are used, so the "tree already cleared"
    state is reproduced identically on every platform: the lock handle is
    back and the lock file is gone *before* either step is forced to fail,
    and nothing has to delete an open file or undo a read-only bit.
    """
    vault.close()
    shutil.rmtree(fixture.tmp_path)


def _force_handle_and_removal_failure_tree_cleared(
    test_case: unittest.TestCase,
) -> tuple[VaultFixture, Vault]:
    """Give ``test_case`` a fixture whose lock handle is not returned and
    whose tree deletion fails, with the tree already cleared before the
    single exit reports -- its last branch.

    The whole failed state is a controllable simulation independent of any
    platform deletion semantics: real work is sealed, the handle is really
    returned and the tree really emptied (ordinary calls), and only then
    are the two teardown steps forced to fail with fixed, verbatim errors
    while the exit's two existence decisions are staged.

    Same recovery contract as the other forced-failure helpers: the
    recovery cleanup is registered *before* the fixture is built, so LIFO
    ordering runs the failing exit first while the patches are live and
    only afterwards restores close/remover and removes the (already empty)
    tree, leaving nothing behind.
    """
    fixture_holder: list[VaultFixture] = []
    vault_holder: list[Vault] = []
    patcher_holder: list = []

    def recover() -> None:
        for patcher in reversed(patcher_holder):
            patcher.stop()
        # The handle was already really returned while the state was built;
        # this only guards against a handle a future edit might leave open,
        # so no unclosed lock ever reaches interpreter shutdown.
        for vault in vault_holder:
            vault.close()
        if fixture_holder:
            shutil.rmtree(fixture_holder[0].tmp_path, ignore_errors=True)

    # Registered FIRST, so LIFO runs the (failing) fixture cleanup while
    # the patches are still live and only then runs this recovery.
    test_case.addCleanup(recover)
    fixture = VaultFixture(
        test_case, tmp_path=_double_failure_tree_path()
    )
    fixture_holder.append(fixture)
    _SIMULATED_PATHS.append(fixture.tmp_path)
    vault = fixture.open()
    vault_holder.append(vault)
    # Genuine work on a genuine vault before the state is staged.
    vault.seal("k", b"m")
    _really_close_and_clear(fixture, vault)
    patchers, _state = _double_failure_branch_patches(fixture, vault)
    patcher_holder.extend(patchers)
    for patcher in patchers:
        patcher.start()
    return fixture, vault


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


class _SimAssertThenHandleAndCleanupFail(unittest.TestCase):
    """The case fails an assertion; teardown then fails to return the lock
    handle AND to delete the tree.  The original assertion must stay the
    verbatim primary failure while the teardown surfaces both of its own
    problems together."""

    def test_body(self) -> None:
        _fixture, vault = _force_handle_and_removal_failure(self)
        vault.seal("k", b"m")
        self.assertTrue(False, "ORIGINAL-ASSERTION-MARKER")


class _SimAssertThenHandleAndRemovalFailTreeCleared(unittest.TestCase):
    """The case fails an assertion; the one teardown then fails to return
    the lock handle AND fails to delete an already-cleared tree.  The
    original assertion must remain the verbatim primary failure while the
    teardown surfaces *both* of its own errors -- the removal error
    verbatim with the handle error chained as its cause and carrying its
    own traceback -- neither swallowing nor rewriting the other."""

    def test_body(self) -> None:
        # The helper seals real material, really returns the handle and
        # really clears the tree, then arms both step failures; the body
        # itself only fails its assertion.
        _force_handle_and_removal_failure_tree_cleared(self)
        self.assertTrue(False, "ORIGINAL-ASSERTION-MARKER")


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

    def test_read_only_leftover_is_made_writable_and_removed_on_retry(self):
        fixture = self._fixture()
        fixture.open().seal("k", b"m")
        # A genuinely read-only file sitting in the tree.
        payload = fixture.tmp_path / "readonly.bin"
        payload.write_bytes(b"read-only leftover")
        payload.chmod(0o444)
        self.assertEqual(payload.stat().st_mode & 0o222, 0)

        # Force the first removal attempt to fail, so the teardown's
        # recovery pass -- chmod every leftover writable, then delete the
        # tree again -- is the path that actually runs.  The second call is
        # the real remover working on the real read-only file.
        real_rmtree = shutil.rmtree
        removal_attempts: list[Path] = []

        def fail_first_removal(path, *args, **kwargs):
            removal_attempts.append(Path(path))
            if len(removal_attempts) == 1:
                raise OSError("simulated first-pass removal failure")
            return real_rmtree(path, *args, **kwargs)

        chmodded: list[tuple[str, int]] = []
        real_chmod = os.chmod

        def recording_chmod(path, mode, *args, **kwargs):
            chmodded.append((str(path), mode))
            return real_chmod(path, mode, *args, **kwargs)

        with mock.patch.object(
            shutil, "rmtree", side_effect=fail_first_removal
        ):
            with mock.patch.object(os, "chmod", side_effect=recording_chmod):
                fixture._cleanup()  # must not raise: the retry succeeds

        # The recovery pass really did restore the read-only file to
        # writable before deleting it again ...
        self.assertIn((str(payload), 0o700), chmodded)
        # ... and the removal really was retried once after the failure.
        self.assertEqual(len(removal_attempts), 2)
        self.assertEqual(removal_attempts[1], fixture.tmp_path)
        # The retry deleted the whole tree: no debris and no read-only
        # attribute left behind for later cases to trip over.
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
        # The handles came back cleanly, so no handle problem is reported:
        # the removal failure stands on its own.
        self.assertNotIn("not returned cleanly", message)
        self.assertIsNone(caught.exception.__cause__)
        # The failure was not swallowed into nothingness.
        self.assertTrue(fixture.tmp_path.exists())

    def test_handle_failure_is_not_swallowed_by_removal_failure(self):
        fixture = VaultFixture(unittest.TestCase())
        self.addCleanup(
            lambda: shutil.rmtree(fixture.tmp_path, ignore_errors=True)
        )
        vault = fixture.open()
        vault.seal("k", b"m")
        with mock.patch.object(
            vault, "close", side_effect=RuntimeError("simulated close failure")
        ):
            with mock.patch.object(
                shutil, "rmtree",
                side_effect=OSError("simulated removal failure"),
            ):
                with self.assertRaises(AssertionError) as caught:
                    fixture._cleanup()
        message = str(caught.exception)
        # The case fails and the directory problem is still named exactly ...
        self.assertIn("teardown could not delete temporary directory", message)
        self.assertIn(str(fixture.tmp_path), message)
        # ... and the handle failure is surfaced alongside it, named in the
        # message and chained as the direct cause -- not swallowed by the
        # removal step.
        self.assertIn("not returned cleanly", message)
        self.assertIn("simulated close failure", message)
        cause = caught.exception.__cause__
        self.assertIsInstance(cause, RuntimeError)
        self.assertEqual(str(cause), "simulated close failure")
        # The simulated failure skipped the real handle return; the mocks
        # are gone now, so close for real and leave no unclosed lock handle
        # (which would surface as a ResourceWarning at interpreter shutdown).
        vault.close()
        # The directory genuinely survived; the safety-net cleanup above
        # removes it for real after this case.
        self.assertTrue(fixture.tmp_path.exists())

    def test_both_failures_survive_when_the_tree_was_already_cleared(self):
        # The last branch of the one teardown exit, reached through a
        # controllable, platform-independent simulation: the temporary
        # tree is emptied for real, after which returning the lock handle
        # and deleting the tree are both forced to fail and the exit's two
        # existence decisions are staged.  The exit reports the removal
        # error verbatim with the handle error chained as its direct cause
        # and carrying its own traceback: both survive intact.
        fixture = VaultFixture(
            unittest.TestCase(), tmp_path=_double_failure_tree_path()
        )
        self.addCleanup(
            lambda: shutil.rmtree(fixture.tmp_path, ignore_errors=True)
        )
        vault = fixture.open()
        vault.seal("k", b"m")
        lock_path = fixture.root / "vault.lock"
        self.assertTrue(lock_path.exists())
        # Reproduce "tree already cleared" with ordinary calls only, so it
        # is identical on every platform: the handle is really back and the
        # tree really gone before either teardown step is forced to fail.
        _really_close_and_clear(fixture, vault)
        patchers, state = _double_failure_branch_patches(fixture, vault)
        for patcher in patchers:
            patcher.start()
        try:
            with self.assertRaises(OSError) as caught:
                fixture._cleanup()
        finally:
            for patcher in reversed(patchers):
                patcher.stop()
            # The handle was already really returned above; this is just the
            # idempotent no-op that guarantees no unclosed lock survives.
            vault.close()

        # Both removal attempts really ran (``self._tmp.cleanup()`` and the
        # recovery retry), and the exit consulted existence exactly twice
        # -- once to retain the error, once to find the tree already clear.
        self.assertEqual(state["rmtree_calls"], 2)
        self.assertEqual(state["exists_calls"], 2)
        # The removal failure is reported VERBATIM, byte-for-byte: it is not
        # the tree-survived AssertionError, is not rewritten into the handle
        # error, and is compared against the whole fixed text rather than a
        # substring of it.
        self.assertNotIsInstance(caught.exception, AssertionError)
        self.assertEqual(str(caught.exception), _SIMULATED_REMOVAL_TEXT)
        self.assertEqual(
            str(caught.exception),
            "SIMULATED-REMOVAL-FAILURE-MARKER: could not delete temporary "
            "directory 'teardown-double-failure-tree'",
        )
        self.assertNotIn(_SIMULATED_CLOSE_TEXT, str(caught.exception))
        # ... and the handle failure survives intact as the chained direct
        # cause, verbatim and WITH ITS OWN ORIGINAL TRACEBACK -- the stack
        # reaches back through the single exit's close call instead of
        # collapsing into a pasted line of text.
        cause = caught.exception.__cause__
        self.assertIsInstance(cause, RuntimeError)
        self.assertEqual(str(cause), _SIMULATED_CLOSE_TEXT)
        self.assertNotIn(_SIMULATED_REMOVAL_TEXT, str(cause))
        self.assertIsNotNone(cause.__traceback__)
        cause_frames = traceback.extract_tb(cause.__traceback__)
        self.assertTrue(cause_frames)
        self.assertIn("_cleanup", [frame.name for frame in cause_frames])
        cause_tb = "".join(
            traceback.format_exception(type(cause), cause, cause.__traceback__)
        )
        self.assertIn("RuntimeError: " + _SIMULATED_CLOSE_TEXT, cause_tb)
        # The tree was really cleared before teardown, handle back and lock
        # file gone, independently of any platform deletion semantics;
        # nothing is left behind for the safety net to find.
        self.assertFalse(lock_path.exists())
        self.assertFalse(fixture.root.exists())
        self.assertFalse(fixture.tmp_path.exists())

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

    def test_original_assertion_stays_primary_when_handle_and_removal_fail(self):
        _SIMULATED_PATHS.clear()
        result = _run(
            unittest.TestSuite([_SimAssertThenHandleAndCleanupFail("test_body")])
        )

        tracebacks = _tracebacks(result)
        original = [
            tb for tb in tracebacks if "ORIGINAL-ASSERTION-MARKER" in tb
        ]
        # The case's own assertion is still the primary failure, reported
        # verbatim and not polluted by either teardown complaint.
        self.assertEqual(len(original), 1)
        self.assertIn("AssertionError", original[0])
        self.assertIn(
            'self.assertTrue(False, "ORIGINAL-ASSERTION-MARKER")', original[0]
        )
        self.assertIn("False is not true : ORIGINAL-ASSERTION-MARKER", original[0])
        self.assertNotIn("teardown could not delete", original[0])
        self.assertNotIn("SIMULATED-CLOSE-MARKER", original[0])
        # The teardown surfaces BOTH of its failures together: the
        # directory it could not delete and the handle it could not return.
        combined = [
            tb for tb in tracebacks if "teardown could not delete" in tb
        ]
        self.assertEqual(len(combined), 1)
        self.assertIn("not returned cleanly", combined[0])
        self.assertIn("SIMULATED-CLOSE-MARKER", combined[0])
        self.assertIn("RuntimeError", combined[0])
        for path in list(_SIMULATED_PATHS):
            self.assertFalse(path.exists())

    def test_original_assertion_stays_primary_when_handle_and_removal_fail_tree_cleared(
        self,
    ):
        # The last branch of the one teardown exit, driven through the
        # controllable, platform-independent simulation: the temporary
        # tree is emptied for real, after which returning the lock handle
        # and deleting it are both forced to fail.  The case's own
        # assertion is still the verbatim primary failure, and the
        # teardown separately surfaces BOTH of its errors -- the removal
        # error rendered verbatim with the handle error chained as its
        # direct cause and carrying its own multi-frame traceback --
        # neither swallowing the other and neither rewritten into the
        # surviving-directory complaint.
        _SIMULATED_PATHS.clear()
        result = _run(
            unittest.TestSuite(
                [_SimAssertThenHandleAndRemovalFailTreeCleared("test_body")]
            )
        )

        tracebacks = _tracebacks(result)
        original = [
            tb for tb in tracebacks if "ORIGINAL-ASSERTION-MARKER" in tb
        ]
        # Exactly one primary failure: the case's own assertion, verbatim,
        # carrying neither teardown complaint.
        self.assertEqual(len(original), 1)
        self.assertIn("AssertionError", original[0])
        self.assertIn(
            'self.assertTrue(False, "ORIGINAL-ASSERTION-MARKER")', original[0]
        )
        self.assertIn("False is not true : ORIGINAL-ASSERTION-MARKER", original[0])
        self.assertNotIn("teardown could not delete", original[0])
        self.assertNotIn(_SIMULATED_CLOSE_TEXT, original[0])
        self.assertNotIn(_SIMULATED_REMOVAL_TEXT, original[0])

        # Exactly one teardown error that keeps BOTH step failures intact:
        # the removal error is the reported exception and the handle error
        # is its chained direct cause.  It is not the tree-survived
        # complaint, because the tree had already been cleared.
        combined = [
            tb
            for tb in tracebacks
            if _SIMULATED_REMOVAL_TEXT in tb or _SIMULATED_CLOSE_TEXT in tb
        ]
        self.assertEqual(len(combined), 1)
        self.assertNotIn("ORIGINAL-ASSERTION-MARKER", combined[0])
        self.assertNotIn("teardown could not delete", combined[0])
        # The removal error is rendered VERBATIM -- the whole fixed text on
        # the OSError line, not a fragment matched by containment -- and
        # names the fixed directory.
        self.assertIn(f"OSError: {_SIMULATED_REMOVAL_TEXT}", combined[0])
        self.assertIn("OSError", combined[0])
        # The handle error is the chained direct cause and, crucially,
        # carries its own original traceback: the stack runs through the
        # single exit's close call into the failing handle, instead of
        # collapsing into a single pasted sentence.
        self.assertIn(f"RuntimeError: {_SIMULATED_CLOSE_TEXT}", combined[0])
        self.assertIn("RuntimeError", combined[0])
        self.assertIn("direct cause", combined[0])
        self.assertIn("vault.close()", combined[0])
        self.assertIn("_cleanup", combined[0])

        # The assertion is a failure and the double-failure teardown is a
        # separate error: two distinct entries, neither swallowing the other.
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(len(result.errors), 1)
        # The tree was really cleared for real; no debris survives.
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

    def test_double_failure_branch_twice_as_subprocess_is_byte_identical_and_clean(
        self,
    ):
        # The class that owns the both-steps-fail cases -- the direct
        # branch exercise and the assertion-primary driven case included --
        # driven through the real unittest command line twice.  After the
        # timing line and temporary names are normalised the two runs are
        # byte-identical, both exit zero and neither tail carries any
        # resource/warning machinery text: the forced failures leave
        # neither a surviving tree nor an unclosed lock handle.
        target = "tests.test_teardown_cleanup.TestTeardownFailureSurfaced"
        env = cli_env()
        command = [sys.executable, "-m", "unittest", target]
        first = subprocess.run(
            command, capture_output=True, text=True, env=env, timeout=180
        )
        second = subprocess.run(
            command, capture_output=True, text=True, env=env, timeout=180
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
