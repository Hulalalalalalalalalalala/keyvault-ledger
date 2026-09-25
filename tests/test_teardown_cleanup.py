"""Teardown cleanup tests: deletion failures are never swallowed.

The product code, the public interface and the three README command line
entry points are untouched; this module only pins the contract of the
single test-fixture teardown path in :mod:`tests._fixtures`:

* teardown runs one fixed order -- release gates/barriers, drain threads,
  drain worker processes, return every vault lock handle, and only then
  delete the whole temporary directory;

* a successful teardown leaves no tree and no lock file behind, repairs a
  read-only tree once and then removes it, and is safe to run twice;

* an occupied lock file cannot block tree removal (on POSIX a live flock
  held by another process is removed with the tree regardless; on Windows
  the same guarantee comes from draining every process first);

* if the temporary directory genuinely cannot be removed, teardown raises
  an ``AssertionError`` naming that directory instead of quietly passing:
  the case fails for it;

* when the case already fails an assertion, that original assertion error
  surfaces verbatim and the cleanup failure is recorded alongside it -- it
  neither masks nor rewrites the original -- and the following case still
  runs;

* ``close()`` repeated is an error-free ``None``-returning no-op and the
  next operation reacquires the lock with identical observable behaviour,
  asserted entirely through the public interface;

* repeating identical inputs in independent temporary directories gives
  byte-identical results, and cases touch nothing outside their temp dirs.

Everything is independent of execution order and deterministic (no
temporary path or timing value is compared or printed)::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import threading
import unittest
from pathlib import Path
from unittest import mock

from keyvault_ledger import Vault
from keyvault_ledger.vault import LOCK_NAME, MANIFEST_NAME
from tests._fixtures import VaultFixture

try:
    import fcntl
except ImportError:  # Windows: tree removal is guarded by draining processes.
    fcntl = None


def _fresh_fixture() -> VaultFixture:
    """Build a fixture whose cleanup is never auto-run.

    The owner is a bare ``TestCase`` whose cleanup stack nobody triggers, so
    the test drives :meth:`VaultFixture._cleanup` itself at the exact point
    it wants to observe, with no second teardown route.
    """
    return VaultFixture(unittest.TestCase())


def _flock_holder(
    lock_path: str,
    ready: "multiprocessing.Event",
    release: "multiprocessing.Event",
) -> None:
    """Hold a genuine exclusive flock on the vault lock until released."""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        ready.set()
        release.wait(30)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _failing_case_body(
    testcase: unittest.TestCase,
    failure: OSError,
    leaked: list[VaultFixture],
) -> None:
    """Body of a case whose own assertion fails while teardown cannot delete.

    Cleanup functions run LIFO: the patch's ``stop`` is registered first and
    the fixture's teardown second, so when the assertion fails the fixture
    teardown runs *first*, while the deletion failure is still injected; the
    patch is only torn down afterwards.
    """
    rmtree_patch = mock.patch(
        "tests._fixtures.shutil.rmtree", side_effect=failure
    )
    rmtree_patch.start()
    testcase.addCleanup(rmtree_patch.stop)

    fixture = VaultFixture(testcase)
    # Force the first deletion attempt (TemporaryDirectory.cleanup) to fail
    # as well; the retry goes through the patched shutil.rmtree.
    fixture._tmp.cleanup = lambda: (_ for _ in ()).throw(failure)
    leaked.append(fixture)

    testcase.fail("ORIGINAL-ASSERTION-MARKER")


class TestTeardownCleanup(unittest.TestCase):
    # ------------------------------------------------------------------
    # successful cleanup
    # ------------------------------------------------------------------

    def test_successful_cleanup_leaves_no_tree_and_no_lock_file(self):
        fixture = _fresh_fixture()
        vault = fixture.open()
        vault.seal("k", b"m")
        lock_path = fixture.tmp_path / "vault" / LOCK_NAME
        self.assertTrue(lock_path.exists())

        fixture._cleanup()

        self.assertFalse(fixture.tmp_path.exists())
        self.assertFalse(lock_path.exists())

    def test_cleanup_is_safe_to_run_twice(self):
        fixture = _fresh_fixture()
        fixture.open().seal("k", b"m")

        fixture._cleanup()
        # A second teardown (all workers already gone, handles already
        # returned, tree already absent) must be an error-free no-op.
        fixture._cleanup()

        self.assertFalse(fixture.tmp_path.exists())

    @unittest.skipUnless(
        hasattr(os, "geteuid") and os.geteuid() != 0,
        "permission bits are not enforced for root",
    )
    def test_read_only_tree_is_repaired_once_then_removed(self):
        fixture = _fresh_fixture()
        fixture.open().seal("k", b"m")

        # Make every leftover read-only, including the directories, the way
        # a case that failed before restoring chmod bits would leave them.
        for base, dirs, files in os.walk(fixture.tmp_path, topdown=False):
            for name in files:
                os.chmod(os.path.join(base, name), 0o400)
            for name in dirs:
                os.chmod(os.path.join(base, name), 0o500)
        os.chmod(fixture.tmp_path, 0o500)

        fixture._cleanup()  # must repair permissions and still remove all

        self.assertFalse(fixture.tmp_path.exists())

    def test_cleanup_drains_a_tracked_thread_before_removing_tree(self):
        fixture = _fresh_fixture()
        gate = threading.Event()

        def parked_worker() -> None:
            gate.wait(5)

        thread = threading.Thread(target=parked_worker)
        fixture.track_gate(gate)
        fixture.track_thread(thread)
        thread.start()
        self.assertTrue(thread.is_alive())

        fixture._cleanup()  # releases the gate and joins the thread first

        self.assertFalse(thread.is_alive())
        self.assertFalse(fixture.tmp_path.exists())

    # ------------------------------------------------------------------
    # deletion failure is exposed, never swallowed
    # ------------------------------------------------------------------

    def test_undeletable_directory_fails_and_names_the_directory(self):
        fixture = _fresh_fixture()
        vault = fixture.open()
        vault.seal("k", b"m")  # lock handles are returned at step 4 first
        # Never let a failure of this test leave the directory behind.
        self.addCleanup(
            lambda: shutil.rmtree(fixture.tmp_path, ignore_errors=True)
        )

        failure = OSError("injected undeletable directory")
        with mock.patch("tests._fixtures.shutil.rmtree", side_effect=failure):
            # Both deletion attempts fail: the TemporaryDirectory first try
            # and the repaired-permission retry.
            fixture._tmp.cleanup = lambda: (_ for _ in ()).throw(failure)
            with self.assertRaises(AssertionError) as caught:
                fixture._cleanup()

        message = str(caught.exception)
        # The failure is loud and says exactly which directory survived.
        self.assertIn(str(fixture.tmp_path), message)
        self.assertIn("could not remove temporary directory", message)
        # The underlying OSError stays chained for diagnosis, not swallowed.
        self.assertIs(caught.exception.__cause__, failure)
        # The directory really was left in place at the moment of failure.
        self.assertTrue(fixture.tmp_path.exists())
        self.assertTrue(
            (fixture.tmp_path / "vault" / MANIFEST_NAME).exists()
        )

        # Once the injection is gone the same directory removes cleanly, so
        # the suite leaves no residue (and no warning at process shutdown).
        del fixture._tmp.cleanup
        fixture._tmp.cleanup()
        self.assertFalse(fixture.tmp_path.exists())

    def test_original_assertion_kept_and_later_case_still_runs(self):
        # Drive a real (in-memory) two-case suite: the first case fails its
        # own assertion while its directory cannot be removed, the second
        # case is ordinary.  unittest runs the cleanup stack and records
        # outcomes on a TestResult, which is exactly what the command line
        # reporter consumes.
        failure = OSError("injected undeletable directory")
        leaked: list[VaultFixture] = []
        ran: list[str] = []

        FailingCase = type(
            "FailingCase",
            (unittest.TestCase,),
            {
                "test_original_then_bad_cleanup": lambda case: (
                    _failing_case_body(case, failure, leaked)
                )
            },
        )

        def following_body(case: unittest.TestCase) -> None:
            ran.append("following")
            case.assertTrue(True)

        FollowingCase = type(
            "FollowingCase",
            (unittest.TestCase,),
            {"test_runs_normally_afterwards": following_body},
        )

        failing_test = FailingCase("test_original_then_bad_cleanup")
        following_test = FollowingCase("test_runs_normally_afterwards")
        suite = unittest.TestSuite([failing_test, following_test])
        result = unittest.TestResult()
        suite.run(result)

        failing_id = failing_test.id()
        failure_ids = [test.id() for test, _traceback in result.failures]

        # Both cases ran; the following one was not stopped by the failure.
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(ran, ["following"])
        # The first case produced exactly two recorded failures -- its own
        # assertion and the teardown failure -- and no errors; the following
        # case produced neither.
        self.assertEqual(failure_ids, [failing_id, failing_id])
        self.assertEqual(
            [test.id() for test, _traceback in result.errors], []
        )

        original_traceback = result.failures[0][1]
        cleanup_traceback = result.failures[1][1]
        # The original assertion is still there, verbatim and first: the
        # cleanup failure neither replaced nor rewrote it.
        self.assertIn("AssertionError: ORIGINAL-ASSERTION-MARKER", original_traceback)
        self.assertIn("could not remove temporary directory", cleanup_traceback)
        self.assertIn(str(leaked[0].tmp_path), cleanup_traceback)

        # Release the injection and remove the surviving directory so the
        # process shutdown stays warning-free and leaves no residue.
        for fixture in leaked:
            try:
                del fixture._tmp.cleanup
                fixture._tmp.cleanup()
            except OSError:
                shutil.rmtree(fixture.tmp_path, ignore_errors=True)
            self.assertFalse(fixture.tmp_path.exists())

    # ------------------------------------------------------------------
    # an occupied lock file never blocks removal
    # ------------------------------------------------------------------

    @unittest.skipUnless(fcntl is not None, "POSIX flock behaviour")
    def test_lock_file_held_by_another_process_does_not_block_removal(self):
        fixture = _fresh_fixture()
        vault = fixture.open()
        vault.seal("k", b"m")
        lock_path = fixture.root / LOCK_NAME

        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_flock_holder,
            args=(str(lock_path), ready, release),
        )
        fixture.track_process(holder)
        holder.start()
        self.assertTrue(ready.wait(10))

        # The lock is genuinely held by a live process; removing the tree
        # must not be obstructed by it.
        shutil.rmtree(fixture.tmp_path)
        self.assertFalse(fixture.tmp_path.exists())

        release.set()
        holder.join(10)
        self.assertEqual(holder.exitcode, 0)
        # The fixture's own teardown now only returns the (still tracked)
        # handle and no-ops on the absent directory.
        fixture._cleanup()
        self.assertFalse(fixture.tmp_path.exists())

    # ------------------------------------------------------------------
    # close(): observable idempotence and unchanged reacquisition
    # ------------------------------------------------------------------

    def test_repeated_close_is_error_free_and_reacquisition_unchanged(self):
        fixture = _fresh_fixture()
        vault = fixture.open()
        self.assertEqual(vault.seal("k", b"one"), 1)
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())

        # Reacquiring the lock changes nothing observable: the sequence
        # continues and every answer matches the pre-close behaviour.
        self.assertEqual(vault.seal("k", b"two"), 2)
        self.assertEqual(vault.load("k", 1), b"one")
        self.assertEqual(vault.load("k"), b"two")
        self.assertEqual(vault.active("k"), 2)
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertIsNone(vault.close())

        # A freshly opened handle observes exactly the persisted state.
        reopened = Vault(fixture.root)
        try:
            self.assertEqual(reopened.versions("k"), [1, 2])
            self.assertEqual(reopened.active("k"), 2)
            self.assertEqual(reopened.load("k", 1), b"one")
            self.assertEqual(reopened.load("k", 2), b"two")
        finally:
            reopened.close()
        fixture._cleanup()

    # ------------------------------------------------------------------
    # determinism: identical inputs, confined to temp directories
    # ------------------------------------------------------------------

    def test_identical_inputs_in_two_temp_dirs_are_byte_identical(self):
        def build() -> tuple[VaultFixture, dict[str, bytes]]:
            fixture = _fresh_fixture()
            vault = fixture.open()
            vault.seal("alpha", b"plain-one")
            vault.derive_seal("alpha", b"passphrase", b"salt", 100, 24)
            vault.seal("beta", b"plain-two")
            vault.set_active("alpha", 1)
            vault.revoke("beta", 1)
            vault.close()

            disk = {
                str(path.relative_to(fixture.root)): path.read_bytes()
                for path in fixture.root.rglob("*")
                if path.is_file() and path.name != LOCK_NAME
            }
            return fixture, disk

        first_fixture, first_disk = build()
        second_fixture, second_disk = build()

        # Everything the vault persisted (minus the pure mutex file) is
        # byte-for-byte identical between two fresh temporary directories.
        self.assertEqual(set(first_disk), set(second_disk))
        self.assertEqual(first_disk, second_disk)

        # Observable answers agree too, and every persisted path stayed
        # inside the owning temporary directory.
        for fixture in (first_fixture, second_fixture):
            vault = Vault(fixture.root)
            try:
                self.assertEqual(vault.versions("alpha"), [1, 2])
                self.assertEqual(vault.versions("beta"), [1])
                self.assertEqual(vault.active("alpha"), 1)
                self.assertEqual(vault.revoked_versions("beta"), [1])
            finally:
                vault.close()
            for path in fixture.root.rglob("*"):
                self.assertTrue(
                    Path(path).resolve().is_relative_to(
                        fixture.tmp_path.resolve()
                    )
                )
            fixture._cleanup()

        self.assertFalse(first_fixture.tmp_path.exists())
        self.assertFalse(second_fixture.tmp_path.exists())


if __name__ == "__main__":
    unittest.main()
