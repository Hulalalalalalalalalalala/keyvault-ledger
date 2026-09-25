"""Shared test fixture with exactly one teardown path.

Every test case owns a :class:`VaultFixture` (built in ``setUp``) and reaches
all of its temporary vaults, workers and synchronisation primitives through
it.  Whether a case succeeds, fails an assertion or raises midway, the *same*
cleanup runs, in one fixed order:

1. unblock every gate (set release/ready events) and abort every barrier, so
   workers parked on contention are let through;
2. drain the in-process threads;
3. drain the worker processes (terminating/killing anything still alive);
4. return every vault lock handle via ``Vault.close`` — idempotent, safe to
   call on handles a case already closed;
5. delete the whole temporary directory tree.

Every step always runs: a failure draining a thread or process, returning a
handle or removing the tree never causes a later step to be skipped.  Such
failures are collected and reported truthfully at the end (raised from the
cleanup as a single error or an ``ExceptionGroup``), never silently
swallowed — unittest then shows them as a teardown ERROR in addition to, never
instead of, the case's own assertion failure.  Temporary-directory removal is
not allowed to ignore errors: anything left behind is reported.

The whole cleanup is idempotent: a second call is a no-op and raises nothing.

There is deliberately no second teardown route and no write-only handle
registry: the tracked lists exist solely for this cleanup to read.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import threading
import unittest
from pathlib import Path

from keyvault_ledger import Vault

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generous, fixed timeouts for cleanup-time draining.  Every gate has just
# been released and every barrier aborted, so healthy workers only need a
# moment; the bounds exist to avoid hanging the whole run.
_THREAD_JOIN_TIMEOUT = 30.0
_PROCESS_JOIN_TIMEOUT = 30.0


def cli_env() -> dict[str, str]:
    """Environment for a ``python -m keyvault_ledger`` subprocess.

    The frozen CLI outputs do not depend on the ambient warning policy: every
    entry point explicitly returns its vault lock handle before exit, so even a
    strict policy that forces warnings on (``error`` plus
    ``always::ResourceWarning``) appends no ``ResourceWarning`` tail line. The
    ambient ``PYTHONWARNINGS`` is therefore left in place and propagates to
    the subprocess, so a whole-suite run launched under the strictest policy
    genuinely exercises the CLI under that policy.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


class VaultFixture:
    """One per test case: temporary space, handles and workers, one exit."""

    def __init__(self, test_case: unittest.TestCase) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        # The conventional vault root of this case; cases that need several
        # independent vaults use :meth:`path`.
        self.root = self.tmp_path / "vault"
        self._vaults: list[Vault] = []
        self._threads: list[threading.Thread] = []
        self._processes: list[multiprocessing.Process] = []
        self._gates: list[object] = []
        self._barriers: list[object] = []
        # Guards the single teardown: once cleanup has run, a repeat call is
        # an error-free no-op.
        self._cleaned_up = False
        # The one and only teardown registration for the whole case.
        test_case.addCleanup(self._cleanup)

    def path(self, name: str) -> Path:
        """Return a fresh path inside the case's temporary directory."""
        return self.tmp_path / name

    def open(self, root: Path | str | None = None) -> Vault:
        """Open (and track for the single teardown) a vault at ``root``."""
        vault = Vault(self.root if root is None else root)
        self._vaults.append(vault)
        return vault

    def track_gate(self, *gates: object) -> None:
        """Track events whose ``set()`` lets a parked worker proceed.

        Both ``threading.Event`` and ``multiprocessing.Event`` expose
        ``set()``, so they share one list.
        """
        self._gates.extend(gates)

    def track_barrier(self, *barriers: object) -> None:
        """Track barriers aborted at teardown so no worker waits on a party
        that a failed assertion never started."""
        self._barriers.extend(barriers)

    def track_thread(self, *threads: threading.Thread) -> None:
        self._threads.extend(threads)

    def track_process(self, *processes: multiprocessing.Process) -> None:
        self._processes.extend(processes)

    # ------------------------------------------------------------------
    # the single teardown exit
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        # One-shot teardown: a repeat call (a case finishing normally after it
        # already ran cleanup explicitly, or unittest re-entering it) is an
        # error-free no-op and never double-drains or double-closes.
        if self._cleaned_up:
            return
        self._cleaned_up = True

        # Every meaningful step below is attempted even if an earlier step
        # fails.  Failures are collected and raised together at the end rather
        # than swallowed, so draining and handle return are reported
        # truthfully while later steps still run.
        errors: list[BaseException] = []

        # 1a. Let every blocked worker through: blocked writers only continue
        #     once the holder is released, and their results must not change.
        #     A gate that cannot be set is tolerated here — the worker it was
        #     meant to unblock then fails to drain in step 2/3, which is
        #     reported.
        for gate in self._gates:
            try:
                gate.set()
            except Exception:
                pass
        # 1b. Break barriers so nobody waits for parties that never arrive.
        for barrier in self._barriers:
            try:
                barrier.abort()
            except Exception:
                pass

        # 2. Drain in-process threads before any handle they may use is
        #    closed (close takes the vault's in-process lock).  A thread that
        #    is still alive after the bounded join is a real teardown failure.
        for thread in self._threads:
            if thread.is_alive():
                thread.join(_THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    errors.append(
                        RuntimeError(
                            f"worker thread did not drain within "
                            f"{_THREAD_JOIN_TIMEOUT:g}s: {thread.name!r}"
                        )
                    )

        # 3. Drain worker processes; only escalate to kill if a graceful exit
        #    does not arrive.  Each worker closes its own vault on the way
        #    out, so a drained process leaves no open lock file behind.
        alive = [p for p in self._processes if p.is_alive()]
        for process in alive:
            process.join(_PROCESS_JOIN_TIMEOUT)
        alive = [p for p in alive if p.is_alive()]
        for process in alive:
            try:
                process.terminate()
            except Exception as exc:
                errors.append(
                    RuntimeError(
                        f"could not terminate {process.name!r}: {exc!r}"
                    )
                )
            process.join(5)
        for process in [p for p in alive if p.is_alive()]:
            try:
                process.kill()
            except Exception as exc:
                errors.append(
                    RuntimeError(f"could not kill {process.name!r}: {exc!r}")
                )
            process.join(5)
            if process.is_alive():
                errors.append(
                    RuntimeError(
                        f"worker process did not exit: {process.name!r}"
                    )
                )

        # 4. Return every lock handle.  close() is idempotent by contract, so
        #    handles a case closed itself are harmless no-ops; a close that
        #    raises is reported rather than swallowed, but never skips the
        #    remaining handles or the directory removal.  Calling close once
        #    per distinct vault is enough: close() itself tolerates repeats.
        seen: set[int] = set()
        for vault in self._vaults:
            if id(vault) in seen:
                continue
            seen.add(id(vault))
            try:
                vault.close()
            except Exception as exc:
                errors.append(
                    RuntimeError(
                        f"returning a vault lock handle failed: {exc!r}"
                    )
                )

        # 5. Delete the whole temporary tree; every handle is back and every
        #    worker gone, so the lock file cannot block removal.  Removal
        #    failures are never silenced: anything that cannot be deleted is
        #    made writable once and the TemporaryDirectory's own cleanup is
        #    retried (rather than rmtree-ing behind its back, which would
        #    leave its implicit finalizer to emit a ResourceWarning at exit).
        #    A failure that survives that retry is reported, not ignored.
        try:
            self._tmp.cleanup()
        except OSError as first_exc:
            # Read-only leftovers (some cases chmod files/dirs and restore
            # them in their own cleanups): make the tree writable once.
            self._make_tree_writable()
            try:
                # Retry through the object itself: on success it detaches its
                # implicit finalizer, so no ResourceWarning tail line follows.
                self._tmp.cleanup()
            except OSError as retry_exc:
                if self.tmp_path.exists():
                    errors.append(
                        RuntimeError(
                            f"could not remove temporary directory "
                            f"{self.tmp_path}: {retry_exc!r} "
                            f"(first error: {first_exc!r})"
                        )
                    )

        # Report all teardown failures together.  unittest records these as a
        # cleanup ERROR in addition to the case's own assertion failure, never
        # replacing it; with exactly one failure that error is re-raised.
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("vault fixture teardown failed", errors)

    def _make_tree_writable(self) -> None:
        """Best-effort: make every path in the temp tree writable/removable."""
        for base, dirs, files in os.walk(self.tmp_path, topdown=False):
            for entry in files + dirs:
                try:
                    os.chmod(os.path.join(base, entry), 0o700)
                except OSError:
                    pass
