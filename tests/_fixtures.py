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
5. delete the whole temporary directory tree.  Removal is never allowed to
   fail silently: if the tree (or any part of it) survives the final
   removal attempt, the case fails with an error naming the directory that
   could not be deleted.  When the handle return above also failed, that
   failure is named and chained in the same error — the removal failure
   never swallows it, and a handle failure with a healthy removal is
   reported on its own.  When the case itself already raised (a failed
   assertion or any other error), that original exception is what the
   runner reports — the cleanup neither swallows nor rewrites it.

There is deliberately no second teardown route and no write-only handle
registry: the tracked lists exist solely for this cleanup to read.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
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

    The frozen CLI outputs must not depend on the ambient warning policy: the
    CLI process returns its vault handle on every exit path, but an inherited
    ``PYTHONWARNINGS`` setting could still append unrelated warning lines to
    its otherwise frozen stderr.  Every subprocess therefore gets the same
    scrubbed environment.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PYTHONWARNINGS", None)
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
        # 1a. Let every blocked worker through: blocked writers only continue
        #     once the holder is released, and their results must not change.
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
        #    closed (close takes the vault's in-process lock).
        for thread in self._threads:
            if thread.is_alive():
                thread.join(_THREAD_JOIN_TIMEOUT)

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
            except Exception:
                pass
            process.join(5)
        for process in [p for p in alive if p.is_alive()]:
            try:
                process.kill()
            except Exception:
                pass
            process.join(5)

        # 4. Return every lock handle.  close() is idempotent by contract, so
        #    handles a case closed itself are harmless no-ops.  A failure here
        #    is not swallowed: it is remembered and surfaced once the removal
        #    attempt is done, unless the case itself already raised, in which
        #    case the case's own exception wins verbatim.
        seen: set[int] = set()
        handle_error: BaseException | None = None
        for vault in self._vaults:
            if id(vault) in seen:
                continue
            seen.add(id(vault))
            try:
                vault.close()
                vault.close()  # repeated release must stay an error-free no-op
            except Exception as exc:
                if handle_error is None:
                    handle_error = exc

        # 5. Delete the whole temporary tree; every handle is back and every
        #    worker gone, so the lock file cannot block removal.  A first
        #    failure gets one recovery pass -- read-only leftovers (some cases
        #    chmod files/dirs and restore them in their own cleanups) are made
        #    writable once -- but a tree that still survives afterwards is a
        #    hard test failure naming the directory, never a silent skip.
        removal_error: BaseException | None = None
        try:
            self._tmp.cleanup()
        except OSError as exc:
            removal_error = exc
            for base, dirs, files in os.walk(self.tmp_path, topdown=False):
                for entry in files + dirs:
                    try:
                        os.chmod(os.path.join(base, entry), 0o700)
                    except OSError:
                        pass
            try:
                shutil.rmtree(self.tmp_path)
            except OSError as retry_exc:
                # The directory vanishing between attempts is still success;
                # only a tree that genuinely survives keeps the error.
                if self.tmp_path.exists():
                    removal_error = retry_exc
                else:
                    removal_error = None
            else:
                removal_error = None

        if self.tmp_path.exists():
            # Name exactly which directory the cleanup could not remove and
            # list whatever is still inside it.  A handle that also failed
            # to come back is surfaced alongside -- named in the message and
            # chained as the direct cause -- never swallowed by the removal
            # failure.
            leftovers = sorted(
                str(path.relative_to(self.tmp_path))
                for path in self.tmp_path.rglob("*")
            )
            message = (
                "teardown could not delete temporary directory "
                f"{self.tmp_path}: remaining entries: {leftovers}"
            )
            if handle_error is not None:
                message += (
                    "; the vault lock handle was not returned cleanly "
                    f"either: {handle_error!r}"
                )
                raise AssertionError(message) from handle_error
            raise AssertionError(message)

        if handle_error is not None:
            if removal_error is not None:
                # Both steps failed yet the tree is gone (a racing remover
                # finished it): keep both failures visible, neither dropped.
                raise removal_error from handle_error
            raise handle_error
        if removal_error is not None:
            raise removal_error
