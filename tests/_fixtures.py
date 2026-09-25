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

There is deliberately no second teardown route and no write-only handle
registry: the tracked lists exist solely for this cleanup to read.  No step
is ever skipped silently: a gate that cannot be released, a thread or process
that survives draining, a handle whose return raises, or a temporary tree
that survives deletion is collected and reported (as one ``RuntimeError``
raised once every step has run) rather than swallowed.  Reporting happens in
addition to — never instead of — the case's own assertion failure, and a
case that failed is still taken through every step so later cases run with
all workers drained, all handles returned and all temporary space gone.
Running the cleanup a second time (it is registered once, but cases may call
it explicitly) is an error-free no-op.
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

    The frozen CLI outputs must stay byte-for-byte stable, and every entry
    point now returns its vault lock handle before exiting, so the ambient
    warning policy is inherited verbatim: under a forced-visible strict
    policy a CLI process must finish on its ordinary ending line with no
    ``ResourceWarning`` tail whatsoever.
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
        # The one and only teardown registration for the whole case.
        self._cleaned = False
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
        # A second call is an error-free no-op: a finished run has nothing
        # left to drain, return or delete.
        if self._cleaned:
            return
        self._cleaned = True
        problems: list[str] = []

        # 1a. Let every blocked worker through: blocked writers only continue
        #     once the holder is released, and their results must not change.
        for index, gate in enumerate(self._gates):
            try:
                gate.set()
            except Exception as exc:
                problems.append(
                    f"gate {index} could not be released: {exc!r}"
                )
        # 1b. Break barriers so nobody waits for parties that never arrive.
        for index, barrier in enumerate(self._barriers):
            try:
                barrier.abort()
            except Exception as exc:
                problems.append(
                    f"barrier {index} could not be aborted: {exc!r}"
                )

        # 2. Drain in-process threads before any handle they may use is
        #    closed (close takes the vault's in-process lock).  A thread that
        #    survives the bound is reported, never silently left running.
        stuck_threads: list[str] = []
        for thread in self._threads:
            if thread.is_alive():
                thread.join(_THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    stuck_threads.append(thread.name)
        if stuck_threads:
            problems.append(
                "threads still alive after drain: " + ", ".join(stuck_threads)
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
                problems.append(
                    f"process {process.name} could not terminate: {exc!r}"
                )
            process.join(5)
        for process in [p for p in alive if p.is_alive()]:
            try:
                process.kill()
            except Exception as exc:
                problems.append(
                    f"process {process.name} could not be killed: {exc!r}"
                )
            process.join(5)
        still_alive = [process.name for process in alive if process.is_alive()]
        if still_alive:
            problems.append(
                "processes still alive after drain: " + ", ".join(still_alive)
            )

        # 4. Return every lock handle.  close() is idempotent by contract, so
        #    handles a case closed itself are harmless no-ops; each is closed
        #    twice to pin that repeated release never reports an error.
        seen: set[int] = set()
        for index, vault in enumerate(self._vaults):
            if id(vault) in seen:
                continue
            seen.add(id(vault))
            try:
                vault.close()
                vault.close()
            except Exception as exc:
                problems.append(
                    f"vault {index} lock handle return failed: {exc!r}"
                )

        # 5. Delete the whole temporary tree; every handle is back and every
        #    worker gone, so the lock file cannot block removal.  Failure is
        #    reported, not skipped: make the tree writable once, retry, then
        #    flag anything that still survives.
        try:
            self._tmp.cleanup()
        except OSError:
            # Read-only leftovers (some cases chmod files/dirs and restore
            # them in their own cleanups): make the tree writable once and
            # remove it outright, so nothing is ever left behind.
            for base, dirs, files in os.walk(self.tmp_path, topdown=False):
                for entry in files + dirs:
                    try:
                        os.chmod(os.path.join(base, entry), 0o700)
                    except OSError:
                        pass
            try:
                if self.tmp_path.exists():
                    shutil.rmtree(self.tmp_path)
            except OSError as exc:
                problems.append(f"temporary tree removal failed: {exc!r}")
        if self.tmp_path.exists():
            problems.append(
                f"temporary directory survived cleanup: {self.tmp_path}"
            )

        if problems:
            # unittest chains a cleanup failure onto the case's own result,
            # so this report never masks or rewrites the assertion the case
            # actually failed with: both surface verbatim.
            raise RuntimeError(
                "fixture teardown problems:\n  - " + "\n  - ".join(problems)
            )
