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
5. delete the whole temporary directory tree.  This step never fails
   silently: read-only leftovers get one writable retry, and if the
   directory still survives that, teardown raises and names it, so the
   case fails (the case's own assertion error, if any, is raised first and
   stays on record verbatim).

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
        #    handles a case closed itself are harmless no-ops.
        seen: set[int] = set()
        for vault in self._vaults:
            if id(vault) in seen:
                continue
            seen.add(id(vault))
            try:
                vault.close()
                vault.close()  # repeated release must stay an error-free no-op
            except Exception:
                # Teardown must never mask the case's own exception.
                pass

        # 5. Delete the whole temporary tree; every handle is back and every
        #    worker gone, so the lock file cannot block removal.  A directory
        #    that survives deletion is a teardown failure, not something to
        #    quietly ignore: make read-only entries writable once and retry;
        #    if it still exists afterwards, raise naming the directory that
        #    could not be removed so the case fails for it.
        try:
            self._tmp.cleanup()
            return
        except OSError:
            self._make_tree_writable(self.tmp_path)

        try:
            shutil.rmtree(self.tmp_path, ignore_errors=False)
        except OSError as exc:
            if not self.tmp_path.exists():
                return
            raise AssertionError(
                f"could not remove temporary directory {self.tmp_path} "
                f"during test teardown: {exc!r}"
            ) from exc
        if self.tmp_path.exists():
            raise AssertionError(
                "could not remove temporary directory "
                f"{self.tmp_path} during test teardown"
            )

    @staticmethod
    def _make_tree_writable(root: Path) -> None:
        """Restore write/delete permission on every entry under ``root``.

        Some cases chmod files or directories read-only and restore them in
        their own cleanups; a failure before that restore would otherwise
        leave the tree unremovable.  Directories also need execute (search)
        permission to delete their contents.
        """
        for base, dirs, files in os.walk(root, topdown=False):
            for entry in files + dirs:
                try:
                    os.chmod(os.path.join(base, entry), 0o700)
                except OSError:
                    pass
        try:
            os.chmod(root, 0o700)
        except OSError:
            pass
