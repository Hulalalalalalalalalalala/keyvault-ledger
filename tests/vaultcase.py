"""Shared fixture: one single teardown path for every vault test case.

Every case in the suite builds its own temporary vault directory and
leaves through exactly one exit, regardless of whether it passed, failed
or raised midway:

1. drain the threads that have not stopped — every registered gate is
   released first, so a thread parked against a lock or waiting to be
   released is let through, then each registered thread is joined;
2. return every registered vault handle (``Vault.close`` is idempotent,
   so a handle a case already closed itself is returned again without
   error, and later operations would simply reacquire the lock);
3. remove the whole temporary directory — no handle and no thread can
   still hold a lock file open at that point, so nothing blocks the
   removal and no residue is left behind.

The same path serves plain cases (which simply register no threads) and
contention cases alike; no case keeps a second cleanup path of its own.
Because the exit is one ``addCleanup`` registered in ``setUp``, an
assertion failure or any other exception mid-case still releases the
lock-contending threads and waits for them to exit, and the original
exception propagates verbatim — the teardown neither swallows nor
rewrites it, and later cases run unaffected.

Cases only register resources: handles come from :meth:`open_vault`,
threads from :meth:`watch_thread`.  There is no registry that is written
but never read — both lists exist solely to be drained by ``_teardown``.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from keyvault_ledger import Vault


class VaultFixtureCase(unittest.TestCase):
    """One temporary vault directory per case and one teardown exit."""

    #: How long the teardown waits for one registered thread to exit.
    THREAD_JOIN_TIMEOUT = 30

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "vault"
        self._teardown_gates: list[threading.Event] = []
        self._teardown_threads: list[threading.Thread] = []
        self._teardown_vaults: list[Vault] = []
        # The single teardown exit.  Registered first so that, cleanups
        # running LIFO, it runs after any case-specific state restoration
        # and always sees the whole registry.
        self.addCleanup(self._teardown)

    def open_vault(self, root: "str | Path | None" = None) -> Vault:
        """Open a vault and hand its handle to the teardown.

        The teardown returns the handle after every registered thread
        has been drained and before the temporary directory is removed —
        whether the case passed, failed or raised.  ``close`` is
        idempotent, so a case that closes its vault itself needs no
        second cleanup path.
        """
        vault = Vault(self.root if root is None else root)
        self._teardown_vaults.append(vault)
        return vault

    def watch_thread(
        self, thread: threading.Thread, *gates: threading.Event
    ) -> None:
        """Register a started thread the teardown must drain.

        ``gates`` are events the teardown releases before joining, so a
        thread parked against a lock or waiting to be released is let
        through first; a contention case that fails midway still drains
        its threads before any handle is returned.  Registering a thread
        that has already finished is harmless.
        """
        self._teardown_gates.extend(gates)
        self._teardown_threads.append(thread)

    def _teardown(self) -> None:
        """The single teardown path: drain threads, return handles, rmdir."""
        try:
            # 1. Let every parked thread through, then wait for each one
            #    to exit before any handle it may hold a lock on is
            #    touched.
            for gate in self._teardown_gates:
                gate.set()
            for thread in self._teardown_threads:
                thread.join(timeout=self.THREAD_JOIN_TIMEOUT)
            # 2. Return the handles; a repeated close is not an error.
            for vault in self._teardown_vaults:
                vault.close()
        finally:
            # 3. Remove the whole directory: with the threads drained and
            #    the handles returned, no lock file is still held open,
            #    so the removal cannot be blocked and leaves no residue.
            self._tmp.cleanup()
