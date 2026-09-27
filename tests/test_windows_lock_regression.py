"""Independent regression cases for the Windows file-lock branch.

``vault._file_lock`` falls back to ``msvcrt.locking`` when ``fcntl`` is
unavailable (Windows).  These cases pin the mutual-exclusion behaviour of
that branch as checkable facts, driving real OS-level inter-process locking
rather than a controllable fake:

* the branch is forced by patching ``keyvault_ledger.vault.fcntl`` away and
  installing an ``msvcrt`` stand-in whose ``locking`` is backed by a genuine
  OS-level inter-process mutex (a byte-range ``fcntl.lockf`` on the very
  descriptor the vault hands it — or the real ``msvcrt`` where that is what
  the platform provides).  ``LK_LOCK`` raises ``OSError`` exactly like the
  C runtime's timed-out wait, so the production acquire/retry/release loop
  runs verbatim against a real system lock;
* several writer processes contending for the one lock allocate version
  numbers strictly from 1, never repeating and never skipping one, and two
  real processes sealing the same key at the same time leave each version
  number exactly once in the persisted records;
* a writer that cannot take the lock waits until it is released: while it
  waits no version is allocated, no half record appears and the manifest is
  never truncated; once it proceeds, the result is byte-for-byte the result
  of the same writes without contention, and historical material is never
  rewritten;
* ``close()`` returns the lock handle, repeats harmlessly, hands the lock
  to a waiting process immediately, and later operations reacquire the lock
  with no observable change;
* the lock file is only a mutual-exclusion device: empty or arbitrary bytes
  in it affect no read, write or reload;
* a reload under the lock only ever swaps a complete snapshot — a
  concurrent reader never observes a half-old/half-new mixture — and a
  reload whose validation fails raises ``ValueError`` while the in-memory
  snapshot and the keys already in hand stay exactly as they were;
* the entry-point error vocabulary (empty id -> ``ValueError``, non-genuine
  integer version -> ``TypeError``, unknown key -> ``KeyError``) holds on
  the Windows branch and never allocates a version;
* repeating the same scripted input yields byte-identical results, and the
  cases are independent of execution order.

Only this test file is added; the product code, the public interface, the
three CLI entry points and every existing test are untouched.  Everything
runs in temporary directories with the standard library only; teardown
drains every process, returns every handle and deletes the temporary tree,
leaving no resource warnings behind::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
import queue as queue_mod
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from keyvault_ledger import Vault
from keyvault_ledger import vault as vault_mod
from keyvault_ledger.vault import (
    ACTIVATIONS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
)
from tests._fixtures import VaultFixture

try:
    import fcntl as _os_fcntl
except ImportError:  # real Windows: the real msvcrt is the stand-in
    _os_fcntl = None


def _build_msvcrt_stand_in() -> "types.ModuleType | None":
    """An ``msvcrt`` module object backed by a real inter-process mutex.

    On POSIX the byte-range lock is delegated to ``fcntl.lockf`` on the same
    descriptor, so contending processes genuinely block each other at the OS
    level; ``LK_LOCK`` raises ``OSError`` on contention, mirroring the C
    runtime giving up after its internal timeout, which is what drives the
    production retry loop.  Where ``fcntl`` does not exist the real
    ``msvcrt`` is used as itself.
    """
    if _os_fcntl is None:
        try:
            import msvcrt as real_msvcrt
        except ImportError:
            return None
        return real_msvcrt

    stand_in = types.ModuleType("msvcrt")
    stand_in.LK_LOCK = 1
    stand_in.LK_UNLCK = 0

    def locking(fd, mode, nbytes):
        if mode == stand_in.LK_UNLCK:
            _os_fcntl.lockf(fd, _os_fcntl.LOCK_UN, nbytes, 0, os.SEEK_SET)
            return
        try:
            _os_fcntl.lockf(
                fd, _os_fcntl.LOCK_EX | _os_fcntl.LOCK_NB, nbytes, 0, os.SEEK_SET
            )
        except OSError:
            # The C runtime's LK_LOCK waits and then raises OSError; the
            # production loop catches it and keeps retrying.
            raise OSError(errno.EDEADLK, "locking region busy") from None

    stand_in.locking = locking
    return stand_in


_MSVCRT_STAND_IN = _build_msvcrt_stand_in()


def _force_windows_branch() -> None:
    """Force ``vault._file_lock`` onto the Windows branch in this process."""
    vault_mod.fcntl = None
    sys.modules["msvcrt"] = _MSVCRT_STAND_IN


# ---------------------------------------------------------------------------
# module-level worker entry points (picklable for every start method)
# ---------------------------------------------------------------------------


def _seal_batch(
    root: str,
    key_id: str,
    payloads: list[bytes],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Seal ``payloads`` through the Windows branch; report the versions."""
    try:
        _force_windows_branch()
        if barrier is not None:
            barrier.wait()
        vault = Vault(root)
        try:
            sealed = [(vault.seal(key_id, payload), payload) for payload in payloads]
        finally:
            vault.close()
        results.put(("ok", sealed))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _hold_lock(
    root: str,
    acquired: "multiprocessing.Event",
    release: "multiprocessing.Event",
) -> None:
    """Take the Windows-branch inter-process lock and hold it until told."""
    _force_windows_branch()
    vault = Vault(root)
    try:
        # The file-lock helper's contract requires the caller to already
        # hold the in-process RLock.
        with vault._lock:
            with vault._file_lock():
                acquired.set()
                if not release.wait(timeout=30):
                    raise RuntimeError("lock holder was never released")
    finally:
        vault.close()


def _observe_snapshots(
    root: str,
    key_id: str,
    payloads: set[bytes],
    total: int,
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier",
) -> None:
    """Reopen/reload under the Windows branch; only complete prefixes may show.

    Every observed state must be exactly versions 1..n with the active
    version at n, every material one of the sealed payloads, and a version's
    material must never change between observations: a half-old/half-new
    mixture or a rewritten historical record fails the worker at once.
    """
    try:
        _force_windows_branch()
        barrier.wait()
        pinned: dict[int, bytes] = {}
        last = 0
        deadline = time.monotonic() + 60
        while last < total:
            if time.monotonic() > deadline:
                raise AssertionError(f"observer stalled at {last}, expected {total}")
            vault = Vault(root)
            try:
                vault.reload()
                visible = vault.versions(key_id)
                if visible != list(range(1, len(visible) + 1)):
                    raise AssertionError(f"non-contiguous versions: {visible}")
                if visible and vault.active(key_id) != len(visible):
                    raise AssertionError(
                        f"active {vault.active(key_id)} != latest {len(visible)}"
                    )
                for version in visible:
                    material = vault.load(key_id, version)
                    if material not in payloads:
                        raise AssertionError(
                            f"version {version} holds unexpected material"
                        )
                    if version in pinned and pinned[version] != material:
                        raise AssertionError(
                            f"version {version} material changed under reload"
                        )
                    pinned[version] = material
                if len(visible) < last:
                    raise AssertionError(
                        f"visible history shrank {last} -> {len(visible)}"
                    )
                last = len(visible)
            finally:
                vault.close()
        results.put(("ok", last))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _close_dance_first(
    root: str,
    key_id: str,
    closed: "multiprocessing.Event",
    peer_done: "multiprocessing.Event",
) -> None:
    """Seal, release the handle (twice), then reacquire after the peer."""
    _force_windows_branch()
    vault = Vault(root)
    try:
        if vault.seal(key_id, b"first-one") != 1:
            raise AssertionError("first seal must allocate version 1")
        vault.close()
        vault.close()  # releasing twice is not an error
        closed.set()
        if not peer_done.wait(timeout=30):
            raise RuntimeError("peer never finished after close")
        # The handle reacquires the lock and continues the shared sequence.
        if vault.seal(key_id, b"first-three") != 3:
            raise AssertionError("seal after close must continue the sequence")
    finally:
        vault.close()


def _close_dance_second(
    root: str,
    key_id: str,
    closed: "multiprocessing.Event",
    peer_done: "multiprocessing.Event",
    results: "multiprocessing.Queue",
) -> None:
    """Wait for the peer's ``close()`` then take the lock and measure it."""
    try:
        _force_windows_branch()
        vault = Vault(root)  # opened early; parked without holding the lock
        try:
            if not closed.wait(timeout=30):
                raise RuntimeError("peer never closed its lock handle")
            started = time.monotonic()
            version = vault.seal(key_id, b"second-two")
            elapsed = time.monotonic() - started
        finally:
            vault.close()  # release before signalling: no stale handle remains
        peer_done.set()
        results.put(("ok", (version, elapsed)))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


# ---------------------------------------------------------------------------
# shared fixture: everything runs on the forced Windows branch
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _MSVCRT_STAND_IN is not None, "no OS mutex available for the Windows branch"
)
class WindowsBranchCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.root = self.fixture.root
        # The whole test — parent side included — runs on the Windows
        # branch; both patches are reverted at teardown so no other test
        # module is affected regardless of execution order.  The
        # ``sys.modules`` entry is swapped by hand rather than through
        # ``mock.patch.dict``: that helper rebuilds the whole mapping at
        # teardown, which would evict modules lazily imported during the
        # patch window (notably ``multiprocessing.connection``) and break
        # their singleton identity.
        #
        # These two patch cleanups are registered after the fixture cleanup,
        # so LIFO order restores the branches first; closing the tracked
        # handles and deleting the directory (the single fixture teardown)
        # happen last.
        fcntl_patch = mock.patch.object(vault_mod, "fcntl", None)
        fcntl_patch.start()
        self.addCleanup(fcntl_patch.stop)
        missing = object()
        previous_msvcrt = sys.modules.get("msvcrt", missing)
        sys.modules["msvcrt"] = _MSVCRT_STAND_IN

        def restore_msvcrt() -> None:
            if previous_msvcrt is missing:
                sys.modules.pop("msvcrt", None)
            else:
                sys.modules["msvcrt"] = previous_msvcrt

        self.addCleanup(restore_msvcrt)

    def open_vault(self) -> Vault:
        """Open a vault; its handle is returned by the single teardown."""
        return self.fixture.open()

    def disk_manifest(self, root: Path | None = None) -> dict:
        root = self.root if root is None else root
        return json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))

    def disk_bytes(self, root: Path | None = None) -> dict[str, bytes]:
        """All vault records on disk keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, so it is deliberately excluded.
        """
        root = self.root if root is None else root
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def run_processes(
        self,
        processes: list[multiprocessing.Process],
        timeout: float = 90,
    ) -> None:
        # Tracked so the single teardown drains them even when an assertion
        # in this case fails while they are running.
        self.fixture.track_process(*processes)
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
                self.fail(f"worker {process.name} hung past {timeout}s")
            self.assertEqual(
                process.exitcode,
                0,
                f"worker {process.name} exited with {process.exitcode}",
            )

    def drain_nowait(self, process_queue: "multiprocessing.Queue") -> list:
        items = []
        while True:
            try:
                items.append(process_queue.get_nowait())
            except queue_mod.Empty:
                return items

    def assert_worker_results_ok(self, process_queue: "multiprocessing.Queue") -> list:
        """Drain a ``(status, payload)`` worker queue and fail on errors."""
        payloads = []
        for status, payload in self.drain_nowait(process_queue):
            self.assertEqual(status, "ok", payload)
            payloads.append(payload)
        return payloads

    def assert_strict_sequence_on_disk(self, key_id: str, total: int) -> None:
        """The persisted records hold 1..total exactly once, active at total."""
        entry = self.disk_manifest()["keys"][key_id]
        versions = [record["version"] for record in entry["versions"]]
        self.assertEqual(versions, list(range(1, total + 1)))
        self.assertEqual(len(set(versions)), total)
        self.assertEqual(entry["active"], total)
        material_files = sorted((self.root / MATERIALS_DIR).rglob("*.bin"))
        self.assertEqual(len(material_files), total)
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])


# ---------------------------------------------------------------------------
# cross-process contention: one strict sequence, each version exactly once
# ---------------------------------------------------------------------------


class TestWindowsContention(WindowsBranchCase):
    def test_two_processes_seal_one_key_each_version_recorded_once(self):
        # The minimal real race: two genuine processes, one key, one lock.
        each = 6
        payloads = [
            [f"proc{proc}-{index}".encode("utf-8") for index in range(each)]
            for proc in range(2)
        ]
        barrier = multiprocessing.Barrier(2)
        self.fixture.track_barrier(barrier)
        results: multiprocessing.Queue = multiprocessing.Queue()
        processes = [
            multiprocessing.Process(
                target=_seal_batch,
                args=(str(self.root), "k", payloads[proc], results, barrier),
            )
            for proc in range(2)
        ]
        self.run_processes(processes)
        reported = self.assert_worker_results_ok(results)
        self.assertEqual(len(reported), 2)

        total = 2 * each
        allocated: dict[int, bytes] = {}
        for sealed in reported:
            self.assertEqual(len(sealed), each)
            for version, payload in sealed:
                # Every version number is reported by exactly one process.
                self.assertNotIn(version, allocated)
                allocated[version] = payload
        self.assertEqual(sorted(allocated), list(range(1, total + 1)))

        # The persisted records show every version number exactly once, and
        # each historical material is byte-for-byte the payload that won it.
        self.assert_strict_sequence_on_disk("k", total)
        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, total + 1)))
        for version, payload in allocated.items():
            self.assertEqual(final.load("k", version), payload)

    def test_many_writers_strict_sequence_without_duplicates_or_gaps(self):
        writers, each = 4, 5
        payloads = [
            [f"w{writer}-{index}".encode("utf-8") for index in range(each)]
            for writer in range(writers)
        ]
        total = writers * each
        barrier = multiprocessing.Barrier(writers)
        self.fixture.track_barrier(barrier)
        results: multiprocessing.Queue = multiprocessing.Queue()
        processes = [
            multiprocessing.Process(
                target=_seal_batch,
                args=(str(self.root), "k", payloads[writer], results, barrier),
            )
            for writer in range(writers)
        ]
        self.run_processes(processes)
        reported = self.assert_worker_results_ok(results)
        self.assertEqual(len(reported), writers)

        allocated: dict[int, bytes] = {}
        for sealed in reported:
            for version, payload in sealed:
                self.assertNotIn(version, allocated)
                allocated[version] = payload
        # From 1, strictly ascending, no duplicates and no gaps.
        self.assertEqual(sorted(allocated), list(range(1, total + 1)))

        self.assert_strict_sequence_on_disk("k", total)
        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, total + 1)))
        self.assertEqual(final.active("k"), total)
        for version, payload in allocated.items():
            self.assertEqual(final.load("k", version), payload)


# ---------------------------------------------------------------------------
# a blocked writer waits, allocates nothing, and lands the uncontended result
# ---------------------------------------------------------------------------


class TestWindowsBlockedWriter(WindowsBranchCase):
    def test_blocked_writer_waits_then_matches_the_uncontended_result(self):
        vault = self.open_vault()
        vault.seal("k", b"m-1")
        manifest_before = (self.root / MANIFEST_NAME).read_bytes()

        acquired = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_hold_lock, args=(str(self.root), acquired, release)
        )
        # On a failed assertion the single teardown releases this gate and
        # drains holder and the blocked sealer: the blocked write proceeds
        # and finishes with its result unchanged.
        self.fixture.track_gate(release)
        self.fixture.track_process(holder)
        holder.start()
        self.assertTrue(acquired.wait(timeout=10))

        results: multiprocessing.Queue = multiprocessing.Queue()
        sealer = multiprocessing.Process(
            target=_seal_batch,
            args=(str(self.root), "k", [b"m-2"], results),
        )
        self.fixture.track_process(sealer)
        sealer.start()

        # The whole waiting window: the sealer stays parked, no version is
        # allocated, and every poll of the manifest finds the complete
        # pre-wait bytes — never a half record, never a truncated listing.
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            self.assertTrue(sealer.is_alive(), "sealer ran while the lock was held")
            self.assertEqual(self.drain_nowait(results), [])
            raw = (self.root / MANIFEST_NAME).read_bytes()
            self.assertEqual(raw, manifest_before)
            manifest = json.loads(raw.decode("utf-8"))
            self.assertEqual(
                [r["version"] for r in manifest["keys"]["k"]["versions"]], [1]
            )
            self.assertEqual(
                sorted(p.name for p in (self.root / MATERIALS_DIR).rglob("*.bin")),
                ["1.bin"],
            )
            self.assertEqual((self.root / REVOCATIONS_NAME).read_bytes(), b"")
            self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])
            # The historical material is untouched while the writer waits.
            (material_file,) = (self.root / MATERIALS_DIR).rglob("*.bin")
            self.assertEqual(material_file.read_bytes(), b"m-1")
            time.sleep(0.05)

        release.set()
        sealer.join(timeout=30)
        self.assertEqual(sealer.exitcode, 0)
        holder.join(timeout=30)
        self.assertEqual(holder.exitcode, 0)
        # The blocked write allocates its version only once the lock is its.
        self.assertEqual(self.assert_worker_results_ok(results), [[(2, b"m-2")]])

        # The contended run lands exactly the same bytes as running the same
        # two seals with no contention at all.
        reference_root = self.fixture.path("reference")
        reference = self.fixture.open(reference_root)
        reference.seal("k", b"m-1")
        reference.seal("k", b"m-2")
        reference.close()
        self.assertEqual(self.disk_bytes(), self.disk_bytes(reference_root))

        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2])
        self.assertEqual(final.load("k", 1), b"m-1")
        self.assertEqual(final.load("k", 2), b"m-2")


# ---------------------------------------------------------------------------
# close(): idempotent release, immediate handoff, transparent reacquire
# ---------------------------------------------------------------------------


class TestWindowsCloseRelease(WindowsBranchCase):
    def test_close_is_idempotent_and_operations_reacquire_unchanged(self):
        # Observable outcomes only, never private state: repeated close()
        # calls return None without error, the next operation works and the
        # durable results are unchanged.
        vault = self.open_vault()
        vault.seal("k", b"one")
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())  # repeated release is not an error

        # Every operation kind reopens the lock and behaves identically.
        self.assertEqual(vault.seal("k", b"two"), 2)
        self.assertEqual(vault.load("k", 1), b"one")
        self.assertEqual(vault.load("k"), b"two")
        self.assertEqual(vault.derive_seal("d", b"pw", b"salt", 100, 16), 1)
        self.assertEqual(vault.seal("k", b"three"), 3)
        vault.revoke("k", 1)
        vault.set_active("k", 2)
        vault.reload()
        self.assertEqual(vault.active("k"), 2)
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())  # still an error-free no-op

        # The reacquired-lock writes landed durably and survive a new open.
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), [1, 2, 3])
        self.assertTrue(reopened.is_revoked("k", 1))
        self.assertEqual(reopened.active("k"), 2)
        self.assertEqual(reopened.load("k", 2), b"two")
        self.assertEqual(reopened.versions("d"), [1])

    def test_waiting_process_takes_the_lock_immediately_after_close(self):
        closed = multiprocessing.Event()
        peer_done = multiprocessing.Event()
        results: multiprocessing.Queue = multiprocessing.Queue()
        # The second process opens first but parks until the first closes,
        # so it measures the handoff latency directly and proves no stale
        # handle blocks it.
        second = multiprocessing.Process(
            target=_close_dance_second,
            args=(str(self.root), "k", closed, peer_done, results),
        )
        first = multiprocessing.Process(
            target=_close_dance_first, args=(str(self.root), "k", closed, peer_done)
        )
        # A failed assertion releases both handoff gates and drains both
        # processes through the single teardown.
        self.fixture.track_gate(closed, peer_done)
        self.fixture.track_process(first, second)
        second.start()
        first.start()
        for process in (first, second):
            process.join(timeout=40)
            self.assertEqual(process.exitcode, 0)
        self.assertTrue(peer_done.wait(timeout=5))
        (version, elapsed), = self.assert_worker_results_ok(results)
        self.assertEqual(version, 2)
        # No stale handle blocks the peer: the seal completes promptly.
        self.assertLess(elapsed, 10.0)

        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2, 3])
        self.assertEqual(final.load("k", 1), b"first-one")
        self.assertEqual(final.load("k", 2), b"second-two")
        self.assertEqual(final.load("k", 3), b"first-three")


# ---------------------------------------------------------------------------
# the lock file is only a mutual-exclusion device
# ---------------------------------------------------------------------------


class TestWindowsLockFileContent(WindowsBranchCase):
    def test_empty_or_arbitrary_lock_bytes_change_nothing(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.seal("k", b"two")

        lock_path = self.root / LOCK_NAME
        # Empty: reads, writes and reloads are unaffected.
        lock_path.write_bytes(b"")
        vault.reload()
        self.assertEqual(vault.load("k", 2), b"two")
        self.assertEqual(vault.seal("k", b"three"), 3)

        # Arbitrary bytes: still no effect on any operation, and the shared
        # version sequence just continues.
        lock_path.write_bytes(b"\x00\xff not key material \xfe\n" * 8)
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        self.assertEqual(vault.seal("k", b"four"), 4)
        vault.revoke("k", 1)
        vault.set_active("k", 2)
        vault.reload()
        self.assertEqual(vault.active("k"), 2)
        self.assertTrue(vault.is_revoked("k", 1))

        # A different process opens, writes a full record and validates,
        # undeterred by the junk bytes in the lock file.
        results: multiprocessing.Queue = multiprocessing.Queue()
        worker = multiprocessing.Process(
            target=_seal_batch, args=(str(self.root), "k", [b"five"], results)
        )
        self.fixture.track_process(worker)
        worker.start()
        worker.join(timeout=30)
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(self.assert_worker_results_ok(results), [[(5, b"five")]])

        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2, 3, 4, 5])
        self.assertEqual(final.load("k", 5), b"five")
        self.assertTrue(final.is_revoked("k", 1))
        # The junk bytes never leaked into any real record.
        marker = b"not key material"
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != LOCK_NAME:
                self.assertNotIn(marker, path.read_bytes())


# ---------------------------------------------------------------------------
# reload swaps only complete snapshots; failure freezes snapshot and disk
# ---------------------------------------------------------------------------


class TestWindowsReloadSemantics(WindowsBranchCase):
    def test_concurrent_reader_only_observes_complete_snapshots(self):
        writers, each = 3, 4
        payloads = [
            [f"r{writer}-{index}".encode("utf-8") for index in range(each)]
            for writer in range(writers)
        ]
        total = writers * each
        all_payloads = {payload for group in payloads for payload in group}
        barrier = multiprocessing.Barrier(writers + 1)
        self.fixture.track_barrier(barrier)
        results: multiprocessing.Queue = multiprocessing.Queue()
        processes = [
            multiprocessing.Process(
                target=_seal_batch,
                args=(str(self.root), "k", payloads[writer], results, barrier),
            )
            for writer in range(writers)
        ]
        observer_results: multiprocessing.Queue = multiprocessing.Queue()
        observer = multiprocessing.Process(
            target=_observe_snapshots,
            args=(str(self.root), "k", all_payloads, total, observer_results, barrier),
        )
        self.fixture.track_process(observer)
        observer.start()
        self.run_processes(processes)
        observer.join(timeout=60)
        self.assertEqual(observer.exitcode, 0)
        self.assertEqual(self.assert_worker_results_ok(observer_results), [total])
        self.assertEqual(len(self.assert_worker_results_ok(results)), writers)

        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, total + 1)))
        self.assertEqual(
            {final.load("k", v) for v in range(1, total + 1)}, all_payloads
        )

    def test_failed_reload_freezes_snapshot_and_disk_then_recovers(self):
        vault = self.open_vault()
        vault.seal("plain", b"plain-one")
        vault.seal("plain", b"plain-two")
        vault.set_active("plain", 1)
        vault.derive_seal("drv", b"passphrase", b"salt-alpha", 500, 24)
        vault.revoke("plain", 2)

        def answers() -> dict:
            state = {}
            for key_id in ("plain", "drv"):
                versions = vault.versions(key_id)
                state[key_id] = {
                    "versions": versions,
                    "active": vault.active(key_id),
                    "revoked": vault.revoked_versions(key_id),
                    "is_revoked": {
                        version: vault.is_revoked(key_id, version)
                        for version in versions
                    },
                    "materials": {
                        version: vault.load(key_id, version) for version in versions
                    },
                    "derivations": {
                        version: vault.derivation(key_id, version)
                        for version in versions
                    },
                }
            state["manifest"] = vault.manifest()
            return state

        expected = answers()
        healthy_disk = self.disk_bytes()

        # Corrupt the manifest out of band; the already-open handle keeps
        # its snapshot while reloads and fresh opens reject the vault.
        corrupt = b"{ not valid json"
        (self.root / MANIFEST_NAME).write_bytes(corrupt)
        failing_disk = self.disk_bytes()

        message = None
        for _ in range(3):
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            if message is None:
                message = str(caught.exception)
            else:
                # Repeating the same input gives the identical failure.
                self.assertEqual(str(caught.exception), message)
            # The in-memory snapshot is exactly as it was, word for word.
            self.assertEqual(answers(), expected)

        # A cold opener on the same directory raises the same ValueError.
        with self.assertRaises(ValueError) as caught:
            Vault(self.root)
        self.assertEqual(str(caught.exception), message)

        # The failed reloads/opens neither added nor removed a disk record,
        # and the keys already in hand stay readable throughout.
        self.assertEqual(self.disk_bytes(), failing_disk)
        self.assertEqual(vault.load("plain", 1), b"plain-one")
        self.assertEqual(vault.load("drv"), expected["drv"]["materials"][1])

        # Undo the corruption: one reload restores the exact healthy state.
        (self.root / MANIFEST_NAME).write_bytes(healthy_disk[MANIFEST_NAME])
        vault.reload()
        self.assertEqual(answers(), expected)
        self.assertEqual(self.disk_bytes(), healthy_disk)
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("plain"), [1, 2])
        self.assertEqual(reopened.active("plain"), 1)
        self.assertTrue(reopened.is_revoked("plain", 2))


# ---------------------------------------------------------------------------
# entry-point error vocabulary on the Windows branch
# ---------------------------------------------------------------------------


class TestWindowsEntryContract(WindowsBranchCase):
    def test_rejections_raise_exactly_and_allocate_nothing(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        manifest_before = (self.root / MANIFEST_NAME).read_bytes()
        journal_before = (self.root / REVOCATIONS_NAME).read_bytes()

        # Empty key id -> ValueError at every entry point.
        for call in (
            lambda: vault.seal("", b"m"),
            lambda: vault.derive_seal("", b"pw", b"salt", 1, 1),
            lambda: vault.load("", 1),
            lambda: vault.revoke("", 1),
            lambda: vault.set_active("", 1),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
            lambda: vault.derivation("", 1),
            lambda: vault.active(""),
        ):
            with self.assertRaises(ValueError):
                call()

        # Non-genuine-integer version -> TypeError (bools and floats do not
        # count).  ``load`` and ``derivation`` accept None as the
        # active-version sentinel, so None is excluded for them.
        for bad_version in (True, False, 1.0, 2.5, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=repr(bad_version)):
                vault.revoke("k", bad_version)
            with self.assertRaises(TypeError, msg=repr(bad_version)):
                vault.is_revoked("k", bad_version)
            with self.assertRaises(TypeError, msg=repr(bad_version)):
                vault.set_active("k", bad_version)
        for bad_version in (True, False, 1.0, 2.5, "1", (1,)):
            with self.assertRaises(TypeError, msg=repr(bad_version)):
                vault.load("k", bad_version)
            with self.assertRaises(TypeError, msg=repr(bad_version)):
                vault.derivation("k", bad_version)
        # A float numerically equal to an existing version cannot read it.
        with self.assertRaises(TypeError):
            vault.load("k", 1.0)
        self.assertEqual(vault.load("k", None), b"m")
        self.assertEqual(vault.derivation("k", None), {})

        # Unknown key or version -> KeyError.
        for call in (
            lambda: vault.load("ghost"),
            lambda: vault.active("ghost"),
            lambda: vault.revoke("ghost", 1),
            lambda: vault.set_active("ghost", 1),
            lambda: vault.is_revoked("ghost", 1),
            lambda: vault.derivation("ghost"),
        ):
            with self.assertRaises(KeyError):
                call()
        for bad_version in (0, 2, 99, -1):
            with self.assertRaises(KeyError, msg=str(bad_version)):
                vault.load("k", bad_version)
            with self.assertRaises(KeyError, msg=str(bad_version)):
                vault.revoke("k", bad_version)
            with self.assertRaises(KeyError, msg=str(bad_version)):
                vault.set_active("k", bad_version)
            with self.assertRaises(KeyError, msg=str(bad_version)):
                vault.is_revoked("k", bad_version)
            with self.assertRaises(KeyError, msg=str(bad_version)):
                vault.derivation("d", bad_version)

        # No rejection produced a new version or touched a record on disk.
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.versions("d"), [1])
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), manifest_before)
        self.assertEqual((self.root / REVOCATIONS_NAME).read_bytes(), journal_before)
        self.assertFalse((self.root / ACTIVATIONS_NAME).exists())
        self.assertEqual(vault.load("k"), b"m")


# ---------------------------------------------------------------------------
# determinism: the same scripted input lands byte-identical results
# ---------------------------------------------------------------------------


class TestWindowsDeterminism(WindowsBranchCase):
    def _scripted_run(self, root: Path) -> None:
        vault = self.fixture.open(root)
        vault.seal("alpha", b"alpha-1")
        vault.derive_seal("alpha", b"pw", b"salt-1", 100, 16)
        vault.seal("beta", b"beta-1")
        vault.seal("beta", b"beta-2")
        vault.set_active("beta", 1)
        vault.revoke("alpha", 1)
        vault.reload()
        vault.close()

    def test_repeated_identical_input_produces_identical_bytes(self):
        first_root = self.fixture.path("first")
        second_root = self.fixture.path("second")
        self._scripted_run(first_root)
        self._scripted_run(second_root)
        # Repeating the same input yields exactly the same persisted state.
        self.assertEqual(self.disk_bytes(first_root), self.disk_bytes(second_root))

        # And both runs answer every query identically.
        first = self.fixture.open(first_root)
        second = self.fixture.open(second_root)
        for key_id in ("alpha", "beta"):
            self.assertEqual(first.versions(key_id), second.versions(key_id))
            self.assertEqual(first.active(key_id), second.active(key_id))
            self.assertEqual(
                first.revoked_versions(key_id), second.revoked_versions(key_id)
            )
            for version in first.versions(key_id):
                self.assertEqual(
                    first.load(key_id, version), second.load(key_id, version)
                )
                self.assertEqual(
                    first.derivation(key_id, version),
                    second.derivation(key_id, version),
                )


if __name__ == "__main__":
    unittest.main()
