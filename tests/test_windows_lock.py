"""Windows-branch (``msvcrt.locking``) multi-process regression tests.

The production ``_file_lock`` falls back to ``msvcrt.locking`` when
``fcntl`` is unavailable (Windows).  The existing suite exercises that
branch only with a controllable in-process fake; the cases here drive it
for real across processes:

* the branch is forced by patching ``keyvault_ledger.vault.fcntl`` away and
  installing an ``msvcrt`` stand-in whose ``locking`` is backed by a
  genuine OS-level inter-process mutex (a byte-range ``fcntl.lockf`` on
  the very file descriptor the vault hands it — or the real ``msvcrt``
  where that is what the platform provides).  ``LK_LOCK`` raises
  ``OSError`` exactly like the C runtime's timed-out wait, so the
  production acquire/retry/release loop runs verbatim;
* competing writer processes sharing one vault directory allocate version
  numbers strictly from 1 with no duplicates and no gaps, and historical
  material is never overwritten;
* a writer that cannot take the lock waits until it is released: it
  allocates no version number while parked, leaves no half record or
  truncated listing behind, and once it proceeds the result is identical
  to the same writes without contention;
* ``close()`` releases the lock handle, repeats harmlessly, hands the lock
  to a waiting process immediately, and later operations reacquire it with
  no observable change;
* a failed ``reload()`` (corrupt derived-material record or activity
  journal) raises ``ValueError`` while the in-memory snapshot — including
  the per-version ``is_revoked`` answers — and the disk records stay
  byte-for-byte frozen, and undoing the corruption restores full
  snapshot/disk correspondence;
* the entry-point error vocabulary (empty id -> ``ValueError``, non-integer
  version -> ``TypeError``, unknown key/version -> ``KeyError``) holds on
  the Windows branch and never produces a new version.

Only tests are added here; the product code, the public interface and the
three CLI entry points are untouched.  Everything runs in temporary
directories with the standard library only and is independent of execution
order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import errno
import hashlib
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
    _dump_manifest,
)
from tests._fixtures import VaultFixture

try:
    import fcntl as _os_fcntl
except ImportError:  # real Windows: the real msvcrt is the stand-in
    _os_fcntl = None


def _build_msvcrt_stand_in() -> "types.ModuleType | None":
    """An ``msvcrt`` module object backed by a real inter-process mutex.

    On POSIX the byte-range lock is delegated to ``fcntl.lockf`` on the
    same descriptor, so contending processes genuinely block each other;
    ``LK_LOCK`` raises ``OSError`` on contention, mirroring the C runtime
    giving up after its internal timeout, which is what drives the
    production retry loop.  Where ``fcntl`` does not exist the real
    ``msvcrt`` is used as itself.
    """
    if _os_fcntl is None:
        try:
            import msvcrt as real_msvcrt
        except ImportError:
            return None
        return real_msvcrt

    module = types.ModuleType("msvcrt")
    module.LK_LOCK = 1
    module.LK_UNLCK = 0

    def locking(fd, mode, nbytes):
        if mode == module.LK_UNLCK:
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

    module.locking = locking
    return module


_MSVCRT_STAND_IN = _build_msvcrt_stand_in()


def _install_windows_branch() -> None:
    """Force ``vault._file_lock`` onto the Windows branch in this process."""
    vault_mod.fcntl = None
    sys.modules["msvcrt"] = _MSVCRT_STAND_IN


# ---------------------------------------------------------------------------
# module-level worker entry points (picklable for every start method)
# ---------------------------------------------------------------------------


def _win_seal_payloads(
    root: str,
    key_id: str,
    payloads: list[bytes],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Seal ``payloads`` through the Windows branch; report the versions."""
    try:
        _install_windows_branch()
        if barrier is not None:
            barrier.wait()
        vault = Vault(root)
        sealed = []
        try:
            for payload in payloads:
                sealed.append((vault.seal(key_id, payload), payload))
        finally:
            vault.close()
        results.put(("ok", sealed))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _win_lock_holder(
    root: str,
    ready: "multiprocessing.Event",
    release: "multiprocessing.Event",
) -> None:
    """Take the Windows-branch inter-process lock and hold it until told."""
    _install_windows_branch()
    vault = Vault(root)
    try:
        # The file-lock helper's contract requires the caller to already
        # hold the in-process RLock.
        with vault._lock:
            with vault._file_lock():
                ready.set()
                if not release.wait(timeout=30):
                    raise RuntimeError("holder was never released")
    finally:
        vault.close()


def _win_prefix_reader(
    root: str,
    key_id: str,
    payloads: set[bytes],
    total: int,
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Reopen/reload under the Windows branch; only complete prefixes may show.

    Every observed state must be exactly versions 1..n with the active
    version at n, every material one of the sealed payloads, and a
    version's material must never change between observations.
    """
    try:
        _install_windows_branch()
        if barrier is not None:
            barrier.wait()
        pinned: dict[int, bytes] = {}
        last = 0
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and last < total:
            vault = Vault(root)
            vault.reload()
            visible = vault.versions(key_id)
            if visible != list(range(1, len(visible) + 1)):
                raise AssertionError(f"non-contiguous versions: {visible}")
            if visible:
                if vault.active(key_id) != len(visible):
                    raise AssertionError(
                        f"active {vault.active(key_id)} != latest {len(visible)}"
                    )
                for version in visible:
                    material = vault.load(key_id, version)
                    if material not in payloads:
                        raise AssertionError(
                            f"version {version} holds unexpected material"
                        )
                    previous = pinned.get(version)
                    if previous is not None and previous != material:
                        raise AssertionError(
                            f"version {version} material changed under reload"
                        )
                    pinned[version] = material
            if len(visible) < last:
                raise AssertionError(f"visible history shrank {last} -> {len(visible)}")
            last = len(visible)
            vault.close()
        if last != total:
            raise AssertionError(f"reader stalled at {last}, expected {total}")
        results.put(("ok", last))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _win_close_dance_a(
    root: str,
    key_id: str,
    closed: "multiprocessing.Event",
    peer_done: "multiprocessing.Event",
) -> None:
    _install_windows_branch()
    vault = Vault(root)
    if vault.seal(key_id, b"a-one") != 1:
        raise AssertionError("first seal must be version 1")
    vault.close()
    vault.close()  # releasing twice is not an error
    closed.set()
    if not peer_done.wait(timeout=30):
        raise RuntimeError("peer never finished after close")
    # The handle reacquires the lock and continues the shared sequence.
    if vault.seal(key_id, b"a-three") != 3:
        raise AssertionError("seal after close must continue the sequence")
    vault.close()


def _win_close_dance_b(
    root: str,
    key_id: str,
    closed: "multiprocessing.Event",
    peer_done: "multiprocessing.Event",
    results: "multiprocessing.Queue",
) -> None:
    try:
        _install_windows_branch()
        vault = Vault(root)  # opened early; parked without holding the lock
        if not closed.wait(timeout=30):
            raise RuntimeError("peer never closed its lock")
        started = time.monotonic()
        version = vault.seal(key_id, b"b-two")
        elapsed = time.monotonic() - started
        vault.close()  # release before signalling: no stale handle may remain
        peer_done.set()
        results.put(("ok", (version, elapsed)))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _MSVCRT_STAND_IN is not None, "no OS mutex available for the Windows branch"
)
class WindowsLockTestCase(unittest.TestCase):
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
        # happen last, exactly like the previous temp-dir cleanup did.
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

    def disk_manifest(self) -> dict:
        return json.loads((self.root / MANIFEST_NAME).read_bytes().decode("utf-8"))

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


# ---------------------------------------------------------------------------
# cross-process contention on the Windows branch
# ---------------------------------------------------------------------------


class TestWindowsBranchContention(WindowsLockTestCase):
    def test_competing_writers_allocate_a_strict_sequence_from_one(self):
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
                target=_win_seal_payloads,
                args=(str(self.root), "k", payloads[writer], results, barrier),
            )
            for writer in range(writers)
        ]
        self.run_processes(processes)
        reported = self.assert_worker_results_ok(results)
        self.assertEqual(len(reported), writers)

        # Every reported (version, payload) pair is unique in its version
        # and together they are exactly 1..total: no duplicates, no gaps,
        # and nothing allocated early.
        allocated: dict[int, bytes] = {}
        for sealed in reported:
            self.assertEqual(len(sealed), each)
            for version, payload in sealed:
                self.assertNotIn(version, allocated)
                allocated[version] = payload
        self.assertEqual(sorted(allocated), list(range(1, total + 1)))

        # The persisted state matches: a strict sequence from 1, and every
        # historical material byte-for-byte the payload that won that slot —
        # nothing was ever overwritten.
        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, total + 1)))
        self.assertEqual(final.active("k"), total)
        for version, payload in allocated.items():
            self.assertEqual(final.load("k", version), payload)
        disk_entry = self.disk_manifest()["keys"]["k"]
        self.assertEqual(
            [record["version"] for record in disk_entry["versions"]],
            list(range(1, total + 1)),
        )
        self.assertEqual(disk_entry["active"], total)
        material_files = sorted((self.root / MATERIALS_DIR).rglob("*.bin"))
        self.assertEqual(len(material_files), total)
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])

    def test_blocked_writer_waits_and_matches_the_uncontended_result(self):
        vault = self.open_vault()
        vault.seal("k", b"m-1")

        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_win_lock_holder, args=(str(self.root), ready, release)
        )
        # On a failed assertion the single teardown releases this gate and
        # drains holder and the blocked sealer: the blocked write proceeds
        # and finishes with its result unchanged.
        self.fixture.track_gate(release)
        self.fixture.track_process(holder)
        holder.start()
        self.assertTrue(ready.wait(timeout=10))

        results: multiprocessing.Queue = multiprocessing.Queue()
        sealer = multiprocessing.Process(
            target=_win_seal_payloads,
            args=(str(self.root), "k", [b"m-2"], results),
        )
        self.fixture.track_process(sealer)
        sealer.start()

        # While the holder keeps the lock the sealer stays parked: no
        # version is allocated, no material appears, no record is half
        # written and no listing is truncated.
        time.sleep(0.6)
        self.assertTrue(sealer.is_alive(), "sealer ran while the lock was held")
        self.assertEqual(self.drain_nowait(results), [])
        self.assertEqual(
            [r["version"] for r in self.disk_manifest()["keys"]["k"]["versions"]],
            [1],
        )
        self.assertEqual(
            sorted(p.name for p in (self.root / MATERIALS_DIR).rglob("*.bin")),
            ["1.bin"],
        )
        self.assertEqual((self.root / REVOCATIONS_NAME).read_bytes(), b"")
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])

        release.set()
        sealer.join(timeout=30)
        self.assertEqual(sealer.exitcode, 0)
        holder.join(timeout=30)
        self.assertEqual(holder.exitcode, 0)
        self.assertEqual(self.assert_worker_results_ok(results), [[(2, b"m-2")]])

        # The contended run lands exactly the same bytes as running the
        # same two seals with no contention at all.
        reference_root = self.fixture.path("reference")
        reference = self.fixture.open(reference_root)
        reference.seal("k", b"m-1")
        reference.seal("k", b"m-2")
        reference.close()
        self.assertEqual(
            self.disk_bytes(),
            self.disk_bytes(reference_root),
        )
        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2])
        self.assertEqual(final.load("k", 1), b"m-1")
        self.assertEqual(final.load("k", 2), b"m-2")

    def test_concurrent_reader_only_sees_complete_snapshots(self):
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
                target=_win_seal_payloads,
                args=(str(self.root), "k", payloads[writer], results, barrier),
            )
            for writer in range(writers)
        ]
        reader_results: multiprocessing.Queue = multiprocessing.Queue()
        reader = multiprocessing.Process(
            target=_win_prefix_reader,
            args=(str(self.root), "k", all_payloads, total, reader_results, barrier),
        )
        self.fixture.track_process(reader)
        reader.start()
        self.run_processes(processes)
        reader.join(timeout=60)
        self.assertEqual(reader.exitcode, 0)
        self.assertEqual(self.assert_worker_results_ok(reader_results), [total])
        self.assertEqual(len(self.assert_worker_results_ok(results)), writers)

        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, total + 1)))
        self.assertEqual(
            {final.load("k", v) for v in range(1, total + 1)}, all_payloads
        )


# ---------------------------------------------------------------------------
# close(): idempotent release and immediate handoff on the Windows branch
# ---------------------------------------------------------------------------


class TestWindowsCloseRelease(WindowsLockTestCase):
    def test_another_process_takes_the_lock_immediately_after_close(self):
        closed = multiprocessing.Event()
        peer_done = multiprocessing.Event()
        results: multiprocessing.Queue = multiprocessing.Queue()
        # B opens first but parks until A closes, so it measures the
        # handoff latency directly and proves no stale handle blocks it.
        b = multiprocessing.Process(
            target=_win_close_dance_b,
            args=(str(self.root), "k", closed, peer_done, results),
        )
        a = multiprocessing.Process(
            target=_win_close_dance_a, args=(str(self.root), "k", closed, peer_done)
        )
        # A failed assertion releases both handoff gates and drains both
        # processes through the single teardown.
        self.fixture.track_gate(closed, peer_done)
        self.fixture.track_process(a, b)
        b.start()
        a.start()
        for process in (a, b):
            process.join(timeout=40)
            self.assertEqual(process.exitcode, 0)
        self.assertTrue(peer_done.wait(timeout=5))
        (version, elapsed), = self.assert_worker_results_ok(results)
        self.assertEqual(version, 2)
        # No stale handle blocks the peer: the seal completes promptly.
        self.assertLess(elapsed, 10.0)

        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2, 3])
        self.assertEqual(final.load("k", 1), b"a-one")
        self.assertEqual(final.load("k", 2), b"b-two")
        self.assertEqual(final.load("k", 3), b"a-three")

    def test_close_is_idempotent_and_operations_reacquire_unchanged(self):
        # Observable outcomes only, never private state: repeated close()
        # calls return None without error, the next operation works and the
        # durable results are unchanged.  The cross-process "lock handed over
        # immediately" half is pinned by the sibling close-dance case above.
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


# ---------------------------------------------------------------------------
# failed reload on the Windows branch: snapshot and disk double invariance,
# with the per-version revocation query folded into the frozen comparison
# ---------------------------------------------------------------------------


class TestWindowsFailedReloadFreeze(WindowsLockTestCase):
    def _build_healthy(self) -> Vault:
        """A vault exercising every record kind at once."""
        vault = self.open_vault()
        vault.seal("plain", b"plain-one")
        vault.seal("plain", b"plain-two")
        vault.set_active("plain", 1)  # bound to latest=2

        vault.derive_seal("drv", b"passphrase-alpha", b"salt-alpha", 1000, 32)
        vault.derive_seal("drv", b"passphrase-bravo", b"salt-bravo", 2500, 48)
        vault.revoke("drv", 1)

        vault.seal("mix", b"mix-one")
        vault.derive_seal("mix", b"passphrase-charlie", b"salt-charlie", 500, 24)
        vault.seal("mix", b"mix-three")
        vault.revoke("mix", 1)
        vault.set_active("mix", 2)  # points at the derived version, latest=3
        return vault

    def _frozen_answers(self, vault: Vault) -> dict:
        """Every observable answer, per-version revocation status included."""
        keys = {}
        for key_id in ("plain", "drv", "mix"):
            versions = vault.versions(key_id)
            keys[key_id] = {
                "versions": versions,
                "active": vault.active(key_id),
                "revoked": vault.revoked_versions(key_id),
                # The per-version revocation query is part of the frozen
                # comparison in every failure window.
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
        return {
            "keys": keys,
            "manifest": vault.manifest(),
            "unknown_versions": vault.versions("never-sealed"),
            "unknown_revoked": vault.revoked_versions("never-sealed"),
        }

    def _restore_disk(self, healthy: dict[str, bytes]) -> None:
        """Put the vault directory back in exactly the captured state."""
        current = self.disk_bytes()
        for rel in current.keys() - healthy.keys():
            (self.root / rel).unlink()
        for rel, data in healthy.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)

    def _assert_snapshot_corresponds_to_disk(self, vault: Vault) -> None:
        """Re-derive every observable answer straight from the disk records."""
        manifest = json.loads((self.root / MANIFEST_NAME).read_bytes().decode("utf-8"))
        self.assertEqual(vault.manifest(), manifest)

        revoked: dict[str, set[int]] = {}
        for line in (self.root / REVOCATIONS_NAME).read_text("utf-8").splitlines():
            record = json.loads(line)
            revoked.setdefault(record["key_id"], set()).add(record["version"])

        last_repoint: dict[str, tuple[int, int]] = {}
        activations = self.root / ACTIVATIONS_NAME
        if activations.exists():
            for line in activations.read_text("utf-8").splitlines():
                record = json.loads(line)
                last_repoint[record["key_id"]] = (
                    record["version"],
                    record["latest"],
                )

        for key_id, entry in manifest["keys"].items():
            records = entry["versions"]
            versions = [record["version"] for record in records]
            self.assertEqual(vault.versions(key_id), versions)
            for record in records:
                version = record["version"]
                data = (self.root / record["file"]).read_bytes()
                self.assertEqual(vault.load(key_id, version), data)
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
                # The per-version revocation query agrees with the journal.
                self.assertEqual(
                    vault.is_revoked(key_id, version),
                    version in revoked.get(key_id, set()),
                )
            expected_active = entry["active"]
            if key_id in last_repoint:
                target, bound_latest = last_repoint[key_id]
                if bound_latest == versions[-1]:
                    expected_active = target
            self.assertEqual(vault.active(key_id), expected_active)
            self.assertEqual(
                set(vault.revoked_versions(key_id)), revoked.get(key_id, set())
            )

    def _assert_failure_then_recovery(self, vault: Vault, corrupt) -> None:
        """Drive one full corruption window and the recovery afterwards.

        The identical corrupt/restore cycle runs twice: repeating the same
        input must give the identical exception and the identical frozen
        state, and recovery must work both times.
        """
        healthy_disk = self.disk_bytes()
        expected = self._frozen_answers(vault)
        message = None
        for cycle in range(2):
            with self.subTest(cycle=cycle):
                corrupt()
                failing_disk = self.disk_bytes()

                # Repeated failed reloads: one deterministic ValueError, and
                # every observable answer — per-version revocation status
                # included — stays frozen word for word.
                for _ in range(3):
                    with self.assertRaises(ValueError) as caught:
                        vault.reload()
                    if message is None:
                        message = str(caught.exception)
                    else:
                        self.assertEqual(str(caught.exception), message)
                    self.assertEqual(self._frozen_answers(vault), expected)

                # A cold opener rejects the very same state identically.
                with self.assertRaises(ValueError) as caught:
                    Vault(self.root)
                self.assertEqual(str(caught.exception), message)

                # The failed reloads/opens neither added nor removed a disk
                # record, and the keys already in hand stay readable.
                self.assertEqual(self.disk_bytes(), failing_disk)
                self.assertEqual(vault.load("mix", 2), expected["keys"]["mix"]["materials"][2])

                # Undo the corruption: one successful reload restores full
                # correspondence between snapshot and disk.
                self._restore_disk(healthy_disk)
                vault.reload()
                self.assertEqual(self._frozen_answers(vault), expected)
                self._assert_snapshot_corresponds_to_disk(vault)
                self.assertEqual(self.disk_bytes(), healthy_disk)

    def test_corrupt_derivation_record_freezes_snapshot_and_disk(self):
        vault = self._build_healthy()

        def corrupt() -> None:
            path = self.root / MANIFEST_NAME
            manifest = json.loads(path.read_bytes().decode("utf-8"))
            record = next(
                record
                for record in manifest["keys"]["mix"]["versions"]
                if record["version"] == 2
            )
            record["derivation"]["salt"] = "!!!not-base64!!!"
            path.write_bytes(_dump_manifest(manifest))

        self._assert_failure_then_recovery(vault, corrupt)

    def test_corrupt_activation_journal_freezes_snapshot_and_disk(self):
        vault = self._build_healthy()

        def corrupt() -> None:
            (self.root / ACTIVATIONS_NAME).write_bytes(
                b'{"key_id": "mix", "version": 99, "latest": 99}\n'
            )

        self._assert_failure_then_recovery(vault, corrupt)


# ---------------------------------------------------------------------------
# entry-point error contract on the Windows branch
# ---------------------------------------------------------------------------


class TestWindowsEntryContract(WindowsLockTestCase):
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
            lambda: vault.revoke("", 1),
            lambda: vault.set_active("", 1),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
            lambda: vault.derivation(""),
        ):
            with self.assertRaises(ValueError):
                call()

        # Non-integer version -> TypeError (bools and floats do not count).
        # ``derivation`` is checked separately: a None version is legal
        # there and resolves to the active version.
        for bad_version in (True, False, 1.0, 2.5, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=repr(bad_version)):
                vault.revoke("k", bad_version)
            with self.assertRaises(TypeError, msg=repr(bad_version)):
                vault.is_revoked("k", bad_version)
            with self.assertRaises(TypeError, msg=repr(bad_version)):
                vault.set_active("k", bad_version)
        for bad_version in (True, False, 1.0, 2.5, "1", (1,)):
            with self.assertRaises(TypeError, msg=repr(bad_version)):
                vault.derivation("k", bad_version)
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

    def test_readback_is_byte_exact_on_the_windows_branch(self):
        vault = self.open_vault()
        payloads = [b"\x00binary\xff", "文本".encode("utf-8"), b""]
        for index, payload in enumerate(payloads, start=1):
            self.assertEqual(vault.seal("k", payload), index)
        vault.reload()
        for index, payload in enumerate(payloads, start=1):
            self.assertEqual(vault.load("k", index), payload)
        reopened = self.open_vault()
        for index, payload in enumerate(payloads, start=1):
            self.assertEqual(reopened.load("k", index), payload)


if __name__ == "__main__":
    unittest.main()
