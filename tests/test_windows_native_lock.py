"""Genuine Windows-system-lock (``msvcrt.locking``) regression tests.

This module is the native counterpart of :mod:`tests.test_windows_lock`.
That suite forces the production ``_file_lock`` onto its Windows branch with
a controllable *stand-in* ``msvcrt`` (a ``fcntl.lockf`` back-end on POSIX);
nothing there ever calls the operating system's Windows locking runtime.
The cases here close that gap:

* they run **only on Windows** (``sys.platform == "win32"``), where the
  production code imports the real standard-library ``msvcrt`` itself — no
  module is patched, no fake lock is installed, no branch is forced.  Every
  seal, wait, release and reload goes through the genuine system-level file
  lock on ``vault.lock``;
* real OS processes contend for one vault directory, so acquire/block/release
  happen exactly as they would in deployment;
* competing writers allocate each key's versions strictly from 1, never
  repeating or skipping one, and two processes sealing one key at the same
  time leave exactly one on-disk record per version number;
* a writer that cannot take the lock parks until release, allocates no
  version and leaves no half record and no truncated listing while it waits,
  then lands exactly the bytes of the uncontended run;
* a reloading reader only ever observes complete snapshots, never a
  half-old/half-new mixture;
* ``close()`` is an idempotent no-op when repeated, hands the lock to a
  waiting process immediately with no stale handle, and later operations
  reacquire it with unchanged behaviour;
* ``vault.lock`` carries no key data: empty or arbitrary bytes (and even its
  absence) never affect any read, write or reload;
* a reload that fails validation raises ``ValueError`` and leaves both the
  in-memory snapshot and the disk records frozen, with keys already in hand
  still readable;
* the entry-point vocabulary holds on the native branch: empty id ->
  ``ValueError``, non-genuine-integer version -> ``TypeError``, unknown
  key/version -> ``KeyError``, with no new version produced;
* results are repeatable and independent of execution order, and the
  documented ``unittest`` invocation reports zero/non-zero exit codes with
  no resource-warning tail.

Only a test file is added here; the product code, the public interface and
the three CLI entry points are untouched.  Every case works inside a
temporary directory only and drains every process/thread/handle through the
single fixture teardown.  On non-Windows platforms the whole module skips,
so the documented command stays green everywhere::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue as queue_mod
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    ACTIVATIONS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
)
from tests._fixtures import REPO_ROOT, VaultFixture, cli_env

IS_WINDOWS = sys.platform == "win32"

_windows_only = unittest.skipUnless(
    IS_WINDOWS,
    "genuine msvcrt.locking system lock is exercised on Windows only",
)


# ---------------------------------------------------------------------------
# module-level worker entry points (importable by the Windows spawn start
# method).  They patch nothing: on Windows the vault takes its real branch.
# ---------------------------------------------------------------------------


def _native_seal_payloads(
    root: str,
    key_id: str,
    payloads: list[bytes],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Seal ``payloads`` through the real Windows lock; report versions."""
    try:
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


def _native_lock_holder(
    root: str,
    ready: "multiprocessing.Event",
    release: "multiprocessing.Event",
) -> None:
    """Take the genuine inter-process lock and hold it until told."""
    vault = Vault(root)
    try:
        # The file-lock helper's contract requires the in-process RLock.
        with vault._lock:
            with vault._file_lock():
                ready.set()
                if not release.wait(timeout=30):
                    raise RuntimeError("holder was never released")
    finally:
        vault.close()


def _native_prefix_reader(
    root: str,
    key_id: str,
    payloads: set[bytes],
    total: int,
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Reopen/reload; only complete 1..n prefixes may ever be visible.

    The visible versions are always exactly 1..n with the active version at
    n, every material is one of the sealed payloads, a version's material
    never changes between observations and the visible history never shrinks.
    """
    try:
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
                raise AssertionError(
                    f"visible history shrank {last} -> {len(visible)}"
                )
            last = len(visible)
            vault.close()
        if last != total:
            raise AssertionError(f"reader stalled at {last}, expected {total}")
        results.put(("ok", last))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _native_close_dance_a(
    root: str,
    key_id: str,
    closed: "multiprocessing.Event",
    peer_done: "multiprocessing.Event",
) -> None:
    vault = Vault(root)
    if vault.seal(key_id, b"a-one") != 1:
        raise AssertionError("first seal must be version 1")
    vault.close()
    vault.close()  # releasing twice is not an error
    closed.set()
    if not peer_done.wait(timeout=30):
        raise RuntimeError("peer never finished after close")
    if vault.seal(key_id, b"a-three") != 3:
        raise AssertionError("seal after close must continue the sequence")
    vault.close()


def _native_close_dance_b(
    root: str,
    key_id: str,
    closed: "multiprocessing.Event",
    peer_done: "multiprocessing.Event",
    results: "multiprocessing.Queue",
) -> None:
    try:
        vault = Vault(root)  # opened early; parked without holding the lock
        if not closed.wait(timeout=30):
            raise RuntimeError("peer never closed its lock")
        started = time.monotonic()
        version = vault.seal(key_id, b"b-two")
        elapsed = time.monotonic() - started
        vault.close()  # release before signalling: no stale handle remains
        peer_done.set()
        results.put(("ok", (version, elapsed)))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _native_lock_content_worker(root: str, results: "multiprocessing.Queue") -> None:
    """Open, read, seal, revoke and reload with the lock file pre-filled."""
    try:
        vault = Vault(root)
        try:
            vault.reload()
            if vault.seal("k", b"from-other-process") != 3:
                raise AssertionError("version sequence broken by lock bytes")
            vault.revoke("k", 1)
            vault.reload()
            if not vault.is_revoked("k", 1):
                raise AssertionError("revocation did not survive reload")
        finally:
            vault.close()
        results.put(("ok", None))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


class NativeWindowsLockTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.root = self.fixture.root

    def open_vault(self, root: Path | str | None = None) -> Vault:
        """Open a tracked vault in the case's temporary directory."""
        return self.fixture.open(self.root if root is None else root)

    def disk_manifest(self) -> dict:
        return json.loads((self.root / MANIFEST_NAME).read_bytes().decode("utf-8"))

    def disk_bytes(self, root: Path | None = None) -> dict[str, bytes]:
        """All vault records on disk keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, so it is deliberately excluded from every byte comparison.
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
        # fails while they are running.
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
# real processes contending for the genuine Windows lock
# ---------------------------------------------------------------------------


@_windows_only
class TestNativeContention(NativeWindowsLockTestCase):
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
                target=_native_seal_payloads,
                args=(str(self.root), "k", payloads[writer], results, barrier),
            )
            for writer in range(writers)
        ]
        self.run_processes(processes)
        reported = self.assert_worker_results_ok(results)
        self.assertEqual(len(reported), writers)

        # Every reported (version, payload) pair owns a unique version and
        # together they are exactly 1..total: no duplicates, no gaps.
        allocated: dict[int, bytes] = {}
        for sealed in reported:
            self.assertEqual(len(sealed), each)
            for version, payload in sealed:
                self.assertNotIn(version, allocated)
                allocated[version] = payload
        self.assertEqual(sorted(allocated), list(range(1, total + 1)))

        # The persisted state agrees: a strict sequence from 1, every
        # historical material byte-for-byte the payload that won the slot.
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
        self.assertEqual(
            sorted((self.root / MATERIALS_DIR).rglob("*.bin")),
            sorted(
                (self.root / record["file"])
                for record in disk_entry["versions"]
            ),
        )
        self.assertEqual(len(list((self.root / MATERIALS_DIR).rglob("*.bin"))), total)
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])
        self.assertEqual((self.root / REVOCATIONS_NAME).read_bytes(), b"")

    def test_two_real_processes_seal_one_key_each_version_lands_once(self):
        # The headline Windows behaviour: two genuine processes sealing the
        # same key at the same time leave exactly one record per version.
        payloads_a = [b"alpha-1", b"alpha-2", b"alpha-3"]
        payloads_b = [b"bravo-1", b"bravo-2", b"bravo-3"]
        barrier = multiprocessing.Barrier(2)
        self.fixture.track_barrier(barrier)
        results: multiprocessing.Queue = multiprocessing.Queue()
        processes = [
            multiprocessing.Process(
                target=_native_seal_payloads,
                args=(str(self.root), "shared", payloads, results, barrier),
            )
            for payloads in (payloads_a, payloads_b)
        ]
        self.run_processes(processes)

        records = self.disk_manifest()["keys"]["shared"]["versions"]
        versions = [record["version"] for record in records]
        # Each version number appears exactly once, from 1, none skipped.
        self.assertEqual(versions, list(range(1, 7)))
        self.assertEqual(len(versions), len(set(versions)))
        # One material file per record, no two records pointing at one file.
        files = [record["file"] for record in records]
        self.assertEqual(len(files), len(set(files)))
        self.assertEqual(
            sorted(path.name for path in (self.root / MATERIALS_DIR).rglob("*.bin")),
            [f"{version}.bin" for version in range(1, 7)],
        )
        # The surviving bytes are exactly the union of the two inputs.
        final = self.open_vault()
        self.assertEqual(
            {final.load("shared", v) for v in range(1, 7)},
            set(payloads_a) | set(payloads_b),
        )

    def test_blocked_writer_waits_then_matches_the_uncontended_result(self):
        vault = self.open_vault()
        vault.seal("k", b"m-1")

        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_native_lock_holder, args=(str(self.root), ready, release)
        )
        # A failed assertion releases the gate and drains both workers via
        # the single teardown; the blocked write then proceeds unchanged.
        self.fixture.track_gate(release)
        self.fixture.track_process(holder)
        holder.start()
        self.assertTrue(ready.wait(timeout=10))

        results: multiprocessing.Queue = multiprocessing.Queue()
        sealer = multiprocessing.Process(
            target=_native_seal_payloads,
            args=(str(self.root), "k", [b"m-2"], results),
        )
        self.fixture.track_process(sealer)
        sealer.start()

        # While the holder owns the lock the sealer stays parked: no version
        # allocated, no material, no half record and no truncated listing.
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

        # The contended run lands exactly the bytes of the same seals with
        # no contention at all.
        reference_root = self.fixture.path("reference")
        reference = self.open_vault(reference_root)
        reference.seal("k", b"m-1")
        reference.seal("k", b"m-2")
        reference.close()
        self.assertEqual(self.disk_bytes(), self.disk_bytes(reference_root))

        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2])
        self.assertEqual(final.load("k", 1), b"m-1")
        self.assertEqual(final.load("k", 2), b"m-2")


# ---------------------------------------------------------------------------
# readers only ever see complete snapshots under the real lock
# ---------------------------------------------------------------------------


@_windows_only
class TestNativeReaderCompleteSnapshots(NativeWindowsLockTestCase):
    def test_concurrent_reader_sees_only_complete_prefixes(self):
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
        writers_ = [
            multiprocessing.Process(
                target=_native_seal_payloads,
                args=(str(self.root), "k", payloads[writer], results, barrier),
            )
            for writer in range(writers)
        ]
        reader_results: multiprocessing.Queue = multiprocessing.Queue()
        reader = multiprocessing.Process(
            target=_native_prefix_reader,
            args=(str(self.root), "k", all_payloads, total, reader_results, barrier),
        )
        self.fixture.track_process(reader)
        reader.start()
        self.run_processes(writers_)
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
# close(): immediate handoff and transparent reacquire
# ---------------------------------------------------------------------------


@_windows_only
class TestNativeCloseRelease(NativeWindowsLockTestCase):
    def test_another_process_takes_the_lock_immediately_after_close(self):
        closed = multiprocessing.Event()
        peer_done = multiprocessing.Event()
        results: multiprocessing.Queue = multiprocessing.Queue()
        # B opens first but parks until A closes, so it measures the handoff
        # latency directly and proves no stale handle blocks it.
        b = multiprocessing.Process(
            target=_native_close_dance_b,
            args=(str(self.root), "k", closed, peer_done, results),
        )
        a = multiprocessing.Process(
            target=_native_close_dance_a, args=(str(self.root), "k", closed, peer_done)
        )
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
        # Released immediately: no residual handle keeps the peer waiting.
        self.assertLess(elapsed, 10.0)

        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2, 3])
        self.assertEqual(final.load("k", 1), b"a-one")
        self.assertEqual(final.load("k", 2), b"b-two")
        self.assertEqual(final.load("k", 3), b"a-three")

    def test_repeated_close_is_harmless_and_operations_reacquire_unchanged(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())  # repeated release stays an error-free no-op

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
        self.assertIsNone(vault.close())

        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), [1, 2, 3])
        self.assertTrue(reopened.is_revoked("k", 1))
        self.assertEqual(reopened.active("k"), 2)
        self.assertEqual(reopened.load("k", 2), b"two")
        self.assertEqual(reopened.versions("d"), [1])


# ---------------------------------------------------------------------------
# the lock file is only a mutex device: its bytes never matter
# ---------------------------------------------------------------------------


@_windows_only
class TestNativeLockFileIsOnlyMutex(NativeWindowsLockTestCase):
    def test_empty_arbitrary_or_absent_lock_bytes_never_affect_anything(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.seal("k", b"two")
        lock_path = self.root / LOCK_NAME

        # Empty bytes.
        lock_path.write_bytes(b"")
        vault.reload()
        self.assertEqual(vault.load("k", 2), b"two")

        marker = b"this is not key material\n\xff\x00"
        lock_path.write_bytes(marker)
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.load("k"), b"two")

        # A cold opener tolerates the arbitrary bytes as well.
        cold = self.open_vault()
        self.assertEqual(cold.load("k", 1), b"one")
        cold.close()

        # A different real process opens, writes a full record and validates
        # while the lock file still carries arbitrary content.
        results: multiprocessing.Queue = multiprocessing.Queue()
        worker = multiprocessing.Process(
            target=_native_lock_content_worker, args=(str(self.root), results)
        )
        self.fixture.track_process(worker)
        worker.start()
        worker.join(timeout=30)
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(self.assert_worker_results_ok(results), [None])

        final = self.open_vault()
        final.reload()
        self.assertEqual(final.versions("k"), [1, 2, 3])
        self.assertEqual(final.load("k", 3), b"from-other-process")
        self.assertTrue(final.is_revoked("k", 1))

        # The marker never leaked into any validated record.
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != LOCK_NAME:
                self.assertNotIn(marker, path.read_bytes())

        # Absence is tolerated too, but only with no held handle: return
        # every tracked handle first (on Windows an open lock file cannot be
        # removed at all), delete the device, and let the next operation
        # recreate and reacquire it transparently.
        vault.close()
        final.close()
        lock_path.unlink()
        self.assertFalse(lock_path.exists())
        fresh = self.open_vault()
        self.assertEqual(fresh.seal("k", b"four"), 4)
        self.assertTrue(lock_path.exists())
        fresh.reload()
        self.assertEqual(fresh.load("k", 4), b"four")


# ---------------------------------------------------------------------------
# failed reload: ValueError, snapshot frozen, held keys still readable
# ---------------------------------------------------------------------------


@_windows_only
class TestNativeReloadFailureFreeze(NativeWindowsLockTestCase):
    def _build_healthy(self) -> Vault:
        vault = self.open_vault()
        vault.seal("plain", b"plain-one")
        vault.seal("plain", b"plain-two")
        vault.set_active("plain", 1)  # bound to latest=2
        vault.derive_seal("drv", b"passphrase", b"drv-salt", 1000, 32)
        vault.revoke("drv", 1)
        return vault

    def _frozen_answers(self, vault: Vault) -> dict:
        keys = {}
        for key_id in ("plain", "drv"):
            versions = vault.versions(key_id)
            keys[key_id] = {
                "versions": versions,
                "active": vault.active(key_id),
                "revoked": vault.revoked_versions(key_id),
                "is_revoked": {
                    version: vault.is_revoked(key_id, version) for version in versions
                },
                "materials": {
                    version: vault.load(key_id, version) for version in versions
                },
                "derivations": {
                    version: vault.derivation(key_id, version) for version in versions
                },
            }
        return {"keys": keys, "manifest": vault.manifest()}

    def _restore_disk(self, healthy: dict[str, bytes]) -> None:
        current = self.disk_bytes()
        for rel in current.keys() - healthy.keys():
            (self.root / rel).unlink()
        for rel, data in healthy.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)

    def test_failed_reload_freezes_snapshot_and_disk(self):
        vault = self._build_healthy()

        def corrupt_manifest() -> None:
            (self.root / MANIFEST_NAME).write_bytes(b"{not valid json")

        def corrupt_derivation() -> None:
            from keyvault_ledger.vault import _dump_manifest

            path = self.root / MANIFEST_NAME
            manifest = json.loads(path.read_bytes().decode("utf-8"))
            record = manifest["keys"]["drv"]["versions"][0]
            record["derivation"]["salt"] = "!!!not-base64!!!"
            path.write_bytes(_dump_manifest(manifest))

        healthy_disk = self.disk_bytes()
        expected = self._frozen_answers(vault)
        for corrupt in (corrupt_manifest, corrupt_derivation):
            with self.subTest(corrupt=corrupt.__name__):
                corrupt()
                failing_disk = self.disk_bytes()

                message = None
                # Repeated identical input -> identical ValueError, frozen
                # answers every time, held keys still readable.
                for _ in range(3):
                    with self.assertRaises(ValueError) as caught:
                        vault.reload()
                    if message is None:
                        message = str(caught.exception)
                    else:
                        self.assertEqual(str(caught.exception), message)
                    self.assertEqual(self._frozen_answers(vault), expected)

                # A cold opener rejects the same state identically.
                with self.assertRaises(ValueError) as caught:
                    Vault(self.root)
                self.assertEqual(str(caught.exception), message)

                # The failed attempts changed no disk byte; keys in hand
                # remain readable.
                self.assertEqual(self.disk_bytes(), failing_disk)
                self.assertEqual(
                    vault.load("plain", 1),
                    expected["keys"]["plain"]["materials"][1],
                )
                self.assertTrue(vault.is_revoked("drv", 1))

                # Undo the corruption: one reload restores full
                # snapshot/disk correspondence, deterministically.
                self._restore_disk(healthy_disk)
                vault.reload()
                self.assertEqual(self._frozen_answers(vault), expected)
                self.assertEqual(self.disk_bytes(), healthy_disk)


# ---------------------------------------------------------------------------
# entry-point error vocabulary on the native Windows branch
# ---------------------------------------------------------------------------


@_windows_only
class TestNativeEntryContract(NativeWindowsLockTestCase):
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

        # Non-genuine-integer version -> TypeError (bools/floats do not count).
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
        # A float equal to an existing version still cannot read it, while
        # the None sentinel resolves to the active version.
        with self.assertRaises(TypeError):
            vault.load("k", 1.0)
        self.assertEqual(vault.load("k", None), b"m")
        self.assertEqual(vault.derivation("k", None), {})

        # Unknown key or a genuine-int version that does not exist -> KeyError.
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

        # No rejection produced a new version or altered a disk record.
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.versions("d"), [1])
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), manifest_before)
        self.assertEqual((self.root / REVOCATIONS_NAME).read_bytes(), journal_before)
        self.assertFalse((self.root / ACTIVATIONS_NAME).exists())
        self.assertEqual(vault.load("k"), b"m")


# ---------------------------------------------------------------------------
# repeatability: the same input lands the same bytes every time
# ---------------------------------------------------------------------------


@_windows_only
class TestNativeDeterminism(NativeWindowsLockTestCase):
    def test_the_same_input_sequence_is_byte_identical_across_runs(self):
        payloads = [b"\x00binary\xff", "文本".encode("utf-8"), b"plain-three"]

        def build(root: Path) -> None:
            vault = self.open_vault(root)
            for payload in payloads:
                vault.seal("k", payload)
            vault.derive_seal("d", b"passphrase", b"fixed-salt", 500, 24)
            vault.revoke("k", 1)
            vault.set_active("k", 2)
            vault.reload()
            vault.close()

        first_root = self.fixture.path("run-one")
        second_root = self.fixture.path("run-two")
        build(first_root)
        build(second_root)

        # Same input in two independent temporary trees: identical durable
        # records (the lock file, a pure device, is excluded).
        self.assertEqual(self.disk_bytes(first_root), self.disk_bytes(second_root))

        # Repeating every read against a fresh opener returns the same
        # answers both times.
        summaries = []
        for root in (first_root, second_root):
            vault = Vault(root)
            summaries.append(
                {
                    "versions": vault.versions("k"),
                    "active": vault.active("k"),
                    "revoked": vault.revoked_versions("k"),
                    "materials": {
                        version: vault.load("k", version)
                        for version in vault.versions("k")
                    },
                    "derived": vault.load("d"),
                    "derivation": vault.derivation("d"),
                }
            )
            vault.close()
        self.assertEqual(summaries[0], summaries[1])


# ---------------------------------------------------------------------------
# execution-order independence: forward, reverse and picked subsets agree
# ---------------------------------------------------------------------------


@_windows_only
class TestNativeOrderIndependence(NativeWindowsLockTestCase):
    @staticmethod
    def _lightweight_classes() -> list[type[unittest.TestCase]]:
        # The self-contained, fast classes of this module: each owns its own
        # temporary directory and drains its own workers.  The heavy
        # many-process contention classes and this driver itself are excluded
        # so the meta-run never re-enters recursively.
        return [
            TestNativeEntryContract,
            TestNativeReloadFailureFreeze,
            TestNativeCloseRelease,
            TestNativeLockFileIsOnlyMutex,
            TestNativeDeterminism,
        ]

    @staticmethod
    def _run(suite: unittest.TestSuite) -> unittest.TestResult:
        result = unittest.TestResult()
        suite.run(result)
        return result

    def test_forward_reverse_and_picked_runs_all_pass_identically(self):
        # Key the case set by (class, method name) and build a FRESH test
        # instance for every ordering: running a TestSuite consumes its leaf
        # slots (unittest replaces each ran case with None), so an instance
        # must never be shared between two suite runs.
        names: list[tuple[type[unittest.TestCase], str]] = []
        for cls in self._lightweight_classes():
            for name in unittest.TestLoader().getTestCaseNames(cls):
                names.append((cls, name))
        self.assertTrue(names)

        def suite_for(order: list[int]) -> unittest.TestSuite:
            return unittest.TestSuite(
                [names[index][0](names[index][1]) for index in order]
            )

        forward_order = list(range(len(names)))
        reverse_order = list(reversed(forward_order))
        pick_order = [2, 0, len(names) - 1]

        forward_result = self._run(suite_for(forward_order))
        self.assertTrue(forward_result.wasSuccessful())
        reverse_result = self._run(suite_for(reverse_order))
        self.assertTrue(reverse_result.wasSuccessful())
        picked_result = self._run(suite_for(pick_order))
        self.assertTrue(picked_result.wasSuccessful())

        # Forward and reverse ran the same case set; the picked subset ran
        # exactly its three cases.  Nothing failed in any ordering, so the
        # conclusion does not depend on execution order.
        self.assertEqual(reverse_result.testsRun, len(names))
        self.assertEqual(forward_result.testsRun, len(names))
        self.assertEqual(picked_result.testsRun, 3)


# ---------------------------------------------------------------------------
# documented runner contract: exit code zero when green, non-zero on failure,
# clean of resource/warning tail text
# ---------------------------------------------------------------------------


_GREEN_PROBE_TEMPLATE = """
import unittest

from keyvault_ledger import Vault


class NativeLockProbe(unittest.TestCase):
    def test_real_lock_round_trip(self):
        root = {root!r}
        vault = Vault(root)
        self.assertEqual(
            [vault.seal("k", b"m-%d" % i) for i in range(1, 4)], [1, 2, 3]
        )
        vault.reload()
        self.assertEqual(vault.load("k", 3), b"m-3")
        vault.close()
        reopened = Vault(root)
        self.assertEqual(reopened.versions("k"), [1, 2, 3])
        reopened.close()


if __name__ == "__main__":
    unittest.main()
"""

_FAILING_PROBE = """
import unittest


class FailingProbe(unittest.TestCase):
    def test_fails(self):
        self.assertTrue(False, "native-runner-failure-marker")


if __name__ == "__main__":
    unittest.main()
"""

_WARNING_RE = re.compile(
    r"resourcewarning|unclosed|exception ignored in|unraisablehook|"
    r"still running",
    re.IGNORECASE,
)


@_windows_only
class TestNativeRunnerContract(NativeWindowsLockTestCase):
    def _discover(self, directory: Path) -> subprocess.CompletedProcess:
        env = dict(cli_env())
        env["PYTHONPATH"] = (
            str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        )
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(directory),
                "-t",
                str(directory),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

    def test_documented_runner_reports_zero_green_and_nonzero_failure(self):
        probe_root = self.fixture.path("probe-vault")
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)

            green_path = directory / "test_native_green_probe.py"
            green_path.write_text(_GREEN_PROBE_TEMPLATE.format(root=str(probe_root)))
            green = self._discover(directory)
            self.assertEqual(green.returncode, 0, green.stderr)
            self.assertTrue(green.stderr.rstrip().endswith("OK"))
            self.assertFalse(_WARNING_RE.search(green.stderr), green.stderr)
            self.assertEqual(green.stdout, "")

            green_path.unlink()
            (directory / "test_native_failing_probe.py").write_text(_FAILING_PROBE)
            failing = self._discover(directory)
            self.assertNotEqual(failing.returncode, 0)
            self.assertIn("FAILED", failing.stderr)
            self.assertIn("native-runner-failure-marker", failing.stderr)
            self.assertFalse(_WARNING_RE.search(failing.stderr), failing.stderr)


if __name__ == "__main__":
    unittest.main()
