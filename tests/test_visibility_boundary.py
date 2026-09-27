"""Regression tests pinning the reload/writer visibility boundary.

The baseline vault capabilities — the inter-process lock, the whole-vault
reload and the exception vocabulary — are already implemented and are
intentionally not touched here.  These cases only freeze observable
behaviour, matching the public documentation:

* real subprocesses and threads interleave ``seal`` / ``derive_seal`` /
  ``revoke`` / ``set_active`` / ``reload`` on one temporary vault directory;
  afterwards every key's version numbers are still a strict 1..n sequence —
  no duplicates, no gaps — and historical material is never rewritten;

* a process holds the lock for one whole record: waiters observe no new
  record at all until the holder's record is complete, and then start their
  own record only once the lock is theirs;

* a reload interleaved with writers only ever observes the complete old
  records or the complete newly persisted ones — a half-old/half-new
  mixture never appears, and a version's material never changes between
  observations;

* a reload that fails validation raises ``ValueError`` while both sides
  stay frozen: the in-memory snapshot answers every query word-for-word as
  before and the on-disk records neither grow nor shrink, byte for byte;
  removing the corruption makes the next reload succeed with the snapshot
  matching the disk records one to one;

* ``vault.lock`` is only a mutual-exclusion device: its contents never
  participate in validation and never leak into any record;

* ``close()`` is idempotent, and later operations reacquire the same lock
  with no observable change in behaviour;

* the entry-point error contract (empty key id -> ``ValueError``;
  non-bytes material/passphrase/salt -> ``TypeError``; non-genuine-integer
  version or derivation parameter -> ``TypeError``, bools and floats
  included; unknown key or version -> ``KeyError``; empty salt or
  non-positive derivation parameter -> ``ValueError``; duplicate
  revocation, repointing at the active or a revoked version ->
  ``ValueError``) raises exactly the documented exception type and a failed
  call leaves not even half a record behind.

Every case works inside a temporary directory, uses the standard library
only, and is independent of execution order.  Running the whole suite must
stay green::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import queue as queue_mod
import threading
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
from tests._fixtures import VaultFixture


# ---------------------------------------------------------------------------
# module-level worker entry points (picklable for every start method)
# ---------------------------------------------------------------------------


def _seal_worker(
    root: str,
    key_id: str,
    payloads: list[bytes],
    barrier: "multiprocessing.managers.Barrier | None" = None,
    gate: "multiprocessing.managers.Event | None" = None,
) -> None:
    if barrier is not None:
        barrier.wait()
    if gate is not None:
        gate.wait(timeout=30)
    vault = Vault(root)
    try:
        for payload in payloads:
            vault.seal(key_id, payload)
    finally:
        vault.close()


def _revoke_worker(
    root: str,
    key_id: str,
    versions: list[int],
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    if barrier is not None:
        barrier.wait()
    vault = Vault(root)
    try:
        for version in versions:
            vault.revoke(key_id, version)
    finally:
        vault.close()


def _boundary_reader(
    root: str,
    key_id: str,
    expected: set[bytes],
    total: int,
    ready: "multiprocessing.Event",
    results: "multiprocessing.Queue",
) -> None:
    """Reload in a loop while a writer appends; every observation must be a
    complete prefix of the final sequence.

    The first complete observation releases the writer through ``ready``, so
    the reader provably saw the complete old records before any new one
    landed.  Afterwards the visible version list must always be exactly
    1..n with the active version at n, the visible count must never shrink,
    and a version's material must never change between observations.
    """
    vault = Vault(root)
    pinned: dict[int, bytes] = {}
    observations: list[int] = []
    try:
        last = 0
        deadline = time.monotonic() + 60
        while last < total and time.monotonic() < deadline:
            vault.reload()
            versions = vault.versions(key_id)
            n = len(versions)
            if versions != list(range(1, n + 1)):
                raise AssertionError(f"torn version listing: {versions}")
            if n < last:
                raise AssertionError(f"visible history shrank {last} -> {n}")
            if n and vault.active(key_id) != n:
                raise AssertionError(
                    f"active {vault.active(key_id)} != latest {n}"
                )
            entry = vault.manifest()["keys"].get(key_id)
            listed = [] if entry is None else [r["version"] for r in entry["versions"]]
            if listed != versions:
                raise AssertionError("manifest listing does not match snapshot")
            for version in versions:
                material = vault.load(key_id, version)
                if material not in expected:
                    raise AssertionError(f"unexpected material at version {version}")
                if version in pinned and pinned[version] != material:
                    raise AssertionError(
                        f"historical material rewritten at version {version}"
                    )
                pinned[version] = material
            observations.append(n)
            last = n
            if not ready.is_set():
                # The complete old records were observed before any writer
                # could start; let the writer go now.
                ready.set()
        if last != total:
            raise AssertionError(f"reader stalled at {last} of {total}")
        results.put(("ok", observations))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))
    finally:
        vault.close()


def _record_holder(
    root: str,
    key_id: str,
    payload: bytes,
    ready: "multiprocessing.Event",
    release: "multiprocessing.Event",
    sealed: "multiprocessing.Event",
) -> None:
    """Hold the real production lock, then write one complete record.

    Uses the vault's own lock helpers (the in-process RLock plus the file
    lock) so the waiters below block on the genuine inter-process lock while
    it is held.  Once ``release`` opens, the holder lets go and writes its
    one whole record through the ordinary seal path; only after that record
    is fully done does it open ``sealed`` for the gated waiters.
    """
    vault = Vault(root)
    with vault._lock:
        with vault._file_lock():
            ready.set()
            if not release.wait(timeout=30):
                raise RuntimeError("holder was never released")
    vault.seal(key_id, payload)
    sealed.set()
    vault.close()


def _reload_worker(
    root: str, rounds: int, gate: "multiprocessing.Event"
) -> None:
    gate.wait(timeout=30)
    for _ in range(rounds):
        vault = Vault(root)
        vault.reload()
        vault.close()


def _peer_seal_worker(
    root: str, key_id: str, payload: bytes, results: "multiprocessing.Queue"
) -> None:
    vault = Vault(root)
    try:
        results.put(("ok", vault.seal(key_id, payload)))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))
    finally:
        vault.close()


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


class VisibilityBoundaryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.root = self.fixture.root

    def open_vault(self) -> Vault:
        """Open a vault; its handle is returned by the single teardown."""
        return self.fixture.open()

    def disk_manifest(self) -> dict:
        return json.loads((self.root / MANIFEST_NAME).read_bytes().decode("utf-8"))

    def journal_records(self, name: str) -> list[dict]:
        path = self.root / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text("utf-8").splitlines()]

    def all_disk_bytes(self) -> dict[Path, bytes]:
        """Every persisted byte, excluding the mutex-only lock file."""
        return {
            path: path.read_bytes()
            for path in self.root.rglob("*")
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

    def drain_queue(self, process_queue: "multiprocessing.Queue") -> list:
        items = []
        while True:
            try:
                items.append(process_queue.get_nowait())
            except queue_mod.Empty:
                return items

    def assert_worker_ok(self, process_queue: "multiprocessing.Queue") -> list:
        """Drain a ``(status, payload)`` worker queue and fail on errors."""
        payloads = []
        for status, payload in self.drain_queue(process_queue):
            self.assertEqual(status, "ok", payload)
            payloads.append(payload)
        return payloads


# ---------------------------------------------------------------------------
# interleaved processes + threads: strict sequences, untouched history
# ---------------------------------------------------------------------------


class TestInterleavedFinalSequence(VisibilityBoundaryTestCase):
    def test_mixed_workers_leave_strict_sequences_and_untouched_history(self):
        vault = self.open_vault()
        # Baseline history of the revocation/repoint key, sealed up front so
        # the revokers race over disjoint, fully deterministic partitions.
        baseline_materials = {
            version: f"base-{version}".encode("utf-8") for version in range(1, 5)
        }
        for version, material in baseline_materials.items():
            self.assertEqual(vault.seal("r", material), version)

        # One shared key sealed concurrently by real subprocesses and
        # in-process threads; every payload is distinct so the final set
        # must match exactly.
        proc_payloads = [
            f"proc-{p}-{i}".encode("utf-8") for p in range(3) for i in range(4)
        ]
        thread_payloads = [
            f"thread-{t}-{i}".encode("utf-8") for t in range(2) for i in range(3)
        ]
        expected_k = set(proc_payloads) | set(thread_payloads)
        total_k = len(expected_k)

        barrier = multiprocessing.Barrier(5)  # 3 sealers + 2 revokers
        self.fixture.track_barrier(barrier)
        processes = [
            multiprocessing.Process(
                target=_seal_worker,
                args=(str(self.root), "k", proc_payloads[p * 4 : (p + 1) * 4], barrier),
            )
            for p in range(3)
        ]
        processes += [
            multiprocessing.Process(
                target=_revoke_worker, args=(str(self.root), "r", versions, barrier)
            )
            for versions in ([1, 2], [3])
        ]

        thread_errors: list[BaseException] = []

        def seal_slice(payloads: list[bytes]) -> None:
            local = Vault(self.root)
            try:
                for payload in payloads:
                    local.seal("k", payload)
            except BaseException as exc:  # pragma: no cover - surfaced below
                thread_errors.append(exc)
            finally:
                local.close()

        def repoint(targets: tuple[int, ...]) -> None:
            local = Vault(self.root)
            try:
                for target in targets:
                    try:
                        local.set_active("r", target)
                    except ValueError:
                        pass  # revoked by a peer or already active: a lost race
            except BaseException as exc:  # pragma: no cover - surfaced below
                thread_errors.append(exc)
            finally:
                local.close()

        seals_done = threading.Event()
        pinned: dict[int, bytes] = {}

        def observe() -> None:
            local = Vault(self.root)
            try:
                while not seals_done.is_set():
                    local.reload()
                    versions = local.versions("k")
                    n = len(versions)
                    if versions != list(range(1, n + 1)):
                        thread_errors.append(
                            AssertionError(f"torn version listing: {versions}")
                        )
                        return
                    if n and local.active("k") != n:
                        thread_errors.append(
                            AssertionError("active pointer not at latest version")
                        )
                        return
                    entry = local.manifest()["keys"].get("k")
                    listed = (
                        [] if entry is None else [r["version"] for r in entry["versions"]]
                    )
                    if listed != versions:
                        thread_errors.append(AssertionError("manifest listing torn"))
                        return
                    for version in versions:
                        material = local.load("k", version)
                        if material not in expected_k:
                            thread_errors.append(
                                AssertionError(f"unexpected material at {version}")
                            )
                            return
                        if version in pinned and pinned[version] != material:
                            thread_errors.append(
                                AssertionError(f"history rewritten at {version}")
                            )
                            return
                        pinned[version] = material
            finally:
                local.close()

        threads = [
            threading.Thread(target=seal_slice, args=(thread_payloads[0:3],)),
            threading.Thread(target=seal_slice, args=(thread_payloads[3:6],)),
            threading.Thread(target=repoint, args=((2, 4, 1),)),
            threading.Thread(target=repoint, args=((3, 1, 4),)),
            threading.Thread(target=observe),
        ]
        # The observer only exits once ``seals_done`` is set; treat it as a
        # gate so a failed assertion still lets every thread drain before
        # handles are returned.
        self.fixture.track_gate(seals_done)
        self.fixture.track_thread(*threads)
        for thread in threads:
            thread.start()
        self.run_processes(processes)
        for thread in threads[:-1]:
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive())
        seals_done.set()
        threads[-1].join(timeout=30)
        self.assertFalse(threads[-1].is_alive())
        self.assertEqual(thread_errors, [])

        # The shared key: one strict 1..n sequence across processes and
        # threads, no duplicates, no gaps, every material byte-exact.
        vault.reload()
        self.assertEqual(vault.versions("k"), list(range(1, total_k + 1)))
        self.assertEqual(
            {vault.load("k", v) for v in range(1, total_k + 1)}, expected_k
        )
        self.assertEqual(vault.active("k"), total_k)

        # The revocation key: disjoint partitions landed exactly once each;
        # the baseline materials are byte-for-byte untouched.
        self.assertEqual(vault.versions("r"), [1, 2, 3, 4])
        self.assertEqual(vault.revoked_versions("r"), [1, 2, 3])
        for version, material in baseline_materials.items():
            self.assertEqual(vault.load("r", version), material)
        self.assertIn(vault.active("r"), [1, 2, 3, 4])

        # Both journals hold whole, unique, well-formed records only.
        revocations = self.journal_records(REVOCATIONS_NAME)
        pairs = sorted((r["key_id"], r["version"]) for r in revocations)
        self.assertEqual(pairs, [("r", 1), ("r", 2), ("r", 3)])
        for record in self.journal_records(ACTIVATIONS_NAME):
            self.assertEqual(set(record), {"key_id", "version", "latest"})
            self.assertEqual(record["key_id"], "r")
            self.assertEqual(record["latest"], 4)
            self.assertTrue(1 <= record["version"] <= 4)

        # A fresh opener validates the whole keyring: a torn append or a
        # half record anywhere would raise ValueError here.
        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, total_k + 1)))
        self.assertEqual(final.revoked_versions("r"), [1, 2, 3])
        self.assertEqual(
            {final.load("k", v) for v in range(1, total_k + 1)}, expected_k
        )


# ---------------------------------------------------------------------------
# one whole record per lock tenure
# ---------------------------------------------------------------------------


class TestWholeRecordLockTenure(VisibilityBoundaryTestCase):
    def test_waiters_see_nothing_until_the_holder_completes_its_record(self):
        vault = self.open_vault()
        for version in range(1, 4):
            vault.seal("k", f"m-{version}".encode("utf-8"))

        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        sealed = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_record_holder,
            args=(str(self.root), "k", b"holder-record", ready, release, sealed),
        )
        # If an assertion fails while the holder parks the waiters, the
        # single teardown sets these gates and drains every process before
        # the parent handle is returned.
        self.fixture.track_gate(release, sealed)
        self.fixture.track_process(holder)
        holder.start()
        self.assertTrue(ready.wait(timeout=10))

        # One waiter per operation kind, each gated on the holder's record
        # being fully done: a sealer and a pure reloader.
        sealer = multiprocessing.Process(
            target=_seal_worker,
            args=(str(self.root), "k", [b"waiter-record"], None, sealed),
        )
        reloader = multiprocessing.Process(
            target=_reload_worker, args=(str(self.root), 1, sealed)
        )
        self.fixture.track_process(sealer, reloader)
        sealer.start()
        reloader.start()

        # While the holder keeps the lock, nothing new is observable: no
        # manifest replacement, no orphan material, no temporary file.
        time.sleep(0.5)
        self.assertTrue(sealer.is_alive(), "waiter ran while the lock was held")
        self.assertTrue(reloader.is_alive(), "reload ran while the lock was held")
        self.assertEqual(
            [r["version"] for r in self.disk_manifest()["keys"]["k"]["versions"]],
            [1, 2, 3],
        )
        self.assertEqual(
            sorted(p.name for p in (self.root / MATERIALS_DIR).rglob("*.bin")),
            ["1.bin", "2.bin", "3.bin"],
        )
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])

        release.set()
        for waiter in (holder, sealer, reloader):
            waiter.join(timeout=30)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(waiter.exitcode, 0)

        # The holder's complete record landed first (the waiters were gated
        # on it), then exactly one complete waiter record.
        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2, 3, 4, 5])
        self.assertEqual(final.load("k", 4), b"holder-record")
        self.assertEqual(final.load("k", 5), b"waiter-record")
        for version in range(1, 4):
            self.assertEqual(final.load("k", version), f"m-{version}".encode("utf-8"))


# ---------------------------------------------------------------------------
# reload sees complete old records or complete new ones, never a mixture
# ---------------------------------------------------------------------------


class TestReloadVisibilityBoundary(VisibilityBoundaryTestCase):
    def test_reload_only_observes_complete_old_or_complete_new_records(self):
        vault = self.open_vault()
        vault.seal("k", b"old-1")

        new_payloads = [f"new-{n}".encode("utf-8") for n in range(2, 9)]
        expected = {b"old-1"} | set(new_payloads)

        ready = multiprocessing.Event()
        results: multiprocessing.Queue = multiprocessing.Queue()
        reader = multiprocessing.Process(
            target=_boundary_reader,
            args=(str(self.root), "k", expected, 8, ready, results),
        )
        # Both processes are drained by the single teardown if an assertion
        # fails while they are still running; the gate lets the sealer
        # proceed even then.
        self.fixture.track_gate(ready)
        self.fixture.track_process(reader)
        reader.start()
        # The sealer does not write a single new record until the reader has
        # observed the complete old one-version list.
        sealer = multiprocessing.Process(
            target=_seal_worker,
            args=(str(self.root), "k", new_payloads, None, ready),
        )
        self.fixture.track_process(sealer)
        sealer.start()
        sealer.join(timeout=60)
        self.assertEqual(sealer.exitcode, 0)
        reader.join(timeout=60)
        self.assertEqual(reader.exitcode, 0)
        observations = self.assert_worker_ok(results)[0]

        # It started at the complete old list, finished at the complete new
        # list, and every state in between was a prefix 1..n that never
        # shrank.
        self.assertEqual(observations[0], 1)
        self.assertEqual(observations[-1], 8)
        self.assertEqual(observations, sorted(observations))
        self.assertTrue(all(1 <= n <= 8 for n in observations))

        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, 9)))
        self.assertEqual({final.load("k", v) for v in range(1, 9)}, expected)


# ---------------------------------------------------------------------------
# failed reload: snapshot and disk double invariance, then recovery
# ---------------------------------------------------------------------------


class TestFailedReloadDoubleInvariance(VisibilityBoundaryTestCase):
    def _build_healthy_vault(self) -> tuple[Vault, dict]:
        vault = self.open_vault()
        materials = {
            1: b"m-1",
            2: b"m-2",
            3: b"m-3",
            4: hashlib.pbkdf2_hmac("sha256", b"pw", b"boundary-salt", 500, dklen=24),
        }
        vault.seal("k", materials[1])
        vault.seal("k", materials[2])
        vault.seal("k", materials[3])
        vault.derive_seal("k", b"pw", b"boundary-salt", 500, 24)
        vault.revoke("k", 1)
        vault.set_active("k", 2)
        facts = {
            "versions": [1, 2, 3, 4],
            "active": 2,
            "revoked": [1],
            "materials": materials,
            "derivation": {"salt": b"boundary-salt", "iterations": 500, "length": 24},
        }
        return vault, facts

    def test_failure_freezes_both_sides_and_recovery_restores_one_to_one(self):
        vault, facts = self._build_healthy_vault()
        healthy_manifest = (self.root / MANIFEST_NAME).read_bytes()

        def answers():
            return (
                vault.versions("k"),
                vault.active("k"),
                tuple(vault.load("k", v) for v in facts["versions"]),
                vault.revoked_versions("k"),
                vault.revoked_versions("never-sealed"),
                vault.is_revoked("k", 1),
                vault.is_revoked("k", 2),
                vault.derivation("k", 4),
                vault.derivation("k", 1),
                vault.manifest(),
            )

        before = answers()
        healthy_disk = self.all_disk_bytes()

        # Corrupt the manifest out of band, underneath the open handle.
        corrupt = b"{not valid json"
        (self.root / MANIFEST_NAME).write_bytes(corrupt)

        # Every failed reload raises ValueError and changes nothing: the
        # snapshot answers word-for-word as before, keys already in hand
        # stay readable, and the disk neither grows nor shrinks a byte.
        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
            self.assertEqual(answers(), before)
            self.assertEqual(self.all_disk_bytes(), healthy_disk | {
                self.root / MANIFEST_NAME: corrupt
            })

        # A cold open on the corrupt directory fails the same way and is
        # equally read-only.
        with self.assertRaises(ValueError):
            Vault(self.root)
        self.assertEqual(
            self.all_disk_bytes(), healthy_disk | {self.root / MANIFEST_NAME: corrupt}
        )

        # Undo the corruption: the same handle reloads successfully and the
        # snapshot corresponds to the disk records one to one.
        (self.root / MANIFEST_NAME).write_bytes(healthy_manifest)
        vault.reload()
        self.assertEqual(answers(), before)
        self.assertEqual(self.all_disk_bytes(), healthy_disk)

        disk_manifest = self.disk_manifest()
        self.assertEqual(vault.manifest(), disk_manifest)
        entry = disk_manifest["keys"]["k"]
        self.assertEqual(
            vault.versions("k"), [r["version"] for r in entry["versions"]]
        )
        for record in entry["versions"]:
            self.assertEqual(
                vault.load("k", record["version"]),
                (self.root / record["file"]).read_bytes(),
            )
        self.assertEqual(
            vault.revoked_versions("k"),
            sorted(r["version"] for r in self.journal_records(REVOCATIONS_NAME)),
        )
        # The activation journal's last bound record decides the pointer.
        activations = self.journal_records(ACTIVATIONS_NAME)
        self.assertEqual(len(activations), 1)
        self.assertEqual(vault.active("k"), activations[0]["version"])

        # A fresh opener sees the exact same one-to-one correspondence.
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), facts["versions"])
        self.assertEqual(reopened.active("k"), facts["active"])
        self.assertEqual(reopened.revoked_versions("k"), facts["revoked"])
        for version, material in facts["materials"].items():
            self.assertEqual(reopened.load("k", version), material)


# ---------------------------------------------------------------------------
# the lock file is only a mutual-exclusion device
# ---------------------------------------------------------------------------


class TestLockFileIsMutexOnly(VisibilityBoundaryTestCase):
    def test_lock_contents_never_matter_to_validation_or_records(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.seal("k", b"two")

        lock_path = self.root / LOCK_NAME
        # Emptying the lock file changes nothing observable.
        lock_path.write_bytes(b"")
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.load("k"), b"two")

        # Filling it with arbitrary bytes changes nothing either: the lock
        # carries no key data and plays no part in validation.
        marker = b"lock-only-payload-\xff"
        lock_path.write_bytes(marker)
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.load("k", 1), b"one")
        self.assertEqual(vault.load("k", 2), b"two")

        # The lock file's bytes never leak into any record.
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != LOCK_NAME:
                self.assertNotIn(marker, path.read_bytes())


# ---------------------------------------------------------------------------
# close(): idempotent release, transparent reacquire
# ---------------------------------------------------------------------------


class TestCloseAndReacquire(VisibilityBoundaryTestCase):
    def test_repeated_close_is_a_noop_and_later_operations_are_unchanged(self):
        vault = self.open_vault()
        self.assertEqual(vault.seal("k", b"one"), 1)
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())  # repeated release is not an error

        # With no handle held, another process takes the very same lock
        # immediately and continues the shared sequence.
        results: multiprocessing.Queue = multiprocessing.Queue()
        peer = multiprocessing.Process(
            target=_peer_seal_worker, args=(str(self.root), "k", b"two", results)
        )
        self.fixture.track_process(peer)
        peer.start()
        peer.join(timeout=30)
        self.assertEqual(peer.exitcode, 0)
        self.assertEqual(self.assert_worker_ok(results), [2])

        # Every operation kind reopens the lock and behaves identically.
        self.assertEqual(vault.seal("k", b"three"), 3)
        self.assertEqual(vault.derive_seal("d", b"pw", b"salt", 100, 16), 1)
        vault.revoke("k", 1)
        vault.set_active("k", 2)
        vault.reload()
        self.assertEqual(vault.load("k"), b"two")
        self.assertEqual(vault.active("k"), 2)
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())  # still an error-free no-op

        # The reacquired-lock writes landed durably and survive a new open.
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), [1, 2, 3])
        self.assertEqual(reopened.active("k"), 2)
        self.assertTrue(reopened.is_revoked("k", 1))
        self.assertEqual(reopened.load("k", 2), b"two")
        self.assertEqual(reopened.versions("d"), [1])


# ---------------------------------------------------------------------------
# entry-point error contract: exact types, no half records
# ---------------------------------------------------------------------------


class TestEntryPointErrorContract(VisibilityBoundaryTestCase):
    def test_rejected_calls_raise_exact_types_and_leave_no_trace(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        vault.revoke("k", 1)
        vault.set_active("k", 2)
        disk_before = self.all_disk_bytes()

        # Empty key id -> ValueError at every entry that takes a key id.
        for call in (
            lambda: vault.seal("", b"m"),
            lambda: vault.derive_seal("", b"pw", b"salt", 1, 1),
            lambda: vault.load(""),
            lambda: vault.revoke("", 1),
            lambda: vault.set_active("", 1),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
            lambda: vault.derivation(""),
        ):
            with self.assertRaises(ValueError, msg=repr(call)):
                call()

        # Material, passphrase or salt of the wrong type -> TypeError.
        for bad in ("text", 1, 1.5, None, [b"x"], object()):
            with self.assertRaises(TypeError, msg=f"material={bad!r}"):
                vault.seal("k", bad)
        for bad in ("text", 1, None, bytearray(b"x"), memoryview(b"x")):
            with self.assertRaises(TypeError, msg=f"password={bad!r}"):
                vault.derive_seal("k", bad, b"salt", 1, 1)
            with self.assertRaises(TypeError, msg=f"salt={bad!r}"):
                vault.derive_seal("k", b"pw", bad, 1, 1)

        # Version or derivation parameter that is not a genuine integer ->
        # TypeError; bools and floats do not count, not even 1.0 or True.
        # (None is the active-version sentinel for load/derivation, not a
        # rejected value; the other versioned entries reject it too.)
        for bad in (1.0, 2.5, True, False, "1", (1,)):
            with self.assertRaises(TypeError, msg=f"version={bad!r}"):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=f"version={bad!r}"):
                vault.derivation("k", bad)
        for bad in (1.0, 2.5, True, False, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=f"version={bad!r}"):
                vault.revoke("k", bad)
            with self.assertRaises(TypeError, msg=f"version={bad!r}"):
                vault.set_active("k", bad)
            with self.assertRaises(TypeError, msg=f"version={bad!r}"):
                vault.is_revoked("k", bad)
            with self.assertRaises(TypeError, msg=f"iterations={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(TypeError, msg=f"length={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)

        # Unknown key or nonexistent version -> KeyError.
        for call in (
            lambda: vault.load("ghost"),
            lambda: vault.active("ghost"),
            lambda: vault.revoke("ghost", 1),
            lambda: vault.set_active("ghost", 1),
            lambda: vault.is_revoked("ghost", 1),
            lambda: vault.derivation("ghost"),
        ):
            with self.assertRaises(KeyError, msg=repr(call)):
                call()
        for bad_version in (0, 4, 99, -1):
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

        # Empty salt or non-positive derivation parameter -> ValueError.
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 1)
        for bad in (0, -1, -1000):
            with self.assertRaises(ValueError, msg=f"iterations={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(ValueError, msg=f"length={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)

        # Duplicate revocation, repointing at the active version and
        # repointing at a revoked version -> ValueError.
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)
        with self.assertRaises(ValueError):
            vault.set_active("k", 1)

        # Not one failed call left half a record behind: every persisted
        # byte is exactly what it was before the rejections.
        self.assertEqual(self.all_disk_bytes(), disk_before)
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        self.assertEqual(vault.versions("d"), [1])
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.revoked_versions("k"), [1])


if __name__ == "__main__":
    unittest.main()
