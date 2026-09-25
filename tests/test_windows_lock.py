"""Multi-process platform tests for the Windows (``msvcrt.locking``) lock branch.

``Vault._file_lock`` has two platform branches: ``fcntl.flock`` where
available and ``msvcrt.locking`` on Windows.  The pre-existing fallback
cases (``TestWindowsLockFallback`` in ``test_vault.py``) drive that branch
with a controllable in-process fake, which can pin the retry control flow
but cannot say anything about real inter-process mutual exclusion.  These
tests close that gap: they force the Windows branch and run it against a
*real* operating-system byte-range lock shared between genuine processes,
so the branch's own acquire, wait, retry and release path is what
serialises the writers:

* several writer processes contending for one vault allocate version
  numbers from 1 upwards, strictly increasing with no duplicates and no
  gaps, and never overwrite historical material;

* a writer that cannot take the lock waits until the holder releases it:
  while it waits no half record and no truncated listing is observable and
  no version number is allocated; once the lock lands the result is
  byte-for-byte identical to the same operations run without contention;

* ``close()`` releases the lock handle, repeated calls are harmless, later
  operations reacquire the lock transparently, and a peer process takes
  the released lock immediately without being stuck on a stale handle;

* a failed reload (corrupt derived-material record or corrupt activity
  journal) raises ``ValueError`` while the in-memory snapshot — including
  the per-version ``is_revoked`` answers, folded into the common frozen
  comparison here — and every on-disk record stay verbatim unchanged, the
  keys already in hand stay readable, and undoing the corruption restores
  full snapshot/disk correspondence;

* the entry-point error vocabulary (empty id -> ``ValueError``, non-integer
  version -> ``TypeError``, unknown key/version -> ``KeyError``) is
  identical on the Windows branch and rejected calls allocate nothing.

How the branch is forced on every platform: ``_force_windows_branch``
hides ``fcntl`` from the vault module and installs a faithful
``msvcrt.locking`` stand-in whose byte-range lock is a genuine
``fcntl.lockf`` call — a real inter-process mutex, not a simulated flag.
A contended ``LK_LOCK`` keeps failing with ``OSError`` exactly the way the
C runtime reports its timed-out wait, so the branch's own retry loop does
the waiting.  On a genuine Windows host ``fcntl`` is already absent and
the real ``msvcrt`` is used unchanged.  Every worker process re-applies
the forcing itself, so the cases do not depend on the multiprocessing
start method.  The three CLI subcommands, the public interface and the
product code are untouched.

Everything happens inside temporary directories, uses the standard library
only and is independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
import queue as queue_mod
import sys
import tempfile
import time
import unittest
from pathlib import Path

import keyvault_ledger.vault as vault_module
from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    ACTIVATIONS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
    _dump_manifest,
)


class _MsvcrtStandIn:
    """Faithful ``msvcrt.locking`` stand-in backed by real OS byte-range locks.

    Only what the vault's Windows branch uses is provided: the ``LK_LOCK``
    / ``LK_UNLCK`` modes and ``locking(fd, mode, nbytes)``.  The byte range
    is locked with ``fcntl.lockf`` — a genuine inter-process mutex shared
    with every other process locking the same file — and a contended
    ``LK_LOCK`` keeps raising ``OSError`` the way the real C runtime
    reports its timed-out wait, so the production retry loop (sleep, then
    ask again) runs for real.  ``stats`` counts the calls so tests can
    prove the branch genuinely locked, waited and released.
    """

    LK_LOCK = 1
    LK_NBLCK = 2
    LK_UNLCK = 3

    def __init__(self) -> None:
        self.stats = {"lock": 0, "unlock": 0, "contended": 0}

    def locking(self, fd, mode, nbytes):
        import fcntl as real_fcntl

        if mode == self.LK_UNLCK:
            self.stats["unlock"] += 1
            real_fcntl.lockf(fd, real_fcntl.LOCK_UN, nbytes, 0, os.SEEK_SET)
            return
        self.stats["lock"] += 1
        # Like the real LK_LOCK: wait a short while, then report failure
        # with OSError; the vault's Windows branch keeps retrying.
        deadline = time.monotonic() + 0.2
        while True:
            try:
                real_fcntl.lockf(
                    fd,
                    real_fcntl.LOCK_EX | real_fcntl.LOCK_NB,
                    nbytes,
                    0,
                    os.SEEK_SET,
                )
                return
            except OSError:
                self.stats["contended"] += 1
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)


_STAND_IN = _MsvcrtStandIn()


def _force_windows_branch():
    """Route ``Vault._file_lock`` through its Windows (``msvcrt``) branch.

    Returns a zero-argument restore callable.  On a genuine Windows host
    ``fcntl`` is already absent and the real ``msvcrt`` is used unchanged;
    elsewhere ``fcntl`` is hidden from the vault module and the faithful
    stand-in above is installed as ``msvcrt``, so the very same branch
    (seek, ``LK_LOCK`` retry loop, ``LK_UNLCK``) runs against a real
    inter-process lock.
    """
    if vault_module.fcntl is None:
        return lambda: None
    # ``subprocess`` (pulled in lazily by ``multiprocessing``) decides
    # whether it runs on Windows by trying ``import msvcrt``; importing it
    # before the stand-in is installed keeps that decision on POSIX.
    import subprocess  # noqa: F401
    import multiprocessing.synchronize  # noqa: F401

    original_fcntl = vault_module.fcntl
    sentinel = object()
    original_msvcrt = sys.modules.get("msvcrt", sentinel)
    vault_module.fcntl = None
    sys.modules["msvcrt"] = _STAND_IN

    def restore() -> None:
        vault_module.fcntl = original_fcntl
        if original_msvcrt is sentinel:
            sys.modules.pop("msvcrt", None)
        else:
            sys.modules["msvcrt"] = original_msvcrt

    return restore


# ---------------------------------------------------------------------------
# module-level worker entry points (picklable for every start method)
# ---------------------------------------------------------------------------


def _windows_seal_worker(
    root: str,
    key_id: str,
    payloads: list[bytes],
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    _force_windows_branch()
    if barrier is not None:
        barrier.wait()
    vault = Vault(root)
    try:
        for payload in payloads:
            vault.seal(key_id, payload)
    finally:
        vault.close()


def _windows_derive_worker(
    root: str,
    key_id: str,
    passwords: list[bytes],
    salt: bytes,
    iterations: int,
    length: int,
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    _force_windows_branch()
    if barrier is not None:
        barrier.wait()
    vault = Vault(root)
    try:
        for password in passwords:
            vault.derive_seal(key_id, password, salt, iterations, length)
    finally:
        vault.close()


def _windows_prefix_reader(
    root: str,
    key_id: str,
    expected_payloads: set[bytes],
    total: int,
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Reopen/reload under the Windows branch; every state must be a prefix.

    The visible versions are always exactly 1..n with the active version
    at n, every visible material is one of the sealed payloads, and a
    version's material never changes between observations: a concurrent
    reader only ever sees a complete old snapshot or a complete new one.
    """
    try:
        _force_windows_branch()
        if barrier is not None:
            barrier.wait()
        pinned: dict[int, bytes] = {}
        last = 0
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            vault = Vault(root)
            vault.reload()
            visible = vault.versions(key_id)
            n = len(visible)
            if visible != list(range(1, n + 1)):
                raise AssertionError(f"torn listing: {visible}")
            if n:
                if vault.active(key_id) != n:
                    raise AssertionError(f"active {vault.active(key_id)} != {n}")
                entry = vault.manifest()["keys"][key_id]
                if [r["version"] for r in entry["versions"]] != visible:
                    raise AssertionError("manifest listing torn")
                for version in visible:
                    material = vault.load(key_id, version)
                    if material not in expected_payloads:
                        raise AssertionError(
                            f"version {version} holds unexpected material"
                        )
                    previous = pinned.get(version)
                    if previous is not None and previous != material:
                        raise AssertionError(
                            f"version {version} material changed under reload"
                        )
                    pinned[version] = material
            if n < last:
                raise AssertionError(f"visible history shrank {last} -> {n}")
            last = n
            vault.close()
            if n >= total:
                break
        results.put(("ok", len(pinned)))
    except BaseException as exc:  # reported to the parent verbatim
        results.put(("error", repr(exc)))


def _windows_lock_holder(
    root: str,
    ready: "multiprocessing.Event",
    release: "multiprocessing.Event",
) -> None:
    """Take the real Windows-branch lock and hold it until told to release."""
    _force_windows_branch()
    vault = Vault(root)
    try:
        with vault._lock:
            with vault._file_lock():
                ready.set()
                if not release.wait(timeout=30):
                    raise RuntimeError("holder was never released")
    finally:
        vault.close()


def _windows_blocked_sealer(
    root: str,
    key_id: str,
    payload: bytes,
    results: "multiprocessing.Queue",
) -> None:
    """Seal one payload, waiting on the Windows-branch lock as long as needed."""
    _force_windows_branch()
    try:
        vault = Vault(root)  # construction itself waits for the lock
        try:
            version = vault.seal(key_id, payload)
        finally:
            vault.close()
        # Prove the wait went through the branch's real retry path.
        results.put(("ok", version, dict(_STAND_IN.stats)))
    except BaseException as exc:  # reported to the parent verbatim
        results.put(("error", repr(exc)))


def _windows_close_a(
    root: str,
    key_id: str,
    closed: "multiprocessing.Event",
    peer_done: "multiprocessing.Event",
) -> None:
    _force_windows_branch()
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


def _windows_close_b(
    root: str,
    key_id: str,
    closed: "multiprocessing.Event",
    peer_done: "multiprocessing.Event",
    results: "multiprocessing.Queue",
) -> None:
    _force_windows_branch()
    try:
        vault = Vault(root)  # opened early; parked here without the lock
        if not closed.wait(timeout=30):
            raise RuntimeError("peer never closed its lock")
        started = time.monotonic()
        version = vault.seal(key_id, b"b-two")
        elapsed = time.monotonic() - started
        vault.close()  # release before signalling: no stale handle remains
        peer_done.set()
        results.put(("ok", version, elapsed))
    except BaseException as exc:  # reported to the parent verbatim
        results.put(("error", repr(exc)))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class WindowsBranchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "vault"
        # True on a genuine Windows host, where the branch under test is
        # the only one and the real msvcrt is used as-is.
        self._native_windows = vault_module.fcntl is None
        self.addCleanup(_force_windows_branch())

    def open_vault(self, root: Path | None = None) -> Vault:
        vault = Vault(self.root if root is None else root)
        # Registered after the temp-dir cleanup, so LIFO ordering releases
        # every lock handle before the temporary directory is removed.
        self.addCleanup(vault.close)
        return vault

    def disk_bytes(self, root: Path) -> dict[str, bytes]:
        """All vault records under ``root``, keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device and carries no
        key data, so it is deliberately excluded.
        """
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def restore_disk(self, root: Path, healthy: dict[str, bytes]) -> None:
        """Put ``root`` back in exactly the captured healthy state."""
        current = self.disk_bytes(root)
        for rel in current.keys() - healthy.keys():
            (root / rel).unlink()
        for rel, data in healthy.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)

    def run_processes(
        self,
        processes: list[multiprocessing.Process],
        timeout: float = 90,
    ) -> None:
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
        """Drain a ``(status, ...)`` worker queue and fail on errors."""
        payloads = []
        for item in self.drain_nowait(process_queue):
            self.assertEqual(item[0], "ok", item)
            payloads.append(item[1:])
        return payloads


# ---------------------------------------------------------------------------
# the branch under test is genuinely the Windows one
# ---------------------------------------------------------------------------


class TestWindowsBranchIsForced(WindowsBranchTestCase):
    def test_operations_route_through_the_windows_branch(self):
        if self._native_windows:
            self.assertIsNone(vault_module.fcntl)
        else:
            self.assertIsNone(vault_module.fcntl)
            self.assertIs(sys.modules["msvcrt"], _STAND_IN)
            before = dict(_STAND_IN.stats)
            vault = self.open_vault()
            vault.seal("k", b"m")
            vault.reload()
            vault.close()
            after = dict(_STAND_IN.stats)
            # Every disk operation locked and released through the stand-in.
            self.assertGreater(after["lock"], before["lock"])
            self.assertGreater(after["unlock"], before["unlock"])
            self.assertEqual(vault.versions("k"), [1])


# ---------------------------------------------------------------------------
# cross-process contention on one shared lock
# ---------------------------------------------------------------------------


class TestWindowsBranchContention(WindowsBranchTestCase):
    def test_competing_writers_share_one_strict_version_sequence(self):
        seal_procs, derive_procs, each = 3, 2, 4
        total = (seal_procs + derive_procs) * each
        salt = b"windows-branch-salt"
        iterations, length = 100, 16
        seal_payloads = {
            f"seal-{p}-{i}".encode("utf-8")
            for p in range(seal_procs)
            for i in range(each)
        }
        passwords = [
            f"pw-{p}-{i}".encode("utf-8")
            for p in range(derive_procs)
            for i in range(each)
        ]
        derived_payloads = {
            hashlib.pbkdf2_hmac("sha256", pw, salt, iterations, dklen=length)
            for pw in passwords
        }
        expected_payloads = seal_payloads | derived_payloads

        barrier = multiprocessing.Barrier(seal_procs + derive_procs + 1)
        workers = [
            multiprocessing.Process(
                target=_windows_seal_worker,
                args=(
                    str(self.root),
                    "k",
                    sorted(
                        f"seal-{p}-{i}".encode("utf-8") for i in range(each)
                    ),
                    barrier,
                ),
            )
            for p in range(seal_procs)
        ]
        workers += [
            multiprocessing.Process(
                target=_windows_derive_worker,
                args=(
                    str(self.root),
                    "k",
                    passwords[p * each : (p + 1) * each],
                    salt,
                    iterations,
                    length,
                    barrier,
                ),
            )
            for p in range(derive_procs)
        ]
        results: multiprocessing.Queue = multiprocessing.Queue()
        reader = multiprocessing.Process(
            target=_windows_prefix_reader,
            args=(
                str(self.root),
                "k",
                expected_payloads,
                total,
                results,
                barrier,
            ),
        )
        reader.start()  # joins the barrier last, so everyone starts together
        self.run_processes(workers)
        reader.join(timeout=60)
        self.assertEqual(reader.exitcode, 0)
        (pinned_count,) = self.assert_worker_results_ok(results)[0]
        self.assertEqual(pinned_count, total)

        final = self.open_vault()
        # From 1 upwards, strictly increasing, no duplicates, no gaps.
        self.assertEqual(final.versions("k"), list(range(1, total + 1)))
        self.assertEqual(final.active("k"), total)
        # Every payload landed exactly once and reads back byte for byte.
        self.assertEqual(
            {final.load("k", v) for v in range(1, total + 1)},
            expected_payloads,
        )
        for version in range(1, total + 1):
            material = final.load("k", version)
            parameters = final.derivation("k", version)
            if parameters:
                # A derived version: parameters travel with the record and
                # the stored bytes are exactly the PBKDF2 output.
                self.assertEqual(
                    parameters,
                    {"salt": salt, "iterations": iterations, "length": length},
                )
                self.assertIn(material, derived_payloads)
                self.assertEqual(len(material), length)
            else:
                self.assertIn(material, seal_payloads)
        # A full reload changes nothing: historical material stays put.
        final.reload()
        self.assertEqual(final.versions("k"), list(range(1, total + 1)))
        self.assertEqual(
            {final.load("k", v) for v in range(1, total + 1)},
            expected_payloads,
        )


# ---------------------------------------------------------------------------
# a blocked writer waits, then lands exactly the uncontended result
# ---------------------------------------------------------------------------


class TestWindowsBranchBlockedWriter(WindowsBranchTestCase):
    def test_blocked_writer_waits_then_matches_uncontended_result(self):
        vault = self.open_vault()
        vault.seal("k", b"seed")
        manifest_v1 = (self.root / MANIFEST_NAME).read_bytes()

        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_windows_lock_holder, args=(str(self.root), ready, release)
        )
        holder.start()
        self.assertTrue(ready.wait(timeout=10))

        results: multiprocessing.Queue = multiprocessing.Queue()
        sealer = multiprocessing.Process(
            target=_windows_blocked_sealer,
            args=(str(self.root), "k", b"blocked-material", results),
        )
        sealer.start()

        # While the holder keeps the lock the sealer waits: no half record,
        # no truncated listing, and crucially no version number allocated.
        time.sleep(0.8)
        self.assertTrue(sealer.is_alive(), "sealer ran while the lock was held")
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), manifest_v1)
        self.assertEqual(
            sorted(p.name for p in (self.root / MATERIALS_DIR).rglob("*.bin")),
            ["1.bin"],
        )
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])
        self.assertEqual(vault.versions("k"), [1])

        release.set()
        sealer.join(timeout=30)
        holder.join(timeout=30)
        self.assertEqual(sealer.exitcode, 0)
        self.assertEqual(holder.exitcode, 0)
        (status, version, stats) = next(iter(self.drain_nowait(results)))
        self.assertEqual(status, "ok")
        # The version was allocated only once the lock was in hand.
        self.assertEqual(version, 2)
        if not self._native_windows:
            # The wait really went through the branch's retry loop.
            self.assertGreaterEqual(stats["contended"], 1)
            self.assertGreaterEqual(stats["lock"], 1)
            self.assertGreaterEqual(stats["unlock"], 1)

        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2])
        self.assertEqual(final.load("k", 1), b"seed")
        self.assertEqual(final.load("k", 2), b"blocked-material")

        # The contended outcome is byte-for-byte the uncontended one: the
        # same operations with nobody holding the lock produce the same
        # records on disk, and repeating the input reproduces the result.
        control_root = Path(self._tmp.name) / "control"
        control = self.open_vault(control_root)
        control.seal("k", b"seed")
        control.seal("k", b"blocked-material")
        self.assertEqual(self.disk_bytes(self.root), self.disk_bytes(control_root))


# ---------------------------------------------------------------------------
# close(): idempotent release, immediate handoff, transparent reacquire
# ---------------------------------------------------------------------------


class TestWindowsBranchCloseRelease(WindowsBranchTestCase):
    def test_another_process_takes_the_lock_immediately_after_close(self):
        closed = multiprocessing.Event()
        peer_done = multiprocessing.Event()
        results: multiprocessing.Queue = multiprocessing.Queue()
        # B opens first but parks until A closes, so it measures the
        # handoff latency directly and proves no stale handle blocks it.
        b = multiprocessing.Process(
            target=_windows_close_b,
            args=(str(self.root), "k", closed, peer_done, results),
        )
        a = multiprocessing.Process(
            target=_windows_close_a, args=(str(self.root), "k", closed, peer_done)
        )
        b.start()
        a.start()
        for process in (a, b):
            process.join(timeout=40)
            self.assertEqual(process.exitcode, 0)
        self.assertTrue(peer_done.wait(timeout=5))
        (status, version, elapsed) = next(iter(self.drain_nowait(results)))
        self.assertEqual(status, "ok")
        self.assertEqual(version, 2)
        # No stale handle blocks the peer: the seal completes promptly.
        self.assertLess(elapsed, 10.0)

        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2, 3])
        self.assertEqual(final.load("k", 1), b"a-one")
        self.assertEqual(final.load("k", 2), b"b-two")
        self.assertEqual(final.load("k", 3), b"a-three")

    def test_close_is_idempotent_and_operations_reacquire_unchanged(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.close()
        self.assertIsNone(vault._lock_fh)
        vault.close()
        vault.close()  # repeated release is not an error
        self.assertIsNone(vault._lock_fh)

        # Every operation kind reopens the lock and behaves identically.
        self.assertEqual(vault.seal("k", b"two"), 2)
        self.assertIsNotNone(vault._lock_fh)
        self.assertEqual(vault.load("k", 1), b"one")
        self.assertEqual(vault.load("k"), b"two")
        self.assertEqual(vault.derive_seal("d", b"pw", b"salt", 100, 16), 1)
        self.assertEqual(vault.seal("k", b"three"), 3)
        vault.revoke("k", 1)
        vault.set_active("k", 2)
        vault.reload()
        self.assertEqual(vault.active("k"), 2)
        self.assertTrue(vault.is_revoked("k", 1))
        vault.close()
        self.assertIsNone(vault._lock_fh)

        # The reacquired-lock writes landed durably and survive a new open.
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), [1, 2, 3])
        self.assertTrue(reopened.is_revoked("k", 1))
        self.assertEqual(reopened.active("k"), 2)
        self.assertEqual(reopened.load("k", 2), b"two")
        self.assertEqual(reopened.versions("d"), [1])


# ---------------------------------------------------------------------------
# failed reload: snapshot (per-version revocation included) and disk frozen
# ---------------------------------------------------------------------------


class TestWindowsBranchFailedReloadFreeze(WindowsBranchTestCase):
    def _build_healthy(self, name: str) -> tuple[Vault, Path]:
        """Build a rich, healthy vault and return ``(handle, root)``."""
        root = Path(self._tmp.name) / name
        vault = self.open_vault(root)

        vault.seal("plain", b"plain-one")
        vault.seal("plain", b"plain-two")
        vault.set_active("plain", 1)  # bound to latest=2

        vault.derive_seal("drv", b"passphrase-alpha", b"salt-alpha", 200, 24)
        vault.derive_seal("drv", b"passphrase-bravo", b"salt-bravo", 300, 32)
        vault.revoke("drv", 1)

        vault.seal("mix", b"mix-one")
        vault.derive_seal("mix", b"passphrase-charlie", b"salt-charlie", 400, 16)
        vault.seal("mix", b"mix-three")
        vault.revoke("mix", 1)
        vault.set_active("mix", 2)  # points at the derived version, latest=3

        return vault, root

    def _answers(self, vault: Vault) -> dict:
        """Every queryable state, per-version revocation status included."""
        keys = {}
        for key_id in ("plain", "drv", "mix"):
            versions = vault.versions(key_id)
            keys[key_id] = {
                "versions": versions,
                "active": vault.active(key_id),
                "revoked": vault.revoked_versions(key_id),
                # The per-version revocation query is folded into the
                # common frozen comparison for every failure window.
                "is_revoked": {
                    v: vault.is_revoked(key_id, v) for v in versions
                },
                "materials": {v: vault.load(key_id, v) for v in versions},
                "derivations": {v: vault.derivation(key_id, v) for v in versions},
            }
        return {
            "keys": keys,
            "manifest": vault.manifest(),
            "unknown_versions": vault.versions("never-sealed"),
            "unknown_revoked": vault.revoked_versions("never-sealed"),
        }

    def _assert_snapshot_corresponds_to_disk(self, vault: Vault, root: Path) -> None:
        """Re-derive every observable answer straight from the disk records."""
        manifest = json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))
        self.assertEqual(vault.manifest(), manifest)

        revoked: dict[str, set[int]] = {}
        for line in (root / REVOCATIONS_NAME).read_text("utf-8").splitlines():
            record = json.loads(line)
            revoked.setdefault(record["key_id"], set()).add(record["version"])

        last_repoint: dict[str, tuple[int, int]] = {}
        activations = root / ACTIVATIONS_NAME
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
                data = (root / record["file"]).read_bytes()
                self.assertEqual(vault.load(key_id, version), data)
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
                self.assertEqual(
                    vault.is_revoked(key_id, version),
                    version in revoked.get(key_id, set()),
                )
                parameters = vault.derivation(key_id, version)
                if "derivation" in record:
                    persisted = record["derivation"]
                    self.assertEqual(
                        parameters["salt"],
                        base64.b64decode(persisted["salt"], validate=True),
                    )
                    self.assertEqual(parameters["iterations"], persisted["iterations"])
                    self.assertEqual(parameters["length"], persisted["length"])
                    self.assertEqual(len(data), persisted["length"])
                else:
                    self.assertEqual(parameters, {})
            expected_active = entry["active"]
            if key_id in last_repoint:
                target, bound_latest = last_repoint[key_id]
                if bound_latest == versions[-1]:
                    expected_active = target
            self.assertEqual(vault.active(key_id), expected_active)
            self.assertEqual(
                set(vault.revoked_versions(key_id)), revoked.get(key_id, set())
            )

    def _assert_failure_then_recovery(
        self,
        vault: Vault,
        root: Path,
        expected: dict,
        corrupt,
    ) -> None:
        """Drive one full corruption window and the recovery afterwards."""
        healthy_disk = self.disk_bytes(root)

        corrupt()
        failing_disk = self.disk_bytes(root)
        self.assertNotEqual(failing_disk, healthy_disk)

        # Three failed reloads: the exception type and message are
        # deterministic, and every observable answer stays frozen —
        # per-version revocation markers included.
        message = None
        for _ in range(3):
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            if message is None:
                message = str(caught.exception)
            else:
                self.assertEqual(str(caught.exception), message)
            self.assertEqual(self._answers(vault), expected)

        # A cold opener rejects the very same state with the same complaint.
        with self.assertRaises(ValueError) as caught:
            Vault(root)
        self.assertEqual(str(caught.exception), message)

        # The failed reloads/open neither added nor removed a disk record.
        self.assertEqual(self.disk_bytes(root), failing_disk)

        # Undo the corruption: one successful reload restores full
        # correspondence between snapshot and disk.
        self.restore_disk(root, healthy_disk)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self._assert_snapshot_corresponds_to_disk(vault, root)
        reopened = self.open_vault(root)
        self.assertEqual(self._answers(reopened), expected)
        self.assertEqual(self.disk_bytes(root), healthy_disk)

        # Repeating the identical failing input gives the identical result,
        # then recovery works a second time as well.
        corrupt()
        with self.assertRaises(ValueError) as caught:
            vault.reload()
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(self._answers(vault), expected)
        self.restore_disk(root, healthy_disk)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self._assert_snapshot_corresponds_to_disk(vault, root)

    def test_corrupt_derivation_record_freezes_then_recovers(self):
        vault, root = self._build_healthy("freeze-derived")
        expected = self._answers(vault)

        def corrupt() -> None:
            path = root / MANIFEST_NAME
            manifest = json.loads(path.read_bytes().decode("utf-8"))
            record = next(
                r
                for r in manifest["keys"]["mix"]["versions"]
                if r["version"] == 2
            )
            record["derivation"]["salt"] = "!!!not-base64!!!"
            path.write_bytes(_dump_manifest(manifest))

        self._assert_failure_then_recovery(vault, root, expected, corrupt)

    def test_corrupt_activation_journal_freezes_then_recovers(self):
        vault, root = self._build_healthy("freeze-activation")
        expected = self._answers(vault)
        payload = b'{"key_id": "mix", "version": 99, "latest": 99}\n'

        def corrupt() -> None:
            (root / ACTIVATIONS_NAME).write_bytes(payload)

        self._assert_failure_then_recovery(vault, root, expected, corrupt)


# ---------------------------------------------------------------------------
# public entry-point error contract on the Windows branch
# ---------------------------------------------------------------------------


class TestWindowsBranchEntryContract(WindowsBranchTestCase):
    def test_error_vocabulary_unchanged_and_nothing_allocated(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        manifest_before = (self.root / MANIFEST_NAME).read_bytes()
        journal_before = (self.root / REVOCATIONS_NAME).read_bytes()

        # Empty key id -> ValueError at every entry point.
        for call in (
            lambda: vault.seal("", b"x"),
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
        for bad in (1.0, 2.5, True, False, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.revoke("k", bad)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.is_revoked("k", bad)
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.set_active("k", bad)
        for bad in (1.0, 2.5, True, False, "1", (1,)):
            # None is the documented "active version" sentinel here.
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.derivation("d", bad)

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

        # No rejected call produced a version, a journal record or any
        # other disk change; the keys in hand read back byte for byte.
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), manifest_before)
        self.assertEqual((self.root / REVOCATIONS_NAME).read_bytes(), journal_before)
        self.assertFalse((self.root / ACTIVATIONS_NAME).exists())
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.versions("d"), [1])
        self.assertEqual(vault.load("k"), b"m")

        # The next genuine seal still draws version 2: nothing was consumed.
        self.assertEqual(vault.seal("k", b"next"), 2)
        self.assertEqual(vault.load("k", 2), b"next")


if __name__ == "__main__":
    unittest.main()
