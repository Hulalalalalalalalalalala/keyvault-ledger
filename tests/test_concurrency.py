"""Concurrency and public-contract regression tests for ``keyvault_ledger``.

These tests pin the observable behaviour promised for several processes (and
several threads) sharing one vault directory:

* a process holding the lock for one complete record blocks every other
  process until it releases — no half record is observable and no listing is
  ever truncated;
* a reload interleaved with seals only ever sees the complete old list or a
  complete newly persisted prefix, never a half-old/half-new mixture;
* a reload that fails whole-vault validation raises ``ValueError`` and leaves
  both the in-memory snapshot and the disk records byte-for-byte unchanged;
* ``vault.lock`` is only a mutual-exclusion device: clearing or filling it
  carries no key data and never participates in validation;
* ``close()`` is idempotent and releases the lock so another process can take
  it immediately, with later operations reacquiring it transparently;
* the entry-point error contract (empty ids, non-bytes material, non-integer
  parameters, unknown keys/versions, duplicate revocation, repointing at the
  already-active version, ...) raises exactly the documented exception type
  and never produces a new version.

Every case works in a temporary directory, uses real processes/threads and
the standard library only.  Running the whole suite must stay green::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import queue as queue_mod
import re
import subprocess
import sys
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
from tests._fixtures import VaultFixture, cli_env


# ---------------------------------------------------------------------------
# module-level worker entry points (picklable for every start method)
# ---------------------------------------------------------------------------


def _seal_payloads(
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


def _derive_payloads(
    root: str,
    key_id: str,
    passwords: list[bytes],
    salt: bytes,
    iterations: int,
    length: int,
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    if barrier is not None:
        barrier.wait()
    vault = Vault(root)
    try:
        for password in passwords:
            vault.derive_seal(key_id, password, salt, iterations, length)
    finally:
        vault.close()


def _revoke_partition(
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
            vault.reload()
    finally:
        vault.close()


def _reload_rounds(
    root: str,
    rounds: int,
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    if barrier is not None:
        barrier.wait()
    for _ in range(rounds):
        vault = Vault(root)
        vault.reload()
        # Both journals must parse as complete whole records throughout:
        # a torn append would show up here as a malformed line.
        _parse_journal(Path(root) / REVOCATIONS_NAME)
        activations = Path(root) / ACTIVATIONS_NAME
        if activations.exists():
            _parse_journal(activations)
        vault.close()


def _parse_journal(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw != b"" and not raw.endswith(b"\n"):
        raise AssertionError(f"journal line not newline-terminated: {path}")
    records = []
    for line in raw.splitlines():
        if not line:
            raise AssertionError(f"empty journal record in {path}")
        record = json.loads(line)
        if not isinstance(record, dict):
            raise AssertionError(f"journal record not an object in {path}")
        records.append(record)
    return records


def _prefix_reader(
    root: str,
    expected: dict[str, set[bytes]],
    total_counts: dict[str, int],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
    ready: "multiprocessing.managers.Event | None" = None,
) -> None:
    """Reopen/reload the vault and verify every visible state is a prefix.

    For each key the visible versions must always be exactly 1..n with the
    active version at n, every visible material one of the sealed payloads,
    and a version's material must never change between observations
    (historical material is never overwritten).  Visible counts never
    shrink, because records are append-only.
    """
    if barrier is not None:
        barrier.wait()
    last_counts = {key_id: 0 for key_id in expected}
    pinned: dict[str, dict[int, bytes]] = {key_id: {} for key_id in expected}
    observations: dict[str, list[int]] = {key_id: [] for key_id in expected}
    ready_signalled = False
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        vault = Vault(root)
        vault.reload()
        manifest = vault.manifest()
        for key_id, payloads in expected.items():
            visible = vault.versions(key_id)
            n = len(visible)
            if visible != list(range(1, n + 1)):
                raise AssertionError(f"non-contiguous versions for {key_id!r}: {visible}")
            if n > 0:
                if vault.active(key_id) != n:
                    raise AssertionError(
                        f"{key_id!r}: active {vault.active(key_id)} != latest {n}"
                    )
                disk_entry = manifest["keys"][key_id]
                disk_versions = [r["version"] for r in disk_entry["versions"]]
                if disk_versions != visible or disk_entry["active"] != n:
                    raise AssertionError(
                        f"{key_id!r}: manifest listing {disk_versions} / "
                        f"active {disk_entry['active']} does not match snapshot"
                    )
                for version in range(1, n + 1):
                    material = vault.load(key_id, version)
                    if material not in payloads:
                        raise AssertionError(
                            f"{key_id!r} version {version} holds unexpected material"
                        )
                    previous = pinned[key_id].get(version)
                    if previous is not None and previous != material:
                        raise AssertionError(
                            f"{key_id!r} version {version} material changed under reload"
                        )
                    pinned[key_id][version] = material
                # Every visible material is unique: appending never reuses a
                # version, so no two slots may resolve to the same payload.
                if len(set(pinned[key_id].values())) != len(pinned[key_id]):
                    raise AssertionError(f"{key_id!r}: duplicated historical material")
            if n < last_counts[key_id]:
                raise AssertionError(
                    f"{key_id!r}: visible history shrank {last_counts[key_id]} -> {n}"
                )
            last_counts[key_id] = n
            observations[key_id].append(n)
        vault.close()
        if ready is not None and not ready_signalled:
            ready.set()
            ready_signalled = True
        if all(last_counts[k] >= total_counts[k] for k in expected):
            break
    results.put(("ok", observations))


def _prefix_reader_guarded(
    root: str,
    expected: dict[str, set[bytes]],
    total_counts: dict[str, int],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
    ready: "multiprocessing.managers.Event | None" = None,
) -> None:
    """Run :func:`_prefix_reader`, surfacing any failure over ``results``."""
    try:
        _prefix_reader(root, expected, total_counts, results, barrier, ready)
    except BaseException as exc:  # reported to the parent verbatim
        results.put(("error", repr(exc)))


def _held_lock_holder(
    root: str,
    key_id: str,
    revoke_version: int,
    ready: "multiprocessing.Event",
    release: "multiprocessing.Event",
) -> None:
    """Hold the inter-process lock across one explicit journal append.

    Uses the vault's own lock helpers (the in-process RLock plus the file
    lock) so the waiters below block on the real production lock, then
    appends one revocation record while holding it.
    """
    vault = Vault(root)
    with vault._lock:
        with vault._file_lock():
            ready.set()
            if not release.wait(timeout=30):
                raise RuntimeError("holder was never released")
            line = (
                json.dumps({"key_id": key_id, "version": revoke_version}, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            with open(Path(root) / REVOCATIONS_NAME, "ab") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
    vault.close()


def _expect_open_value_error(root: str, results: "multiprocessing.Queue") -> None:
    vault = None
    try:
        vault = Vault(root)
    except ValueError as exc:
        results.put(type(exc).__name__)
        if vault is not None:
            vault.close()
        return
    vault.close()
    results.put("no-exception")


def _faulted_state_reader(
    root: str,
    facts: dict,
    opened: "multiprocessing.Event",
    begin: "multiprocessing.Event",
    recovered: "multiprocessing.Event",
    errors: "multiprocessing.Queue",
) -> None:
    """Hold a healthy handle across disk corruption and re-restoration."""
    vault = Vault(root)
    healthy_manifest = vault.manifest()

    def check_snapshot(label: str) -> None:
        try:
            if vault.versions("k") != facts["versions"]:
                raise AssertionError("versions changed")
            if vault.active("k") != facts["active"]:
                raise AssertionError("active changed")
            for version, material in facts["materials"].items():
                if vault.load("k", version) != material:
                    raise AssertionError(f"material for {version} changed")
            if vault.revoked_versions("k") != facts["revoked"]:
                raise AssertionError("revoked set changed")
            if not vault.is_revoked("k", 1):
                raise AssertionError("revocation marker lost")
            if any(vault.is_revoked("k", v) for v in (2, 3, 4)):
                raise AssertionError("spurious revocation marker")
            if vault.revoked_versions("never-sealed") != []:
                raise AssertionError("unknown-key query changed")
            if vault.derivation("k", 4) != facts["derivation"]:
                raise AssertionError("derivation parameters changed")
            if vault.derivation("k", 1) != {}:
                raise AssertionError("plain version gained derivation data")
            try:
                vault.load("never-sealed")
                raise AssertionError("unknown key became loadable")
            except KeyError:
                pass
            try:
                vault.derivation("never-sealed")
                raise AssertionError("unknown derivation became queryable")
            except KeyError:
                pass
            if vault.manifest() != healthy_manifest:
                raise AssertionError("manifest copy changed")
        except BaseException as exc:  # surfaced to the parent verbatim
            errors.put(f"snapshot changed ({label}): {exc!r}")

    opened.set()
    if not begin.wait(timeout=30):
        errors.put("reader never told to begin")
        return
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        try:
            vault.reload()
            errors.put("reload unexpectedly succeeded while corrupt")
        except ValueError:
            pass
        check_snapshot("during corrupt window")
    if not recovered.wait(timeout=30):
        errors.put("reader never saw recovery")
        vault.close()
        return
    vault.reload()
    check_snapshot("after recovery")
    vault.close()


def _cli_hammer(
    root: str,
    rounds: int,
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier",
) -> None:
    barrier.wait()
    # The frozen output must not depend on the ambient warning policy (the
    # CLI subprocess gets a scrubbed environment via cli_env).
    env = cli_env()
    line_pattern = re.compile(r"^([^\t]+)\tactive=(\d+)\tversions=(\d+(?:,\d+)*)$")
    commands = []
    try:
        for index in range(rounds):
            command = "reload" if index % 2 else "versions"
            result = subprocess.run(
                [sys.executable, "-m", "keyvault_ledger", "--root", root, command],
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                raise AssertionError(f"cli {command} failed: {result.stderr!r}")
            if command == "reload":
                if result.stdout != "reloaded\n" or result.stderr != "":
                    raise AssertionError("frozen reload output changed")
            else:
                for line in result.stdout.splitlines():
                    match = line_pattern.match(line)
                    if match is None:
                        raise AssertionError(f"malformed versions line: {line!r}")
                    versions = [int(v) for v in match.group(3).split(",")]
                    active = int(match.group(2))
                    if versions != list(range(1, len(versions) + 1)):
                        raise AssertionError(
                            f"truncated/non-contiguous listing: {line!r}"
                        )
                    if active != versions[-1]:
                        raise AssertionError(f"active not at latest version: {line!r}")
                if result.stderr != "":
                    raise AssertionError("versions wrote to stderr")
            commands.append(command)
        results.put(("ok", commands))
    except BaseException as exc:  # surface worker failures to the parent
        results.put(("error", repr(exc)))


def _lock_content_worker(root: str) -> None:
    vault = Vault(root)
    try:
        vault.reload()
        if vault.seal("k", b"from-other-process") != 3:
            raise AssertionError("version sequence broken after clearing lock file")
        vault.revoke("k", 1)
        vault.reload()
    finally:
        vault.close()


def _close_dance_a(
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
    # The handle reacquires the lock and continues the shared sequence.
    if vault.seal(key_id, b"a-three") != 3:
        raise AssertionError("seal after close must continue the sequence")
    vault.close()


def _close_dance_b(
    root: str,
    key_id: str,
    closed: "multiprocessing.Event",
    peer_done: "multiprocessing.Event",
    results: "multiprocessing.Queue",
) -> None:
    vault = Vault(root)  # opened early; parked here without holding the lock
    if not closed.wait(timeout=30):
        raise RuntimeError("peer never closed its lock")
    started = time.monotonic()
    version = vault.seal(key_id, b"b-two")
    elapsed = time.monotonic() - started
    vault.close()  # release before signalling: no stale handle may remain
    peer_done.set()
    results.put((version, elapsed))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class ConcurrencyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.root = self.fixture.root

    def open_vault(self) -> Vault:
        """Open a vault; its handle is returned by the single teardown."""
        return self.fixture.open()

    def disk_manifest(self) -> dict:
        return json.loads((self.root / MANIFEST_NAME).read_bytes().decode("utf-8"))

    def journal_records(self, name: str = REVOCATIONS_NAME) -> list[dict]:
        path = self.root / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text("utf-8").splitlines()]

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
# cross-process lock serialisation
# ---------------------------------------------------------------------------


class TestCrossProcessLockSerialisation(ConcurrencyTestCase):
    def test_waiters_block_until_holder_releases_with_no_half_records(self):
        vault = self.open_vault()
        for version in range(1, 4):
            vault.seal("k", f"m-{version}".encode("utf-8"))

        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_held_lock_holder, args=(str(self.root), "k", 1, ready, release)
        )
        # If an assertion fails while the holder parks the waiters, the
        # single teardown sets this gate and drains every process before the
        # parent handle is returned: the blocked writes proceed and finish,
        # their results unchanged.
        self.fixture.track_gate(release)
        self.fixture.track_process(holder)
        holder.start()
        self.assertTrue(ready.wait(timeout=10))

        # Start one of every operation kind only after the holder owns the
        # lock: a sealer, a revoker and a pure reader must all be parked
        # (construction itself takes the lock, so they block on open).
        sealer = multiprocessing.Process(
            target=_seal_payloads, args=(str(self.root), "k", [b"blocked-material"])
        )
        revoker = multiprocessing.Process(
            target=_revoke_partition, args=(str(self.root), "k", [2])
        )
        reloader = multiprocessing.Process(target=_reload_rounds, args=(str(self.root), 1))
        self.fixture.track_process(sealer, revoker, reloader)
        for waiter in (sealer, revoker, reloader):
            waiter.start()

        # While the holder keeps the lock (and has not written yet) every
        # waiter is alive, and nothing on disk has started appearing:
        # no manifest replacement, no orphan material, no partial journal.
        time.sleep(0.5)
        for waiter in (sealer, revoker, reloader):
            self.assertTrue(waiter.is_alive(), "waiter ran while the lock was held")
        self.assertEqual(
            [r["version"] for r in self.disk_manifest()["keys"]["k"]["versions"]],
            [1, 2, 3],
        )
        self.assertEqual((self.root / REVOCATIONS_NAME).read_bytes(), b"")
        self.assertEqual(
            sorted(p.name for p in (self.root / MATERIALS_DIR).rglob("*.bin")),
            ["1.bin", "2.bin", "3.bin"],
        )
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])

        release.set()
        for waiter in (sealer, revoker, reloader):
            waiter.join(timeout=30)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(waiter.exitcode, 0)
        holder.join(timeout=30)
        self.assertEqual(holder.exitcode, 0)

        # Exactly one complete record per operation landed; nothing torn.
        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2, 3, 4])
        self.assertEqual(final.load("k", 4), b"blocked-material")
        # Historical material survived untouched.
        for version in range(1, 4):
            self.assertEqual(final.load("k", version), f"m-{version}".encode("utf-8"))
        records = self.journal_records()
        self.assertEqual(sorted(r["version"] for r in records), [1, 2])
        self.assertTrue(all(r["key_id"] == "k" for r in records))
        final.reload()
        self.assertEqual(final.revoked_versions("k"), [1, 2])

    def test_concurrent_seals_revokes_and_reloads_never_tear_journals(self):
        vault = self.open_vault()
        total = 12
        for version in range(1, total + 1):
            vault.seal("k", f"m-{version}".encode("utf-8"))

        procs = 3
        barrier = multiprocessing.Barrier(procs + 1 + 2)  # revokers + sealer + readers
        self.fixture.track_barrier(barrier)
        workers = [
            multiprocessing.Process(
                target=_revoke_partition,
                args=(
                    str(self.root),
                    "k",
                    [v for v in range(1, total + 1) if v % procs == p],
                    barrier,
                ),
            )
            for p in range(procs)
        ]
        workers.append(
            multiprocessing.Process(
                target=_seal_payloads,
                args=(
                    str(self.root),
                    "k",
                    [b"fresh-1", b"fresh-2", b"fresh-3"],
                    barrier,
                ),
            )
        )
        workers += [
            multiprocessing.Process(target=_reload_rounds, args=(str(self.root), 25, barrier))
            for _ in range(2)
        ]
        self.run_processes(workers)

        # Journal: exactly `total` whole, unique records, one per version.
        raw = (self.root / REVOCATIONS_NAME).read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        records = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(len(records), total)
        self.assertEqual(sorted(r["version"] for r in records), list(range(1, total + 1)))
        self.assertEqual(len({(r["key_id"], r["version"]) for r in records}), total)

        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, total + 4)))
        self.assertEqual(final.revoked_versions("k"), list(range(1, total + 1)))
        for version in range(1, total + 1):
            self.assertEqual(final.load("k", version), f"m-{version}".encode("utf-8"))
        self.assertEqual(final.load("k", total + 1), b"fresh-1")
        self.assertEqual(final.load("k", total + 3), b"fresh-3")


# ---------------------------------------------------------------------------
# reload interleaved with seals: complete old list or complete new prefix
# ---------------------------------------------------------------------------


class TestReloadSealInterleaving(ConcurrencyTestCase):
    def test_every_reload_sees_a_complete_prefix_never_a_mixture(self):
        vault = self.open_vault()
        vault.seal("k", b"m-1")

        new_count = 8
        payloads = [b"m-1"] + [
            f"sealed-{n}".encode("utf-8") for n in range(2, new_count + 1)
        ]

        results: multiprocessing.Queue = multiprocessing.Queue()
        ready = multiprocessing.Event()
        gate = multiprocessing.Event()
        # The reader gates the sealer: it does not seal a single new version
        # until the reader has observed the complete old one-version list.
        reader = multiprocessing.Process(
            target=_prefix_reader_guarded,
            args=(
                str(self.root),
                {"k": set(payloads)},
                {"k": new_count},
                results,
                None,
                ready,
            ),
        )
        # Both processes are drained by the single teardown if an assertion
        # fails while they are still running.
        self.fixture.track_process(reader)
        reader.start()
        self.assertTrue(ready.wait(timeout=15))
        gate.set()
        sealer = multiprocessing.Process(
            target=_seal_payloads,
            args=(str(self.root), "k", payloads[1:], None, gate),
        )
        self.fixture.track_process(sealer)
        sealer.start()
        sealer.join(timeout=60)
        self.assertEqual(sealer.exitcode, 0)
        reader.join(timeout=60)
        self.assertEqual(reader.exitcode, 0)
        observations = self.assert_worker_results_ok(results)[0]["k"]

        # It started at the old complete list, finished at the new complete
        # list, and every state in between was a prefix 1..n.
        self.assertEqual(observations[0], 1)
        self.assertEqual(observations[-1], new_count)
        self.assertEqual(observations, sorted(observations))
        self.assertTrue(all(1 <= n <= new_count for n in observations))

        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, new_count + 1)))
        self.assertEqual(
            {final.load("k", v) for v in range(1, new_count + 1)}, set(payloads)
        )

    def test_two_keys_sealed_plain_and_derived_stay_self_consistent(self):
        plain_each, derived_each = 5, 4
        plain_payloads = [
            f"plain-{p}-{i}".encode("utf-8")
            for p in range(2)
            for i in range(plain_each)
        ]
        salt = b"interleave-salt"
        derived_passwords = [
            f"pw-{p}-{i}".encode("utf-8")
            for p in range(2)
            for i in range(derived_each)
        ]
        derived_payloads = [
            hashlib.pbkdf2_hmac("sha256", password, salt, 100, dklen=16)
            for password in derived_passwords
        ]
        barrier = multiprocessing.Barrier(5)
        # Aborted at teardown if an assertion fails before every party has
        # shown up, so no worker waits forever for a missing party.
        self.fixture.track_barrier(barrier)
        workers = [
            multiprocessing.Process(
                target=_seal_payloads,
                args=(
                    str(self.root),
                    "alpha",
                    plain_payloads[p * plain_each : (p + 1) * plain_each],
                    barrier,
                ),
            )
            for p in range(2)
        ]
        workers += [
            multiprocessing.Process(
                target=_derive_payloads,
                args=(
                    str(self.root),
                    "beta",
                    derived_passwords[p * derived_each : (p + 1) * derived_each],
                    salt,
                    100,
                    16,
                    barrier,
                ),
            )
            for p in range(2)
        ]
        results: multiprocessing.Queue = multiprocessing.Queue()
        reader = multiprocessing.Process(
            target=_prefix_reader_guarded,
            args=(
                str(self.root),
                {"alpha": set(plain_payloads), "beta": set(derived_payloads)},
                {"alpha": 2 * plain_each, "beta": 2 * derived_each},
                results,
                barrier,
            ),
        )
        reader.start()  # joins the barrier last, then everyone starts together
        self.fixture.track_process(reader)
        self.run_processes(workers)
        reader.join(timeout=60)
        self.assertEqual(reader.exitcode, 0)
        observed = self.assert_worker_results_ok(results)[0]
        self.assertEqual(observed["alpha"][-1], 2 * plain_each)
        self.assertEqual(observed["beta"][-1], 2 * derived_each)

        final = self.open_vault()
        self.assertEqual(final.versions("alpha"), list(range(1, 2 * plain_each + 1)))
        self.assertEqual(final.versions("beta"), list(range(1, 2 * derived_each + 1)))
        self.assertEqual(
            {final.load("alpha", v) for v in final.versions("alpha")},
            set(plain_payloads),
        )
        self.assertEqual(
            {final.load("beta", v) for v in final.versions("beta")},
            set(derived_payloads),
        )

    def test_cli_listings_under_concurrent_seals_are_never_truncated(self):
        rounds = 6
        sealers = 3
        each = 4
        barrier = multiprocessing.Barrier(sealers + 1)
        self.fixture.track_barrier(barrier)
        workers = [
            multiprocessing.Process(
                target=_seal_payloads,
                args=(
                    str(self.root),
                    f"key-{p}",
                    [f"{p}-{i}".encode("utf-8") for i in range(each)],
                    barrier,
                ),
            )
            for p in range(sealers)
        ]
        results: multiprocessing.Queue = multiprocessing.Queue()
        hammer = multiprocessing.Process(
            target=_cli_hammer, args=(str(self.root), rounds, results, barrier)
        )
        self.fixture.track_process(hammer)
        hammer.start()
        self.run_processes(workers)
        hammer.join(timeout=60)
        self.assertEqual(hammer.exitcode, 0)
        commands = self.assert_worker_results_ok(results)[0]
        self.assertEqual(sorted(commands).count("reload"), rounds // 2)

        final = self.open_vault()
        for p in range(sealers):
            self.assertEqual(final.versions(f"key-{p}"), list(range(1, each + 1)))


# ---------------------------------------------------------------------------
# failed reload: snapshot and disk double invariance
# ---------------------------------------------------------------------------


class TestFailedReloadInvariance(ConcurrencyTestCase):
    def _build_healthy_vault(self) -> tuple[Vault, dict]:
        vault = self.open_vault()
        materials = {
            1: b"m-1",
            2: b"m-2",
            3: b"m-3",
            4: hashlib.pbkdf2_hmac("sha256", b"pw", b"facts-salt", 500, dklen=24),
        }
        vault.seal("k", materials[1])
        vault.seal("k", materials[2])
        vault.seal("k", materials[3])
        vault.derive_seal("k", b"pw", b"facts-salt", 500, 24)
        vault.revoke("k", 1)
        vault.set_active("k", 2)
        facts = {
            "versions": [1, 2, 3, 4],
            "active": 2,
            "revoked": [1],
            "materials": materials,
            "derivation": {"salt": b"facts-salt", "iterations": 500, "length": 24},
        }
        return vault, facts

    def _all_disk_bytes(self) -> dict[Path, bytes]:
        return {
            path: path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def test_value_error_keeps_snapshot_and_disk_verbatim_under_concurrency(self):
        _vault, facts = self._build_healthy_vault()
        healthy_manifest = (self.root / MANIFEST_NAME).read_bytes()

        opened = multiprocessing.Event()
        begin = multiprocessing.Event()
        recovered = multiprocessing.Event()
        errors: multiprocessing.Queue = multiprocessing.Queue()
        reader = multiprocessing.Process(
            target=_faulted_state_reader,
            args=(str(self.root), facts, opened, begin, recovered, errors),
        )
        # The reader parks on ``begin`` and later ``recovered``; on a failed
        # assertion the single teardown releases those gates and drains the
        # reader (and every opener) before the parent handle is returned.
        self.fixture.track_gate(begin, recovered)
        self.fixture.track_process(reader)
        reader.start()
        self.assertTrue(opened.wait(timeout=10))

        # Corrupt the manifest out of band; the long-lived reader keeps its
        # snapshot while fresh openers must reject the vault with ValueError.
        corrupt = b"{not valid json"
        (self.root / MANIFEST_NAME).write_bytes(corrupt)
        disk_during_failure = self._all_disk_bytes()
        begin.set()

        fresh_results: multiprocessing.Queue = multiprocessing.Queue()
        openers = [
            multiprocessing.Process(
                target=_expect_open_value_error, args=(str(self.root), fresh_results)
            )
            for _ in range(3)
        ]
        for opener in openers:
            opener.start()
        self.fixture.track_process(*openers)
        for opener in openers:
            opener.join(timeout=30)
            self.assertEqual(opener.exitcode, 0)
        self.assertEqual(self.drain_nowait(fresh_results), ["ValueError"] * 3)

        # Let the reader finish its corrupt-window polling, then make sure it
        # reported nothing while waiting for recovery.
        time.sleep(7)
        self.assertEqual(self.drain_nowait(errors), [])

        # The failed opens/reloads changed nothing on disk, byte for byte.
        self.assertEqual(self._all_disk_bytes(), disk_during_failure)
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), corrupt)

        # Restore: the same handle reloads successfully and a fresh opener
        # sees the exact healthy content again.
        (self.root / MANIFEST_NAME).write_bytes(healthy_manifest)
        recovered.set()
        reader.join(timeout=30)
        self.assertEqual(reader.exitcode, 0)
        self.assertEqual(self.drain_nowait(errors), [])

        self.assertEqual(
            self._all_disk_bytes()[self.root / MANIFEST_NAME], healthy_manifest
        )
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), facts["versions"])
        self.assertEqual(reopened.active("k"), facts["active"])
        self.assertEqual(reopened.revoked_versions("k"), facts["revoked"])
        for version, material in facts["materials"].items():
            self.assertEqual(reopened.load("k", version), material)

    def test_repeated_failed_reload_queries_match_word_for_word(self):
        vault, facts = self._build_healthy_vault()
        # Snapshot the healthy answers, then corrupt the disk underneath a
        # handle that is already open.
        def answers():
            return (
                vault.versions("k"),
                vault.active("k"),
                tuple(vault.load("k", v) for v in facts["versions"]),
                vault.revoked_versions("k"),
                vault.revoked_versions("unknown-key"),
                vault.derivation("k", 4),
                vault.derivation("k", 1),
                vault.manifest(),
            )

        before = answers()
        (self.root / MANIFEST_NAME).write_bytes(b"garbage")

        for _ in range(5):
            with self.assertRaises(ValueError):
                vault.reload()
            self.assertEqual(answers(), before)
        # Disk gained nor lost any record.
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), b"garbage")
        self.assertEqual(
            len((self.root / REVOCATIONS_NAME).read_bytes().splitlines()), 1
        )
        self.assertEqual(
            len((self.root / ACTIVATIONS_NAME).read_bytes().splitlines()), 1
        )


# ---------------------------------------------------------------------------
# lock file carries no key data and takes no part in validation
# ---------------------------------------------------------------------------


class TestLockFileIsOnlyAMutex(ConcurrencyTestCase):
    def test_clearing_lock_file_does_not_make_reload_fail(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.seal("k", b"two")

        lock_path = self.root / LOCK_NAME
        lock_path.write_bytes(b"")
        vault.reload()
        self.assertEqual(vault.load("k", 2), b"two")

        # Filling it with arbitrary bytes changes nothing: the lock never
        # carries key data and never participates in validation.
        lock_path.write_bytes(b"this is not key material\n\xff")
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.load("k"), b"two")

        # A different process opens, writes a full record and validates.
        worker = multiprocessing.Process(
            target=_lock_content_worker, args=(str(self.root),)
        )
        self.fixture.track_process(worker)
        worker.start()
        worker.join(timeout=30)
        self.assertEqual(worker.exitcode, 0)

        final = self.open_vault()
        self.assertEqual(final.versions("k"), [1, 2, 3])
        self.assertEqual(final.load("k", 3), b"from-other-process")
        self.assertTrue(final.is_revoked("k", 1))
        final.reload()
        self.assertTrue(final.is_revoked("k", 1))

    def test_lock_file_bytes_never_appear_in_records(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        marker = b"lock-only-payload"
        (self.root / LOCK_NAME).write_bytes(marker)
        vault.reload()
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != LOCK_NAME:
                self.assertNotIn(marker, path.read_bytes())


# ---------------------------------------------------------------------------
# close(): idempotent release, immediate handoff, transparent reacquire
# ---------------------------------------------------------------------------


class TestCloseRelease(ConcurrencyTestCase):
    def test_another_process_takes_the_lock_immediately_after_close(self):
        closed = multiprocessing.Event()
        peer_done = multiprocessing.Event()
        results: multiprocessing.Queue = multiprocessing.Queue()
        # B opens first but parks until A closes, so it measures the handoff
        # latency directly and proves no stale handle blocks it.
        b = multiprocessing.Process(
            target=_close_dance_b,
            args=(str(self.root), "k", closed, peer_done, results),
        )
        a = multiprocessing.Process(
            target=_close_dance_a, args=(str(self.root), "k", closed, peer_done)
        )
        # A failed assertion in this case releases both handoff gates and
        # drains both processes through the single teardown.
        self.fixture.track_gate(closed, peer_done)
        self.fixture.track_process(a, b)
        b.start()
        a.start()
        for process in (a, b):
            process.join(timeout=40)
            self.assertEqual(process.exitcode, 0)
        self.assertTrue(peer_done.wait(timeout=5))
        version, elapsed = results.get(timeout=5)
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
        self.assertIsNone(vault.close())
        # Repeated release is observable only as an error-free no-op: never
        # rely on internal handle state.
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())

        # Every operation kind reopens the lock and behaves identically:
        # the correct answers are observable through the public interface.
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

        # The reacquired-lock writes landed durably and survive a new open.
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), [1, 2, 3])
        self.assertTrue(reopened.is_revoked("k", 1))
        self.assertEqual(reopened.active("k"), 2)
        self.assertEqual(reopened.load("k", 2), b"two")
        self.assertEqual(reopened.versions("d"), [1])


# ---------------------------------------------------------------------------
# in-process thread interleaving
# ---------------------------------------------------------------------------


class TestThreadedInterleaving(ConcurrencyTestCase):
    def test_threads_share_one_lock_and_all_records_stay_consistent(self):
        vault = self.open_vault()

        # Phase 1: concurrent seals allocate one strict sequence.
        thread_count, each = 6, 6
        barrier = threading.Barrier(thread_count)
        self.fixture.track_barrier(barrier)
        phase_a = {
            f"a-{t}-{i}".encode("utf-8")
            for t in range(thread_count)
            for i in range(each)
        }
        errors: list[BaseException] = []

        def seal_phase(thread_index: int) -> None:
            local = Vault(self.root)
            try:
                barrier.wait()
                for i in range(each):
                    local.seal("k", f"a-{thread_index}-{i}".encode("utf-8"))
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            finally:
                local.close()

        threads = [
            threading.Thread(target=seal_phase, args=(t,)) for t in range(thread_count)
        ]
        # The single teardown aborts the barrier and drains these threads
        # before returning any handle if an assertion fails midway.
        self.fixture.track_thread(*threads)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        vault.reload()  # the parent handle's snapshot predates the workers
        self.assertEqual(vault.versions("k"), list(range(1, thread_count * each + 1)))

        # Phase 2: seals, revokes, repoints and a reloading reader all hit
        # one shared vault directory concurrently.
        extra = 12
        phase_b = sorted(f"b-{i}".encode("utf-8") for i in range(extra))
        sealers_done = threading.Event()
        reader_errors: list[BaseException] = []

        def seal_slice(payloads: list[bytes]) -> None:
            local = Vault(self.root)
            try:
                for payload in payloads:
                    local.seal("k", payload)
            finally:
                local.close()

        def revoke_all() -> None:
            local = Vault(self.root)
            try:
                for candidate in list(range(1, thread_count * each + 1)) * 2:
                    try:
                        local.revoke("k", candidate)
                    except (ValueError, KeyError):
                        pass  # duplicate race, never an integrity failure
                    try:
                        local.reload()
                    except ValueError as exc:  # pragma: no cover - would be a bug
                        reader_errors.append(AssertionError(f"reload failed mid-flight: {exc}"))
            finally:
                local.close()

        def repoint() -> None:
            local = Vault(self.root)
            try:
                for target in (30, 5, 20, 10, 35, 2, 25, 15):
                    try:
                        local.set_active("k", target)
                    except (ValueError, KeyError):
                        pass  # revoked by another thread, already active, or not sealed yet
            finally:
                local.close()

        def reader() -> None:
            local = Vault(self.root)
            try:
                valid_materials = phase_a | set(phase_b)
                while not sealers_done.is_set():
                    local.reload()
                    versions = local.versions("k")
                    if versions != list(range(1, len(versions) + 1)):
                        reader_errors.append(AssertionError(f"torn versions: {versions}"))
                        return
                    manifest = local.manifest()["keys"]["k"]
                    if [r["version"] for r in manifest["versions"]] != versions:
                        reader_errors.append(AssertionError("manifest torn"))
                        return
                    # Repoints never touch the manifest: its active pointer is
                    # always the latest sealed version.
                    if manifest["active"] != versions[-1]:
                        reader_errors.append(AssertionError("manifest active not latest"))
                        return
                    if local.active("k") not in versions:
                        reader_errors.append(AssertionError("active pointer outside versions"))
                        return
                    for version in versions:
                        if local.load("k", version) not in valid_materials:
                            reader_errors.append(AssertionError("unexpected material"))
                            return
            finally:
                local.close()

        workers = [
            threading.Thread(target=seal_slice, args=(phase_b[0::2],)),
            threading.Thread(target=seal_slice, args=(phase_b[1::2],)),
            threading.Thread(target=revoke_all),
            threading.Thread(target=revoke_all),
            threading.Thread(target=repoint),
            threading.Thread(target=repoint),
            threading.Thread(target=reader),
        ]
        # The reader only exits its loop once ``sealers_done`` is set; treat
        # it as a gate so a failed assertion still lets the reader finish and
        # every thread drains before handles are returned.
        self.fixture.track_gate(sealers_done)
        self.fixture.track_thread(*workers)
        for worker in workers:
            worker.start()
        for worker in workers[:-1]:
            worker.join(timeout=30)
            self.assertFalse(worker.is_alive())
        sealers_done.set()
        workers[-1].join(timeout=30)
        self.assertFalse(workers[-1].is_alive())
        self.assertEqual(reader_errors, [])

        # A fresh opener validates both journals and all materials: a torn
        # append or half record anywhere would raise ValueError here.
        final = self.open_vault()
        final_count = thread_count * each + extra
        self.assertEqual(final.versions("k"), list(range(1, final_count + 1)))
        self.assertEqual(
            {final.load("k", v) for v in range(1, final_count + 1)},
            phase_a | set(phase_b),
        )
        self.assertIn(final.active("k"), final.versions("k"))
        # Every baseline version sealed before phase 2 ends up revoked.
        self.assertEqual(
            set(final.revoked_versions("k")),
            set(range(1, thread_count * each + 1)),
        )
        # The revocation journal holds whole, unique records only.
        revocations = self.journal_records()
        pairs = [(r["key_id"], r["version"]) for r in revocations]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(key_id == "k" for key_id, _ in pairs))
        # The activation journal holds whole well-formed records only.
        for record in self.journal_records(ACTIVATIONS_NAME):
            self.assertEqual(record["key_id"], "k")
            self.assertIsInstance(record["version"], int)
            self.assertIsInstance(record["latest"], int)


# ---------------------------------------------------------------------------
# public entry-point error contract
# ---------------------------------------------------------------------------


class TestEntryPointContract(ConcurrencyTestCase):
    def test_empty_key_id_rejected_at_every_entry_without_new_version(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        manifest_before = (self.root / MANIFEST_NAME).read_bytes()

        with self.assertRaises(ValueError):
            vault.seal("", b"m")
        with self.assertRaises(ValueError):
            vault.revoke("", 1)
        with self.assertRaises(ValueError):
            vault.set_active("", 1)
        with self.assertRaises(ValueError):
            vault.derive_seal("", b"pw", b"salt", 1, 1)
        with self.assertRaises(ValueError):
            vault.is_revoked("", 1)
        with self.assertRaises(ValueError):
            vault.revoked_versions("")
        with self.assertRaises(ValueError):
            vault.derivation("")

        # No entry produced a new version or a journal record.
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), manifest_before)
        self.assertEqual((self.root / REVOCATIONS_NAME).read_bytes(), b"")

    def test_material_must_be_bytes_like(self):
        vault = self.open_vault()
        for bad in ("text", 1, 1.5, None, [b"x"], {"k": b"v"}, object()):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.seal("k", bad)
        # Genuine bytes-like buffers are accepted and byte-exact.
        self.assertEqual(vault.seal("k", bytearray(b"array")), 1)
        self.assertEqual(vault.seal("k", memoryview(b"view")), 2)
        self.assertEqual(vault.load("k", 1), b"array")
        self.assertEqual(vault.load("k", 2), b"view")

    def test_password_and_salt_must_be_bytes(self):
        vault = self.open_vault()
        for bad in ("text", 1, None, [b"x"], bytearray(b"x"), memoryview(b"x")):
            with self.assertRaises(TypeError, msg=f"password={bad!r}"):
                vault.derive_seal("k", bad, b"salt", 1, 1)
            with self.assertRaises(TypeError, msg=f"salt={bad!r}"):
                vault.derive_seal("k", b"pw", bad, 1, 1)

    def test_iterations_and_length_must_be_true_positive_ints(self):
        vault = self.open_vault()
        # bool is an int subclass: it must not pass as a genuine int.
        for bad in (True, False, 1.0, 2.5, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=f"iterations={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(TypeError, msg=f"length={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)
        for bad in (0, -1, -1000):
            with self.assertRaises(ValueError, msg=f"iterations={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(ValueError, msg=f"length={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)
        # Empty salt is a ValueError, distinct from the type checks above.
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 1)
        self.assertEqual(vault.versions("k"), [])

    def test_unknown_key_and_missing_version_raise_key_error_everywhere(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)

        with self.assertRaises(KeyError):
            vault.load("ghost")
        with self.assertRaises(KeyError):
            vault.active("ghost")
        with self.assertRaises(KeyError):
            vault.revoke("ghost", 1)
        with self.assertRaises(KeyError):
            vault.set_active("ghost", 1)
        with self.assertRaises(KeyError):
            vault.is_revoked("ghost", 1)
        with self.assertRaises(KeyError):
            vault.derivation("ghost")

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

    def test_duplicate_revoke_and_repoint_at_active_are_value_errors(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        # v2 is active by default; repointing at the current active
        # version is an error.
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)

        # On a key with no revocation: repoint away, then repointing at the
        # newly active version is the same ValueError; repointing back works.
        vault.seal("r", b"r1")
        vault.seal("r", b"r2")
        vault.set_active("r", 1)
        with self.assertRaises(ValueError):
            vault.set_active("r", 1)
        vault.set_active("r", 2)
        with self.assertRaises(ValueError):
            vault.set_active("r", 2)
        self.assertEqual(vault.active("r"), 2)

    def test_revoked_versions_unknown_key_is_empty_but_empty_id_raises(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        self.assertEqual(vault.revoked_versions("never-sealed"), [])
        self.assertEqual(vault.revoked_versions("k"), [])
        with self.assertRaises(ValueError):
            vault.revoked_versions("")

    def test_rejected_derive_seal_creates_no_version_or_files(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        manifest_before = (self.root / MANIFEST_NAME).read_bytes()
        materials_before = sorted(
            path.read_bytes()
            for path in (self.root / MATERIALS_DIR).rglob("*.bin")
        )
        bad_calls = (
            lambda: vault.derive_seal("", b"pw", b"salt", 1, 1),
            lambda: vault.derive_seal("k", "pw", b"salt", 1, 1),
            lambda: vault.derive_seal("k", b"pw", "salt", 1, 1),
            lambda: vault.derive_seal("k", b"pw", b"", 1, 1),
            lambda: vault.derive_seal("k", b"pw", b"salt", True, 1),
            lambda: vault.derive_seal("k", b"pw", b"salt", 1, 0),
            lambda: vault.derive_seal("k", b"pw", b"salt", 1.5, 1),
        )
        for call in bad_calls:
            with self.assertRaises((TypeError, ValueError)):
                call()
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), manifest_before)
        self.assertEqual(
            sorted(
                path.read_bytes()
                for path in (self.root / MATERIALS_DIR).rglob("*.bin")
            ),
            materials_before,
        )
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.load("k"), b"v1")


if __name__ == "__main__":
    unittest.main()
