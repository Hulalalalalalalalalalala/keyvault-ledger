"""Visibility-boundary regression tests for whole-vault reload vs writers.

The baseline vault already implements cross-process mutual exclusion, the
whole-vault reload and every documented exception; this module adds no
product behaviour and changes no on-disk format.  It pins, against real
subprocesses and real threads racing inside one temporary vault directory,
the observable boundaries promised for that concurrency:

* a process (or thread) finishes one whole record -- a version with its
  material, or one complete journal line -- before releasing the lock, and
  every other writer waits until it holds the lock itself before starting
  its own record, so after the interleaving every key's version numbers are
  still strictly ascending, with no duplicate and no gap, and historical
  material once observed is never rewritten;

* a reload interleaved with seals, derives, revokes and repoints only ever
  resolves a query against the complete old records or the complete newly
  persisted ones: every observation is internally whole (contiguous
  versions 1..n, the snapshot agreeing with the persisted manifest, every
  material matching its digest, every derivation record visible together
  with its parameters) and visible history never shrinks or mixes;

* a reload whose whole-vault validation fails raises exactly ``ValueError``:
  the in-memory snapshot stays word-for-word what it was, keys already in
  hand stay readable, a cold opener on the same directory raises the same
  ``ValueError``, and the records on disk neither gain nor lose a byte;
  undoing the damage and reloading restores the vault, snapshot and disk
  records corresponding one to one;

* ``vault.lock`` is only a mutual-exclusion device: empty or arbitrary bytes
  in it never participate in validation and never appear in a record;

* returning the lock handle with ``close()`` is harmless to repeat, hands
  the same lock to waiting processes immediately, and later operations
  reopen and reacquire it transparently with completely unchanged
  behaviour;

* a writer that loses a race (a duplicate revocation, a repoint at the
  version that just became active) fails with ``ValueError`` and leaves not
  even half a record behind: the journal gains exactly one complete,
  newline-terminated line per winning call.

The three README subcommands keep their behaviour; their requirements are
out of scope here.  Every case reads and writes only inside a temporary
directory, is repeatable with identical results, and is drained, returned
and deleted by the shared single teardown::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import queue as queue_mod
import threading
import time
import unittest
from collections import Counter
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


def _seal_payloads(
    root: str,
    key_id: str,
    payloads: list[bytes],
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    if barrier is not None:
        barrier.wait()
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


def _plan_writer(
    root: str,
    key_id: str,
    plan: list[tuple],
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Run one fixed serial plan of mixed write operations."""
    if barrier is not None:
        barrier.wait()
    vault = Vault(root)
    try:
        for step in plan:
            kind = step[0]
            if kind == "seal":
                vault.seal(key_id, step[1])
            elif kind == "derive":
                _, password, salt, iterations, length = step
                vault.derive_seal(key_id, password, salt, iterations, length)
            elif kind == "revoke":
                vault.revoke(key_id, step[1])
            elif kind == "repoint":
                vault.set_active(key_id, step[1])
            else:  # pragma: no cover - guards the test itself
                raise AssertionError(f"unknown plan step {kind!r}")
    finally:
        vault.close()


def _single_attempt(
    root: str,
    kind: str,
    key_id: str,
    version: int,
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier",
) -> None:
    """All racers attempt exactly one write; exactly one may win."""
    try:
        barrier.wait()
        vault = Vault(root)
        try:
            if kind == "revoke":
                vault.revoke(key_id, version)
            elif kind == "repoint":
                vault.set_active(key_id, version)
            else:  # pragma: no cover - guards the test itself
                raise AssertionError(f"unknown attempt kind {kind!r}")
        finally:
            vault.close()
        results.put("ok")
    except ValueError:
        # The only legitimate loss: another racer completed the same record
        # first (duplicate revocation, or the target became active).
        results.put("ValueError")
    except BaseException as exc:  # any torn state would surface this way
        results.put(f"error: {exc!r}")


def _open_and_report(root: str, results: "multiprocessing.Queue") -> None:
    vault = None
    try:
        vault = Vault(root)
    except ValueError:
        results.put("ValueError")
        if vault is not None:
            vault.close()
        return
    vault.close()
    results.put("no-exception")


def _assert_complete_state(
    vault: Vault, key_ids: list[str], pinned: dict[str, dict[int, bytes]]
) -> None:
    """Assert one observation is a complete state, never a half mixture.

    Self-contained: it derives every expectation from the snapshot's own
    persisted manifest rather than from a payload schedule, so it can run in
    a process that knows nothing about the writers.  Contiguous 1..n
    versions, the snapshot listings agreeing with the manifest (whose
    active field is always the newest sealed version), a valid active
    pointer, a consistent revoked set, every material matching its digest,
    a derivation record visible exactly together with its parameters, and
    a version's material never changing once seen.
    """
    manifest = vault.manifest()
    for key_id in key_ids:
        versions = vault.versions(key_id)
        if versions != list(range(1, len(versions) + 1)):
            raise AssertionError(
                f"{key_id!r}: non-contiguous versions observed: {versions}"
            )
        if not versions:
            continue
        entry = manifest["keys"].get(key_id)
        if entry is None:
            raise AssertionError(f"{key_id!r}: missing from the snapshot manifest")
        records = entry["versions"]
        if [record["version"] for record in records] != versions:
            raise AssertionError(f"{key_id!r}: manifest listing disagrees with query")
        # Repoints live only in the activation journal: the persisted
        # manifest always names the newest sealed version as active.
        if entry["active"] != versions[-1]:
            raise AssertionError(f"{key_id!r}: manifest active is not the newest")
        active = vault.active(key_id)
        if active not in versions:
            raise AssertionError(f"{key_id!r}: active pointer {active} outside versions")
        revoked = vault.revoked_versions(key_id)
        if revoked != sorted(set(revoked)) or not set(revoked) <= set(versions):
            raise AssertionError(f"{key_id!r}: bad revoked listing {revoked}")
        for version in revoked:
            if not vault.is_revoked(key_id, version):
                raise AssertionError(f"{key_id!r}: revoked listing/status disagree")

        for record in records:
            version = record["version"]
            material = vault.load(key_id, version)
            if hashlib.sha256(material).hexdigest() != record["sha256"]:
                raise AssertionError(
                    f"{key_id!r} version {version}: material/digest disagreement"
                )
            parameters = vault.derivation(key_id, version)
            persisted = record.get("derivation")
            if persisted is None:
                if parameters != {}:
                    raise AssertionError(
                        f"{key_id!r} version {version}: parameters without a record"
                    )
            else:
                if parameters["iterations"] != persisted["iterations"]:
                    raise AssertionError(
                        f"{key_id!r} version {version}: half-visible iterations"
                    )
                if parameters["length"] != persisted["length"]:
                    raise AssertionError(
                        f"{key_id!r} version {version}: half-visible length"
                    )
                if len(material) != persisted["length"]:
                    raise AssertionError(
                        f"{key_id!r} version {version}: material/length disagreement"
                    )
                if parameters["salt"] != base64.b64decode(
                    persisted["salt"], validate=True
                ):
                    raise AssertionError(
                        f"{key_id!r} version {version}: half-visible salt"
                    )
            previous = pinned[key_id].get(version)
            if previous is None:
                pinned[key_id][version] = material
            elif previous != material:
                # Historical material is append-only and never rewritten.
                raise AssertionError(
                    f"{key_id!r} version {version}: historical material changed"
                )


def _visibility_observer(
    root: str,
    totals: dict[str, int],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Reload in a loop while writers interleave; every state must be whole.

    Visible version counts may only stay put or grow (never shrink) and may
    only rest on states that are complete; a half-old/half-new observation
    would fail :func:`_assert_complete_state`.
    """
    try:
        if barrier is not None:
            barrier.wait()
        vault = Vault(root)
        pinned = {key_id: {} for key_id in totals}
        last_counts = {key_id: 0 for key_id in totals}
        observations = {key_id: [] for key_id in totals}
        rounds = 0
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            vault.reload()
            rounds += 1
            _assert_complete_state(vault, list(totals), pinned)
            for key_id in totals:
                count = len(vault.versions(key_id))
                if count < last_counts[key_id]:
                    raise AssertionError(
                        f"{key_id!r}: visible history shrank "
                        f"{last_counts[key_id]} -> {count}"
                    )
                last_counts[key_id] = count
                observations[key_id].append(count)
            if all(last_counts[key_id] >= totals[key_id] for key_id in totals):
                break
        else:  # pragma: no cover - only on the 120s deadline
            raise AssertionError(f"observer stalled at {last_counts}, expected {totals}")
        vault.close()
        results.put(("ok", {"rounds": rounds, "observations": observations}))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _gated_old_state_reader(
    root: str,
    old_counts: dict[str, int],
    totals: dict[str, int],
    results: "multiprocessing.Queue",
    ready: "multiprocessing.Event",
) -> None:
    """Pin the complete old records, then keep reading after writers start.

    The parent starts no writer until this reader has reloaded and reported
    the complete old state (``ready``), so the first observation is
    deterministically the old one; every later observation must again be a
    complete prefix.
    """
    try:
        vault = Vault(root)
        pinned = {key_id: {} for key_id in old_counts}
        observations = {key_id: [] for key_id in old_counts}
        vault.reload()
        _assert_complete_state(vault, list(old_counts), pinned)
        for key_id, count in old_counts.items():
            if len(vault.versions(key_id)) != count:
                raise AssertionError(
                    f"{key_id!r}: expected the complete old state {count}"
                )
            observations[key_id].append(count)
        ready.set()

        deadline = time.monotonic() + 120
        last_counts = dict(old_counts)
        while time.monotonic() < deadline:
            vault.reload()
            _assert_complete_state(vault, list(old_counts), pinned)
            for key_id in old_counts:
                count = len(vault.versions(key_id))
                if count < last_counts[key_id]:
                    raise AssertionError(f"{key_id!r}: visible history shrank")
                last_counts[key_id] = count
                observations[key_id].append(count)
            if all(last_counts[key_id] >= totals[key_id] for key_id in old_counts):
                break
        else:  # pragma: no cover - only on the 120s deadline
            raise AssertionError("gated reader never reached the final totals")
        vault.close()
        results.put(("ok", observations))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


class VisibilityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.root = self.fixture.root

    def open_vault(self, root: Path | str | None = None) -> Vault:
        """Open a vault; its handle is returned by the single teardown."""
        return self.fixture.open(self.root if root is None else root)

    def drain_nowait(self, process_queue: "multiprocessing.Queue") -> list:
        items = []
        while True:
            try:
                items.append(process_queue.get_nowait())
            except queue_mod.Empty:
                return items

    def join_processes(
        self,
        processes: list[multiprocessing.Process],
        timeout: float = 150,
    ) -> None:
        self.fixture.track_process(*processes)
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout)
            if process.is_alive():
                process.kill()
                process.join(5)
                self.fail(f"worker {process.name} hung past {timeout}s")
            self.assertEqual(
                process.exitcode,
                0,
                f"worker {process.name} exited with {process.exitcode}",
            )

    def disk_manifest(self) -> dict:
        return json.loads(
            (self.root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )

    def journal_records(self, name: str) -> list[dict]:
        path = self.root / name
        if not path.exists():
            return []
        raw = path.read_bytes()
        if raw != b"" and not raw.endswith(b"\n"):
            self.fail(f"{name} holds a torn, non-terminated line")
        records = []
        for line in raw.splitlines():
            if not line:
                self.fail(f"{name} holds an empty record")
            record = json.loads(line)
            records.append(record)
        return records

    def assert_snapshot_matches_disk(self, vault: Vault) -> None:
        """Re-derive every observable answer straight from the disk records.

        No private state is consulted: this parses the manifest and both
        journals itself and requires the public snapshot to match one to one.
        """
        manifest = self.disk_manifest()
        self.assertEqual(vault.manifest(), manifest)

        revoked: dict[str, set[int]] = {}
        for record in self.journal_records(REVOCATIONS_NAME):
            self.assertEqual(set(record), {"key_id", "version"})
            self.assertIsInstance(record["key_id"], str)
            self.assertIsInstance(record["version"], int)
            revoked.setdefault(record["key_id"], set()).add(record["version"])

        last_repoint: dict[str, tuple[int, int]] = {}
        for record in self.journal_records(ACTIVATIONS_NAME):
            self.assertEqual(set(record), {"key_id", "version", "latest"})
            self.assertIsInstance(record["key_id"], str)
            self.assertIsInstance(record["version"], int)
            self.assertIsInstance(record["latest"], int)
            last_repoint[record["key_id"]] = (
                record["version"],
                record["latest"],
            )

        for key_id, entry in manifest["keys"].items():
            records = entry["versions"]
            versions = [record["version"] for record in records]
            self.assertEqual(versions, list(range(1, len(versions) + 1)))
            self.assertEqual(vault.versions(key_id), versions)
            self.assertEqual(entry["active"], versions[-1])
            for record in records:
                version = record["version"]
                data = (self.root / record["file"]).read_bytes()
                self.assertEqual(vault.load(key_id, version), data)
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), record["sha256"]
                )
                parameters = vault.derivation(key_id, version)
                if "derivation" in record:
                    persisted = record["derivation"]
                    self.assertEqual(
                        parameters["salt"],
                        base64.b64decode(persisted["salt"], validate=True),
                    )
                    self.assertEqual(
                        parameters["iterations"], persisted["iterations"]
                    )
                    self.assertEqual(parameters["length"], persisted["length"])
                else:
                    self.assertEqual(parameters, {})
            expected_active = versions[-1]
            repoint = last_repoint.get(key_id)
            if repoint is not None and repoint[1] == versions[-1]:
                expected_active = repoint[0]
            self.assertEqual(vault.active(key_id), expected_active)
            self.assertEqual(
                set(vault.revoked_versions(key_id)),
                revoked.get(key_id, set()),
            )


# ---------------------------------------------------------------------------
# real processes and threads interleaved: strict sequence, fixed history,
# and only whole visible states
# ---------------------------------------------------------------------------


class TestInterleavedVersionSequenceAndVisibility(VisibilityTestCase):
    def test_processes_and_threads_interleave_with_whole_records_only(self):
        salt, iterations, length = b"boundary-salt", 100, 16

        alpha_payloads = [
            f"alpha-{writer}-{index}".encode("utf-8")
            for writer in range(2)
            for index in range(6)
        ]
        beta_passwords = [f"beta-pw-{index}".encode("utf-8") for index in range(8)]
        beta_materials = {
            hashlib.pbkdf2_hmac(
                "sha256", password, salt, iterations, dklen=length
            )
            for password in beta_passwords
        }
        gamma_password = b"gamma-pw"
        gamma_derived = hashlib.pbkdf2_hmac(
            "sha256", gamma_password, salt, iterations, dklen=length
        )
        gamma_plan = [
            ("seal", b"gamma-1"),
            ("seal", b"gamma-2"),
            ("derive", gamma_password, salt, iterations, length),
            ("revoke", 1),
            ("seal", b"gamma-4"),
            ("repoint", 2),
            ("seal", b"gamma-5"),
        ]

        delta_seal_payloads = [
            f"delta-{thread}-{index}".encode("utf-8")
            for thread in range(3)
            for index in range(4)
        ]
        delta_passwords = [b"delta-pw-0", b"delta-pw-1"]
        delta_derived = {
            hashlib.pbkdf2_hmac(
                "sha256", password, salt, iterations, dklen=length
            )
            for password in delta_passwords
        }

        totals = {"alpha": 12, "beta": 8, "gamma": 5, "delta": 14}

        # Five process writers and two observers start at one barrier.
        barrier = multiprocessing.Barrier(7)
        # Aborted at teardown if an assertion fails before every party
        # arrives, so no worker waits on a missing party.
        self.fixture.track_barrier(barrier)
        workers = [
            multiprocessing.Process(
                target=_seal_payloads,
                args=(
                    str(self.root),
                    "alpha",
                    alpha_payloads[0:6],
                    barrier,
                ),
            ),
            multiprocessing.Process(
                target=_seal_payloads,
                args=(
                    str(self.root),
                    "alpha",
                    alpha_payloads[6:12],
                    barrier,
                ),
            ),
            multiprocessing.Process(
                target=_derive_payloads,
                args=(
                    str(self.root),
                    "beta",
                    beta_passwords[0:4],
                    salt,
                    iterations,
                    length,
                    barrier,
                ),
            ),
            multiprocessing.Process(
                target=_derive_payloads,
                args=(
                    str(self.root),
                    "beta",
                    beta_passwords[4:8],
                    salt,
                    iterations,
                    length,
                    barrier,
                ),
            ),
            multiprocessing.Process(
                target=_plan_writer,
                args=(str(self.root), "gamma", gamma_plan, barrier),
            ),
        ]
        results: multiprocessing.Queue = multiprocessing.Queue()
        observers = [
            multiprocessing.Process(
                target=_visibility_observer,
                args=(str(self.root), totals, results, barrier),
            )
            for _ in range(2)
        ]

        # In-process threads race the same directory on the fourth key:
        # three sealers and one deriver behind a thread barrier.
        thread_barrier = threading.Barrier(4)
        self.fixture.track_barrier(thread_barrier)
        thread_errors: list[BaseException] = []

        def thread_seals(payloads: list[bytes]) -> None:
            handle = Vault(self.root)
            try:
                thread_barrier.wait()
                for payload in payloads:
                    handle.seal("delta", payload)
            except BaseException as exc:  # pragma: no cover - surfaced below
                thread_errors.append(exc)
            finally:
                handle.close()

        def thread_derives() -> None:
            handle = Vault(self.root)
            try:
                thread_barrier.wait()
                for password in delta_passwords:
                    handle.derive_seal(
                        "delta", password, salt, iterations, length
                    )
            except BaseException as exc:  # pragma: no cover - surfaced below
                thread_errors.append(exc)
            finally:
                handle.close()

        threads = [
            threading.Thread(
                target=thread_seals,
                args=(delta_seal_payloads[thread * 4 : (thread + 1) * 4],),
            )
            for thread in range(3)
        ]
        threads.append(threading.Thread(target=thread_derives))
        self.fixture.track_thread(*threads)

        for thread in threads:
            thread.start()
        self.join_processes(workers + observers)
        for thread in threads:
            thread.join(timeout=60)
            self.assertFalse(thread.is_alive())
        self.assertEqual(thread_errors, [])

        # Both observers reached the complete totals and never reported a
        # torn, mixed or shrinking state.  Their first observation need not
        # be the empty/old prefix (the in-process threads do not share the
        # process barrier and may have already written); the deterministic
        # complete-old -> complete-new edge is pinned separately below.
        reports = self.drain_nowait(results)
        self.assertEqual(len(reports), 2, reports)
        for status, payload in reports:
            self.assertEqual(status, "ok", payload)
            self.assertGreaterEqual(payload["rounds"], 1)
            for key_id, total in totals.items():
                counts = payload["observations"][key_id]
                # Every observation rested on a complete state and ended at
                # the complete newly persisted one.
                self.assertEqual(counts[-1], total)
                self.assertEqual(counts, sorted(counts))
                self.assertTrue(all(0 <= count <= total for count in counts))

        # The final state re-derived from the persisted records: strict 1..n
        # sequences and snapshot/disk one to one.
        final = self.open_vault()
        final.reload()
        self.assert_snapshot_matches_disk(final)

        self.assertEqual(final.versions("alpha"), list(range(1, 13)))
        self.assertEqual(
            {final.load("alpha", v) for v in range(1, 13)},
            set(alpha_payloads),
        )
        self.assertEqual(final.versions("beta"), list(range(1, 9)))
        self.assertEqual(
            {final.load("beta", v) for v in range(1, 9)}, beta_materials
        )
        for version in range(1, 9):
            self.assertEqual(
                final.derivation("beta", version),
                {"salt": salt, "iterations": iterations, "length": length},
            )

        self.assertEqual(final.versions("gamma"), [1, 2, 3, 4, 5])
        self.assertEqual(final.load("gamma", 1), b"gamma-1")
        self.assertEqual(final.load("gamma", 2), b"gamma-2")
        self.assertEqual(final.load("gamma", 3), gamma_derived)
        self.assertEqual(final.load("gamma", 4), b"gamma-4")
        self.assertEqual(final.load("gamma", 5), b"gamma-5")
        self.assertEqual(final.revoked_versions("gamma"), [1])
        # The last seal superseded the repoint, exactly as documented.
        self.assertEqual(final.active("gamma"), 5)
        self.assertEqual(
            [
                record
                for record in self.journal_records(ACTIVATIONS_NAME)
                if record["key_id"] == "gamma"
            ],
            [{"key_id": "gamma", "version": 2, "latest": 4}],
        )
        self.assertEqual(
            [
                record
                for record in self.journal_records(REVOCATIONS_NAME)
                if record["key_id"] == "gamma"
            ],
            [{"key_id": "gamma", "version": 1}],
        )

        self.assertEqual(final.versions("delta"), list(range(1, 15)))
        self.assertEqual(
            {final.load("delta", v) for v in range(1, 15)},
            set(delta_seal_payloads) | delta_derived,
        )

        # No half-written record and no orphan material survived.
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])
        self.assertEqual(
            len(list((self.root / MATERIALS_DIR).rglob("*.bin"))),
            sum(totals.values()),
        )


# ---------------------------------------------------------------------------
# deterministic complete-old -> complete-new boundary
# ---------------------------------------------------------------------------


class TestCompleteOldToCompleteNewBoundary(VisibilityTestCase):
    def test_reader_pins_old_then_only_sees_complete_new_states(self):
        # Seed the complete old state: two keys, one plain and one derived.
        vault = self.open_vault()
        salt, iterations, length = b"gate-salt", 100, 16
        vault.seal("alpha", b"old-a-1")
        vault.seal("alpha", b"old-a-2")
        vault.derive_seal("beta", b"old-pw", salt, iterations, length)

        old_counts = {"alpha": 2, "beta": 1}
        new_alpha = [f"new-a-{i}".encode("utf-8") for i in range(3, 8)]
        new_beta_passwords = [f"new-pw-{i}".encode("utf-8") for i in range(2, 5)]
        new_beta_materials = [
            hashlib.pbkdf2_hmac("sha256", password, salt, iterations, dklen=length)
            for password in new_beta_passwords
        ]
        totals = {"alpha": 7, "beta": 4}

        ready = multiprocessing.Event()
        results: multiprocessing.Queue = multiprocessing.Queue()
        reader = multiprocessing.Process(
            target=_gated_old_state_reader,
            args=(str(self.root), old_counts, totals, results, ready),
        )
        # The single teardown drains the reader if an assertion fails while
        # it is still polling.
        self.fixture.track_process(reader)
        reader.start()
        self.assertTrue(ready.wait(timeout=15))

        # Only now, with the complete old state pinned, do the writers start.
        alpha_writer = multiprocessing.Process(
            target=_seal_payloads,
            args=(str(self.root), "alpha", new_alpha),
        )
        beta_writer = multiprocessing.Process(
            target=_derive_payloads,
            args=(
                str(self.root),
                "beta",
                new_beta_passwords,
                salt,
                iterations,
                length,
            ),
        )
        self.join_processes([alpha_writer, beta_writer])

        # Markers and the repoint land after the seals through the parent
        # handle, each as one complete record, so the reader still only
        # crosses complete states while it drains towards the totals.
        vault.reload()
        vault.revoke("alpha", 1)
        vault.set_active("alpha", 2)
        reader.join(timeout=60)
        self.assertEqual(reader.exitcode, 0)
        status, observations = self.drain_nowait(results)[0]
        self.assertEqual(status, "ok", observations)

        # The first observation of each key is deterministically the
        # complete old count; the last is the complete new count; nothing in
        # between was a half-old/half-new mixture (the reader validates every
        # observation), and visible counts never went backwards.
        for key_id, old_count in old_counts.items():
            counts = observations[key_id]
            self.assertEqual(counts[0], old_count)
            self.assertEqual(counts[-1], totals[key_id])
            self.assertEqual(counts, sorted(counts))

        final = self.open_vault()
        self.assert_snapshot_matches_disk(final)
        self.assertEqual(final.versions("alpha"), list(range(1, 8)))
        self.assertEqual(final.versions("beta"), list(range(1, 5)))
        self.assertEqual(final.load("alpha", 1), b"old-a-1")
        self.assertEqual(final.load("alpha", 2), b"old-a-2")
        self.assertEqual(final.active("alpha"), 2)
        self.assertTrue(final.is_revoked("alpha", 1))
        for index, material in zip(range(3, 8), new_alpha):
            self.assertEqual(final.load("alpha", index), material)
        for version, material in zip(range(2, 5), new_beta_materials):
            self.assertEqual(final.load("beta", version), material)


# ---------------------------------------------------------------------------
# failed whole-vault reload: the double invariance, then one-to-one recovery
# ---------------------------------------------------------------------------


class TestFailedReloadDoubleInvariance(VisibilityTestCase):
    def _record_file(self, key_id: str, version: int) -> Path:
        manifest = self.disk_manifest()
        record = next(
            record
            for record in manifest["keys"][key_id]["versions"]
            if record["version"] == version
        )
        return self.root / record["file"]

    def _disk_bytes(self) -> dict[str, bytes]:
        """All vault records keyed by relative path; the lock is excluded:
        it is only a mutual-exclusion device and carries no key data."""
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def _answers(self, vault: Vault) -> dict:
        answer = {"manifest": vault.manifest()}
        for key_id in ("k", "other"):
            versions = vault.versions(key_id)
            answer[key_id] = {
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
        answer["unknown"] = {
            "versions": vault.versions("never-sealed"),
            "revoked": vault.revoked_versions("never-sealed"),
        }
        return answer

    def _restore(self, healthy: dict[str, bytes]) -> None:
        for relative in self._disk_bytes().keys() - healthy.keys():
            (self.root / relative).unlink()
        for relative, data in healthy.items():
            path = self.root / relative
            if not path.exists() or path.read_bytes() != data:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

    def test_failed_reload_freezes_both_sides_then_recovers_one_to_one(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.seal("k", b"two")
        vault.seal("k", b"three")
        vault.derive_seal("k", b"passphrase", b"salt", 100, 16)
        vault.revoke("k", 1)
        vault.set_active("k", 2)
        vault.seal("other", b"other-one")

        expected = self._answers(vault)
        healthy_disk = self._disk_bytes()

        def tamper_material() -> None:
            path = self._record_file("k", 3)
            self.assertEqual(path.read_bytes(), b"three")
            path.write_bytes(b"THREE-tampered")

        def duplicate_revocation() -> None:
            with (self.root / REVOCATIONS_NAME).open("ab") as fh:
                fh.write(b'{"key_id": "k", "version": 1}\n')

        for name, damage in (
            ("material_mismatch", tamper_material),
            ("duplicate_revocation", duplicate_revocation),
        ):
            with self.subTest(case=name):
                self._restore(healthy_disk)
                vault.reload()
                self.assertEqual(self._answers(vault), expected)

                damage()
                failing_disk = self._disk_bytes()

                # Repeated failed reloads raise exactly ValueError and the
                # answers stay word-for-word what they were before.
                for _ in range(3):
                    with self.assertRaises(ValueError):
                        vault.reload()
                    self.assertEqual(self._answers(vault), expected)

                # Keys already in hand stay readable, including the active
                # version the snapshot was repointed at.
                self.assertEqual(vault.load("k"), b"two")
                self.assertEqual(vault.load("k", 4), expected["k"]["materials"][4])
                self.assertTrue(vault.is_revoked("k", 1))

                # A cold opener rejects the same state the same way.
                opener_results: multiprocessing.Queue = multiprocessing.Queue()
                opener = multiprocessing.Process(
                    target=_open_and_report,
                    args=(str(self.root), opener_results),
                )
                self.fixture.track_process(opener)
                opener.start()
                opener.join(timeout=30)
                self.assertEqual(opener.exitcode, 0)
                self.assertEqual(self.drain_nowait(opener_results), ["ValueError"])

                # The failed reloads and the rejected opener neither added
                # nor removed a disk record.
                self.assertEqual(self._disk_bytes(), failing_disk)

                # Undo the damage: one full reload restores normal service,
                # snapshot and disk records corresponding one to one.
                self._restore(healthy_disk)
                vault.reload()
                self.assertEqual(self._answers(vault), expected)
                self.assert_snapshot_matches_disk(vault)

                reopened = self.open_vault()
                reopened.reload()
                self.assertEqual(self._answers(reopened), expected)
                self.assert_snapshot_matches_disk(reopened)
                self.assertEqual(self._disk_bytes(), healthy_disk)


# ---------------------------------------------------------------------------
# vault.lock: mutual exclusion only, never key data, never validation
# ---------------------------------------------------------------------------


class TestLockFileIsOnlyMutex(VisibilityTestCase):
    def test_lock_bytes_never_participate_in_validation_or_records(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.seal("k", b"two")
        vault.seal("k", b"three")

        lock_path = self.root / LOCK_NAME
        # Empty, then arbitrary bytes (including non-UTF-8): the lock never
        # carries key data and plays no part in validation.
        lock_path.write_bytes(b"")
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        marker = b"not key material, not json either\n\xff\x00"
        lock_path.write_bytes(marker)
        vault.reload()
        self.assertEqual(vault.load("k"), b"three")

        # A different process opens, completes a whole record and validates
        # while the lock still holds arbitrary content.
        worker = multiprocessing.Process(
            target=_plan_writer,
            args=(
                str(self.root),
                "k",
                [("seal", b"four"), ("revoke", 1)],
                None,
            ),
        )
        self.join_processes([worker])
        lock_path.write_bytes(marker + b"more\n")
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2, 3, 4])
        self.assertEqual(vault.load("k", 4), b"four")
        self.assertTrue(vault.is_revoked("k", 1))

        # A cold opener validates the whole keyring and ignores the bytes.
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), [1, 2, 3, 4])
        self.assert_snapshot_matches_disk(reopened)

        # The marker never leaked into any persisted record.
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != LOCK_NAME:
                self.assertNotIn(b"not key material", path.read_bytes())


# ---------------------------------------------------------------------------
# close(): repeated release, immediate handoff, transparent reacquire
# ---------------------------------------------------------------------------


class TestCloseReleaseHandoff(VisibilityTestCase):
    def test_repeated_close_cycles_handoff_and_reacquire_unchanged(self):
        vault = self.open_vault()
        self.assertEqual(vault.seal("k", b"seed"), 1)

        salt, iterations, length = b"handoff-salt", 100, 16
        # Three rounds: the handle is returned (twice) while two subprocess
        # writers race for the same lock, then the same handle reacquires it
        # and continues the strict sequence.  Round 2 derives instead of
        # seals so every reacquiring write kind is exercised.
        for round_index in range(3):
            self.assertIsNone(vault.close())
            self.assertIsNone(vault.close())  # repeated release is harmless

            barrier = multiprocessing.Barrier(2)
            self.fixture.track_barrier(barrier)
            externals = [
                multiprocessing.Process(
                    target=_seal_payloads,
                    args=(
                        str(self.root),
                        "k",
                        [f"ext-{round_index}-{writer}".encode("utf-8")],
                        barrier,
                    ),
                )
                for writer in range(2)
            ]
            self.join_processes(externals)

            if round_index == 1:
                version = vault.derive_seal(
                    "k",
                    f"local-{round_index}".encode("utf-8"),
                    salt,
                    iterations,
                    length,
                )
            else:
                version = vault.seal("k", f"local-{round_index}".encode("utf-8"))
            # Externals finish two whole records each round before the local
            # handle reacquires, so its version is fixed at 4, 7, 10.
            self.assertEqual(version, 4 + 3 * round_index)
            vault.reload()
            self.assertEqual(
                vault.versions("k"), list(range(1, version + 1))
            )
            self.assertEqual(vault.active("k"), version)

        self.assertEqual(vault.versions("k"), list(range(1, 11)))
        for round_index in range(3):
            local_version = 4 + 3 * round_index
            if round_index == 1:
                self.assertEqual(
                    vault.load("k", local_version),
                    hashlib.pbkdf2_hmac(
                        "sha256",
                        f"local-{round_index}".encode("utf-8"),
                        salt,
                        iterations,
                        dklen=length,
                    ),
                )
                self.assertEqual(
                    vault.derivation("k", local_version),
                    {"salt": salt, "iterations": iterations, "length": length},
                )
            else:
                self.assertEqual(
                    vault.load("k", local_version),
                    f"local-{round_index}".encode("utf-8"),
                )
            external_versions = (2 + 3 * round_index, 3 + 3 * round_index)
            external_materials = {
                vault.load("k", version) for version in external_versions
            }
            self.assertEqual(
                external_materials,
                {
                    f"ext-{round_index}-{writer}".encode("utf-8")
                    for writer in range(2)
                },
            )
        self.assertEqual(vault.load("k", 1), b"seed")

        # Repeated release stays an error-free no-op; a fresh opener sees a
        # fully corresponding vault and the strict unbroken sequence.
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())
        self.assertIsNone(vault.close())
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), list(range(1, 11)))
        self.assert_snapshot_matches_disk(reopened)


# ---------------------------------------------------------------------------
# losing writers: exactly one whole record per record, never half a one
# ---------------------------------------------------------------------------


class TestLosingWritersLeaveNoHalfRecord(VisibilityTestCase):
    def _race_one_record(
        self,
        root: Path,
        *,
        kind: str,
        racers: int,
        use_processes: bool,
    ) -> list[str]:
        if use_processes:
            results: "multiprocessing.Queue" = multiprocessing.Queue()
            barrier = multiprocessing.Barrier(racers)
            self.fixture.track_barrier(barrier)
            processes = [
                multiprocessing.Process(
                    target=_single_attempt,
                    args=(str(root), kind, "k", 1, results, barrier),
                )
                for _ in range(racers)
            ]
            self.join_processes(processes)
            return sorted(self.drain_nowait(results))

        barrier = threading.Barrier(racers)
        self.fixture.track_barrier(barrier)
        lock = threading.Lock()
        outcomes: list[str] = []

        def attempt() -> None:
            handle = Vault(root)
            try:
                barrier.wait()
                try:
                    if kind == "revoke":
                        handle.revoke("k", 1)
                    else:
                        handle.set_active("k", 1)
                    outcome = "ok"
                except ValueError:
                    outcome = "ValueError"
                with lock:
                    outcomes.append(outcome)
            finally:
                handle.close()

        threads = [threading.Thread(target=attempt) for _ in range(racers)]
        self.fixture.track_thread(*threads)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive())
        return sorted(outcomes)

    def test_duplicate_revoke_race_leaves_one_complete_line(self):
        for use_processes in (False, True):
            with self.subTest(processes=use_processes):
                # Each mode gets a fresh vault directory so every sequence
                # starts at version 1 and the journal holds exactly one line.
                root = self.fixture.path(f"revoke-{use_processes}")
                vault = self.open_vault(root)
                vault.seal("k", b"v1")
                vault.seal("k", b"v2")

                outcomes = self._race_one_record(
                    root, kind="revoke", racers=6, use_processes=use_processes
                )
                # One winner, five losers; arrival order is deliberately not
                # part of the contract, only the multiset of outcomes is.
                self.assertEqual(Counter(outcomes), {"ok": 1, "ValueError": 5})

                # Exactly one complete, newline-terminated record landed;
                # the five losers left not even half a line behind.
                records = [
                    json.loads(line)
                    for line in (root / REVOCATIONS_NAME).read_text("utf-8").splitlines()
                ]
                self.assertEqual(records, [{"key_id": "k", "version": 1}])
                self.assertEqual(
                    (root / REVOCATIONS_NAME).read_bytes(),
                    b'{"key_id": "k", "version": 1}\n',
                )
                vault.reload()
                self.assertTrue(vault.is_revoked("k", 1))
                self.assertFalse(vault.is_revoked("k", 2))
                with self.assertRaises(ValueError):
                    vault.revoke("k", 1)

    def test_repoint_race_leaves_one_complete_line(self):
        for use_processes in (False, True):
            with self.subTest(processes=use_processes):
                root = self.fixture.path(f"repoint-{use_processes}")
                vault = self.open_vault(root)
                vault.seal("k", b"v1")
                vault.seal("k", b"v2")

                outcomes = self._race_one_record(
                    root, kind="repoint", racers=6, use_processes=use_processes
                )
                self.assertEqual(Counter(outcomes), {"ok": 1, "ValueError": 5})

                records = [
                    json.loads(line)
                    for line in (root / ACTIVATIONS_NAME).read_text("utf-8").splitlines()
                ]
                self.assertEqual(
                    records, [{"key_id": "k", "version": 1, "latest": 2}]
                )
                vault.reload()
                self.assertEqual(vault.active("k"), 1)
                self.assertEqual(vault.load("k"), b"v1")
                # The target now being active is the same ValueError, and the
                # failed repeat appends nothing.
                with self.assertRaises(ValueError):
                    vault.set_active("k", 1)
                records_after = [
                    json.loads(line)
                    for line in (root / ACTIVATIONS_NAME).read_text("utf-8").splitlines()
                ]
                self.assertEqual(records_after, records)


if __name__ == "__main__":
    unittest.main()
