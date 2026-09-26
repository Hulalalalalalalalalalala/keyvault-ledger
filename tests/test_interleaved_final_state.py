"""Deeper interleaving regression tests pinning the final state on disk.

The baseline vault capabilities and the existing concurrency cases are
already in place; this module only adds cases -- the product code, the
public interface and the three README subcommands are untouched.  Real
subprocesses and in-process threads interleave seals, passphrase-derived
seals, revocations, active-version repoints and whole-vault reloads
against one temporary vault directory, and every case pins the *final*
state by reading the persisted records back independently and comparing
them against the query results:

* after the interleaving, every key's version numbers are still a strict
  ``1..n`` sequence -- no duplicates, no gaps -- and every historical
  material is present, byte-for-byte what was sealed, never rewritten;

* a derived seal interleaved with whole-vault reloads is observed
  atomically: the derivation record and its material are either both
  visible or both absent, never one without the other;

* a repoint racing a seal lands on exactly one of the two serial
  outcomes (repoint-then-seal or seal-then-repoint), and each serial
  order is reachable by genuinely concurrent processes;

* a revocation racing a repoint at the same version allows both serial
  orders, and the losing call leaves not even half a record behind;

* a reload interleaved with writers only ever sees the complete old
  listing or the complete newly persisted records, never a half-old/
  half-new mixture;

* a reload that fails whole-vault validation raises ``ValueError``, the
  in-memory snapshot is frozen word for word and the keys already in
  hand stay readable; failed and rejected calls write nothing -- the
  on-disk manifest, journals and materials neither grow nor shrink;

* the entry-point error contract (empty key id -> ``ValueError``;
  non-bytes material/password/salt and non-genuine-integer versions or
  derivation parameters -> ``TypeError``, bools and floats included;
  empty salt or non-positive iteration count/length -> ``ValueError``;
  unknown key/version -> ``KeyError``; duplicate revocation, repointing
  at the active version or at a revoked version -> ``ValueError``) holds,
  and every rejected call leaves no record behind;

* repeating the same fixed script in two fresh vaults produces
  byte-identical records and identical answers.

Every case reads and writes only inside its own temporary directory,
uses real subprocesses/threads and the standard library only, drains its
workers and returns its handles through the shared fixture, and is
independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
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


def _mixed_seal_derive(
    root: str,
    key_id: str,
    payloads: list[bytes],
    passwords: list[bytes],
    salt: bytes,
    iterations: int,
    length: int,
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Alternate plain and derived seals: odd versions plain, even derived."""
    if barrier is not None:
        barrier.wait()
    vault = Vault(root)
    try:
        for payload, password in zip(payloads, passwords):
            vault.seal(key_id, payload)
            vault.derive_seal(key_id, password, salt, iterations, length)
    finally:
        vault.close()


def _churn_worker(
    root: str,
    key_id: str,
    candidates: list[int],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Revoke/repoint/reload in a loop; lost races are expected outcomes."""
    try:
        if barrier is not None:
            barrier.wait()
        vault = Vault(root)
        try:
            outcomes = []
            for version in candidates:
                try:
                    vault.revoke(key_id, version)
                    outcomes.append(("revoke", version, "ok"))
                except (ValueError, KeyError):
                    # A duplicate revocation, a repoint race or a version
                    # not sealed yet: an outcome, never an integrity failure.
                    outcomes.append(("revoke", version, "rejected"))
                try:
                    vault.set_active(key_id, version)
                    outcomes.append(("repoint", version, "ok"))
                except (ValueError, KeyError):
                    outcomes.append(("repoint", version, "rejected"))
                vault.reload()
        finally:
            vault.close()
        results.put(("ok", outcomes))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _perform(vault: Vault, op: str, key_id: str, arg) -> str:
    """One vault call; a lost race is an outcome, never an exception."""
    try:
        if op == "seal":
            vault.seal(key_id, arg)
        elif op == "revoke":
            vault.revoke(key_id, arg)
        elif op == "repoint":
            vault.set_active(key_id, arg)
        else:  # pragma: no cover - guarded by the callers
            raise AssertionError(f"unknown op {op!r}")
        return "ok"
    except ValueError:
        return "value-error"
    except KeyError:
        return "key-error"


def _gated_call(
    root: str,
    op: str,
    key_id: str,
    arg,
    parked: "multiprocessing.Event",
    gate: "multiprocessing.Event",
    results: "multiprocessing.Queue",
) -> None:
    """Open the vault, park until ``gate`` opens, then perform one call."""
    try:
        vault = Vault(root)
        try:
            parked.set()
            if not gate.wait(timeout=30):
                raise RuntimeError(f"{op} worker was never released")
            results.put((op, _perform(vault, op, key_id, arg)))
        finally:
            vault.close()
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", f"{op}: {exc!r}"))


def _raced_call(
    root: str,
    op: str,
    key_id: str,
    arg,
    barrier: "multiprocessing.Barrier",
    results: "multiprocessing.Queue",
) -> None:
    """Perform one call the moment ``barrier`` releases every racer."""
    try:
        barrier.wait()
        vault = Vault(root)
        try:
            results.put((op, _perform(vault, op, key_id, arg)))
        finally:
            vault.close()
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", f"{op}: {exc!r}"))


def _complete_state_observer(
    root: str,
    key_id: str,
    valid_materials: set[bytes],
    expected_params: dict,
    total: int,
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Reload in a loop: only complete old or complete new states may appear.

    Every observed state must be versions ``1..n`` (never shrinking), the
    manifest must agree with the snapshot, every material must be one of
    the sealed payloads and immutable across observations, and a
    derivation record must be visible exactly together with its material.
    A torn record anywhere would make the validating reload itself fail.
    """
    try:
        if barrier is not None:
            barrier.wait()
        pinned: dict[int, bytes] = {}
        seen = 0
        deadline = time.monotonic() + 60
        while seen < total:
            if time.monotonic() > deadline:
                raise AssertionError(f"observer stalled at {seen}, expected {total}")
            vault = Vault(root)
            vault.reload()
            visible = vault.versions(key_id)
            if visible != list(range(1, len(visible) + 1)):
                raise AssertionError(f"non-contiguous versions: {visible}")
            if len(visible) < seen:
                raise AssertionError(
                    f"visible history shrank {seen} -> {len(visible)}"
                )
            if visible:
                entry = vault.manifest()["keys"][key_id]
                records = {record["version"]: record for record in entry["versions"]}
                if sorted(records) != visible:
                    raise AssertionError("manifest listing does not match snapshot")
                if entry["active"] != visible[-1]:
                    raise AssertionError("manifest active is not the latest version")
                for version in visible:
                    record = records[version]
                    params = vault.derivation(key_id, version)
                    # Co-visibility: the derivation record and the material
                    # are either both there or both absent.
                    if ("derivation" in record) != (params != {}):
                        raise AssertionError(
                            f"version {version}: derivation record and "
                            "material not visible together"
                        )
                    material = vault.load(key_id, version)
                    if material not in valid_materials:
                        raise AssertionError(
                            f"version {version}: unexpected material"
                        )
                    if params:
                        if params != expected_params:
                            raise AssertionError(
                                f"version {version}: wrong derivation parameters"
                            )
                        if len(material) != params["length"]:
                            raise AssertionError(
                                f"version {version}: derived length mismatch"
                            )
                    previous = pinned.get(version)
                    if previous is not None and previous != material:
                        raise AssertionError(
                            f"version {version} material changed under reload"
                        )
                    pinned[version] = material
            seen = len(visible)
            vault.close()
        results.put(("ok", seen))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _reload_loop(
    root: str,
    rounds: int,
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Reload repeatedly; both journals must parse as whole records."""
    if barrier is not None:
        barrier.wait()
    for _ in range(rounds):
        vault = Vault(root)
        vault.reload()
        for name in (REVOCATIONS_NAME, ACTIVATIONS_NAME):
            path = Path(root) / name
            if not path.exists():
                continue
            raw = path.read_bytes()
            if raw != b"" and not raw.endswith(b"\n"):
                raise AssertionError(f"torn journal record in {name}")
            for line in raw.splitlines():
                if not line:
                    raise AssertionError(f"empty journal record in {name}")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise AssertionError(f"journal record not an object in {name}")
        vault.close()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class InterleavingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.root = self.fixture.root

    def open_vault(self, root: Path | str | None = None) -> Vault:
        """Open a vault; its handle is returned by the single teardown."""
        return self.fixture.open(root)

    def disk_records(self, root: Path | str) -> dict[str, bytes]:
        """Every persisted record as relative-path -> bytes.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, so it is deliberately excluded.
        """
        root = Path(root)
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def journal_records(self, root: Path | str, name: str) -> list[dict]:
        path = Path(root) / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_bytes().splitlines()]

    def run_gated_pair(self, root: Path, first: tuple, second: tuple) -> dict:
        """Interleave two single-call workers in a forced order.

        Both workers open the vault and park; the parent releases them one
        at a time in the given order, waiting for each outcome before
        releasing the next -- a genuine two-process interleaving whose
        serial order is pinned.  Returns ``{op: outcome}``.
        """
        results: multiprocessing.Queue = multiprocessing.Queue()
        workers = {}
        for op, key_id, arg in (first, second):
            parked = multiprocessing.Event()
            gate = multiprocessing.Event()
            # Released by the single teardown if an assertion fails while
            # the worker is parked, so the worker always drains.
            self.fixture.track_gate(gate)
            proc = multiprocessing.Process(
                target=_gated_call,
                args=(str(root), op, key_id, arg, parked, gate, results),
            )
            self.fixture.track_process(proc)
            workers[op] = (proc, parked, gate)
        for proc, _, _ in workers.values():
            proc.start()
        for _, parked, _ in workers.values():
            self.assertTrue(parked.wait(timeout=15), "worker never parked")
        outcomes = {}
        for op, _, _ in (first, second):
            workers[op][2].set()
            got_op, outcome = results.get(timeout=30)
            # A worker-side failure arrives as ("error", repr): mismatching
            # the expected op surfaces it here with the payload as context.
            self.assertEqual(got_op, op, outcome)
            outcomes[got_op] = outcome
        for proc, _, _ in workers.values():
            proc.join(timeout=30)
            self.assertEqual(proc.exitcode, 0)
        return outcomes

    def run_raced_pair(self, root: Path, first: tuple, second: tuple) -> list:
        """Release two single-call workers on one barrier; collect outcomes."""
        barrier = multiprocessing.Barrier(2)
        # Aborted by the single teardown if an assertion fails before both
        # parties arrive, so no worker waits forever.
        self.fixture.track_barrier(barrier)
        results: multiprocessing.Queue = multiprocessing.Queue()
        procs = [
            multiprocessing.Process(
                target=_raced_call,
                args=(str(root), op, key_id, arg, barrier, results),
            )
            for op, key_id, arg in (first, second)
        ]
        self.fixture.track_process(*procs)
        for proc in procs:
            proc.start()
        outcomes = [results.get(timeout=30), results.get(timeout=30)]
        for proc in procs:
            proc.join(timeout=30)
            self.assertEqual(proc.exitcode, 0)
        return sorted(outcomes)

    def assert_worker_results_ok(self, process_queue: "multiprocessing.Queue") -> list:
        """Drain a ``(status, payload)`` worker queue and fail on errors."""
        payloads = []
        while True:
            try:
                status, payload = process_queue.get_nowait()
            except queue_mod.Empty:
                return payloads
            self.assertEqual(status, "ok", payload)
            payloads.append(payload)

    # ------------------------------------------------------------------
    # the final-state cross-check: queries against the persisted records
    # ------------------------------------------------------------------

    def assert_final_state_matches_disk(self, vault: Vault, root: Path) -> None:
        """Re-derive every observable answer straight from the disk records.

        This never consults the vault's private state: it parses the
        manifest and both journals itself and requires the public snapshot
        to match exactly -- versions strictly increasing with no gaps,
        historical materials byte-for-byte their files, the active pointer
        what the append-only journals imply, the revoked set what the
        revocation journal records, and the materials directory holding
        exactly one file per recorded version, nothing extra.
        """
        root = Path(root)
        manifest = json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))
        self.assertEqual(vault.manifest(), manifest)

        revoked: dict[str, set[int]] = {}
        seen_revocations: set[tuple] = set()
        for line in (root / REVOCATIONS_NAME).read_bytes().splitlines():
            record = json.loads(line)
            marker = (record["key_id"], record["version"])
            # The append-only journal holds whole, unique records only.
            self.assertNotIn(marker, seen_revocations)
            seen_revocations.add(marker)
            revoked.setdefault(record["key_id"], set()).add(record["version"])

        last_repoint: dict[str, tuple[int, int]] = {}
        activations = root / ACTIVATIONS_NAME
        if activations.exists():
            for line in activations.read_bytes().splitlines():
                record = json.loads(line)
                last_repoint[record["key_id"]] = (
                    record["version"],
                    record["latest"],
                )

        material_files: set[str] = set()
        for key_id, entry in manifest["keys"].items():
            records = entry["versions"]
            versions = [record["version"] for record in records]
            # Strictly increasing, no duplicates, no gaps skipped on disk.
            self.assertEqual(versions, sorted(versions))
            self.assertEqual(len(versions), len(set(versions)))
            self.assertEqual(vault.versions(key_id), versions)
            # The manifest's active pointer is always the latest sealed
            # version; repoints live only in the activation journal.
            self.assertEqual(entry["active"], versions[-1])
            for record in records:
                version = record["version"]
                data = (root / record["file"]).read_bytes()
                material_files.add(str(Path(record["file"])))
                # The snapshot serves the exact bytes persisted on disk,
                # and those bytes match the manifest digest.
                self.assertEqual(vault.load(key_id, version), data)
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), record["sha256"]
                )
                params = vault.derivation(key_id, version)
                if "derivation" in record:
                    self.assertNotEqual(params, {})
                    self.assertEqual(
                        params["iterations"], record["derivation"]["iterations"]
                    )
                    self.assertEqual(
                        params["length"], record["derivation"]["length"]
                    )
                    self.assertEqual(len(data), params["length"])
                else:
                    self.assertEqual(params, {})
            expected_active = versions[-1]
            if key_id in last_repoint:
                target, bound_latest = last_repoint[key_id]
                if bound_latest == versions[-1]:
                    expected_active = target
            self.assertEqual(vault.active(key_id), expected_active)
            self.assertEqual(
                set(vault.revoked_versions(key_id)), revoked.get(key_id, set())
            )
            for version in versions:
                self.assertEqual(
                    vault.is_revoked(key_id, version),
                    version in revoked.get(key_id, set()),
                )

        # The materials directory holds exactly the recorded files: one per
        # version, none missing, none extra, no half-written temp file.
        on_disk = {
            str(path.relative_to(root))
            for path in (root / MATERIALS_DIR).rglob("*.bin")
        }
        self.assertEqual(on_disk, material_files)
        self.assertEqual(list(root.glob("**/*.tmp.*")), [])


# ---------------------------------------------------------------------------
# strict version sequences and intact history after full interleaving
# ---------------------------------------------------------------------------


class TestStrictSequenceAfterInterleaving(InterleavingTestCase):
    def test_processes_and_threads_leave_strict_sequences_and_intact_history(self):
        salt = b"strict-salt"
        iterations, length = 100, 16
        alpha_payloads = [
            f"alpha-{p}-{i}".encode("utf-8") for p in range(2) for i in range(5)
        ]
        beta_passwords = [
            f"beta-{p}-{i}".encode("utf-8") for p in range(2) for i in range(4)
        ]
        thread_alpha = [b"thread-alpha-0", b"thread-alpha-1", b"thread-alpha-2"]
        thread_beta = [b"thread-beta-0", b"thread-beta-1"]
        alpha_total = len(alpha_payloads) + len(thread_alpha)
        beta_total = len(beta_passwords) + len(thread_beta)

        barrier = multiprocessing.Barrier(5)  # 2 sealers + 2 derivers + 1 churn
        # Aborted by the single teardown if an assertion fails before every
        # party has shown up, so no worker waits for a missing party.
        self.fixture.track_barrier(barrier)
        churn_results: multiprocessing.Queue = multiprocessing.Queue()
        workers = [
            multiprocessing.Process(
                target=_seal_payloads,
                args=(str(self.root), "alpha", alpha_payloads[0:5], barrier),
            ),
            multiprocessing.Process(
                target=_seal_payloads,
                args=(str(self.root), "alpha", alpha_payloads[5:10], barrier),
            ),
            multiprocessing.Process(
                target=_derive_payloads,
                args=(str(self.root), "beta", beta_passwords[0:4], salt, iterations, length, barrier),
            ),
            multiprocessing.Process(
                target=_derive_payloads,
                args=(str(self.root), "beta", beta_passwords[4:8], salt, iterations, length, barrier),
            ),
            multiprocessing.Process(
                target=_churn_worker,
                args=(
                    str(self.root),
                    "alpha",
                    list(range(1, alpha_total + 1)),
                    churn_results,
                    barrier,
                ),
            ),
        ]
        # Drained by the single teardown even if an assertion below fails
        # while they are still running.
        self.fixture.track_process(*workers)
        for worker in workers:
            worker.start()

        # In-process threads interleave more seals and derived seals on the
        # same keys, plus a churn thread revoking/repointing/reloading.
        errors: list[BaseException] = []

        def thread_seals() -> None:
            local = Vault(self.root)
            try:
                for payload in thread_alpha:
                    local.seal("alpha", payload)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            finally:
                local.close()

        def thread_derives() -> None:
            local = Vault(self.root)
            try:
                for password in thread_beta:
                    local.derive_seal("beta", password, salt, iterations, length)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            finally:
                local.close()

        def thread_churn() -> None:
            local = Vault(self.root)
            try:
                for version in range(1, alpha_total + 1):
                    try:
                        local.revoke("alpha", version)
                    except (ValueError, KeyError):
                        pass  # lost race, never an integrity failure
                    try:
                        local.set_active("alpha", version)
                    except (ValueError, KeyError):
                        pass  # revoked, already active or not sealed yet
                    local.reload()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            finally:
                local.close()

        threads = [
            threading.Thread(target=thread_seals),
            threading.Thread(target=thread_derives),
            threading.Thread(target=thread_churn),
        ]
        # The single teardown drains these threads before returning any
        # handle if an assertion fails midway.
        self.fixture.track_thread(*threads)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        for worker in workers:
            worker.join(timeout=60)
            self.assertFalse(worker.is_alive())
            self.assertEqual(worker.exitcode, 0)
        self.assertEqual(len(self.assert_worker_results_ok(churn_results)), 1)

        # The final state, pinned against the persisted records.
        final = self.open_vault()
        self.assertEqual(final.versions("alpha"), list(range(1, alpha_total + 1)))
        self.assertEqual(final.versions("beta"), list(range(1, beta_total + 1)))

        # Historical material: one distinct payload per version slot, every
        # sealed payload present, nothing rewritten.
        alpha_materials = {final.load("alpha", v) for v in range(1, alpha_total + 1)}
        self.assertEqual(alpha_materials, set(alpha_payloads) | set(thread_alpha))
        self.assertEqual(len(alpha_materials), alpha_total)
        expected_beta = {
            hashlib.pbkdf2_hmac("sha256", password, salt, iterations, dklen=length)
            for password in beta_passwords + thread_beta
        }
        beta_materials = {final.load("beta", v) for v in range(1, beta_total + 1)}
        self.assertEqual(beta_materials, expected_beta)
        self.assertEqual(len(beta_materials), beta_total)

        # Derivation records: every beta version carries the exact
        # parameters, no alpha version carries any.
        for version in range(1, beta_total + 1):
            self.assertEqual(
                final.derivation("beta", version),
                {"salt": salt, "iterations": iterations, "length": length},
            )
        for version in range(1, alpha_total + 1):
            self.assertEqual(final.derivation("alpha", version), {})

        # The revocation journal holds whole, unique records only, and the
        # revoked set it implies is exactly what the queries report.
        journal = self.journal_records(self.root, REVOCATIONS_NAME)
        markers = {(record["key_id"], record["version"]) for record in journal}
        self.assertEqual(len(journal), len(markers))
        self.assertTrue(all(key_id == "alpha" for key_id, _ in markers))
        self.assertEqual(
            final.revoked_versions("alpha"),
            sorted(version for _, version in markers),
        )
        self.assertEqual(final.revoked_versions("beta"), [])

        # A full reload, then the independent disk cross-check: queries and
        # persisted records describe the very same final state.
        final.reload()
        self.assert_final_state_matches_disk(final, self.root)


# ---------------------------------------------------------------------------
# derived seal vs whole-vault reload: record and material co-visible
# ---------------------------------------------------------------------------


class TestDeriveSealReloadCoVisibility(InterleavingTestCase):
    def test_derivation_record_and_material_are_visible_together_or_not_at_all(self):
        self.open_vault()  # create the vault before the observer starts
        salt = b"co-visible-salt"
        iterations, length = 200, 24
        passwords = [f"pw-{i}".encode("utf-8") for i in range(10)]
        valid = {
            hashlib.pbkdf2_hmac("sha256", password, salt, iterations, dklen=length)
            for password in passwords
        }
        barrier = multiprocessing.Barrier(2)
        self.fixture.track_barrier(barrier)
        results: multiprocessing.Queue = multiprocessing.Queue()
        observer = multiprocessing.Process(
            target=_complete_state_observer,
            args=(
                str(self.root),
                "d",
                valid,
                {"salt": salt, "iterations": iterations, "length": length},
                len(passwords),
                results,
                barrier,
            ),
        )
        deriver = multiprocessing.Process(
            target=_derive_payloads,
            args=(str(self.root), "d", passwords, salt, iterations, length, barrier),
        )
        # Both processes are drained by the single teardown if an assertion
        # fails while they are still running.
        self.fixture.track_process(observer, deriver)
        observer.start()
        deriver.start()
        deriver.join(timeout=60)
        self.assertEqual(deriver.exitcode, 0)
        observer.join(timeout=60)
        self.assertEqual(observer.exitcode, 0)
        self.assertEqual(self.assert_worker_results_ok(results), [len(passwords)])

        # The final state: every version derived, parameters exact, material
        # byte-for-byte a fresh PBKDF2 of its password.
        final = self.open_vault()
        self.assertEqual(final.versions("d"), list(range(1, len(passwords) + 1)))
        for version in range(1, len(passwords) + 1):
            self.assertEqual(
                final.derivation("d", version),
                {"salt": salt, "iterations": iterations, "length": length},
            )
            material = final.load("d", version)
            self.assertIn(material, valid)
            self.assertEqual(len(material), length)
        self.assertEqual(
            {final.load("d", v) for v in range(1, len(passwords) + 1)}, valid
        )
        self.assert_final_state_matches_disk(final, self.root)


# ---------------------------------------------------------------------------
# reload vs writers: complete old listing or complete new records
# ---------------------------------------------------------------------------


class TestReloadSeesOnlyCompleteListings(InterleavingTestCase):
    def test_reload_during_mixed_writes_never_sees_a_half_new_state(self):
        self.open_vault()  # create the vault before the observers start
        salt = b"mixed-salt"
        iterations, length = 150, 20
        plain = [f"plain-{i}".encode("utf-8") for i in range(6)]
        passwords = [f"dpw-{i}".encode("utf-8") for i in range(6)]
        derived = [
            hashlib.pbkdf2_hmac("sha256", password, salt, iterations, dklen=length)
            for password in passwords
        ]
        valid = set(plain) | set(derived)
        total = len(plain) + len(passwords)

        barrier = multiprocessing.Barrier(3)
        self.fixture.track_barrier(barrier)
        results: multiprocessing.Queue = multiprocessing.Queue()
        writer = multiprocessing.Process(
            target=_mixed_seal_derive,
            args=(str(self.root), "m", plain, passwords, salt, iterations, length, barrier),
        )
        observer = multiprocessing.Process(
            target=_complete_state_observer,
            args=(
                str(self.root),
                "m",
                valid,
                {"salt": salt, "iterations": iterations, "length": length},
                total,
                results,
                barrier,
            ),
        )
        reloader = multiprocessing.Process(
            target=_reload_loop, args=(str(self.root), 30, barrier)
        )
        self.fixture.track_process(writer, observer, reloader)
        for proc in (writer, observer, reloader):
            proc.start()
        for proc in (writer, observer, reloader):
            proc.join(timeout=60)
            self.assertFalse(proc.is_alive())
            self.assertEqual(proc.exitcode, 0)
        self.assertEqual(self.assert_worker_results_ok(results), [total])

        # The single writer alternated strictly, so the final layout is
        # fully determined: odd versions plain, even versions derived.
        final = self.open_vault()
        self.assertEqual(final.versions("m"), list(range(1, total + 1)))
        for index in range(len(plain)):
            self.assertEqual(final.load("m", 2 * index + 1), plain[index])
            self.assertEqual(final.load("m", 2 * index + 2), derived[index])
            self.assertEqual(final.derivation("m", 2 * index + 1), {})
            self.assertEqual(
                final.derivation("m", 2 * index + 2),
                {"salt": salt, "iterations": iterations, "length": length},
            )
        self.assert_final_state_matches_disk(final, self.root)


# ---------------------------------------------------------------------------
# repoint vs seal: the final active version is one of two serial outcomes
# ---------------------------------------------------------------------------


class TestSetActiveVersusSeal(InterleavingTestCase):
    def _build_base(self, root: Path) -> Vault:
        vault = self.fixture.open(root)
        for index in range(1, 4):
            vault.seal("k", f"m-{index}".encode("utf-8"))
        return vault

    def test_each_forced_order_gives_its_own_final_state(self):
        # repoint -> seal: the seal supersedes the repoint, active is 4.
        root_a = self.fixture.path("repoint-then-seal")
        self._build_base(root_a)
        outcomes = self.run_gated_pair(
            root_a, ("repoint", "k", 1), ("seal", "k", b"m-4")
        )
        self.assertEqual(outcomes, {"repoint": "ok", "seal": "ok"})
        handle_a = self.fixture.open(root_a)
        self.assertEqual(handle_a.active("k"), 4)
        self.assertEqual(
            self.journal_records(root_a, ACTIVATIONS_NAME),
            [{"key_id": "k", "latest": 3, "version": 1}],
        )
        self.assert_final_state_matches_disk(handle_a, root_a)

        # seal -> repoint: the repoint binds to the new latest, active is 1.
        root_b = self.fixture.path("seal-then-repoint")
        self._build_base(root_b)
        outcomes = self.run_gated_pair(
            root_b, ("seal", "k", b"m-4"), ("repoint", "k", 1)
        )
        self.assertEqual(outcomes, {"seal": "ok", "repoint": "ok"})
        handle_b = self.fixture.open(root_b)
        self.assertEqual(handle_b.active("k"), 1)
        self.assertEqual(
            self.journal_records(root_b, ACTIVATIONS_NAME),
            [{"key_id": "k", "latest": 4, "version": 1}],
        )
        self.assert_final_state_matches_disk(handle_b, root_b)

        # The two serial outcomes are genuinely different final states.
        self.assertNotEqual(self.disk_records(root_a), self.disk_records(root_b))

    def test_raced_repoint_and_seal_lands_on_a_serial_outcome(self):
        # The two reference outcomes, produced serially in their own vaults.
        ref_a_root = self.fixture.path("ref-repoint-first")
        ref_a = self._build_base(ref_a_root)
        ref_a.set_active("k", 1)
        ref_a.seal("k", b"m-4")
        self.assertEqual(ref_a.active("k"), 4)
        ref_a_picture = (4, self.disk_records(ref_a_root))

        ref_b_root = self.fixture.path("ref-seal-first")
        ref_b = self._build_base(ref_b_root)
        ref_b.seal("k", b"m-4")
        ref_b.set_active("k", 1)
        self.assertEqual(ref_b.active("k"), 1)
        ref_b_picture = (1, self.disk_records(ref_b_root))

        for round_index in range(6):
            root = self.fixture.path(f"race-{round_index}")
            self._build_base(root)
            outcomes = self.run_raced_pair(
                root, ("repoint", "k", 1), ("seal", "k", b"m-4")
            )
            # Both calls always succeed here; only their order varies.
            self.assertEqual(outcomes, [("repoint", "ok"), ("seal", "ok")])
            handle = self.fixture.open(root)
            picture = (handle.active("k"), self.disk_records(root))
            # The final state is byte-for-byte one of the two serial
            # outcomes -- never a mixture of the two.
            self.assertIn(picture, (ref_a_picture, ref_b_picture))
            self.assert_final_state_matches_disk(handle, root)


# ---------------------------------------------------------------------------
# revoke vs repoint: both orders allowed, the loser leaves no record
# ---------------------------------------------------------------------------


class TestRevokeVersusSetActive(InterleavingTestCase):
    def _build_base(self, root: Path) -> Vault:
        vault = self.fixture.open(root)
        for index in range(1, 4):
            vault.seal("k", f"m-{index}".encode("utf-8"))
        return vault

    def test_both_orders_occur_and_the_loser_leaves_no_record(self):
        # repoint -> revoke: both succeed; revocation never moves the
        # pointer, so the revoked version stays the active one.
        root_a = self.fixture.path("repoint-then-revoke")
        self._build_base(root_a)
        outcomes = self.run_gated_pair(
            root_a, ("repoint", "k", 2), ("revoke", "k", 2)
        )
        self.assertEqual(outcomes, {"repoint": "ok", "revoke": "ok"})
        handle_a = self.fixture.open(root_a)
        self.assertEqual(handle_a.active("k"), 2)
        self.assertTrue(handle_a.is_revoked("k", 2))
        self.assertEqual(handle_a.revoked_versions("k"), [2])
        self.assertEqual(handle_a.load("k", 2), b"m-2")  # revoked stays readable
        self.assertEqual(
            self.journal_records(root_a, ACTIVATIONS_NAME),
            [{"key_id": "k", "latest": 3, "version": 2}],
        )
        self.assertEqual(
            self.journal_records(root_a, REVOCATIONS_NAME),
            [{"key_id": "k", "version": 2}],
        )
        picture_a = (2, self.disk_records(root_a))
        self.assert_final_state_matches_disk(handle_a, root_a)

        # revoke -> repoint: the repoint at a revoked version is rejected
        # with ValueError and leaves not even half a record behind.
        root_b = self.fixture.path("revoke-then-repoint")
        self._build_base(root_b)
        outcomes = self.run_gated_pair(
            root_b, ("revoke", "k", 2), ("repoint", "k", 2)
        )
        self.assertEqual(outcomes, {"revoke": "ok", "repoint": "value-error"})
        handle_b = self.fixture.open(root_b)
        self.assertEqual(handle_b.active("k"), 3)
        self.assertTrue(handle_b.is_revoked("k", 2))
        # The losing repoint wrote nothing: no activations journal at all,
        # and the revocation journal holds exactly one whole record.
        self.assertFalse((root_b / ACTIVATIONS_NAME).exists())
        self.assertEqual(
            self.journal_records(root_b, REVOCATIONS_NAME),
            [{"key_id": "k", "version": 2}],
        )
        picture_b = (3, self.disk_records(root_b))
        self.assertNotEqual(picture_a, picture_b)
        self.assert_final_state_matches_disk(handle_b, root_b)

        # Unforced races: every round lands on exactly one of the two
        # serial outcomes, and the journals always hold whole records.
        for round_index in range(6):
            root = self.fixture.path(f"race-{round_index}")
            self._build_base(root)
            outcomes = self.run_raced_pair(
                root, ("repoint", "k", 2), ("revoke", "k", 2)
            )
            handle = self.fixture.open(root)
            picture = (handle.active("k"), self.disk_records(root))
            if outcomes == [("repoint", "ok"), ("revoke", "ok")]:
                self.assertEqual(picture, picture_a)
            else:
                self.assertEqual(
                    outcomes, [("repoint", "value-error"), ("revoke", "ok")]
                )
                self.assertEqual(picture, picture_b)
            self.assert_final_state_matches_disk(handle, root)


# ---------------------------------------------------------------------------
# failed reload: snapshot frozen, disk untouched, keys still readable
# ---------------------------------------------------------------------------


class TestFailedReloadKeepsSnapshotAndDisk(InterleavingTestCase):
    def test_value_error_freezes_snapshot_and_disk_while_keys_stay_readable(self):
        vault = self.open_vault()
        vault.seal("k", b"one")
        vault.seal("k", b"two")
        vault.derive_seal("k", b"pw", b"freeze-salt", 300, 24)
        vault.revoke("k", 1)
        vault.set_active("k", 2)

        def answers() -> tuple:
            return (
                vault.versions("k"),
                vault.active("k"),
                tuple(vault.load("k", v) for v in (1, 2, 3)),
                vault.revoked_versions("k"),
                vault.is_revoked("k", 1),
                vault.is_revoked("k", 2),
                vault.derivation("k", 3),
                vault.derivation("k", 1),
                vault.manifest(),
            )

        before = answers()
        healthy_disk = self.disk_records(self.root)

        corrupt = b"{ not valid json"
        (self.root / MANIFEST_NAME).write_bytes(corrupt)
        corrupt_disk = self.disk_records(self.root)

        message = None
        for _ in range(3):
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            if message is None:
                message = str(caught.exception)
            else:
                # Repeating the identical failing input gives the identical
                # complaint, word for word.
                self.assertEqual(str(caught.exception), message)
            # The in-memory snapshot is frozen word for word and the keys
            # already in hand stay readable.
            self.assertEqual(answers(), before)

        # The failed reloads wrote nothing: the disk is byte-for-byte the
        # corrupted state, neither grown nor shrunk.
        self.assertEqual(self.disk_records(self.root), corrupt_disk)
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), corrupt)

        # Restore: one successful reload restores full correspondence
        # between the snapshot and the persisted records.
        (self.root / MANIFEST_NAME).write_bytes(healthy_disk[MANIFEST_NAME])
        vault.reload()
        self.assertEqual(answers(), before)
        self.assertEqual(self.disk_records(self.root), healthy_disk)
        self.assert_final_state_matches_disk(vault, self.root)


# ---------------------------------------------------------------------------
# rejected calls: exact exception types, not a single record written
# ---------------------------------------------------------------------------


class TestRejectedCallsLeaveNoRecord(InterleavingTestCase):
    def test_every_rejected_call_raises_its_type_and_disk_is_untouched(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        vault.revoke("k", 1)

        disk_before = self.disk_records(self.root)

        def answers() -> tuple:
            return (
                vault.versions("k"),
                vault.versions("d"),
                vault.active("k"),
                vault.active("d"),
                vault.revoked_versions("k"),
                vault.load("k", 1),
                vault.load("k", 2),
                vault.derivation("d", 1),
                vault.manifest(),
            )

        answers_before = answers()

        # Empty key id -> ValueError at every entry point.
        for call in (
            lambda: vault.seal("", b"m"),
            lambda: vault.derive_seal("", b"pw", b"salt", 1, 1),
            lambda: vault.load("", 1),
            lambda: vault.revoke("", 2),
            lambda: vault.set_active("", 2),
            lambda: vault.is_revoked("", 2),
            lambda: vault.revoked_versions(""),
            lambda: vault.derivation("", 1),
        ):
            with self.assertRaises(ValueError):
                call()

        # Non-bytes-like material, non-bytes password or salt -> TypeError.
        for bad in ("text", 1, 1.5, None, [b"x"], {"k": b"v"}, object()):
            with self.assertRaises(TypeError, msg=f"material={bad!r}"):
                vault.seal("k", bad)
        for bad in ("text", 1, None, bytearray(b"x"), memoryview(b"x")):
            with self.assertRaises(TypeError, msg=f"password={bad!r}"):
                vault.derive_seal("k", bad, b"salt", 1, 1)
            with self.assertRaises(TypeError, msg=f"salt={bad!r}"):
                vault.derive_seal("k", b"pw", bad, 1, 1)

        # Non-genuine-integer versions -> TypeError; bools and floats do
        # not count.  ``None`` is the active-version sentinel of ``load``
        # and ``derivation`` only, so it is rejected just like the others
        # at the entries that have no sentinel.
        for bad in (1.0, 2.5, True, False, "1", (1,)):
            with self.assertRaises(TypeError, msg=f"load {bad!r}"):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("k", bad)
        for bad in (1.0, 2.5, True, False, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("k", bad)
            with self.assertRaises(TypeError, msg=f"revoke {bad!r}"):
                vault.revoke("k", bad)
            with self.assertRaises(TypeError, msg=f"set_active {bad!r}"):
                vault.set_active("k", bad)
        for bad in (True, False, 1.0, 2.5, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=f"iterations={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(TypeError, msg=f"length={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)

        # Empty salt, non-positive iteration count or length -> ValueError.
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 1)
        for bad in (0, -1, -1000):
            with self.assertRaises(ValueError, msg=f"iterations={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(ValueError, msg=f"length={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)

        # Unknown key or version -> KeyError.
        for call in (
            lambda: vault.load("ghost"),
            lambda: vault.active("ghost"),
            lambda: vault.revoke("ghost", 1),
            lambda: vault.set_active("ghost", 1),
            lambda: vault.is_revoked("ghost", 1),
            lambda: vault.derivation("ghost"),
            lambda: vault.load("k", 99),
            lambda: vault.revoke("k", 99),
            lambda: vault.set_active("k", 99),
            lambda: vault.is_revoked("k", 99),
            lambda: vault.derivation("d", 99),
        ):
            with self.assertRaises(KeyError):
                call()

        # Duplicate revocation, repointing at the current active version
        # and repointing at a revoked version -> ValueError.
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)  # 2 is the active version
        with self.assertRaises(ValueError):
            vault.set_active("k", 1)  # 1 is revoked

        # Not one rejected call wrote a record: the manifest, both journals
        # and every material are byte-for-byte what they were -- the
        # activations journal was never even created -- and the in-memory
        # answers are unchanged.
        self.assertEqual(self.disk_records(self.root), disk_before)
        self.assertFalse((self.root / ACTIVATIONS_NAME).exists())
        self.assertEqual(answers(), answers_before)

        # A full reload stays healthy and answers identically afterwards.
        vault.reload()
        self.assertEqual(answers(), answers_before)
        self.assert_final_state_matches_disk(vault, self.root)


# ---------------------------------------------------------------------------
# determinism: the same script replayed gives byte-identical records
# ---------------------------------------------------------------------------


class TestDeterministicReplay(InterleavingTestCase):
    def test_same_script_in_two_fresh_vaults_gives_byte_identical_records(self):
        def script(root: Path) -> Vault:
            vault = self.fixture.open(root)
            vault.seal("k", b"alpha")
            vault.derive_seal("k", b"pw-1", b"replay-salt", 120, 16)
            vault.seal("k", b"omega")
            vault.revoke("k", 1)
            vault.set_active("k", 2)
            vault.reload()
            vault.seal("other", b"other-one")
            vault.derive_seal("other", b"pw-2", b"replay-salt-2", 130, 20)
            vault.revoke("other", 1)
            vault.reload()
            return vault

        root_a = self.fixture.path("replay-a")
        root_b = self.fixture.path("replay-b")
        first = script(root_a)
        second = script(root_b)

        # The persisted records are byte-for-byte identical.
        self.assertEqual(self.disk_records(root_a), self.disk_records(root_b))

        def answers(vault: Vault) -> tuple:
            return (
                vault.versions("k"),
                vault.active("k"),
                tuple(vault.load("k", v) for v in (1, 2, 3)),
                vault.revoked_versions("k"),
                vault.derivation("k", 2),
                vault.derivation("k", 1),
                vault.versions("other"),
                vault.active("other"),
                vault.revoked_versions("other"),
                vault.manifest(),
            )

        # So are the observable answers, and repeating the reads -- before
        # and after another reload -- changes none of them.
        self.assertEqual(answers(first), answers(second))
        self.assertEqual(answers(first), answers(first))
        first.reload()
        second.reload()
        self.assertEqual(answers(first), answers(second))
        self.assert_final_state_matches_disk(first, root_a)
        self.assert_final_state_matches_disk(second, root_b)


if __name__ == "__main__":
    unittest.main()
