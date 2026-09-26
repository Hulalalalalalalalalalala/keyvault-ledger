"""Deeper interleaving regression tests: the final state after real
subprocesses and threads race through one vault directory.

The baseline vault capabilities, the exception vocabulary and the three CLI
entry points are already implemented and are intentionally not touched here.
These cases only freeze observable behaviour.  Every case works in a
temporary vault directory and drives real ``multiprocessing`` subprocesses
and in-process threads through interleaved ``seal`` / ``derive_seal`` /
``revoke`` / ``set_active`` / ``reload`` calls, then pins the final state by
comparing the persisted records (manifest, both journals, material files)
against the query results:

* after the interleaving, every key's version numbers are still a strict
  1..n sequence -- no duplicates, no gaps -- and every historical material
  reads back byte for byte, never rewritten;

* a ``derive_seal`` interleaved with whole-vault reloads is atomic to any
  observer: the derivation record and its material are either both fully
  visible or neither exists yet;

* a ``set_active`` racing a ``seal`` leaves the active pointer on exactly
  one of the two outcomes the two serial orders would give, and the
  persisted journal record says which one;

* a ``revoke`` racing a ``set_active`` at the same version lands in exactly
  one of the two serial orders; both orders are reachable, and the losing
  call leaves not even half a record behind;

* a reload interleaved with writes only ever observes the complete old
  listing or the complete newly persisted records, never a half-old /
  half-new mixture;

* a whole-vault reload that fails validation raises ``ValueError`` while
  the in-memory snapshot stays verbatim and keys already in hand remain
  readable; calls that fail or are blocked behind the lock during the
  failure window write nothing -- the on-disk listing and materials neither
  grow nor shrink;

* the entry-point error contract (empty key id -> ``ValueError``;
  non-bytes material/passphrase/salt -> ``TypeError``; non-genuine-integer
  version or derivation parameter -> ``TypeError``, bools and floats
  included; empty salt or non-positive iterations/length -> ``ValueError``;
  unknown key/version -> ``KeyError``; duplicate revocation, repointing at
  the active or a revoked version -> ``ValueError``) holds and every
  rejected call leaves the disk and the snapshot untouched.

The three README subcommands keep their behaviour; their requirements are
out of scope here.  Everything runs inside temporary directories, is
independent of execution order and is drained, returned and deleted by the
shared fixture teardown::

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


def _pbkdf2(password: bytes, salt: bytes, iterations: int, length: int) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password, salt, iterations, dklen=length
    )


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


def _revoke_exact(
    root: str,
    key_id: str,
    versions: list[int],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Revoke pre-seeded versions; every call must succeed exactly once."""
    try:
        if barrier is not None:
            barrier.wait()
        vault = Vault(root)
        try:
            for version in versions:
                vault.revoke(key_id, version)
        finally:
            vault.close()
        results.put(("ok", list(versions)))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _repoint_attempts(
    root: str,
    key_id: str,
    targets: list[int],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Repoint at pre-seeded targets; only ``ValueError`` may legitimately
    lose (the target became the active version between two attempts)."""
    try:
        if barrier is not None:
            barrier.wait()
        vault = Vault(root)
        outcomes = []
        try:
            for target in targets:
                try:
                    vault.set_active(key_id, target)
                    outcomes.append("ok")
                except ValueError:
                    outcomes.append("ValueError")
        finally:
            vault.close()
        results.put(("ok", outcomes))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _one_shot(
    root: str,
    kind: str,
    key_id: str,
    argument,
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Perform exactly one call and report how it ended."""
    try:
        if barrier is not None:
            barrier.wait()
        vault = Vault(root)
        try:
            if kind == "seal":
                vault.seal(key_id, argument)
            elif kind == "repoint":
                vault.set_active(key_id, argument)
            elif kind == "revoke":
                vault.revoke(key_id, argument)
            else:  # pragma: no cover - guards the test itself
                raise AssertionError(f"unknown one-shot kind {kind!r}")
        finally:
            vault.close()
        results.put((kind, "ok"))
    except BaseException as exc:
        results.put((kind, type(exc).__name__))


def _check_snapshot_consistent(vault: Vault, key_ids: list[str]) -> None:
    """Validate one reloaded snapshot against itself, never against a fresh
    disk read (which could belong to a newer state than the snapshot).

    Every visible state must be a complete one: contiguous versions 1..n,
    the snapshot manifest agreeing with the queries, every material loading
    and matching its digest, and a derivation record visible exactly
    together with its parameters -- never one without the other.
    """
    manifest = vault.manifest()
    for key_id in key_ids:
        versions = vault.versions(key_id)
        if versions != list(range(1, len(versions) + 1)):
            raise AssertionError(f"non-contiguous versions for {key_id!r}: {versions}")
        if not versions:
            continue
        entry = manifest["keys"].get(key_id)
        if entry is None:
            raise AssertionError(f"{key_id!r} missing from the snapshot manifest")
        records = entry["versions"]
        if [record["version"] for record in records] != versions:
            raise AssertionError(f"{key_id!r}: manifest listing does not match")
        # Repoints never touch the manifest: its active field is always the
        # newest sealed version.
        if entry["active"] != versions[-1]:
            raise AssertionError(f"{key_id!r}: manifest active not the newest")
        active = vault.active(key_id)
        if active not in versions:
            raise AssertionError(f"{key_id!r}: active {active} outside versions")
        revoked = vault.revoked_versions(key_id)
        if revoked != sorted(set(revoked)) or not set(revoked) <= set(versions):
            raise AssertionError(f"{key_id!r}: bad revoked listing {revoked}")
        for record in records:
            version = record["version"]
            material = vault.load(key_id, version)
            if hashlib.sha256(material).hexdigest() != record["sha256"]:
                raise AssertionError(f"{key_id!r} version {version}: digest mismatch")
            parameters = vault.derivation(key_id, version)
            persisted = record.get("derivation")
            if persisted is None:
                if parameters != {}:
                    raise AssertionError(
                        f"{key_id!r} version {version}: derivation parameters "
                        "without a persisted record"
                    )
            else:
                # The record and its parameters/material are visible as one
                # unit; a half-visible derive would trip one of these.
                if parameters.get("iterations") != persisted["iterations"]:
                    raise AssertionError(
                        f"{key_id!r} version {version}: iterations mismatch"
                    )
                if parameters.get("length") != persisted["length"]:
                    raise AssertionError(
                        f"{key_id!r} version {version}: length mismatch"
                    )
                if len(material) != persisted["length"]:
                    raise AssertionError(
                        f"{key_id!r} version {version}: material/length mismatch"
                    )


def _reload_observer(
    root: str,
    key_ids: list[str],
    totals: dict[str, int],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Reload in a loop while writers interleave; every observed state must
    be a complete old or complete new one, never a mixture."""
    try:
        if barrier is not None:
            barrier.wait()
        vault = Vault(root)
        rounds = 0
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                vault.reload()
                rounds += 1
                _check_snapshot_consistent(vault, key_ids)
                if all(len(vault.versions(k)) >= totals[k] for k in key_ids):
                    break
        finally:
            vault.close()
        results.put(("ok", rounds))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _derive_observer(
    root: str,
    key_id: str,
    salt: bytes,
    iterations: int,
    length: int,
    expected_materials: set[bytes],
    total: int,
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Watch derive_seal race reloads: a derivation record and its material
    must be visible together, or the version must not exist at all."""
    try:
        if barrier is not None:
            barrier.wait()
        vault = Vault(root)
        seen = 0
        try:
            deadline = time.monotonic() + 60
            while True:
                vault.reload()
                versions = vault.versions(key_id)
                if versions != list(range(1, len(versions) + 1)):
                    raise AssertionError(f"non-contiguous versions: {versions}")
                for version in versions:
                    parameters = vault.derivation(key_id, version)
                    if parameters != {
                        "salt": salt,
                        "iterations": iterations,
                        "length": length,
                    }:
                        raise AssertionError(
                            f"version {version}: parameters {parameters!r} not "
                            "visible together with the record"
                        )
                    material = vault.load(key_id, version)
                    if material not in expected_materials:
                        raise AssertionError(
                            f"version {version}: material not one of the "
                            "derived payloads"
                        )
                    if len(material) != length:
                        raise AssertionError(
                            f"version {version}: material shorter than declared"
                        )
                seen = len(versions)
                if seen >= total:
                    break
                if time.monotonic() > deadline:
                    raise AssertionError(f"observer stalled at {seen} of {total}")
        finally:
            vault.close()
        results.put(("ok", seen))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _mixed_plan_writer(
    root: str,
    key_id: str,
    plan: list[tuple],
    results: "multiprocessing.Queue",
    barrier: "multiprocessing.managers.Barrier | None" = None,
) -> None:
    """Run a fixed sequence of seal/derive/revoke/repoint operations."""
    try:
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
        results.put(("ok", len(plan)))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))


def _lock_holder(
    root: str,
    ready: "multiprocessing.Event",
    release: "multiprocessing.Event",
) -> None:
    """Hold the inter-process lock (the vault's own helpers) until released."""
    vault = Vault(root)
    with vault._lock:
        with vault._file_lock():
            ready.set()
            if not release.wait(timeout=30):
                raise RuntimeError("holder was never released")
    vault.close()


def _seal_after_gate(
    root: str,
    key_id: str,
    payload: bytes,
    opened: "multiprocessing.Event",
    go: "multiprocessing.Event",
    results: "multiprocessing.Queue",
) -> None:
    """Open a handle while the vault is healthy, park, then seal on cue and
    report the exact outcome type."""
    vault = Vault(root)
    opened.set()
    if not go.wait(timeout=30):
        vault.close()
        results.put(("error", "gate was never released"))
        return
    try:
        vault.seal(key_id, payload)
    except BaseException as exc:
        results.put((type(exc).__name__,))
    else:
        results.put(("ok",))
    finally:
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


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class InterleavingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.root = self.fixture.root

    def open_vault(self) -> Vault:
        """Open a vault; its handle is returned by the single teardown."""
        return self.fixture.open()

    def disk_manifest(self) -> dict:
        return json.loads((self.root / MANIFEST_NAME).read_bytes().decode("utf-8"))

    def disk_bytes(self) -> dict[str, bytes]:
        """Every persisted record keyed by relative path; the lock file is
        only a mutual-exclusion device and is deliberately excluded."""
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def journal_records(self, name: str = REVOCATIONS_NAME) -> list[dict]:
        path = self.root / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text("utf-8").splitlines()]

    def observable_answers(self, vault: Vault, key_ids: list[str]) -> dict:
        """Every state a reader can query, in a comparable plain structure."""
        answers = {"manifest": vault.manifest()}
        for key_id in key_ids:
            versions = vault.versions(key_id)
            answers[key_id] = {
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
        answers["unknown"] = {
            "versions": vault.versions("never-sealed"),
            "revoked": vault.revoked_versions("never-sealed"),
        }
        return answers

    def assert_queries_match_disk(self, vault: Vault) -> None:
        """Pin the final state: every query answer re-derived independently
        from the persisted manifest, both journals and the material files.

        Version numbers must be a strict 1..n sequence per key (no
        duplicates, no gaps), every persisted material byte must read back
        through ``load`` and match its digest, a derivation record must come
        with matching parameters, and the revoked set and active pointer
        must be exactly what the append-only journals imply.
        """
        manifest = self.disk_manifest()
        self.assertEqual(vault.manifest(), manifest)

        revoked: dict[str, set[int]] = {}
        for record in self.journal_records():
            revoked.setdefault(record["key_id"], set()).add(record["version"])
        last_repoint: dict[str, dict] = {}
        for record in self.journal_records(ACTIVATIONS_NAME):
            last_repoint[record["key_id"]] = record

        for key_id, entry in manifest["keys"].items():
            numbers = [record["version"] for record in entry["versions"]]
            self.assertEqual(numbers, list(range(1, len(numbers) + 1)), key_id)
            self.assertEqual(vault.versions(key_id), numbers, key_id)
            # The manifest's active field is always the newest sealed
            # version; repoints live only in the activation journal.
            self.assertEqual(entry["active"], numbers[-1], key_id)
            for record in entry["versions"]:
                version = record["version"]
                data = (self.root / record["file"]).read_bytes()
                self.assertEqual(vault.load(key_id, version), data)
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), record["sha256"]
                )
                parameters = vault.derivation(key_id, version)
                persisted = record.get("derivation")
                if persisted is None:
                    self.assertEqual(parameters, {})
                else:
                    self.assertEqual(
                        parameters["iterations"], persisted["iterations"]
                    )
                    self.assertEqual(parameters["length"], persisted["length"])
                    self.assertEqual(len(data), persisted["length"])
            expected_active = numbers[-1]
            repoint = last_repoint.get(key_id)
            if repoint is not None and repoint["latest"] == numbers[-1]:
                expected_active = repoint["version"]
            self.assertEqual(vault.active(key_id), expected_active, key_id)
            self.assertEqual(
                set(vault.revoked_versions(key_id)), revoked.get(key_id, set())
            )

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
# every operation kind interleaved across processes and threads
# ---------------------------------------------------------------------------


class TestFullInterleavingFinalState(InterleavingTestCase):
    def test_final_state_is_strict_and_history_is_untouched(self):
        vault = self.open_vault()
        seed_payloads = [f"seed-{i}".encode("utf-8") for i in range(1, 7)]
        for payload in seed_payloads:
            vault.seal("race", payload)
        thread_seeds = [f"t-seed-{i}".encode("utf-8") for i in range(1, 5)]
        for payload in thread_seeds:
            vault.seal("threaded", payload)

        proc_payloads = {
            "a": [f"proc-a-{i}".encode("utf-8") for i in range(4)],
            "b": [f"proc-b-{i}".encode("utf-8") for i in range(4)],
        }
        derive_passwords = [f"derive-{i}".encode("utf-8") for i in range(3)]
        salt, iterations, length = b"full-salt", 80, 20
        derived_materials = [
            _pbkdf2(password, salt, iterations, length)
            for password in derive_passwords
        ]
        race_total = 6 + 8 + 3

        # Revoke and repoint targets are disjoint pre-seeded versions, so a
        # revoke can never race a repoint at the same version here.
        results: multiprocessing.Queue = multiprocessing.Queue()
        repoint_results: multiprocessing.Queue = multiprocessing.Queue()
        barrier = multiprocessing.Barrier(7)
        # Aborted at teardown if an assertion fails before every party has
        # shown up, so no worker waits forever for a missing party.
        self.fixture.track_barrier(barrier)
        workers = [
            multiprocessing.Process(
                target=_seal_payloads,
                args=(str(self.root), "race", proc_payloads["a"], barrier),
            ),
            multiprocessing.Process(
                target=_seal_payloads,
                args=(str(self.root), "race", proc_payloads["b"], barrier),
            ),
            multiprocessing.Process(
                target=_derive_payloads,
                args=(
                    str(self.root),
                    "race",
                    derive_passwords,
                    salt,
                    iterations,
                    length,
                    barrier,
                ),
            ),
            multiprocessing.Process(
                target=_revoke_exact,
                args=(str(self.root), "race", [4, 5, 6], results, barrier),
            ),
            multiprocessing.Process(
                target=_repoint_attempts,
                args=(
                    str(self.root),
                    "race",
                    [1, 2, 3] * 4,
                    repoint_results,
                    barrier,
                ),
            ),
        ]
        observers = [
            multiprocessing.Process(
                target=_reload_observer,
                args=(
                    str(self.root),
                    ["race", "threaded"],
                    {"race": race_total, "threaded": 15},
                    results,
                    barrier,
                ),
            )
            for _ in range(2)
        ]

        # In-process threads interleave the same operation kinds on the
        # second key of the very same vault directory.
        thread_payloads = [
            f"t-seal-{t}-{i}".encode("utf-8") for t in range(3) for i in range(3)
        ]
        thread_passwords = [b"t-dw-0", b"t-dw-1"]
        thread_salt, thread_iterations, thread_length = b"thread-salt", 70, 18
        thread_derived = [
            _pbkdf2(password, thread_salt, thread_iterations, thread_length)
            for password in thread_passwords
        ]
        thread_barrier = threading.Barrier(6)
        self.fixture.track_barrier(thread_barrier)
        thread_errors: list[BaseException] = []
        repoint_successes: list[int] = []

        def guard(fn) -> None:
            try:
                fn()
            except BaseException as exc:  # pragma: no cover - surfaced below
                thread_errors.append(exc)

        def thread_seals(payloads: list[bytes]) -> None:
            def work() -> None:
                local = Vault(self.root)
                try:
                    thread_barrier.wait()
                    for payload in payloads:
                        local.seal("threaded", payload)
                finally:
                    local.close()

            guard(work)

        def thread_derives() -> None:
            def work() -> None:
                local = Vault(self.root)
                try:
                    thread_barrier.wait()
                    for password in thread_passwords:
                        local.derive_seal(
                            "threaded",
                            password,
                            thread_salt,
                            thread_iterations,
                            thread_length,
                        )
                finally:
                    local.close()

            guard(work)

        def thread_revokes() -> None:
            def work() -> None:
                local = Vault(self.root)
                try:
                    thread_barrier.wait()
                    for version in (1, 2):
                        local.revoke("threaded", version)
                finally:
                    local.close()

            guard(work)

        def thread_repoints() -> None:
            def work() -> None:
                local = Vault(self.root)
                try:
                    thread_barrier.wait()
                    for _ in range(4):
                        try:
                            local.set_active("threaded", 3)
                            repoint_successes.append(1)
                        except ValueError:
                            pass  # already the active version: a legal loss
                finally:
                    local.close()

            guard(work)

        threads = [
            threading.Thread(
                target=thread_seals, args=(thread_payloads[t * 3 : (t + 1) * 3],)
            )
            for t in range(3)
        ]
        threads += [
            threading.Thread(target=thread_derives),
            threading.Thread(target=thread_revokes),
            threading.Thread(target=thread_repoints),
        ]
        # The single teardown aborts the barriers and drains every thread
        # and process before returning any handle if an assertion fails.
        self.fixture.track_thread(*threads)
        for process in workers + observers:
            process.start()
        self.fixture.track_process(*(workers + observers))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            self.assertFalse(thread.is_alive())
        self.assertEqual(thread_errors, [])
        for process in workers + observers:
            process.join(timeout=90)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
                self.fail(f"worker {process.name} hung")
            self.assertEqual(process.exitcode, 0)

        # The revoker and both observers reported success; the repointer's
        # attempts only ever ended "ok" or with the documented ValueError.
        reports = self.drain_nowait(results)
        self.assertEqual(len(reports), 3, reports)
        for status, payload in reports:
            self.assertEqual(status, "ok", payload)
        repoint_reports = self.assert_worker_results_ok(repoint_results)
        self.assertEqual(len(repoint_reports), 1)
        outcomes = repoint_reports[0]
        self.assertEqual(len(outcomes), 12)
        self.assertTrue(set(outcomes) <= {"ok", "ValueError"})

        # The final state, pinned against the persisted records: strict
        # 1..n sequences, untouched history, journals implying exactly the
        # queried revoked set and active pointer.
        final = self.open_vault()
        self.assert_queries_match_disk(final)

        race_versions = final.versions("race")
        self.assertEqual(race_versions, list(range(1, race_total + 1)))
        for index, payload in enumerate(seed_payloads, start=1):
            self.assertEqual(final.load("race", index), payload)
        self.assertEqual(
            {final.load("race", v) for v in race_versions[6:]},
            set(proc_payloads["a"]) | set(proc_payloads["b"]) | set(derived_materials),
        )
        self.assertEqual(final.revoked_versions("race"), [4, 5, 6])
        # Revocation is a marker only: the revoked history reads back too.
        for version in (4, 5, 6):
            self.assertTrue(final.is_revoked("race", version))
            self.assertEqual(final.load("race", version), seed_payloads[version - 1])
        # The active pointer is one of the repoint targets or the newest
        # version, exactly as the journal replay implies.
        self.assertIn(final.active("race"), {1, 2, 3, race_total})

        threaded_versions = final.versions("threaded")
        self.assertEqual(threaded_versions, list(range(1, 16)))
        for index, payload in enumerate(thread_seeds, start=1):
            self.assertEqual(final.load("threaded", index), payload)
        self.assertEqual(
            {final.load("threaded", v) for v in threaded_versions[4:]},
            set(thread_payloads) | set(thread_derived),
        )
        self.assertEqual(final.revoked_versions("threaded"), [1, 2])
        self.assertIn(final.active("threaded"), {3, 15})

        # Every successful repoint appended exactly one well-formed record;
        # the losers appended nothing, not even half a line.
        race_records = [
            r for r in self.journal_records(ACTIVATIONS_NAME) if r["key_id"] == "race"
        ]
        self.assertEqual(len(race_records), outcomes.count("ok"))
        threaded_records = [
            r
            for r in self.journal_records(ACTIVATIONS_NAME)
            if r["key_id"] == "threaded"
        ]
        self.assertEqual(len(threaded_records), len(repoint_successes))
        for record in race_records + threaded_records:
            self.assertIsInstance(record["version"], int)
            self.assertIsInstance(record["latest"], int)
            self.assertTrue(1 <= record["version"] <= record["latest"])
        self.assertEqual(
            sorted(r["version"] for r in self.journal_records() if r["key_id"] == "race"),
            [4, 5, 6],
        )
        self.assertEqual(
            sorted(
                r["version"] for r in self.journal_records() if r["key_id"] == "threaded"
            ),
            [1, 2],
        )

        # The derivation records travelled with exactly the derived
        # versions, parameters intact.
        manifest = self.disk_manifest()
        for key_id, expected_parameters, expected_count in (
            ("race", {"salt": salt, "iterations": iterations, "length": length}, 3),
            (
                "threaded",
                {
                    "salt": thread_salt,
                    "iterations": thread_iterations,
                    "length": thread_length,
                },
                2,
            ),
        ):
            derived_versions = [
                record["version"]
                for record in manifest["keys"][key_id]["versions"]
                if "derivation" in record
            ]
            self.assertEqual(len(derived_versions), expected_count, key_id)
            for version in derived_versions:
                self.assertEqual(
                    final.derivation(key_id, version), expected_parameters
                )

        # No half-written record and no orphan survived the interleaving.
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])
        material_files = list((self.root / MATERIALS_DIR).rglob("*.bin"))
        self.assertEqual(len(material_files), race_total + 15)


# ---------------------------------------------------------------------------
# derive_seal vs reload: record and material are visible together or not at all
# ---------------------------------------------------------------------------


class TestDeriveSealReloadAtomicity(InterleavingTestCase):
    def test_derivation_record_and_material_appear_together_or_not_at_all(self):
        passwords = [f"pw-{i}".encode("utf-8") for i in range(8)]
        salt, iterations, length = b"atomic-salt", 90, 24
        expected_materials = {
            _pbkdf2(password, salt, iterations, length) for password in passwords
        }
        barrier = multiprocessing.Barrier(3)
        self.fixture.track_barrier(barrier)
        derivers = [
            multiprocessing.Process(
                target=_derive_payloads,
                args=(
                    str(self.root),
                    "d",
                    passwords[half * 4 : (half + 1) * 4],
                    salt,
                    iterations,
                    length,
                    barrier,
                ),
            )
            for half in range(2)
        ]
        results: multiprocessing.Queue = multiprocessing.Queue()
        observer = multiprocessing.Process(
            target=_derive_observer,
            args=(
                str(self.root),
                "d",
                salt,
                iterations,
                length,
                expected_materials,
                len(passwords),
                results,
                barrier,
            ),
        )
        self.run_processes(derivers + [observer])
        # The observer reached the complete final state and never saw a
        # record without its material or vice versa.
        self.assertEqual(self.assert_worker_results_ok(results), [len(passwords)])

        final = self.open_vault()
        self.assertEqual(final.versions("d"), list(range(1, len(passwords) + 1)))
        self.assertEqual(
            {final.load("d", v) for v in final.versions("d")}, expected_materials
        )
        for version in final.versions("d"):
            self.assertEqual(
                final.derivation("d", version),
                {"salt": salt, "iterations": iterations, "length": length},
            )
        self.assert_queries_match_disk(final)


# ---------------------------------------------------------------------------
# set_active vs seal: the final pointer is one of the two serial outcomes
# ---------------------------------------------------------------------------


class TestSetActiveVsSealFinalPointer(InterleavingTestCase):
    def test_both_serial_orders_are_pinned(self):
        vault = self.open_vault()
        # Serial order 1: repoint, then seal -- the new seal supersedes the
        # repoint and becomes active, as if the key had never been repointed.
        for payload in (b"a1", b"a2", b"a3"):
            vault.seal("a", payload)
        vault.set_active("a", 1)
        self.assertEqual(vault.seal("a", b"a4"), 4)
        self.assertEqual(vault.active("a"), 4)
        vault.reload()
        self.assertEqual(vault.active("a"), 4)

        # Serial order 2: seal, then repoint -- the repoint lands on the
        # newest sequence and the pointer moves to the target.
        for payload in (b"b1", b"b2", b"b3"):
            vault.seal("b", payload)
        self.assertEqual(vault.seal("b", b"b4"), 4)
        vault.set_active("b", 1)
        self.assertEqual(vault.active("b"), 1)
        vault.reload()
        self.assertEqual(vault.active("b"), 1)
        self.assertEqual(vault.load("b"), b"b1")

    def test_race_final_pointer_is_one_of_the_two_serial_outcomes(self):
        vault = self.open_vault()
        for round_index in range(6):
            key_id = f"repoint-race-{round_index}"
            for payload in (b"s1", b"s2", b"s3"):
                vault.seal(key_id, payload)

            results: multiprocessing.Queue = multiprocessing.Queue()
            barrier = multiprocessing.Barrier(2)
            self.fixture.track_barrier(barrier)
            repointer = multiprocessing.Process(
                target=_one_shot,
                args=(str(self.root), "repoint", key_id, 1, results, barrier),
            )
            sealer = multiprocessing.Process(
                target=_one_shot,
                args=(str(self.root), "seal", key_id, b"race-new", results, barrier),
            )
            self.run_processes([repointer, sealer])
            # Neither call can legitimately fail: the target exists, is not
            # revoked and is not the active version on either serial order.
            self.assertEqual(
                sorted(self.drain_nowait(results)),
                [("repoint", "ok"), ("seal", "ok")],
            )

            vault.reload()
            records = [
                r
                for r in self.journal_records(ACTIVATIONS_NAME)
                if r["key_id"] == key_id
            ]
            # Exactly one complete record landed; its bound newest version
            # says which serial order the race took.
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["version"], 1)
            self.assertIn(record["latest"], (3, 4))
            if record["latest"] == 3:
                # repoint, then seal: the seal superseded the repoint.
                expected_active = 4
            else:
                # seal, then repoint: the pointer moved to the target.
                expected_active = 1
            self.assertEqual(vault.active(key_id), expected_active)
            self.assertIn(vault.active(key_id), (1, 4))
            self.assertEqual(vault.versions(key_id), [1, 2, 3, 4])
            self.assertEqual(vault.load(key_id, 4), b"race-new")
            self.assertEqual(vault.load(key_id, 1), b"s1")
            self.assertEqual(vault.load(key_id), vault.load(key_id, expected_active))


# ---------------------------------------------------------------------------
# revoke vs set_active at the same version: two serial orders, no half record
# ---------------------------------------------------------------------------


class TestRevokeVsRepointRace(InterleavingTestCase):
    def test_both_serial_orders_occur_and_are_pinned(self):
        vault = self.open_vault()
        # Serial order 1: repoint, then revoke -- both succeed; revocation
        # never moves the pointer, so the revoked target stays active.
        vault.seal("a", b"a1")
        vault.seal("a", b"a2")
        vault.set_active("a", 1)
        vault.revoke("a", 1)
        self.assertEqual(vault.active("a"), 1)
        self.assertTrue(vault.is_revoked("a", 1))
        self.assertEqual(vault.load("a"), b"a1")
        vault.reload()
        self.assertEqual(vault.active("a"), 1)

        # Serial order 2: revoke, then repoint -- the repoint loses with
        # ValueError and leaves no record at all.
        vault.seal("b", b"b1")
        vault.seal("b", b"b2")
        vault.revoke("b", 1)
        with self.assertRaises(ValueError):
            vault.set_active("b", 1)
        self.assertEqual(vault.active("b"), 2)
        self.assertTrue(vault.is_revoked("b", 1))
        # The loser's journal holds not even half a record for "b".
        self.assertEqual(
            [r for r in self.journal_records(ACTIVATIONS_NAME) if r["key_id"] == "b"],
            [],
        )
        self.assertEqual(
            [r for r in self.journal_records(ACTIVATIONS_NAME) if r["key_id"] == "a"],
            [{"key_id": "a", "version": 1, "latest": 2}],
        )

    def test_race_outcome_is_exactly_one_serial_order(self):
        vault = self.open_vault()
        for round_index in range(6):
            key_id = f"revoke-race-{round_index}"
            vault.seal(key_id, b"r1")
            vault.seal(key_id, b"r2")  # active is 2, so 1 is a legal target

            results: multiprocessing.Queue = multiprocessing.Queue()
            barrier = multiprocessing.Barrier(2)
            self.fixture.track_barrier(barrier)
            revoker = multiprocessing.Process(
                target=_one_shot,
                args=(str(self.root), "revoke", key_id, 1, results, barrier),
            )
            repointer = multiprocessing.Process(
                target=_one_shot,
                args=(str(self.root), "repoint", key_id, 1, results, barrier),
            )
            self.run_processes([revoker, repointer])
            outcomes = dict(self.drain_nowait(results))
            # Revoking an existing, never-revoked version always succeeds.
            self.assertEqual(outcomes.pop("revoke"), "ok")
            repoint_outcome = outcomes.pop("repoint")
            self.assertEqual(outcomes, {})
            self.assertIn(repoint_outcome, ("ok", "ValueError"))

            vault.reload()
            key_activations = [
                r
                for r in self.journal_records(ACTIVATIONS_NAME)
                if r["key_id"] == key_id
            ]
            if repoint_outcome == "ok":
                # Serial order "repoint, then revoke": one complete record,
                # pointer at the (now revoked) target.
                self.assertEqual(
                    key_activations,
                    [{"key_id": key_id, "version": 1, "latest": 2}],
                )
                self.assertEqual(vault.active(key_id), 1)
            else:
                # Serial order "revoke, then repoint": the loser left no
                # record behind, not even half a line.
                self.assertEqual(key_activations, [])
                self.assertEqual(vault.active(key_id), 2)
            # Either way the revocation landed exactly once, completely.
            self.assertEqual(
                [r for r in self.journal_records() if r["key_id"] == key_id],
                [{"key_id": key_id, "version": 1}],
            )
            self.assertEqual(vault.revoked_versions(key_id), [1])
            self.assertTrue(vault.is_revoked(key_id, 1))
            self.assertFalse(vault.is_revoked(key_id, 2))
            # History is untouched: the revoked material still reads back.
            self.assertEqual(vault.load(key_id, 1), b"r1")
            self.assertEqual(vault.load(key_id, 2), b"r2")
            self.assertEqual(vault.versions(key_id), [1, 2])


# ---------------------------------------------------------------------------
# reload vs a mixed writer: only complete old or complete new states
# ---------------------------------------------------------------------------


class TestReloadSeesOnlyCompleteStates(InterleavingTestCase):
    def test_observer_never_sees_a_half_old_half_new_mixture(self):
        vault = self.open_vault()
        vault.seal("k", b"seed-1")
        vault.seal("k", b"seed-2")

        salt, iterations, length = b"plan-salt", 90, 24
        plan = [
            ("seal", b"w-1"),
            ("seal", b"w-2"),
            ("seal", b"w-3"),
            ("derive", b"w-pw-1", salt, iterations, length),
            ("revoke", 1),
            ("seal", b"w-4"),
            ("seal", b"w-5"),
            ("derive", b"w-pw-2", salt, iterations, length),
            ("repoint", 2),
        ]
        results: multiprocessing.Queue = multiprocessing.Queue()
        barrier = multiprocessing.Barrier(2)
        self.fixture.track_barrier(barrier)
        writer = multiprocessing.Process(
            target=_mixed_plan_writer,
            args=(str(self.root), "k", plan, results, barrier),
        )
        observer = multiprocessing.Process(
            target=_reload_observer,
            args=(str(self.root), ["k"], {"k": 9}, results, barrier),
        )
        self.run_processes([writer, observer])
        # The writer completed its whole plan and the observer never saw a
        # half-old/half-new state on any of its reload rounds.
        payloads = self.assert_worker_results_ok(results)
        self.assertEqual(len(payloads), 2)
        self.assertIn(len(plan), payloads)

        # The final state is exactly the serial execution of the plan.
        final = self.open_vault()
        self.assertEqual(final.versions("k"), list(range(1, 10)))
        self.assertEqual(final.active("k"), 2)
        self.assertEqual(final.revoked_versions("k"), [1])
        self.assertEqual(final.load("k"), b"seed-2")
        expected_materials = {
            1: b"seed-1",
            2: b"seed-2",
            3: b"w-1",
            4: b"w-2",
            5: b"w-3",
            6: _pbkdf2(b"w-pw-1", salt, iterations, length),
            7: b"w-4",
            8: b"w-5",
            9: _pbkdf2(b"w-pw-2", salt, iterations, length),
        }
        for version, material in expected_materials.items():
            self.assertEqual(final.load("k", version), material)
        self.assertEqual(
            self.journal_records(ACTIVATIONS_NAME),
            [{"key_id": "k", "version": 2, "latest": 9}],
        )
        self.assertEqual(self.journal_records(), [{"key_id": "k", "version": 1}])
        self.assert_queries_match_disk(final)


# ---------------------------------------------------------------------------
# failed and blocked calls write nothing; a failed reload freezes the snapshot
# ---------------------------------------------------------------------------


class TestBlockedAndFailedCallsLeaveNoRecord(InterleavingTestCase):
    def _build_healthy(self) -> Vault:
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.derive_seal("k", b"pw", b"salt", 100, 16)
        vault.seal("k", b"v3")
        vault.revoke("k", 1)
        vault.set_active("k", 2)
        return vault

    def test_blocked_and_failing_calls_during_corruption_write_nothing(self):
        vault = self._build_healthy()
        healthy_disk = self.disk_bytes()
        answers_before = self.observable_answers(vault, ["k"])

        # A sealer with an already-open healthy handle parks on its gate.
        opened = multiprocessing.Event()
        go = multiprocessing.Event()
        sealer_results: multiprocessing.Queue = multiprocessing.Queue()
        sealer = multiprocessing.Process(
            target=_seal_after_gate,
            args=(str(self.root), "k", b"blocked-material", opened, go, sealer_results),
        )
        self.fixture.track_gate(go)
        self.fixture.track_process(sealer)
        sealer.start()
        self.assertTrue(opened.wait(timeout=10))

        # A holder takes the inter-process lock and keeps it.
        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_lock_holder, args=(str(self.root), ready, release)
        )
        # On a failed assertion the single teardown releases these gates and
        # drains every process before returning any handle.
        self.fixture.track_gate(release)
        self.fixture.track_process(holder)
        holder.start()
        self.assertTrue(ready.wait(timeout=10))

        # Corrupt the manifest out of band while the lock is held, then let
        # the parked sealer go: its seal blocks behind the holder.  A fresh
        # opener blocks in its constructor the same way.
        corrupt = b"{corrupt manifest"
        (self.root / MANIFEST_NAME).write_bytes(corrupt)
        corrupt_disk = dict(healthy_disk)
        corrupt_disk[MANIFEST_NAME] = corrupt
        go.set()
        opener_results: multiprocessing.Queue = multiprocessing.Queue()
        opener = multiprocessing.Process(
            target=_expect_open_value_error, args=(str(self.root), opener_results)
        )
        self.fixture.track_process(opener)
        opener.start()

        # While the holder keeps the lock, both are parked and the disk is
        # byte-for-byte the corrupted state: nothing half written.
        time.sleep(0.5)
        self.assertTrue(sealer.is_alive(), "sealer ran while the lock was held")
        self.assertTrue(opener.is_alive(), "opener ran while the lock was held")
        self.assertEqual(self.disk_bytes(), corrupt_disk)
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])

        # Release: the blocked seal and the blocked open both fail their
        # validation with ValueError and write nothing.
        release.set()
        sealer.join(timeout=30)
        self.assertEqual(sealer.exitcode, 0)
        opener.join(timeout=30)
        self.assertEqual(opener.exitcode, 0)
        holder.join(timeout=30)
        self.assertEqual(holder.exitcode, 0)
        self.assertEqual(self.drain_nowait(sealer_results), [("ValueError",)])
        self.assertEqual(self.drain_nowait(opener_results), ["ValueError"])

        # The failed and blocked calls left no record: the disk is still
        # exactly the corrupted state, and the live handle's failed reload
        # keeps its snapshot verbatim with keys in hand still readable.
        self.assertEqual(self.disk_bytes(), corrupt_disk)
        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
            self.assertEqual(self.observable_answers(vault, ["k"]), answers_before)
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.load("k", 3), b"v3")
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertEqual(vault.active("k"), 2)

        # Undo the corruption: the same handle recovers, the blocked seal's
        # payload never appears, and the sequence continues without reuse.
        (self.root / MANIFEST_NAME).write_bytes(healthy_disk[MANIFEST_NAME])
        vault.reload()
        self.assertEqual(self.observable_answers(vault, ["k"]), answers_before)
        self.assertEqual(vault.seal("k", b"v4"), 4)
        self.assertEqual(vault.versions("k"), [1, 2, 3, 4])
        self.assertEqual(vault.active("k"), 4)
        vault.reload()
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), [1, 2, 3, 4])
        self.assertEqual(reopened.load("k", 4), b"v4")
        self.assertEqual(reopened.load("k", 1), b"v1")
        self.assertTrue(reopened.is_revoked("k", 1))
        self.assert_queries_match_disk(reopened)

    def test_rejected_calls_leave_disk_and_snapshot_untouched(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        vault.seal("r", b"r1")
        vault.seal("r", b"r2")
        vault.set_active("r", 1)
        vault.derive_seal("d", b"pw", b"salt", 100, 16)

        disk_before = self.disk_bytes()
        answers_before = self.observable_answers(vault, ["k", "r", "d"])

        # Empty key id -> ValueError at every entry that takes an id.
        for call in (
            lambda: vault.seal("", b"x"),
            lambda: vault.derive_seal("", b"pw", b"salt", 1, 1),
            lambda: vault.load("", 1),
            lambda: vault.load("", 1.0),
            lambda: vault.revoke("", 1),
            lambda: vault.set_active("", 1),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
            lambda: vault.derivation(""),
            lambda: vault.derivation("", 1),
        ):
            with self.assertRaises(ValueError):
                call()

        # Material, passphrase or salt of the wrong type -> TypeError.
        for bad in ("text", 1, 1.5, None, [b"x"], {"k": b"v"}, object()):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.seal("k", bad)
        for bad in ("text", 1, None, [b"x"], bytearray(b"x"), memoryview(b"x")):
            with self.assertRaises(TypeError, msg=f"password={bad!r}"):
                vault.derive_seal("k", bad, b"salt", 1, 1)
            with self.assertRaises(TypeError, msg=f"salt={bad!r}"):
                vault.derive_seal("k", b"pw", bad, 1, 1)

        # Version or derivation parameter that is not a genuine integer ->
        # TypeError; bools and floats do not count.
        for bad in (1.0, 2.5, True, False, "1", (1,)):
            with self.assertRaises(TypeError, msg=f"load {bad!r}"):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=f"revoke {bad!r}"):
                vault.revoke("k", bad)
            with self.assertRaises(TypeError, msg=f"set_active {bad!r}"):
                vault.set_active("k", bad)
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("k", bad)
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("k", bad)
        for bad in (True, False, 1.0, 2.5, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=f"iterations={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(TypeError, msg=f"length={bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)

        # Empty salt, non-positive iterations/length -> ValueError.
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
        ):
            with self.assertRaises(KeyError):
                call()
        for missing in (0, 3, 99, -1):
            with self.assertRaises(KeyError, msg=str(missing)):
                vault.load("k", missing)
            with self.assertRaises(KeyError, msg=str(missing)):
                vault.revoke("k", missing)
            with self.assertRaises(KeyError, msg=str(missing)):
                vault.set_active("k", missing)
            with self.assertRaises(KeyError, msg=str(missing)):
                vault.is_revoked("k", missing)
            with self.assertRaises(KeyError, msg=str(missing)):
                vault.derivation("d", missing)

        # Duplicate revocation, repointing at the current active version and
        # repointing at a revoked version -> ValueError.
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)
        with self.assertRaises(ValueError):
            vault.set_active("k", 1)
        with self.assertRaises(ValueError):
            vault.set_active("r", 1)

        # Not one of the rejected calls wrote anything: the persisted
        # records are byte-for-byte what they were and the snapshot answers
        # are frozen word for word.
        self.assertEqual(self.disk_bytes(), disk_before)
        self.assertEqual(self.observable_answers(vault, ["k", "r", "d"]), answers_before)
        self.assertEqual(list(self.root.glob("**/*.tmp.*")), [])

        # The vault keeps working: the next seal continues the sequence and
        # a reload confirms full correspondence with the disk records.
        self.assertEqual(vault.seal("k", b"v3"), 3)
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        self.assert_queries_match_disk(vault)


if __name__ == "__main__":
    unittest.main()
