"""Regression tests pinning the crash site of a failed *first* seal.

The baseline vault already implements sealing, reading, revocation,
derivation, repointing and the whole-vault reload; this module adds no
product behaviour.  It freezes what is observable when the very first
``seal`` of a directory dies midway:

* the material lands on disk first and the manifest is replaced only
  afterwards, so a failure while writing the material leaves the initial
  manifest (no keys, no versions) and no ``.tmp`` file;

* a failure while replacing the manifest -- after the material has already
  landed -- removes that just-written material as well, so the directory
  holds neither half a manifest record nor an orphan material: no version
  1 appears in any listing and no ``1.bin`` survives;

* the next call starts again from the same directory and seals
  successfully: the version number is 1 (never reused, never skipped) and
  the material reads back byte for byte, after a library retry, a full
  ``reload()`` or a cold reopen;

* a failure window on the *first* derived seal leaves the same empty
  crash site and the restart seal still starts at 1;

* when several processes (and threads) race the very first seal of one
  shared directory, every seal succeeds, the versions run strictly
  1..n with no duplicate and no gap, each version holds one unique payload
  and historical material is never rewritten -- plain seals alone, and
  plain seals mixed with ``derive_seal`` on one shared sequence.

Everything happens inside temporary directories, uses the standard
library only, drives real processes through the shared fixture's single
teardown, and is independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import threading
import unittest
from pathlib import Path
from unittest import mock

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    ACTIVATIONS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
    _atomic_write,
)
from tests._fixtures import VaultFixture


# ---------------------------------------------------------------------------
# module-level worker entry points (picklable for every start method)
# ---------------------------------------------------------------------------


def _plain_seal_once(
    root: str, payload: bytes, results: "multiprocessing.Queue"
) -> None:
    vault = Vault(root)
    try:
        results.put(("ok", vault.seal("shared", payload)))
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", repr(exc)))
    finally:
        vault.close()


def _plain_seal_many(root: str, payloads: list[bytes]) -> None:
    vault = Vault(root)
    try:
        for payload in payloads:
            vault.seal("shared", payload)
    finally:
        vault.close()


def _derive_seal_many(root: str, passwords: list[bytes], salt: bytes) -> None:
    vault = Vault(root)
    try:
        for password in passwords:
            vault.derive_seal("shared", password, salt, 100, 16)
    finally:
        vault.close()


class FirstSealCrashTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        # Each scenario gets its own vault subdirectory, so cases never
        # share disk state and execution order cannot matter.
        self.root = self.fixture.tmp_path / "vaults"

    def open_vault(self, root: Path) -> Vault:
        """Open a vault through the case's single teardown path."""
        return self.fixture.open(root)

    # ------------------------------------------------------------------
    # disk-shape helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_files(root: Path) -> dict[str, bytes]:
        """All vault records on disk keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, so it is deliberately excluded.
        """
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def _material_files(self, root: Path) -> set[Path]:
        return set((root / MATERIALS_DIR).rglob("*.bin"))

    def _tmp_files(self, root: Path) -> list[Path]:
        return list(root.rglob("*.tmp.*"))

    def _disk_manifest(self, root: Path) -> dict:
        return json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))

    def _assert_initial_empty_vault(self, root: Path) -> None:
        """The crash site must look exactly like a freshly opened vault."""
        # The initial manifest is intact and records no key or version.
        manifest = self._disk_manifest(root)
        self.assertEqual(manifest["format"], 1)
        self.assertEqual(manifest["keys"], {})
        # No half material and no temp file anywhere under the directory.
        # The per-key directory node is created just before the material
        # write; a dead seal removes the material file but can leave the
        # empty directory node behind.  An empty node is not an orphan
        # material: what must be absent is every file (``*.bin`` or temp).
        materials_dir = root / MATERIALS_DIR
        self.assertTrue(materials_dir.is_dir())
        self.assertEqual(self._material_files(root), set())
        self.assertEqual(self._tmp_files(root), [])
        self.assertEqual(
            [path for path in materials_dir.rglob("*") if path.is_file()],
            [],
        )
        # The append-only journals are exactly the initial state: the
        # revocation journal exists and is empty, the activation journal
        # was never created.
        self.assertEqual((root / REVOCATIONS_NAME).read_bytes(), b"")
        self.assertFalse((root / ACTIVATIONS_NAME).exists())

    @staticmethod
    def _fail_first_manifest_write():
        """Patch helper: die on the first manifest replacement only."""
        real_atomic_write = _atomic_write

        def fail_first_manifest(path, data):
            if path.name == MANIFEST_NAME:
                raise OSError("simulated manifest failure")
            return real_atomic_write(path, data)

        return mock.patch(
            "keyvault_ledger.vault._atomic_write",
            side_effect=fail_first_manifest,
        )

    @staticmethod
    def _fail_first_material_write():
        """Patch helper: die writing the first material file only."""
        real_atomic_write = _atomic_write

        def fail_first_material(path, data):
            if MATERIALS_DIR in path.parts:
                raise OSError("simulated material write failure")
            return real_atomic_write(path, data)

        return mock.patch(
            "keyvault_ledger.vault._atomic_write",
            side_effect=fail_first_material,
        )

    # ------------------------------------------------------------------
    # the two failure windows of the first seal
    # ------------------------------------------------------------------

    def test_failure_writing_first_material_leaves_initial_vault(self):
        root = self.root / "material-failure"
        vault = self.open_vault(root)
        records_before = self._record_files(root)

        with self._fail_first_material_write():
            with self.assertRaises(OSError):
                vault.seal("first-key", b"first-material")

        # Nothing half landed: the manifest is the initial one, there is no
        # material and the failed atomic write cleaned its own temp file.
        self.assertEqual(self._record_files(root), records_before)
        self._assert_initial_empty_vault(root)

    def test_failure_replacing_first_manifest_removes_the_orphan_material(self):
        root = self.root / "manifest-failure"
        vault = self.open_vault(root)
        records_before = self._record_files(root)

        with self._fail_first_manifest_write():
            with self.assertRaises(OSError):
                vault.seal("first-key", b"first-material")

        # The material landed first and was removed again when the manifest
        # flip died: no orphan material, no half manifest version, no temp.
        self.assertEqual(self._record_files(root), records_before)
        self._assert_initial_empty_vault(root)
        # The live handle's snapshot never published the dead version.
        self.assertEqual(vault.versions("first-key"), [])
        self.assertEqual(vault.manifest()["keys"], {})

    def test_failure_at_either_stage_is_observed_on_cold_reopen_too(self):
        # A fresh process opening the directory after either crash must see
        # the same empty, healthy vault, not a ValueError or a half version.
        for index, patcher in enumerate(
            (self._fail_first_material_write, self._fail_first_manifest_write)
        ):
            root = self.root / f"cold-crash-{index}"
            vault = self.open_vault(root)
            with patcher():
                with self.assertRaises(OSError):
                    vault.seal("first-key", b"first-material")
            reopened = self.open_vault(root)
            self.assertEqual(reopened.manifest()["keys"], {})
            self.assertEqual(reopened.versions("first-key"), [])
            self._assert_initial_empty_vault(root)

    def test_first_derived_seal_failure_leaves_the_same_empty_site(self):
        root = self.root / "derived-manifest-failure"
        vault = self.open_vault(root)

        with self._fail_first_manifest_write():
            with self.assertRaises(OSError):
                vault.derive_seal("first-key", b"passphrase", b"salty", 100, 16)

        self._assert_initial_empty_vault(root)
        # A direct first seal after the dead derived seal still starts at 1.
        self.assertEqual(vault.seal("first-key", b"restarted"), 1)
        self.assertEqual(vault.load("first-key"), b"restarted")

    # ------------------------------------------------------------------
    # restart from the same directory
    # ------------------------------------------------------------------

    def test_next_call_restarts_at_version_one_byte_for_byte(self):
        root = self.root / "restart-same-handle"
        vault = self.open_vault(root)
        material = bytes(range(256)) + b"\x00\xff\n\r restart"

        with self._fail_first_manifest_write():
            with self.assertRaises(OSError):
                vault.seal("first-key", material)

        # The very next call on the same handle starts over and seals
        # successfully; the version number is 1 and the bytes are exact.
        self.assertEqual(vault.seal("first-key", material), 1)
        self.assertEqual(vault.active("first-key"), 1)
        self.assertEqual(vault.versions("first-key"), [1])
        self.assertEqual(vault.load("first-key"), material)
        self.assertEqual(vault.load("first-key", 1), material)
        # The disk record is a normal version-1 record whose file holds the
        # exact bytes under the per-key material directory.
        manifest = self._disk_manifest(root)
        records = manifest["keys"]["first-key"]["versions"]
        self.assertEqual([record["version"] for record in records], [1])
        rel_file = records[0]["file"]
        self.assertTrue(rel_file.startswith(f"{MATERIALS_DIR}/"))
        self.assertTrue(rel_file.endswith("/1.bin"))
        self.assertEqual((root / rel_file).read_bytes(), material)
        self.assertEqual(self._material_files(root), {root / rel_file})

    def test_restart_after_reload_and_after_cold_reopen_starts_at_one(self):
        root = self.root / "restart-cold"
        vault = self.open_vault(root)
        with self._fail_first_manifest_write():
            with self.assertRaises(OSError):
                vault.seal("first-key", b"first-material")
        # The crash site is exactly the initial empty vault.
        self._assert_initial_empty_vault(root)

        # A cold opener on the same directory sees an empty, healthy vault.
        reopened = self.open_vault(root)
        self.assertEqual(reopened.versions("first-key"), [])
        reopened.reload()
        material = b"after-the-crash"
        self.assertEqual(reopened.seal("first-key", material), 1)
        self.assertEqual(reopened.load("first-key"), material)
        reopened.reload()
        self.assertEqual(reopened.versions("first-key"), [1])
        self.assertEqual(reopened.active("first-key"), 1)

        # A second fresh opener reads version 1 back byte for byte.
        again = self.open_vault(root)
        self.assertEqual(again.versions("first-key"), [1])
        self.assertEqual(again.load("first-key", 1), material)
        self.assertEqual(again.load("first-key"), material)

    def test_restarted_first_seal_extends_normally_to_version_two(self):
        root = self.root / "restart-extends"
        vault = self.open_vault(root)
        real_atomic_write = _atomic_write
        failed = {"n": 0}

        def fail_first_manifest_once(path, data):
            if path.name == MANIFEST_NAME and failed["n"] == 0:
                failed["n"] += 1
                raise OSError("simulated manifest failure")
            return real_atomic_write(path, data)

        with mock.patch(
            "keyvault_ledger.vault._atomic_write",
            side_effect=fail_first_manifest_once,
        ):
            with self.assertRaises(OSError):
                vault.seal("k", b"v1")

        self.assertEqual(vault.seal("k", b"v1-real"), 1)
        # Version 1 was not reused: the next seal takes 2 and the version-1
        # material is still byte for byte the successful payload.
        self.assertEqual(vault.seal("k", b"v2"), 2)
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.load("k", 1), b"v1-real")
        self.assertEqual(vault.load("k", 2), b"v2")
        manifest = self._disk_manifest(root)
        self.assertEqual(
            self._material_files(root),
            {
                root / record["file"]
                for record in manifest["keys"]["k"]["versions"]
            },
        )

    # ------------------------------------------------------------------
    # several processes racing the very first seal
    # ------------------------------------------------------------------

    def _run_workers(self, workers: list[multiprocessing.Process]) -> None:
        self.fixture.track_process(*workers)
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=120)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
                self.fail(f"worker {worker.name} hung")
            self.assertEqual(
                worker.exitcode,
                0,
                f"worker {worker.name} exited with {worker.exitcode}",
            )

    def test_two_processes_racing_first_seal_both_succeed_versions_from_one(self):
        root = self.root / "two-process-race"
        # The directory is initialised by nobody first: the two processes
        # race the constructor and the first seal itself.
        results: multiprocessing.Queue = multiprocessing.Queue()
        workers = [
            multiprocessing.Process(
                target=_plain_seal_once,
                args=(str(root), f"proc-{p}".encode(), results),
            )
            for p in range(2)
        ]
        self._run_workers(workers)

        outcomes = []
        while not results.empty():
            outcomes.append(results.get())
        self.assertEqual(len(outcomes), 2)
        self.assertEqual([status for status, _ in outcomes], ["ok", "ok"])
        # Versions 1 and 2 were allocated one each, strictly from 1.
        self.assertEqual(sorted(version for _, version in outcomes), [1, 2])

        vault = self.open_vault(root)
        self.assertEqual(vault.versions("shared"), [1, 2])
        self.assertEqual(vault.active("shared"), 2)
        self.assertEqual(
            {vault.load("shared", 1), vault.load("shared", 2)},
            {b"proc-0", b"proc-1"},
        )
        # No half record, orphan or temp file at the crash-prone first seal.
        self.assertEqual(self._tmp_files(root), [])
        self.assertEqual(
            sorted(path.name for path in self._material_files(root)),
            ["1.bin", "2.bin"],
        )
        vault.reload()
        self.assertEqual(vault.versions("shared"), [1, 2])

    def test_many_processes_racing_first_seals_stay_strict_and_immutable(self):
        root = self.root / "many-process-race"
        procs, seals_each = 5, 3
        workers = [
            multiprocessing.Process(
                target=_plain_seal_many,
                args=(
                    str(root),
                    [f"proc-{p}-seal-{i}".encode() for i in range(seals_each)],
                ),
            )
            for p in range(procs)
        ]
        self._run_workers(workers)

        total = procs * seals_each
        vault = self.open_vault(root)
        # Strict 1..n: no duplicate, no gap, no skipped/reused number.
        self.assertEqual(vault.versions("shared"), list(range(1, total + 1)))
        self.assertEqual(vault.active("shared"), total)
        payloads = {
            f"proc-{p}-seal-{i}".encode()
            for p in range(procs)
            for i in range(seals_each)
        }
        materials = {vault.load("shared", v) for v in range(1, total + 1)}
        # Every payload landed exactly once and reads back byte for byte.
        self.assertEqual(materials, payloads)
        self.assertEqual(len(materials), total)

        # Historical material is never rewritten: every material file
        # matches its manifest digest, and a full reload serves the exact
        # same bytes again for every version.
        manifest = self._disk_manifest(root)
        for record in manifest["keys"]["shared"]["versions"]:
            data = (root / record["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
            self.assertEqual(vault.load("shared", record["version"]), data)
        vault.reload()
        self.assertEqual(vault.versions("shared"), list(range(1, total + 1)))
        self.assertEqual(
            {vault.load("shared", v) for v in range(1, total + 1)}, payloads
        )
        self.assertEqual(self._tmp_files(root), [])

    def test_mixed_plain_and_derived_first_seals_share_one_sequence(self):
        root = self.root / "mixed-race"
        seal_procs, each = 3, 3
        salt = b"first-seal-salt"
        workers = [
            multiprocessing.Process(
                target=_plain_seal_many,
                args=(
                    str(root),
                    [f"plain-{p}-{i}".encode() for i in range(each)],
                ),
            )
            for p in range(seal_procs)
        ]
        passwords = [
            f"pw-{p}-{i}".encode() for p in range(2) for i in range(each)
        ]
        workers += [
            multiprocessing.Process(
                target=_derive_seal_many,
                args=(
                    str(root),
                    [f"pw-{p}-{i}".encode() for i in range(each)],
                    salt,
                ),
            )
            for p in range(2)
        ]
        self._run_workers(workers)

        total = (seal_procs + 2) * each
        vault = self.open_vault(root)
        # Direct and derived first seals draw from one strict sequence.
        self.assertEqual(vault.versions("shared"), list(range(1, total + 1)))
        self.assertEqual(vault.active("shared"), total)
        plain_payloads = {
            f"plain-{p}-{i}".encode()
            for p in range(seal_procs)
            for i in range(each)
        }
        # Material -> passphrase, so every derived version can be checked by
        # re-running PBKDF2 with the very passphrase that produced it.
        derived_by_material = {
            hashlib.pbkdf2_hmac("sha256", password, salt, 100, dklen=16): password
            for password in passwords
        }
        materials = {vault.load("shared", v) for v in range(1, total + 1)}
        self.assertEqual(materials, plain_payloads | set(derived_by_material))
        self.assertEqual(len(materials), total)
        for version in range(1, total + 1):
            material = vault.load("shared", version)
            params = vault.derivation("shared", version)
            if material in derived_by_material:
                # Derived version: parameters round-trip and re-running
                # PBKDF2 reproduces the stored bytes exactly.
                self.assertEqual(
                    params,
                    {"salt": salt, "iterations": 100, "length": 16},
                )
                self.assertEqual(
                    material,
                    hashlib.pbkdf2_hmac(
                        "sha256",
                        derived_by_material[material],
                        salt,
                        100,
                        dklen=16,
                    ),
                )
            else:
                # A directly sealed version carries an empty record.
                self.assertEqual(params, {})
                self.assertIn(material, plain_payloads)
        self.assertEqual(self._tmp_files(root), [])

    def test_in_process_threads_racing_first_seal_share_the_sequence(self):
        root = self.root / "thread-race"
        thread_count, each = 6, 4
        barrier = threading.Barrier(thread_count)
        # Aborted at teardown if an assertion fails before every party
        # arrives, so no worker waits forever for a missing party.
        self.fixture.track_barrier(barrier)
        errors: list[BaseException] = []

        def seal_slice(thread_index: int) -> None:
            local = Vault(root)
            try:
                barrier.wait()
                for i in range(each):
                    local.seal("shared", f"t-{thread_index}-{i}".encode())
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            finally:
                local.close()

        threads = [
            threading.Thread(target=seal_slice, args=(t,))
            for t in range(thread_count)
        ]
        self.fixture.track_thread(*threads)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

        total = thread_count * each
        vault = self.open_vault(root)
        self.assertEqual(vault.versions("shared"), list(range(1, total + 1)))
        self.assertEqual(vault.active("shared"), total)
        payloads = {
            f"t-{t}-{i}".encode()
            for t in range(thread_count)
            for i in range(each)
        }
        self.assertEqual(
            {vault.load("shared", v) for v in range(1, total + 1)}, payloads
        )
        self.assertEqual(len(self._material_files(root)), total)
        self.assertEqual(self._tmp_files(root), [])

    # ------------------------------------------------------------------
    # determinism: two independently crashed directories recover alike
    # ------------------------------------------------------------------

    def test_two_crashed_directories_restart_identically(self):
        material = b"same-material"

        def crash_then_restart(name: str) -> dict:
            root = self.root / name
            vault = self.open_vault(root)
            with self._fail_first_manifest_write():
                with self.assertRaises(OSError):
                    vault.seal("first-key", material)
            crash_site = sorted(self._record_files(root))
            self.assertEqual(vault.seal("first-key", material), 1)
            return {
                "crash_site": crash_site,
                "versions": vault.versions("first-key"),
                "active": vault.active("first-key"),
                "material": vault.load("first-key"),
                "records_after": sorted(self._record_files(root)),
            }

        first = crash_then_restart("determinism-a")
        second = crash_then_restart("determinism-b")
        # The relative record layout at the crash site and after restart is
        # built from deterministic names, so it is identical across the two
        # independent directories; every answer is identical too.
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
