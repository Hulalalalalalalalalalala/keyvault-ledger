"""Regression tests for the failed-first-seal scene and the entry contract.

The baseline capabilities (seal, load, revoke, derive, repoint, reload) are
already implemented and are intentionally not touched here.  These cases only
freeze observable behaviour:

* a first seal that fails halfway leaves no half manifest, no half version
  and no orphan material behind: the persisted manifest is byte-for-byte the
  initial empty one, no ``.bin`` or temporary file survives, and the next
  call -- from the same handle, from a fresh handle on the same directory --
  seals successfully as version 1 with the material reading back byte for
  byte;
* two processes racing their first seal of the same new key both succeed and
  the versions come out strictly increasing from 1, never duplicated and
  never skipped, with historical material never overwritten;
* every entry point raises exactly the documented exception type: an empty
  key id is ``ValueError`` and is checked before the version's type; a
  non-genuine-integer version is ``TypeError`` (floats and bools do not
  count, numeric equality does not admit them); an unknown key or version is
  ``KeyError`` for reads, derivation and revocation queries and repoints
  alike; non-bytes material/passphrase/salt is ``TypeError`` and an empty
  salt ``ValueError``; a non-positive iteration count or length is
  ``ValueError`` and a non-genuine-integer one ``TypeError``; a repeated
  revocation, a repoint at the active version and a repoint at a revoked
  version are ``ValueError`` and a failed call appends no record; the
  revoked-version listing of an unknown key is an empty list;
* a missing, unreadable or mismatched material and a malformed activation
  record (missing field, extra field, wrongly typed value, broken JSON)
  make the whole-vault reload fail with ``ValueError`` while the in-memory
  snapshot and the disk records stay untouched; removing or correcting the
  corruption and reloading again restores the one-to-one correspondence
  between snapshot and disk.

Every case works in its own temporary directory, uses the standard library
only, returns its lock handle through the shared fixture and is independent
of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import queue as queue_mod
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


def _first_seal_worker(
    root: str,
    key_id: str,
    payload: bytes,
    barrier: "multiprocessing.managers.Barrier",
    results: "multiprocessing.Queue",
) -> None:
    """Seal exactly one version of ``key_id`` and report the version back."""
    try:
        barrier.wait(timeout=30)
        vault = Vault(root)
        try:
            version = vault.seal(key_id, payload)
        finally:
            vault.close()
    except BaseException as exc:  # surfaced to the parent verbatim
        results.put(("error", key_id, repr(exc)))
        return
    results.put(("ok", key_id, version, payload))


class FirstSealTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.root

    def open_vault(self, root: Path | str | None = None) -> Vault:
        # Tracked so the single fixture cleanup returns the lock handle and
        # removes the temporary tree however the case ends.
        return self.fixture.open(root)

    # ------------------------------------------------------------------
    # observation helpers
    # ------------------------------------------------------------------

    def disk_inventory(self, root: Path) -> dict[str, bytes]:
        """Every vault record on disk keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, so it is deliberately excluded.
        """
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def assert_no_material_or_temp_files(self, root: Path) -> None:
        self.assertEqual(list(root.rglob("*.bin")), [])
        self.assertEqual(list(root.rglob("*.tmp.*")), [])

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

    @staticmethod
    def drain(results: "multiprocessing.Queue") -> list:
        items = []
        while True:
            try:
                items.append(results.get_nowait())
            except queue_mod.Empty:
                return items


# ---------------------------------------------------------------------------
# the scene a failed first seal leaves behind
# ---------------------------------------------------------------------------


class TestFailedFirstSealScene(FirstSealTestCase):
    def _fail_write_for(self, predicate, message: str):
        """Patch ``_atomic_write`` to fail exactly the matching write."""
        real_atomic_write = _atomic_write

        def flaky(path, data):
            if predicate(path):
                raise OSError(message)
            return real_atomic_write(path, data)

        return mock.patch(
            "keyvault_ledger.vault._atomic_write", side_effect=flaky
        )

    def _assert_pristine_fresh_scene(
        self, vault: Vault, initial_disk: dict[str, bytes]
    ) -> None:
        """The directory looks exactly like a just-opened fresh vault."""
        # No half manifest, no half version, no orphan material: the
        # persisted records are byte-for-byte the initial ones and neither a
        # material file nor a temporary write survives.
        self.assertEqual(self.disk_inventory(self.root), initial_disk)
        self.assert_no_material_or_temp_files(self.root)
        self.assertFalse((self.root / ACTIVATIONS_NAME).exists())
        # The in-memory snapshot is the empty one the vault opened with.
        initial_manifest = json.loads(initial_disk[MANIFEST_NAME].decode("utf-8"))
        self.assertEqual(vault.manifest(), initial_manifest)
        self.assertEqual(vault.versions("k"), [])
        self.assertEqual(vault.revoked_versions("k"), [])
        with self.assertRaises(KeyError):
            vault.load("k")
        with self.assertRaises(KeyError):
            vault.active("k")

    def test_failed_material_write_leaves_no_trace_and_retry_is_version_one(self):
        vault = self.open_vault()
        initial_disk = self.disk_inventory(self.root)
        payload = b"first-seal-payload"

        # The material write itself fails halfway through the first seal.
        with self._fail_write_for(
            lambda path: path.suffix == ".bin", "simulated material failure"
        ):
            with self.assertRaises(OSError):
                vault.seal("k", payload)

        self._assert_pristine_fresh_scene(vault, initial_disk)

        # The very next call on the same handle seals version 1.
        self.assertEqual(vault.seal("k", payload), 1)
        self.assertEqual(vault.load("k"), payload)
        self.assertEqual(vault.load("k", 1), payload)
        self.assertEqual(vault.active("k"), 1)
        vault.reload()
        self.assertEqual(vault.load("k", 1), payload)
        # A fresh handle on the same directory reads it back byte for byte.
        reopened = self.open_vault()
        self.assertEqual(reopened.versions("k"), [1])
        self.assertEqual(reopened.load("k", 1), payload)

    def test_failed_manifest_write_removes_orphan_and_retry_is_version_one(self):
        vault = self.open_vault()
        initial_disk = self.disk_inventory(self.root)
        payload = bytes(range(256))

        # The material lands, then the manifest replacement fails: the
        # just-written material must not survive as an orphan.
        with self._fail_write_for(
            lambda path: path.name == MANIFEST_NAME, "simulated manifest failure"
        ):
            with self.assertRaises(OSError):
                vault.seal("k", payload)

        self._assert_pristine_fresh_scene(vault, initial_disk)

        # Retrying from the same directory starts the sequence at 1, not 2.
        self.assertEqual(vault.seal("k", payload), 1)
        self.assertEqual(vault.load("k", 1), payload)
        self.assertEqual(vault.versions("k"), [1])

    def test_failed_first_seal_leaves_directory_fresh_for_a_cold_opener(self):
        vault = self.open_vault()
        initial_disk = self.disk_inventory(self.root)
        payload = b"cold-opener-payload"

        with self._fail_write_for(
            lambda path: path.name == MANIFEST_NAME, "simulated manifest failure"
        ):
            with self.assertRaises(OSError):
                vault.seal("k", payload)
        self._assert_pristine_fresh_scene(vault, initial_disk)

        # A brand-new handle on the same directory opens the still-fresh
        # vault without complaint and its first seal is version 1.
        cold = self.open_vault()
        self.assertEqual(cold.manifest(), vault.manifest())
        self.assertEqual(cold.seal("k", payload), 1)
        self.assertEqual(cold.load("k", 1), payload)
        # The handle that saw the failure picks the record up on reload.
        vault.reload()
        self.assertEqual(vault.versions("k"), [1])
        self.assertEqual(vault.load("k", 1), payload)

    def test_failed_first_derive_seal_leaves_no_trace_and_retries_at_one(self):
        vault = self.open_vault()
        initial_disk = self.disk_inventory(self.root)
        password, salt, iterations, length = b"pw", b"salt-one", 100, 24

        with self._fail_write_for(
            lambda path: path.name == MANIFEST_NAME, "simulated manifest failure"
        ):
            with self.assertRaises(OSError):
                vault.derive_seal("k", password, salt, iterations, length)

        self._assert_pristine_fresh_scene(vault, initial_disk)

        # The retry derives and seals version 1; the bytes read back equal a
        # fresh PBKDF2 run with the same parameters.
        self.assertEqual(
            vault.derive_seal("k", password, salt, iterations, length), 1
        )
        expected = hashlib.pbkdf2_hmac(
            "sha256", password, salt, iterations, dklen=length
        )
        self.assertEqual(vault.load("k", 1), expected)
        self.assertEqual(
            vault.derivation("k", 1),
            {"salt": salt, "iterations": iterations, "length": length},
        )

    def test_repeated_failed_first_seal_is_deterministic(self):
        vault = self.open_vault()
        initial_disk = self.disk_inventory(self.root)

        with self._fail_write_for(
            lambda path: path.name == MANIFEST_NAME, "simulated manifest failure"
        ):
            messages = []
            for _ in range(3):
                with self.assertRaises(OSError) as caught:
                    vault.seal("k", b"payload")
                messages.append(str(caught.exception))
            # The same failing input fails the same way every time, and the
            # scene stays pristine no matter how often the failure repeats.
            self.assertEqual(messages, [messages[0]] * 3)
            self._assert_pristine_fresh_scene(vault, initial_disk)

        self.assertEqual(vault.seal("k", b"payload"), 1)
        self.assertEqual(vault.load("k", 1), b"payload")


# ---------------------------------------------------------------------------
# two processes racing their first seal
# ---------------------------------------------------------------------------


class TestConcurrentFirstSeal(FirstSealTestCase):
    def _race_first_seals(
        self, root: Path, assignments: list[tuple[str, bytes]]
    ) -> dict[tuple[str, int], bytes]:
        """Run one first-seal worker per assignment, all released together.

        Returns ``{(key_id, version): payload}`` as reported by the workers.
        """
        barrier = multiprocessing.Barrier(len(assignments) + 1)
        self.fixture.track_barrier(barrier)
        results: multiprocessing.Queue = multiprocessing.Queue()
        workers = [
            multiprocessing.Process(
                target=_first_seal_worker,
                args=(str(root), key_id, payload, barrier, results),
                name=f"first-seal-{index}",
            )
            for index, (key_id, payload) in enumerate(assignments)
        ]
        self.fixture.track_process(*workers)
        for worker in workers:
            worker.start()
        # Release every worker at the same moment so the seals genuinely
        # race for the lock.
        barrier.wait(timeout=30)
        for worker in workers:
            worker.join(90)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
                self.fail(f"worker {worker.name} hung")
            self.assertEqual(worker.exitcode, 0, f"worker {worker.name}")

        reported = {}
        for item in self.drain(results):
            self.assertEqual(item[0], "ok", item)
            _, key_id, version, payload = item
            reported[(key_id, version)] = payload
        return reported

    def test_two_processes_race_first_seal_of_one_new_key(self):
        self.open_vault()  # initialise the fresh vault directory
        reported = self._race_first_seals(
            self.root, [("k", b"payload-from-A"), ("k", b"payload-from-B")]
        )

        # Both first seals succeeded and the versions came out strictly
        # increasing from 1: no duplicate, no gap.
        self.assertEqual(
            sorted(version for (_, version) in reported), [1, 2]
        )

        vault = self.open_vault()
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.active("k"), 2)
        # Each version holds exactly the payload its winner reported...
        self.assertEqual(vault.load("k", 1), reported[("k", 1)])
        self.assertEqual(vault.load("k", 2), reported[("k", 2)])
        # ...and historical material is never overwritten: version 1 reads
        # back the same bytes across repeated reloads and a cold open.
        for _ in range(3):
            vault.reload()
            self.assertEqual(vault.load("k", 1), reported[("k", 1)])
        reopened = self.open_vault()
        self.assertEqual(reopened.load("k", 1), reported[("k", 1)])
        self.assertEqual(reopened.load("k", 2), reported[("k", 2)])

        # On disk: exactly the two materials and a manifest listing 1, 2.
        materials = sorted(
            path.name for path in (self.root / MATERIALS_DIR).rglob("*.bin")
        )
        self.assertEqual(materials, ["1.bin", "2.bin"])
        manifest = json.loads(
            (self.root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )
        entry = manifest["keys"]["k"]
        self.assertEqual(
            [record["version"] for record in entry["versions"]], [1, 2]
        )
        self.assertEqual(entry["active"], 2)

    def test_two_processes_first_seal_distinct_new_keys_each_start_at_one(self):
        self.open_vault()
        reported = self._race_first_seals(
            self.root, [("alpha", b"alpha-one"), ("beta", b"beta-one")]
        )
        self.assertEqual(
            reported, {("alpha", 1): b"alpha-one", ("beta", 1): b"beta-one"}
        )

        vault = self.open_vault()
        vault.reload()
        for key_id, payload in (("alpha", b"alpha-one"), ("beta", b"beta-one")):
            self.assertEqual(vault.versions(key_id), [1])
            self.assertEqual(vault.active(key_id), 1)
            self.assertEqual(vault.load(key_id, 1), payload)

    def test_two_processes_create_the_vault_and_race_the_first_seal(self):
        # The directory does not exist yet: both processes create the vault
        # itself while racing their first seal of the same new key.
        root = self.fixture.path("raced-vault")
        self.assertFalse(root.exists())
        reported = self._race_first_seals(
            root, [("k", b"raced-one"), ("k", b"raced-two")]
        )
        self.assertEqual(
            sorted(version for (_, version) in reported), [1, 2]
        )

        vault = self.open_vault(root)
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.load("k", 1), reported[("k", 1)])
        self.assertEqual(vault.load("k", 2), reported[("k", 2)])
        vault.reload()
        self.assertEqual(vault.load("k", 1), reported[("k", 1)])


# ---------------------------------------------------------------------------
# the entry-point exception contract, pinned end to end
# ---------------------------------------------------------------------------


class TestEntryExceptionContract(FirstSealTestCase):
    def test_empty_key_id_is_value_error_and_checked_before_the_version(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")

        # Every entry rejects the empty id with ValueError...
        with self.assertRaises(ValueError):
            vault.seal("", b"m")
        with self.assertRaises(ValueError):
            vault.derive_seal("", b"pw", b"salt", 1, 1)
        with self.assertRaises(ValueError):
            vault.load("")
        with self.assertRaises(ValueError):
            vault.load("", 1)
        with self.assertRaises(ValueError):
            vault.derivation("")
        with self.assertRaises(ValueError):
            vault.derivation("", 1)
        with self.assertRaises(ValueError):
            vault.is_revoked("", 1)
        with self.assertRaises(ValueError):
            vault.revoked_versions("")
        with self.assertRaises(ValueError):
            vault.revoke("", 1)
        with self.assertRaises(ValueError):
            vault.set_active("", 1)

        # ...and the identifier is judged before the version's type: an
        # empty id with a non-integer version is still ValueError, never
        # TypeError, at every versioned entry.
        for bad_version in (1.0, True, "1", None, (1,)):
            with self.assertRaises(ValueError, msg=f"load {bad_version!r}"):
                vault.load("", bad_version)
            with self.assertRaises(ValueError, msg=f"derivation {bad_version!r}"):
                vault.derivation("", bad_version)
            with self.assertRaises(ValueError, msg=f"is_revoked {bad_version!r}"):
                vault.is_revoked("", bad_version)
            with self.assertRaises(ValueError, msg=f"revoke {bad_version!r}"):
                vault.revoke("", bad_version)
            with self.assertRaises(ValueError, msg=f"set_active {bad_version!r}"):
                vault.set_active("", bad_version)
        # The write entries validate the identifier ahead of the material
        # and the derivation parameters too.
        with self.assertRaises(ValueError):
            vault.seal("", "not-bytes")
        with self.assertRaises(ValueError):
            vault.derive_seal("", "pw", "salt", "x", "y")

    def test_non_integer_version_is_type_error_floats_bools_never_count(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")

        # 1.0 == 1 and True == 1: numeric equality must not admit them.
        self.assertTrue(1.0 == 1 and True == 1)
        for bad in (1.0, 2.0, 2.5, True, False, "1", (1,)):
            with self.assertRaises(TypeError, msg=f"load {bad!r}"):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("k", bad)
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("k", bad)
            with self.assertRaises(TypeError, msg=f"revoke {bad!r}"):
                vault.revoke("k", bad)
            with self.assertRaises(TypeError, msg=f"set_active {bad!r}"):
                vault.set_active("k", bad)
        # is_revoked has no active-version sentinel: None is illegal there.
        with self.assertRaises(TypeError):
            vault.is_revoked("k", None)
        # The genuine integers still reach the records.
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertFalse(vault.is_revoked("k", 1))

    def test_unknown_key_or_version_is_key_error_with_one_semantics(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)

        # Unknown key: reads, derivation and revocation queries and the
        # repoint all answer KeyError.
        with self.assertRaises(KeyError):
            vault.load("ghost")
        with self.assertRaises(KeyError):
            vault.load("ghost", 1)
        with self.assertRaises(KeyError):
            vault.derivation("ghost")
        with self.assertRaises(KeyError):
            vault.derivation("ghost", 1)
        with self.assertRaises(KeyError):
            vault.is_revoked("ghost", 1)
        with self.assertRaises(KeyError):
            vault.revoke("ghost", 1)
        with self.assertRaises(KeyError):
            vault.set_active("ghost", 1)

        # A genuine-int version that was never sealed: same KeyError.
        for missing in (0, 2, 99, -1):
            with self.assertRaises(KeyError, msg=f"load {missing}"):
                vault.load("k", missing)
            with self.assertRaises(KeyError, msg=f"derivation {missing}"):
                vault.derivation("k", missing)
            with self.assertRaises(KeyError, msg=f"is_revoked {missing}"):
                vault.is_revoked("k", missing)
            with self.assertRaises(KeyError, msg=f"revoke {missing}"):
                vault.revoke("k", missing)
            with self.assertRaises(KeyError, msg=f"set_active {missing}"):
                vault.set_active("k", missing)

    def test_material_password_and_salt_type_rules(self):
        vault = self.open_vault()
        # seal takes bytes-like objects only.
        for bad in ("text", 123, 1.5, None, [b"x"], {"k": b"v"}, object()):
            with self.assertRaises(TypeError, msg=f"material {bad!r}"):
                vault.seal("k", bad)
        # The passphrase and the salt must be genuine bytes: bytearray and
        # memoryview do not pass here even though seal accepts them.
        for bad in ("text", 1, None, bytearray(b"x"), memoryview(b"x")):
            with self.assertRaises(TypeError, msg=f"password {bad!r}"):
                vault.derive_seal("k", bad, b"salt", 1, 1)
            with self.assertRaises(TypeError, msg=f"salt {bad!r}"):
                vault.derive_seal("k", b"pw", bad, 1, 1)
        # An empty salt is a ValueError, distinct from the type checks.
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 1)
        self.assertEqual(vault.versions("k"), [])

    def test_iterations_and_length_must_be_genuine_positive_ints(self):
        vault = self.open_vault()
        # bools and floats are not genuine integers, even 1.0 and True.
        for bad in (True, False, 1.0, 2.5, "1", None, (1,)):
            with self.assertRaises(TypeError, msg=f"iterations {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(TypeError, msg=f"length {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)
        for bad in (0, -1, -1000):
            with self.assertRaises(ValueError, msg=f"iterations {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(ValueError, msg=f"length {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)
        self.assertEqual(vault.versions("k"), [])

    def test_failed_revoke_and_repoint_leave_no_record_behind(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        revocations_before = (self.root / REVOCATIONS_NAME).read_bytes()

        # Repeating the same revocation is a ValueError...
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        # ...and the failed call appended no half record.
        self.assertEqual(
            (self.root / REVOCATIONS_NAME).read_bytes(), revocations_before
        )
        self.assertEqual(vault.revoked_versions("k"), [1])

        # Repointing at the version already active is a ValueError, and so
        # is repointing at a revoked version; neither appends a record.
        self.assertFalse((self.root / ACTIVATIONS_NAME).exists())
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)
        with self.assertRaises(ValueError):
            vault.set_active("k", 1)
        self.assertFalse((self.root / ACTIVATIONS_NAME).exists())
        self.assertEqual(vault.active("k"), 2)

        # A legal repoint appends exactly one record; failing again at the
        # now-active version leaves the journal byte-for-byte untouched.
        vault.seal("k", b"v3")
        vault.set_active("k", 2)
        journal_before = (self.root / ACTIVATIONS_NAME).read_bytes()
        self.assertEqual(len(journal_before.splitlines()), 1)
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)
        self.assertEqual(
            (self.root / ACTIVATIONS_NAME).read_bytes(), journal_before
        )
        self.assertEqual(vault.active("k"), 2)

    def test_revoked_versions_of_unknown_key_is_an_empty_list(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        self.assertEqual(vault.revoked_versions("never-sealed"), [])
        self.assertEqual(vault.revoked_versions("k"), [])
        with self.assertRaises(ValueError):
            vault.revoked_versions("")

    def test_every_rejected_call_leaves_disk_and_snapshot_untouched(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        vault.revoke("k", 1)
        vault.set_active("k", 2)  # legal: neither revoked nor active

        disk_before = self.disk_inventory(self.root)
        answers_before = {
            "versions-k": vault.versions("k"),
            "versions-d": vault.versions("d"),
            "active-k": vault.active("k"),
            "active-d": vault.active("d"),
            "revoked-k": vault.revoked_versions("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "load-k3": vault.load("k", 3),
            "derivation-d": vault.derivation("d", 1),
            "manifest": vault.manifest(),
        }

        rejected = (
            (ValueError, lambda: vault.seal("", b"m")),
            (TypeError, lambda: vault.seal("k", "not-bytes")),
            (ValueError, lambda: vault.derive_seal("", b"pw", b"salt", 1, 1)),
            (TypeError, lambda: vault.derive_seal("k", "pw", b"salt", 1, 1)),
            (ValueError, lambda: vault.derive_seal("k", b"pw", b"", 1, 1)),
            (TypeError, lambda: vault.derive_seal("k", b"pw", b"salt", 1.0, 1)),
            (ValueError, lambda: vault.derive_seal("k", b"pw", b"salt", 0, 1)),
            (ValueError, lambda: vault.load("")),
            (ValueError, lambda: vault.load("", 1.0)),
            (TypeError, lambda: vault.load("k", 1.0)),
            (TypeError, lambda: vault.load("k", True)),
            (KeyError, lambda: vault.load("ghost")),
            (KeyError, lambda: vault.load("k", 99)),
            (ValueError, lambda: vault.derivation("")),
            (TypeError, lambda: vault.derivation("k", 1.0)),
            (KeyError, lambda: vault.derivation("ghost")),
            (KeyError, lambda: vault.derivation("k", 99)),
            (ValueError, lambda: vault.is_revoked("", 1)),
            (TypeError, lambda: vault.is_revoked("k", None)),
            (KeyError, lambda: vault.is_revoked("ghost", 1)),
            (KeyError, lambda: vault.is_revoked("k", 99)),
            (ValueError, lambda: vault.revoked_versions("")),
            (ValueError, lambda: vault.revoke("", 1)),
            (TypeError, lambda: vault.revoke("k", 1.0)),
            (KeyError, lambda: vault.revoke("ghost", 1)),
            (KeyError, lambda: vault.revoke("k", 99)),
            (ValueError, lambda: vault.revoke("k", 1)),  # already revoked
            (ValueError, lambda: vault.set_active("", 1)),
            (TypeError, lambda: vault.set_active("k", 1.0)),
            (KeyError, lambda: vault.set_active("ghost", 1)),
            (KeyError, lambda: vault.set_active("k", 99)),
            (ValueError, lambda: vault.set_active("k", 2)),  # already active
            (ValueError, lambda: vault.set_active("k", 1)),  # revoked target
        )

        for index, (expected_type, call) in enumerate(rejected):
            with self.assertRaises(expected_type, msg=f"rejected call {index}"):
                call()

        # The disk records neither gained nor lost a byte...
        self.assertEqual(self.disk_inventory(self.root), disk_before)
        # ...and every observable answer is exactly what it was before.
        answers_after = {
            "versions-k": vault.versions("k"),
            "versions-d": vault.versions("d"),
            "active-k": vault.active("k"),
            "active-d": vault.active("d"),
            "revoked-k": vault.revoked_versions("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "load-k3": vault.load("k", 3),
            "derivation-d": vault.derivation("d", 1),
            "manifest": vault.manifest(),
        }
        self.assertEqual(answers_after, answers_before)
        self.assertEqual(vault.revoked_versions("k"), [1])
        self.assertEqual(vault.active("k"), 2)


# ---------------------------------------------------------------------------
# reload failure invariance and recovery
# ---------------------------------------------------------------------------


class TestReloadFailureInvariance(FirstSealTestCase):
    def _build_healthy(self, name: str) -> Vault:
        root = self.tmp_path / name
        vault = self.open_vault(root)
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        vault.set_active("k", 1)  # bound to latest=2
        vault.derive_seal("d", b"pw-one", b"salt-one", 100, 24)
        vault.derive_seal("d", b"pw-two", b"salt-two", 200, 32)
        vault.revoke("d", 1)
        return vault

    def _answers(self, vault: Vault) -> dict:
        keys = {}
        for key_id in ("k", "d"):
            versions = vault.versions(key_id)
            keys[key_id] = {
                "versions": versions,
                "active": vault.active(key_id),
                "revoked": vault.revoked_versions(key_id),
                "materials": {v: vault.load(key_id, v) for v in versions},
                "derivations": {v: vault.derivation(key_id, v) for v in versions},
            }
        return {
            "keys": keys,
            "manifest": vault.manifest(),
            "unknown_versions": vault.versions("never-sealed"),
            "unknown_revoked": vault.revoked_versions("never-sealed"),
        }

    def _restore_disk(self, root: Path, healthy: dict[str, bytes]) -> None:
        """Put ``root`` back in exactly the captured healthy state."""
        for path in root.rglob("*"):
            if path.name == LOCK_NAME:
                continue
            rel = str(path.relative_to(root))
            if path.is_dir():
                if rel in healthy:
                    # A directory squatting where a record file belongs.
                    path.rmdir()
            elif rel not in healthy:
                path.unlink()
        for rel, data in healthy.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)

    def _assert_corresponds_to_disk(self, vault: Vault, root: Path) -> None:
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
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), record["sha256"]
                )
            expected_active = entry["active"]
            if key_id in last_repoint:
                target, bound_latest = last_repoint[key_id]
                if bound_latest == versions[-1]:
                    expected_active = target
            self.assertEqual(vault.active(key_id), expected_active)
            self.assertEqual(
                set(vault.revoked_versions(key_id)),
                revoked.get(key_id, set()),
            )

    def _assert_failure_then_recovery(
        self, vault: Vault, root: Path, corrupt
    ) -> None:
        expected = self._answers(vault)
        healthy_disk = self.disk_inventory(root)

        corrupt()
        failing_disk = self.disk_inventory(root)

        # The failure is deterministic: same type, same message, and every
        # observable answer frozen across repeated failed reloads.
        message = None
        for _ in range(3):
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            if message is None:
                message = str(caught.exception)
            else:
                self.assertEqual(str(caught.exception), message)
            self.assertEqual(self._answers(vault), expected)

        # The keys already in hand stay readable while the failure persists.
        self.assertEqual(vault.load("k"), b"k-v1")
        self.assertEqual(vault.load("k", 2), b"k-v2")

        # A cold opener on the same directory raises the same ValueError.
        with self.assertRaises(ValueError) as caught:
            Vault(root)
        self.assertEqual(str(caught.exception), message)

        # The failed reloads neither added nor removed a disk record.
        self.assertEqual(self.disk_inventory(root), failing_disk)

        # Undoing the corruption restores the vault: one reload and the
        # snapshot corresponds to the disk records one to one again.
        self._restore_disk(root, healthy_disk)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self._assert_corresponds_to_disk(vault, root)
        reopened = self.open_vault(root)
        self.assertEqual(self._answers(reopened), expected)
        self._assert_corresponds_to_disk(reopened, root)
        self.assertEqual(self.disk_inventory(root), healthy_disk)

    def _material_file(self, root: Path, key_id: str, version: int) -> Path:
        manifest = json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))
        record = next(
            r for r in manifest["keys"][key_id]["versions"] if r["version"] == version
        )
        return root / record["file"]

    def test_missing_material_fails_reload_and_recovers(self):
        vault = self._build_healthy("reload-missing")
        root = self.tmp_path / "reload-missing"
        target = self._material_file(root, "k", 1)
        self._assert_failure_then_recovery(vault, root, target.unlink)

    def test_mismatched_material_fails_reload_and_recovers(self):
        vault = self._build_healthy("reload-mismatch")
        root = self.tmp_path / "reload-mismatch"
        target = self._material_file(root, "d", 2)
        self._assert_failure_then_recovery(
            vault, root, lambda: target.write_bytes(b"tampered material")
        )

    def test_unreadable_material_fails_reload_and_recovers(self):
        vault = self._build_healthy("reload-unreadable")
        root = self.tmp_path / "reload-unreadable"
        target = self._material_file(root, "k", 2)

        def corrupt():
            # A directory where the material file should be cannot be read
            # back as bytes on any platform.
            target.unlink()
            target.mkdir()

        self._assert_failure_then_recovery(vault, root, corrupt)

    def test_malformed_activation_records_fail_reload_and_recover(self):
        bad_payloads = {
            # a field missing from the documented three-field shape
            "missing_key_id": b'{"version": 1, "latest": 2}\n',
            "missing_version": b'{"key_id": "k", "latest": 2}\n',
            "missing_latest": b'{"key_id": "k", "version": 1}\n',
            # an extra field beyond the documented three
            "extra_field": (
                b'{"key_id": "k", "version": 1, "latest": 2, "note": "x"}\n'
            ),
            # wrongly typed values
            "float_version": b'{"key_id": "k", "version": 1.0, "latest": 2}\n',
            "bool_version": b'{"key_id": "k", "version": true, "latest": 2}\n',
            "string_latest": b'{"key_id": "k", "version": 1, "latest": "2"}\n',
            "null_latest": b'{"key_id": "k", "version": 1, "latest": null}\n',
            # broken JSON
            "broken_json": b"{not json\n",
        }
        for index, (name, payload) in enumerate(bad_payloads.items()):
            with self.subTest(case=name):
                root = self.tmp_path / f"reload-activations-{index}"
                vault = self._build_healthy(f"reload-activations-{index}")
                self._assert_failure_then_recovery(
                    vault,
                    root,
                    lambda payload=payload, root=root: (
                        root / ACTIVATIONS_NAME
                    ).write_bytes(payload),
                )


if __name__ == "__main__":
    unittest.main()
