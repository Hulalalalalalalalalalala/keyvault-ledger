"""Regression tests pinning the active-version query to the shared
empty-id rule of every other read query.

Baseline state: every read query raised ``ValueError`` for an empty key
id except ``active``, which let the empty id fall through to the snapshot
lookup and answered ``KeyError``.  The entry validation of ``active`` was
aligned with the other reads, so this module nails down:

* an empty key id raises ``ValueError`` when it actually reaches each read
  entry -- ``load`` (with and without the active-version sentinel),
  ``derivation`` (likewise), ``active``, ``is_revoked`` and
  ``revoked_versions`` -- and the shared message is
  ``"key_id must not be empty"``; ``versions`` is the one read without an
  id gate and keeps answering an empty list for an empty or unknown id;

* non-empty behaviour is word-for-word unchanged: the active version is the
  most recently sealed one, a repoint makes ``active`` report the pointed-at
  historical version, sealing new material afterwards lands on a fresh
  version and becomes active again as if never repointed, and all of that
  survives reopening the vault;

* the rest of the read vocabulary is untouched: unknown key / nonexistent
  version -> ``KeyError``, a version that is not a genuine integer ->
  ``TypeError`` (bools and floats do not count), ``revoked_versions`` of an
  unknown key -> ``[]``, and ``derivation`` of a directly sealed version ->
  ``{}``;

* every blocked call is strictly read-only: the manifest, both journals,
  the materials and their mtimes neither grow nor shrink and the in-memory
  snapshot answers identically before and after; material written earlier
  reads back byte for byte;

* a whole-vault reload that fails (a corrupt activity journal) keeps the
  snapshot frozen and the disk untouched, and after the journal is restored
  a reload and a cold reopen both succeed with snapshot and disk records
  corresponding one to one;

* the same sequence of calls raises the same exception types on every
  repetition, in any test order.

Everything happens inside temporary directories, uses the standard library
only, returns its lock handles through the shared fixture and is
independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import base64
import json
import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    ACTIVATIONS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    REVOCATIONS_NAME,
)
from tests._fixtures import VaultFixture


class ActiveEmptyIdTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.root

    def open_vault(self, root: Path | None = None) -> Vault:
        # Tracked so the single fixture cleanup returns the lock handle and
        # removes the temporary tree however the case ends.
        return self.fixture.open(root)

    def _disk_bytes(self) -> dict:
        files = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                files[path.relative_to(self.root)] = (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
        return files

    def _rich_vault(self) -> Vault:
        """Key ``k``: two plain versions, repointed at version 1.

        Key ``d`` carries one derived version; ``plain`` one direct version.
        """
        vault = self.open_vault()
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        vault.set_active("k", 1)
        vault.derive_seal("d", b"pw", b"salty", 100, 16)
        vault.seal("plain", b"p1")
        return vault


class TestEmptyKeyIdIsValueErrorAtEveryReadEntry(ActiveEmptyIdTestCase):
    def test_each_read_entry_actually_raises_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        # Every entry is triggered for real, one assertion each: the empty
        # id is an argument problem (ValueError), including the queries that
        # accept the active-version sentinel and omit a version altogether.
        with self.assertRaises(ValueError):
            vault.active("")
        with self.assertRaises(ValueError):
            vault.load("")
        with self.assertRaises(ValueError):
            vault.load("", None)
        with self.assertRaises(ValueError):
            vault.load("", version=None)
        with self.assertRaises(ValueError):
            vault.derivation("")
        with self.assertRaises(ValueError):
            vault.derivation("", None)
        with self.assertRaises(ValueError):
            vault.derivation("", version=None)
        with self.assertRaises(ValueError):
            vault.is_revoked("", 1)
        with self.assertRaises(ValueError):
            vault.revoked_versions("")

    def test_active_shares_the_shared_empty_id_message(self):
        vault = self.open_vault()
        for call in (
            lambda: vault.active(""),
            lambda: vault.load(""),
            lambda: vault.derivation(""),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
        ):
            with self.assertRaises(ValueError) as caught:
                call()
            self.assertEqual(str(caught.exception), "key_id must not be empty")

    def test_active_empty_id_is_checked_before_anything_else(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        # The id gate fires at the entry: no state, sealed or revoked, can
        # change the exception type.  (active itself carries no version
        # argument, so there is nothing for it to be ordered against.)
        with self.assertRaises(ValueError):
            vault.active("")

    def test_versions_keeps_treating_empty_id_as_an_unknown_key(self):
        # ``versions`` is the one read query without an id gate: its empty
        # id answer is the same empty list an unknown non-empty id gets.
        vault = self.open_vault()
        vault.seal("k", b"m")
        self.assertEqual(vault.versions(""), [])
        self.assertEqual(vault.versions("never-sealed"), [])
        self.assertEqual(vault.versions("k"), [1])


class TestNonEmptyBehaviourIsUnchanged(ActiveEmptyIdTestCase):
    def test_active_is_the_latest_sealed_version(self):
        vault = self.open_vault()
        self.assertEqual(vault.seal("k", b"v1"), 1)
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.seal("k", b"v2"), 2)
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.derive_seal("k", b"pw", b"salt", 100, 8), 3)
        self.assertEqual(vault.active("k"), 3)

    def test_repoint_then_new_seal_moves_active_and_allocates_new_version(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        # Repointed key reports the version it points back at.
        vault.set_active("k", 1)
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"v1")
        self.assertEqual(vault.load("k", 2), b"v2")
        # Sealing new material lands on the next version and becomes active
        # again, exactly as if the key had never been repointed.
        self.assertEqual(vault.seal("k", b"v3"), 3)
        self.assertEqual(vault.active("k"), 3)
        self.assertEqual(vault.load("k"), b"v3")
        self.assertEqual(vault.load("k", 1), b"v1")

    def test_repoint_and_active_survive_reopen(self):
        vault = self._rich_vault()
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.active("d"), 1)
        vault.close()
        reopened = self.open_vault()
        self.assertEqual(reopened.active("k"), 1)
        self.assertEqual(reopened.load("k"), b"k-v1")
        self.assertEqual(reopened.active("d"), 1)
        self.assertEqual(reopened.active("plain"), 1)
        # A later seal after reopening still supersedes the repoint.
        self.assertEqual(reopened.seal("k", b"k-v3"), 3)
        self.assertEqual(reopened.active("k"), 3)

    def test_unknown_non_empty_id_still_key_error(self):
        vault = self._rich_vault()
        with self.assertRaises(KeyError):
            vault.active("never-sealed")
        with self.assertRaises(KeyError):
            vault.load("never-sealed")
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed")
        with self.assertRaises(KeyError):
            vault.is_revoked("never-sealed", 1)
        for missing in (0, 99, -1):
            with self.assertRaises(KeyError, msg=f"active n/a; load {missing}"):
                vault.load("k", missing)
            with self.assertRaises(KeyError, msg=f"is_revoked {missing}"):
                vault.is_revoked("k", missing)
            with self.assertRaises(KeyError, msg=f"derivation {missing}"):
                vault.derivation("k", missing)

    def test_non_integer_version_still_type_error_at_the_versioned_reads(self):
        vault = self._rich_vault()
        # For load/derivation None is the documented active-version
        # sentinel, not an illegal version; everything else non-genuine-int
        # is rejected with TypeError before the lookup.
        for bad in (1.0, True, False, "1"):
            with self.assertRaises(TypeError, msg=f"load {bad!r}"):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("k", bad)
        # is_revoked has no sentinel: even a genuine None is an illegal
        # version there, and a float numerically equal to version 1 fails.
        for bad in (1.0, True, False, "1", None):
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("k", bad)
        # The sentinel still resolves to the active version on the two
        # reads that have one.
        self.assertEqual(vault.load("k", None), b"k-v1")
        self.assertEqual(vault.derivation("k", None), {})

    def test_other_read_semantics_keep_their_answers(self):
        vault = self._rich_vault()
        # Unknown keys give an empty revoked listing, not an error.
        self.assertEqual(vault.revoked_versions("never-sealed"), [])
        vault.revoke("plain", 1)
        self.assertEqual(vault.revoked_versions("plain"), [1])
        # A directly sealed version answers the derivation query with {}.
        self.assertEqual(vault.derivation("plain", 1), {})
        self.assertEqual(vault.derivation("k", 1), {})
        # The derived version still hands back its parameters and bytes.
        record = vault.derivation("d", 1)
        self.assertEqual(
            record, {"salt": b"salty", "iterations": 100, "length": 16}
        )
        self.assertFalse(vault.is_revoked("d", 1))


class TestBlockedReadsAreReadOnlyAndDeterministic(ActiveEmptyIdTestCase):
    def test_blocked_calls_touch_neither_disk_nor_snapshot(self):
        vault = self._rich_vault()
        before_files = self._disk_bytes()
        snapshot_before = {
            "versions-k": vault.versions("k"),
            "versions-d": vault.versions("d"),
            "versions-plain": vault.versions("plain"),
            "active-k": vault.active("k"),
            "active-d": vault.active("d"),
            "active-plain": vault.active("plain"),
            "revoked-k": vault.revoked_versions("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "derivation-d": vault.derivation("d", 1),
            "manifest": vault.manifest(),
        }

        rejected = (
            lambda: vault.active(""),
            lambda: vault.load(""),
            lambda: vault.load("", None),
            lambda: vault.derivation(""),
            lambda: vault.derivation("", None),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
            lambda: vault.active("ghost"),
            lambda: vault.load("ghost"),
            lambda: vault.derivation("ghost"),
            lambda: vault.is_revoked("ghost", 1),
            lambda: vault.load("k", 99),
            lambda: vault.derivation("k", 99),
            lambda: vault.is_revoked("k", 99),
            lambda: vault.load("k", 1.0),
            lambda: vault.derivation("k", True),
            lambda: vault.is_revoked("k", None),
        )
        for call in rejected:
            with self.assertRaises((ValueError, TypeError, KeyError)):
                call()

        # Same files, same bytes, untouched mtimes: the blocked calls left
        # no trace on the manifest, either journal or any material.
        self.assertEqual(sorted(self._disk_bytes()), sorted(before_files))
        for rel, (data, mtime) in before_files.items():
            path = self.root / rel
            self.assertEqual(path.read_bytes(), data, str(rel))
            self.assertEqual(path.stat().st_mtime_ns, mtime, str(rel))

        snapshot_after = {
            "versions-k": vault.versions("k"),
            "versions-d": vault.versions("d"),
            "versions-plain": vault.versions("plain"),
            "active-k": vault.active("k"),
            "active-d": vault.active("d"),
            "active-plain": vault.active("plain"),
            "revoked-k": vault.revoked_versions("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "derivation-d": vault.derivation("d", 1),
            "manifest": vault.manifest(),
        }
        self.assertEqual(snapshot_after, snapshot_before)

    def test_existing_material_reads_back_byte_for_byte(self):
        vault = self.open_vault()
        payloads = [b"", bytes(range(256)), b"\x00\xff\nsuffix", "λ-key".encode()]
        for payload in payloads:
            vault.seal("k", payload)
        for call in (
            lambda: vault.active(""),
            lambda: vault.load(""),
            lambda: vault.derivation(""),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
            lambda: vault.active("ghost"),
        ):
            with self.assertRaises((ValueError, KeyError)):
                call()
        for index, payload in enumerate(payloads, start=1):
            self.assertIs(type(vault.load("k", index)), bytes)
            self.assertEqual(vault.load("k", index), payload)

    def test_exception_sequence_is_deterministic(self):
        vault = self._rich_vault()
        calls = (
            lambda: vault.active(""),
            lambda: vault.active("ghost"),
            lambda: vault.load(""),
            lambda: vault.load("k", 1.0),
            lambda: vault.load("ghost", 1),
            lambda: vault.derivation(""),
            lambda: vault.derivation("k", True),
            lambda: vault.derivation("ghost"),
            lambda: vault.is_revoked("", 1),
            lambda: vault.is_revoked("k", None),
            lambda: vault.is_revoked("ghost", 1),
            lambda: vault.revoked_versions(""),
        )
        expected = [
            ValueError,  # active empty id (the aligned entry)
            KeyError,    # active unknown non-empty id
            ValueError,  # load empty id
            TypeError,   # load float version
            KeyError,    # load unknown key
            ValueError,  # derivation empty id
            TypeError,   # derivation bool version
            KeyError,    # derivation unknown key
            ValueError,  # is_revoked empty id
            TypeError,   # is_revoked None version
            KeyError,    # is_revoked unknown key
            ValueError,  # revoked listing empty id
        ]

        def run() -> list:
            seen = []
            for call in calls:
                try:
                    call()
                except Exception as exc:  # noqa: BLE001 - the type is the result
                    seen.append(type(exc))
                else:
                    seen.append("no exception")
            return seen

        first, second, third = run(), run(), run()
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(third, expected)


class TestReloadFailureThenRecoveryRoundTrip(ActiveEmptyIdTestCase):
    def _answers(self, vault: Vault) -> dict:
        """Every public read answer, in a comparable structure."""
        keys = {}
        for key_id in ("k", "d", "plain"):
            versions = vault.versions(key_id)
            keys[key_id] = {
                "versions": versions,
                "active": vault.active(key_id),
                "revoked": vault.revoked_versions(key_id),
                "materials": {v: vault.load(key_id, v) for v in versions},
                "derivations": {
                    v: vault.derivation(key_id, v) for v in versions
                },
            }
        return {
            "keys": keys,
            "manifest": vault.manifest(),
            "unknown-versions": vault.versions("never-sealed"),
            "unknown-revoked": vault.revoked_versions("never-sealed"),
        }

    def _disk_records(self) -> dict[Path, bytes]:
        return {
            path: path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def test_failed_reload_freezes_snapshot_then_recovery_matches_disk(self):
        vault = self._rich_vault()
        expected = self._answers(vault)
        healthy_disk = self._disk_records()

        # A corrupt activity journal makes the whole reload fail.
        journal = self.root / ACTIVATIONS_NAME
        journal.write_bytes(b"{not json\n")

        # During the failure: ValueError every time, snapshot frozen word
        # for word, disk neither grown nor shrunk, and a cold opener on the
        # same directory fails the same way.
        damaged_disk = self._disk_records()
        message = None
        for _ in range(3):
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            if message is None:
                message = str(caught.exception)
            else:
                self.assertEqual(str(caught.exception), message)
            self.assertEqual(self._answers(vault), expected)
            # The aligned entry still reports the repointed version from
            # the frozen snapshot, and the empty id still fails at entry.
            self.assertEqual(vault.active("k"), 1)
            with self.assertRaises(ValueError):
                vault.active("")
        with self.assertRaises(ValueError) as caught:
            Vault(self.root)
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(self._disk_records(), damaged_disk)

        # Restore the healthy journal: a reload succeeds and the snapshot
        # corresponds to the disk records one to one.
        journal.write_bytes(healthy_disk[self.root / ACTIVATIONS_NAME])
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self.assertEqual(self._disk_records(), healthy_disk)
        self._assert_snapshot_matches_disk(vault)

        # A cold reopen after recovery gives the identical answers and also
        # matches the disk records one to one.
        reopened = self.open_vault()
        self.assertEqual(self._answers(reopened), expected)
        self._assert_snapshot_matches_disk(reopened)

    def _assert_snapshot_matches_disk(self, vault: Vault) -> None:
        """Re-derive every answer independently from the on-disk records."""
        manifest = json.loads(
            (self.root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )
        self.assertEqual(vault.manifest(), manifest)

        last_repoint: dict[str, tuple[int, int]] = {}
        activations = self.root / ACTIVATIONS_NAME
        if activations.exists():
            for line in activations.read_text("utf-8").splitlines():
                record = json.loads(line)
                last_repoint[record["key_id"]] = (
                    record["version"],
                    record["latest"],
                )

        revoked_map: dict[str, set[int]] = {}
        for line in (self.root / REVOCATIONS_NAME).read_text("utf-8").splitlines():
            record = json.loads(line)
            revoked_map.setdefault(record["key_id"], set()).add(record["version"])

        for key_id, entry in manifest["keys"].items():
            records = entry["versions"]
            versions = [record["version"] for record in records]
            self.assertEqual(vault.versions(key_id), versions)
            for record in records:
                version = record["version"]
                data = (self.root / record["file"]).read_bytes()
                # The snapshot serves the exact persisted bytes.
                self.assertEqual(vault.load(key_id, version), data)
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
                    # A directly sealed version has no derivation record.
                    self.assertEqual(parameters, {})
            expected_active = entry["active"]
            if key_id in last_repoint:
                target, bound_latest = last_repoint[key_id]
                if bound_latest == versions[-1]:
                    expected_active = target
            self.assertEqual(vault.active(key_id), expected_active)
            self.assertEqual(
                set(vault.revoked_versions(key_id)),
                revoked_map.get(key_id, set()),
            )


if __name__ == "__main__":
    unittest.main()
