"""Regression tests pinning the documented ``activations.jsonl`` record
shape and binding rules.

The README documents the activation journal publicly: one JSON object per
line with exactly the fields ``key_id`` (the key identifier), ``version``
(the version pointed at) and ``latest`` (the key's newest sealed version at
write time, the binding value), serialized with sorted keys, appended in
file order.  A record is honoured only while its ``latest`` still equals
the key's newest sealed version; a vault that was never repointed has no
such file.  These cases freeze exactly that contract:

* a hand-written record that matches the documented shape applies on a
  whole-vault ``reload()`` -- the repoint takes effect;
* repeated repoints append one record each, the last one wins, and the
  historical records stay in the file byte-for-byte, never rewritten;
* a record with a missing field, an extra field, a wrongly typed field,
  broken JSON, a pointer at a key/version that never existed, or a
  ``latest`` beyond the key's newest sealed version at that point makes
  ``reload()`` raise exactly ``ValueError`` -- the in-memory snapshot stays
  frozen, the keys already in hand stay readable, and the disk records
  neither grow nor shrink;
* removing or correcting the bad record and reloading recovers: the
  snapshot and the disk records correspond one-to-one again.

The repointing behaviour itself (its exception vocabulary, the append-only
write, the reload swap semantics) is intentionally not touched here.

Everything happens inside a temporary directory, uses the standard library
only and is independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import ACTIVATIONS_NAME, LOCK_NAME, MANIFEST_NAME
from tests._fixtures import VaultFixture

APP_MATERIALS = {1: b"app-one", 2: b"app-two", 3: b"app-three"}
OTHER_MATERIAL = b"other-one"


class ActivationJournalRecordTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        # Each scenario gets its own vault subdirectory, so cases never
        # share disk state and execution order cannot matter.
        self.root = self.fixture.tmp_path / "vaults"

    def _open(self, root: Path) -> Vault:
        """Open a vault through the case's single teardown path."""
        return self.fixture.open(root)

    # ------------------------------------------------------------------
    # fixture construction
    # ------------------------------------------------------------------

    def _build(self, name: str, *, repoint: bool) -> tuple[Vault, Path]:
        """Build a healthy vault and return ``(handle, root)``.

        Key ``app`` has versions 1..3 (active 3), key ``other`` has one
        version.  With ``repoint=True`` the active version of ``app`` is
        repointed at 1, so the vault holds one valid journal record.
        """
        root = self.root / name
        vault = self._open(root)
        for version in sorted(APP_MATERIALS):
            vault.seal("app", APP_MATERIALS[version])
        vault.seal("other", OTHER_MATERIAL)
        if repoint:
            vault.set_active("app", 1)  # bound to latest=3
        return vault, root

    # ------------------------------------------------------------------
    # observable-state snapshots
    # ------------------------------------------------------------------

    def _answers(self, vault: Vault) -> dict:
        """Every state a reader can query, in a comparable plain structure."""
        keys = {}
        for key_id in ("app", "other"):
            versions = vault.versions(key_id)
            keys[key_id] = {
                "versions": versions,
                "active": vault.active(key_id),
                "materials": {v: vault.load(key_id, v) for v in versions},
            }
        return {
            "keys": keys,
            "manifest": vault.manifest(),
            "unknown_versions": vault.versions("never-sealed"),
        }

    def _disk_bytes(self, root: Path) -> dict[str, bytes]:
        """All vault records on disk keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, so it is deliberately excluded.
        """
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def _restore_disk(self, root: Path, healthy: dict[str, bytes]) -> None:
        """Put ``root`` back in exactly the captured healthy state."""
        current = self._disk_bytes(root)
        for rel in current.keys() - healthy.keys():
            (root / rel).unlink()
        for rel, data in healthy.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)

    # ------------------------------------------------------------------
    # independent snapshot<->disk correspondence check
    # ------------------------------------------------------------------

    def _assert_corresponds(self, vault: Vault, root: Path) -> None:
        """Re-derive every observable answer straight from the disk records.

        This never consults the vault's private state: it parses the
        manifest and the activation journal itself and requires the public
        snapshot to match exactly.
        """
        manifest = json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))
        self.assertEqual(vault.manifest(), manifest)

        last_repoint: dict[str, tuple[int, int]] = {}
        journal = root / ACTIVATIONS_NAME
        if journal.exists():
            for line in journal.read_text("utf-8").splitlines():
                record = json.loads(line)
                self.assertEqual(set(record), {"key_id", "version", "latest"})
                last_repoint[record["key_id"]] = (
                    record["version"],
                    record["latest"],
                )

        for key_id, entry in manifest["keys"].items():
            records = entry["versions"]
            versions = [record["version"] for record in records]
            self.assertEqual(vault.versions(key_id), versions)
            for record in records:
                # The snapshot serves the exact bytes persisted on disk.
                self.assertEqual(
                    vault.load(key_id, record["version"]),
                    (root / record["file"]).read_bytes(),
                )
            expected_active = entry["active"]
            if key_id in last_repoint:
                target, bound_latest = last_repoint[key_id]
                if bound_latest == versions[-1]:
                    expected_active = target
            self.assertEqual(vault.active(key_id), expected_active)

    # ------------------------------------------------------------------
    # the core failure/recovery contract
    # ------------------------------------------------------------------

    def _assert_rejected_then_recovers(
        self,
        vault: Vault,
        root: Path,
        expected: dict,
        corrupt,
    ) -> None:
        """Drive one full corruption window and the recovery afterwards."""
        healthy_disk = self._disk_bytes(root)

        corrupt()
        failing_disk = self._disk_bytes(root)

        # Three failed reloads: the exception type and message are
        # deterministic, and every observable answer stays frozen.
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

        # The failed reloads/open neither added nor removed a disk record:
        # the directory is byte-for-byte the corrupted state just captured.
        self.assertEqual(self._disk_bytes(root), failing_disk)

        # Undo the corruption: one successful reload restores full
        # correspondence between snapshot and disk.
        self._restore_disk(root, healthy_disk)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self._assert_corresponds(vault, root)
        reopened = self._open(root)
        self.assertEqual(self._answers(reopened), expected)
        self._assert_corresponds(reopened, root)
        self.assertEqual(self._disk_bytes(root), healthy_disk)


# ---------------------------------------------------------------------------
# the documented record shape, byte for byte
# ---------------------------------------------------------------------------


class TestRecordShape(ActivationJournalRecordTestCase):
    def test_no_journal_until_first_repoint(self):
        vault, root = self._build("no-journal", repoint=False)
        journal = root / ACTIVATIONS_NAME
        # A vault that was never repointed simply has no such file...
        self.assertFalse(journal.exists())
        vault.reload()
        self.assertFalse(journal.exists())
        # ...and a failed repoint (here: the version already active)
        # leaves no record behind, not even half of one.
        with self.assertRaises(ValueError):
            vault.set_active("app", 3)
        self.assertFalse(journal.exists())
        self.assertEqual(vault.active("app"), 3)

    def test_record_lands_byte_for_byte_as_documented(self):
        vault, root = self._build("record-bytes", repoint=False)
        vault.set_active("app", 2)
        # Exactly one line, fields serialized with sorted keys, newline
        # terminated -- byte-for-byte the documented shape.
        self.assertEqual(
            (root / ACTIVATIONS_NAME).read_bytes(),
            b'{"key_id": "app", "latest": 3, "version": 2}\n',
        )
        self.assertEqual(vault.active("app"), 2)
        self.assertEqual(vault.load("app"), APP_MATERIALS[2])

    def test_handwritten_valid_record_applies_after_reload(self):
        vault, root = self._build("handwritten", repoint=False)
        expected_before = self._answers(vault)
        # A record written out of band, exactly matching the documented
        # shape, is honoured by the next whole-vault reload.
        (root / ACTIVATIONS_NAME).write_bytes(
            b'{"key_id": "app", "latest": 3, "version": 2}\n'
        )
        vault.reload()
        self.assertEqual(vault.active("app"), 2)
        self.assertEqual(vault.load("app"), APP_MATERIALS[2])
        # Everything else is exactly as before the hand-written repoint.
        self.assertEqual(vault.versions("app"), [1, 2, 3])
        self.assertEqual(vault.load("app", 3), APP_MATERIALS[3])
        self.assertEqual(vault.active("other"), 1)
        self.assertEqual(vault.manifest(), expected_before["manifest"])
        self._assert_corresponds(vault, root)
        # A cold opener lands on the same repointed state.
        reopened = self._open(root)
        self.assertEqual(reopened.active("app"), 2)
        self.assertEqual(reopened.load("app"), APP_MATERIALS[2])
        self._assert_corresponds(reopened, root)

    def test_repeated_repoints_last_wins_and_history_is_kept(self):
        vault, root = self._build("repeated", repoint=False)
        vault.set_active("app", 1)
        vault.set_active("app", 2)
        journal = root / ACTIVATIONS_NAME
        # One appended record per repoint, in append order; the historical
        # record is never rewritten.
        expected_bytes = (
            b'{"key_id": "app", "latest": 3, "version": 1}\n'
            b'{"key_id": "app", "latest": 3, "version": 2}\n'
        )
        self.assertEqual(journal.read_bytes(), expected_bytes)
        self.assertEqual(vault.active("app"), 2)

        # The last record wins, and the reload is read-only: the journal
        # keeps both records byte-for-byte.
        vault.reload()
        self.assertEqual(vault.active("app"), 2)
        self.assertEqual(vault.load("app"), APP_MATERIALS[2])
        self.assertEqual(journal.read_bytes(), expected_bytes)
        reopened = self._open(root)
        self.assertEqual(reopened.active("app"), 2)
        self.assertEqual(journal.read_bytes(), expected_bytes)
        self._assert_corresponds(reopened, root)

    def test_later_seal_supersedes_repoint_and_record_stays(self):
        vault, root = self._build("superseded", repoint=True)
        journal = root / ACTIVATIONS_NAME
        record_bytes = journal.read_bytes()
        self.assertEqual(
            record_bytes, b'{"key_id": "app", "latest": 3, "version": 1}\n'
        )
        self.assertEqual(vault.active("app"), 1)

        # Sealing a higher version makes the new version active again,
        # exactly as if the key had never been repointed; the superseded
        # record stays in the journal, untouched.
        vault.seal("app", b"app-four")
        self.assertEqual(vault.active("app"), 4)
        self.assertEqual(vault.load("app"), b"app-four")
        self.assertEqual(journal.read_bytes(), record_bytes)
        vault.reload()
        self.assertEqual(vault.active("app"), 4)
        self.assertEqual(journal.read_bytes(), record_bytes)
        self._assert_corresponds(vault, root)


# ---------------------------------------------------------------------------
# malformed records: rejected on reload, then recovery
# ---------------------------------------------------------------------------


class TestMalformedRecordsRejected(ActivationJournalRecordTestCase):
    def _write_activations(self, root: Path, data: bytes) -> None:
        (root / ACTIVATIONS_NAME).write_bytes(data)

    def test_malformed_records_are_rejected_then_recover(self):
        bad_payloads = {
            # a field missing
            "missing_key_id": b'{"version": 1, "latest": 3}\n',
            "missing_version": b'{"key_id": "app", "latest": 3}\n',
            "missing_latest": b'{"key_id": "app", "version": 1}\n',
            # an extra field
            "extra_field": (
                b'{"key_id": "app", "version": 1, "latest": 3, "note": "x"}\n'
            ),
            # wrongly typed fields
            "string_version": b'{"key_id": "app", "version": "1", "latest": 3}\n',
            "float_version": b'{"key_id": "app", "version": 1.0, "latest": 3}\n',
            "bool_version": b'{"key_id": "app", "version": true, "latest": 3}\n',
            "string_latest": b'{"key_id": "app", "version": 1, "latest": "3"}\n',
            "null_latest": b'{"key_id": "app", "version": 1, "latest": null}\n',
            "int_key_id": b'{"key_id": 7, "version": 1, "latest": 3}\n',
            "empty_key_id": b'{"key_id": "", "version": 1, "latest": 3}\n',
            # broken JSON / record shape
            "not_json": b"{not json\n",
            "not_object": b"[1, 2]\n",
            "empty_record": b"\n",
            "invalid_utf8": b"\xff\xfe\n",
            # pointers at keys/versions that never existed
            "unknown_key": b'{"key_id": "ghost", "version": 1, "latest": 1}\n',
            "unknown_version": b'{"key_id": "app", "version": 99, "latest": 99}\n',
            # the binding value beyond the key's newest sealed version
            "latest_beyond_newest": b'{"key_id": "app", "version": 1, "latest": 9}\n',
            "version_above_bound_latest": (
                b'{"key_id": "app", "version": 3, "latest": 2}\n'
            ),
            # one good line does not excuse a bad one
            "good_then_bad": (
                b'{"key_id": "app", "latest": 3, "version": 2}\n'
                b'{"key_id": "app", "version": 1}\n'
            ),
        }

        for index, (name, payload) in enumerate(bad_payloads.items()):
            with self.subTest(case=name):
                vault, root = self._build(f"malformed-{index}", repoint=True)
                expected = self._answers(vault)
                self._assert_rejected_then_recovers(
                    vault,
                    root,
                    expected,
                    lambda payload=payload, root=root: self._write_activations(
                        root, payload
                    ),
                )

    def test_failed_reload_keeps_keys_readable_and_disk_untouched(self):
        vault, root = self._build("frozen", repoint=True)
        expected = self._answers(vault)
        payload = b'{"key_id": "app", "version": 1, "latest": 3, "x": 0}\n'
        self._write_activations(root, payload)

        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
        # The keys already in hand stay readable, with the previously
        # validated repoint still in effect.
        self.assertEqual(vault.active("app"), 1)
        self.assertEqual(vault.load("app"), APP_MATERIALS[1])
        self.assertEqual(vault.load("app", 3), APP_MATERIALS[3])
        self.assertEqual(vault.load("other"), OTHER_MATERIAL)
        self.assertEqual(self._answers(vault), expected)
        # The corrupt bytes are neither repaired nor removed by reload.
        self.assertEqual((root / ACTIVATIONS_NAME).read_bytes(), payload)

    def test_removing_the_bad_record_recovers(self):
        vault, root = self._build("remove-bad", repoint=False)
        journal = root / ACTIVATIONS_NAME
        journal.write_bytes(b'{"key_id": "app", "version": 1, "latest": 9}\n')
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.active("app"), 3)

        # Removing the bad record returns the vault to the never-repointed
        # state: no journal, the manifest's active version applies.
        journal.unlink()
        vault.reload()
        self.assertEqual(vault.active("app"), 3)
        self.assertEqual(vault.load("app"), APP_MATERIALS[3])
        self.assertFalse(journal.exists())
        self._assert_corresponds(vault, root)

    def test_correcting_the_bad_record_recovers(self):
        vault, root = self._build("correct-bad", repoint=False)
        journal = root / ACTIVATIONS_NAME
        journal.write_bytes(b'{"key_id": "app", "version": 1, "latest": 9}\n')
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.active("app"), 3)

        # Correcting the record in place makes the reload succeed and the
        # repoint apply; snapshot and disk correspond one-to-one again.
        journal.write_bytes(b'{"key_id": "app", "latest": 3, "version": 1}\n')
        vault.reload()
        self.assertEqual(vault.active("app"), 1)
        self.assertEqual(vault.load("app"), APP_MATERIALS[1])
        self._assert_corresponds(vault, root)
        reopened = self._open(root)
        self.assertEqual(reopened.active("app"), 1)
        self._assert_corresponds(reopened, root)


if __name__ == "__main__":
    unittest.main()
