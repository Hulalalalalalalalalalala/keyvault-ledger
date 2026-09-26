"""Regression tests pinning the documented ``activations.jsonl`` record shape.

The README documents the activation journal's record shape and binding
rules: one JSON object per line carrying exactly the fields ``key_id`` (the
repointed key), ``version`` (the historical version pointed at) and
``latest`` (the binding value — the key's newest sealed version at write
time), in write order; a repoint applies only while its bound ``latest``
still equals the key's newest sealed version; a vault that was never
repointed has no such file.  These cases freeze that contract end to end:

* a record written by ``set_active`` lands on disk byte-for-byte as
  documented, and a hand-written record following the documented shape is
  honoured by a full ``reload()``;
* repeated repoints append one line each, the last one wins, and the
  historical lines stay in the file unmodified;
* a record missing a field, carrying an extra field, holding a wrongly
  typed value, written as broken JSON, pointing at a key or version that
  never existed, or bound to a ``latest`` beyond the key's newest sealed
  version makes ``reload()`` raise ``ValueError``, while the in-memory
  snapshot stays frozen, the keys already in hand stay readable and the
  disk records neither grow nor shrink;
* removing or correcting the bad record and reloading again restores the
  vault, with the snapshot matching the disk records one to one.

The repoint entry-point contract itself (empty key id -> ``ValueError``,
non-integer version -> ``TypeError``, unknown key/version -> ``KeyError``,
active or revoked target -> ``ValueError``, a failed call appends nothing)
is covered by the existing suites and intentionally not re-litigated here.

Everything happens inside temporary directories, uses the standard library
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

# The documented line, byte-for-byte as ``set_active`` writes it (one JSON
# object per line, fields sorted, trailing newline).
HEALTHY_LINE = b'{"key_id": "k", "latest": 3, "version": 1}\n'


class ActivationJournalRecordTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        # Each scenario gets its own vault subdirectory, so cases never
        # share disk state and execution order cannot matter.
        self.root = self.fixture.tmp_path / "vaults"

    def _open(self, root: Path) -> Vault:
        """Open a vault through the case's single teardown path."""
        return self.fixture.open(root)

    @staticmethod
    def _journal(root: Path) -> Path:
        return root / ACTIVATIONS_NAME

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

    def _build(self, name: str) -> tuple[Vault, Path]:
        """Build a healthy vault: key ``k`` with three versions, repointed
        at version 1 (bound to ``latest`` 3), plus a second key ``other``
        whose readability must survive every failure window."""
        root = self.root / name
        vault = self._open(root)
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        vault.seal("k", b"k-v3")
        vault.set_active("k", 1)  # bound to latest=3
        vault.seal("other", b"other-v1")
        return vault, root

    def _answers(self, vault: Vault) -> dict:
        """Every state a reader can query, in a comparable plain structure."""
        return {
            "k": {
                "versions": vault.versions("k"),
                "active": vault.active("k"),
                "materials": {
                    version: vault.load("k", version)
                    for version in vault.versions("k")
                },
            },
            "other": {
                "versions": vault.versions("other"),
                "active": vault.active("other"),
                "materials": {1: vault.load("other", 1)},
            },
            "manifest": vault.manifest(),
        }

    def _assert_corresponds(self, vault: Vault, root: Path) -> None:
        """Re-derive every observable answer straight from the disk records.

        This never consults the vault's private state: it parses the
        manifest and the activation journal itself, applies the documented
        binding rule (the last record whose ``latest`` still equals the
        key's newest sealed version wins) and requires the public snapshot
        to match exactly.
        """
        manifest = json.loads(
            (root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )
        self.assertEqual(vault.manifest(), manifest)

        last_bound: dict[str, tuple[int, int]] = {}
        journal = root / ACTIVATIONS_NAME
        if journal.exists():
            for line in journal.read_text("utf-8").splitlines():
                record = json.loads(line)
                last_bound[record["key_id"]] = (
                    record["version"],
                    record["latest"],
                )

        for key_id, entry in manifest["keys"].items():
            versions = [record["version"] for record in entry["versions"]]
            self.assertEqual(vault.versions(key_id), versions)
            for record in entry["versions"]:
                self.assertEqual(
                    vault.load(key_id, record["version"]),
                    (root / record["file"]).read_bytes(),
                )
            expected_active = entry["active"]
            if key_id in last_bound:
                target, bound_latest = last_bound[key_id]
                if bound_latest == versions[-1]:
                    expected_active = target
            self.assertEqual(vault.active(key_id), expected_active)


class TestDocumentedRecordShape(ActivationJournalRecordTestCase):
    def test_never_repointed_vault_has_no_journal_and_reloads_clean(self):
        root = self.root / "never-repointed"
        vault = self._open(root)
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        # Opening, sealing and reloading never create the journal.
        self.assertFalse(self._journal(root).exists())
        vault.reload()
        self.assertFalse(self._journal(root).exists())
        self.assertEqual(vault.active("k"), 2)
        reopened = self._open(root)
        self.assertFalse(self._journal(root).exists())
        self.assertEqual(reopened.active("k"), 2)

    def test_first_repoint_creates_journal_with_the_documented_line(self):
        root = self.root / "first-repoint"
        vault = self._open(root)
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        self.assertFalse(self._journal(root).exists())

        vault.set_active("k", 1)
        # Byte-for-byte the documented shape: one JSON object per line with
        # exactly the three documented fields.
        self.assertEqual(
            self._journal(root).read_bytes(),
            b'{"key_id": "k", "latest": 2, "version": 1}\n',
        )
        record = json.loads(self._journal(root).read_text("utf-8"))
        self.assertEqual(set(record), {"key_id", "version", "latest"})
        self.assertEqual(record["key_id"], "k")
        self.assertEqual(record["version"], 1)
        # The binding value is the newest sealed version at write time.
        self.assertEqual(record["latest"], 2)

    def test_handwritten_record_matching_the_documented_shape_applies(self):
        root = self.root / "handwritten"
        vault = self._open(root)
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        vault.seal("k", b"k-v3")
        # Never repointed: no journal exists yet.
        journal = self._journal(root)
        self.assertFalse(journal.exists())

        # Hand-write exactly the documented line: repoint "k" at version 2,
        # bound to the newest sealed version 3.
        journal.write_bytes(b'{"key_id": "k", "latest": 3, "version": 2}\n')
        disk_before = self._disk_bytes(root)
        vault.reload()
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"k-v2")
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        # Reload is read-only: the hand-written file is untouched.
        self.assertEqual(self._disk_bytes(root), disk_before)
        # A fresh opener lands on the same state.
        reopened = self._open(root)
        self.assertEqual(reopened.active("k"), 2)
        self.assertEqual(reopened.load("k"), b"k-v2")
        self._assert_corresponds(reopened, root)

    def test_repeated_repoints_last_wins_and_history_stays(self):
        root = self.root / "repeated"
        vault = self._open(root)
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        vault.seal("k", b"k-v3")
        vault.set_active("k", 1)
        vault.set_active("k", 3)
        vault.set_active("k", 2)

        journal = self._journal(root)
        # One line per repoint, in write order, each bound to the newest
        # sealed version 3.
        self.assertEqual(
            journal.read_bytes().splitlines(keepends=True),
            [
                b'{"key_id": "k", "latest": 3, "version": 1}\n',
                b'{"key_id": "k", "latest": 3, "version": 3}\n',
                b'{"key_id": "k", "latest": 3, "version": 2}\n',
            ],
        )
        history = journal.read_bytes()
        vault.reload()
        # The last record wins and the historical lines are not rewritten.
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"k-v2")
        self.assertEqual(journal.read_bytes(), history)
        reopened = self._open(root)
        self.assertEqual(reopened.active("k"), 2)
        self.assertEqual(journal.read_bytes(), history)
        self._assert_corresponds(reopened, root)

    def test_newer_seal_supersedes_repoint_without_touching_journal(self):
        root = self.root / "superseded"
        vault = self._open(root)
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        vault.set_active("k", 1)  # bound to latest=2
        history = self._journal(root).read_bytes()

        vault.seal("k", b"k-v3")
        # The repoint was bound to newest-sealed 2, so version 3 is active
        # again; the historical record stays in the journal, inert.
        self.assertEqual(vault.active("k"), 3)
        vault.reload()
        self.assertEqual(vault.active("k"), 3)
        self.assertEqual(vault.load("k"), b"k-v3")
        self.assertEqual(self._journal(root).read_bytes(), history)
        self.assertEqual(self._open(root).active("k"), 3)
        self._assert_corresponds(vault, root)


class TestMalformedRecords(ActivationJournalRecordTestCase):
    def test_malformed_records_are_rejected_then_recover(self):
        bad_payloads = {
            # missing fields
            "missing_key_id": b'{"latest": 3, "version": 1}\n',
            "missing_version": b'{"key_id": "k", "latest": 3}\n',
            "missing_latest": b'{"key_id": "k", "version": 1}\n',
            # an extra field beyond the documented three
            "extra_field": (
                b'{"key_id": "k", "latest": 3, "version": 1, "note": "x"}\n'
            ),
            # broken JSON / not an object / blank line
            "broken_json": b"{not json\n",
            "not_an_object": b"[1, 2]\n",
            "empty_line": b"\n",
            # wrongly typed fields
            "key_id_not_string": b'{"key_id": 7, "latest": 3, "version": 1}\n',
            "version_string": b'{"key_id": "k", "latest": 3, "version": "1"}\n',
            "version_float": b'{"key_id": "k", "latest": 3, "version": 1.0}\n',
            "version_bool": b'{"key_id": "k", "latest": 3, "version": true}\n',
            "latest_string": b'{"key_id": "k", "latest": "3", "version": 1}\n',
            "latest_bool": b'{"key_id": "k", "latest": true, "version": 1}\n',
            # pointing at a key or version that never existed
            "unknown_key": b'{"key_id": "ghost", "latest": 1, "version": 1}\n',
            "unknown_version": b'{"key_id": "k", "latest": 3, "version": 9}\n',
            # binding value beyond the key's newest sealed version
            "latest_above_newest_sealed": (
                b'{"key_id": "k", "latest": 9, "version": 1}\n'
            ),
            "version_above_bound_latest": (
                b'{"key_id": "k", "latest": 2, "version": 3}\n'
            ),
            # a valid record followed by a corrupt one
            "good_then_bad": HEALTHY_LINE + b"{broken\n",
        }

        for index, (name, payload) in enumerate(bad_payloads.items()):
            with self.subTest(case=name):
                vault, root = self._build(f"bad-{index}")
                expected = self._answers(vault)
                healthy_disk = self._disk_bytes(root)

                self._journal(root).write_bytes(payload)
                failing_disk = self._disk_bytes(root)

                # The exception type and message are deterministic, and
                # every observable answer stays frozen across failures.
                message = None
                for _ in range(3):
                    with self.assertRaises(ValueError) as caught:
                        vault.reload()
                    if message is None:
                        message = str(caught.exception)
                    else:
                        self.assertEqual(str(caught.exception), message)
                    self.assertEqual(self._answers(vault), expected)

                # The keys already in hand stay readable.
                self.assertEqual(vault.load("k"), b"k-v1")
                self.assertEqual(vault.load("k", 2), b"k-v2")
                self.assertEqual(vault.load("other"), b"other-v1")

                # A cold opener rejects the same state with the same
                # complaint.
                with self.assertRaises(ValueError) as caught:
                    Vault(root)
                self.assertEqual(str(caught.exception), message)

                # The failed reloads neither grew nor shrank the disk
                # records: the directory is byte-for-byte the corrupted
                # state just captured.
                self.assertEqual(self._disk_bytes(root), failing_disk)

                # Removing or correcting the bad record restores the vault:
                # one successful reload re-establishes the one-to-one
                # correspondence between snapshot and disk records.
                self._journal(root).write_bytes(HEALTHY_LINE)
                vault.reload()
                self.assertEqual(self._answers(vault), expected)
                self._assert_corresponds(vault, root)
                self.assertEqual(self._disk_bytes(root), healthy_disk)
                reopened = self._open(root)
                self.assertEqual(self._answers(reopened), expected)
                self._assert_corresponds(reopened, root)

    def test_correcting_the_record_applies_the_corrected_repoint(self):
        vault, root = self._build("corrected")
        # An extra field makes the record corrupt...
        self._journal(root).write_bytes(
            b'{"key_id": "k", "latest": 3, "version": 2, "note": 1}\n'
        )
        with self.assertRaises(ValueError):
            vault.reload()
        # ...and while it is corrupt the validated snapshot is frozen.
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"k-v1")

        # Correcting the record (dropping the extra field) makes the reload
        # succeed and the corrected repoint takes effect.
        self._journal(root).write_bytes(
            b'{"key_id": "k", "latest": 3, "version": 2}\n'
        )
        vault.reload()
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"k-v2")
        self._assert_corresponds(vault, root)

    def test_removing_the_bad_line_keeps_the_surviving_records(self):
        vault, root = self._build("truncated")
        # A valid record followed by a corrupt trailing line fails...
        self._journal(root).write_bytes(HEALTHY_LINE + b"{broken\n")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.active("k"), 1)
        # ...removing the bad line (not rewriting the good one) recovers.
        self._journal(root).write_bytes(HEALTHY_LINE)
        vault.reload()
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"k-v1")
        self.assertEqual(self._journal(root).read_bytes(), HEALTHY_LINE)
        self._assert_corresponds(vault, root)


if __name__ == "__main__":
    unittest.main()
