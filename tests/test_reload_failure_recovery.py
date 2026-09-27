"""Regression tests for reload failure and recovery around materials and
the activity journal.

The baseline vault already implements whole-vault validation and reload;
this module adds no product behaviour.  It pins, end to end, what happens
when the on-disk keyring is damaged out of band:

* a stored material that is missing, that cannot be read (its path turned
  into a directory, or its file made unreadable) or whose bytes do not
  match the manifest record makes a whole ``reload()`` raise exactly
  ``ValueError`` -- for plain and for derived versions;

* an activity-journal record that is missing a field, carrying an extra
  field, holding a wrongly typed value or written as broken JSON makes
  ``reload()`` raise exactly ``ValueError`` (a good record followed by a
  corrupt trailing line included);

* while such a failure persists the in-memory snapshot is frozen word for
  word: the keys already in hand all stay readable and the answers for
  versions, active version, revocation markers, derivation parameters and
  the manifest are byte-for-byte identical before and after every failed
  reload; repeating any query returns the identical answer, and a cold
  opener on the same directory raises the same ``ValueError``;

* a failed reload is strictly read-only: the on-disk manifest, both
  journals and every material neither gain nor lose a byte;

* once the damage is undone (the bytes restored, or the bad activity line
  removed/corrected) another full ``reload()`` succeeds and the snapshot
  matches the disk records one to one, as does a fresh opener;

* the same damage applied to two independently built vaults yields the
  same exception type/message and the same frozen answers.

Everything happens inside temporary directories, uses the standard
library only and is independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
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

# The one healthy activity record the fixture vault carries: key "a"
# repointed at version 1, bound to its newest sealed version 3.
HEALTHY_ACTIVATION = b'{"key_id": "a", "latest": 3, "version": 1}\n'


class MaterialActivityReloadTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        # Each scenario gets its own vault subdirectory.
        self.root = self.fixture.tmp_path / "vaults"

    def open_vault(self, root: Path) -> Vault:
        """Open a vault through the case's single teardown path."""
        return self.fixture.open(root)

    # ------------------------------------------------------------------
    # fixture construction: several keys, both material kinds, one repoint
    # ------------------------------------------------------------------

    def _build_healthy(self, name: str) -> tuple[Vault, Path]:
        """Build a multi-key vault and return ``(handle, root)``.

        ``a`` has three plain versions and is repointed at version 1
        (bound to newest sealed 3), version 2 revoked; ``b`` has one
        derived version; ``c`` has one plain version.  Every other key must
        stay readable while any one record is damaged.
        """
        root = self.root / name
        vault = self.open_vault(root)
        vault.seal("a", b"a-v1")
        vault.seal("a", b"a-v2")
        vault.seal("a", b"a-v3")
        vault.set_active("a", 1)  # activations.jsonl: bound latest=3
        vault.revoke("a", 2)
        vault.derive_seal("b", b"passphrase-b", b"salt-b", 1000, 32)
        vault.seal("c", b"c-v1")
        return vault, root

    # ------------------------------------------------------------------
    # observable-state snapshots
    # ------------------------------------------------------------------

    def _answers(self, vault: Vault) -> dict:
        """Every state a reader can query, in a comparable structure."""
        keys = {}
        for key_id in ("a", "b", "c"):
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
            "unknown_versions": vault.versions("never-sealed"),
            "unknown_revoked": vault.revoked_versions("never-sealed"),
        }

    def _disk_records(self, root: Path) -> dict[str, object]:
        """All vault records on disk keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device, so it is excluded.
        A record made unreadable out of band (a chmod-damage case) cannot
        expose its bytes, so it is represented by a stable
        ``(UNREADABLE, size, mtime, mode)`` marker instead: the point under
        test is that a failed reload neither changes nor removes it, which
        its size/mtime/mode prove just as well.
        """
        records: dict[str, object] = {}
        for path in root.rglob("*"):
            if not path.is_file() or path.name == LOCK_NAME:
                continue
            stat = path.stat()
            try:
                records[str(path.relative_to(root))] = path.read_bytes()
            except PermissionError:
                records[str(path.relative_to(root))] = (
                    "UNREADABLE",
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_mode & 0o777,
                )
        return records

    def _restore_disk(self, root: Path, healthy: dict[str, bytes]) -> None:
        """Put ``root`` back in exactly the captured healthy state."""
        current = self._disk_records(root)
        for rel in current.keys() - healthy.keys():
            (root / rel).unlink()
        for rel, data in healthy.items():
            path = root / rel
            # A permission-damage case may have left the file unreadable;
            # make it writable before restoring the healthy bytes.
            try:
                if path.exists() and not os.access(path, os.W_OK):
                    path.chmod(0o600)
            except OSError:
                pass
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)

    def _assert_corresponds(self, vault: Vault, root: Path) -> None:
        """Re-derive every answer straight from the disk records.

        No private vault state is consulted: the manifest, both journals
        and the material files are parsed independently and must explain
        every public answer exactly.
        """
        manifest = json.loads(
            (root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )
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
                # The snapshot serves the exact persisted bytes and they
                # match the manifest digest.
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
                set(vault.revoked_versions(key_id)),
                revoked.get(key_id, set()),
            )

    # ------------------------------------------------------------------
    # the shared failure window
    # ------------------------------------------------------------------

    def _assert_failure_window(
        self, vault: Vault, root: Path, expected: dict
    ) -> None:
        """Assert freeze during the failure; returns after leaving damage.

        The caller has *already* damaged the disk.  This drives three
        failed reloads and a cold open, requires identical complaints and
        frozen answers, and verifies the disk neither grew nor shrank.  It
        does NOT repair -- the caller restores and calls
        :meth:`_assert_recovery`.
        """
        damaged_disk = self._disk_records(root)

        message = None
        for _ in range(3):
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            if message is None:
                message = str(caught.exception)
            else:
                self.assertEqual(str(caught.exception), message)
            # Snapshot frozen word for word; every key stays readable.
            self.assertEqual(self._answers(vault), expected)

        # A cold opener rejects the same state with the same complaint.
        with self.assertRaises(ValueError) as caught:
            Vault(root)
        self.assertEqual(str(caught.exception), message)

        # The failed reloads/open changed not one byte on disk.
        self.assertEqual(self._disk_records(root), damaged_disk)

    def _assert_recovery(
        self, vault: Vault, root: Path, expected: dict, healthy_disk: dict
    ) -> None:
        """Restore the healthy bytes and require one-to-one correspondence."""
        self._restore_disk(root, healthy_disk)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self._assert_corresponds(vault, root)
        self.assertEqual(self._disk_records(root), healthy_disk)
        reopened = self.open_vault(root)
        self.assertEqual(self._answers(reopened), expected)
        self._assert_corresponds(reopened, root)

    def _material_path(
        self, root: Path, key_id: str, version: int
    ) -> Path:
        manifest = json.loads(
            (root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )
        record = next(
            record
            for record in manifest["keys"][key_id]["versions"]
            if record["version"] == version
        )
        return root / record["file"]


# ---------------------------------------------------------------------------
# material missing / unreadable / mismatching
# ---------------------------------------------------------------------------


class TestMaterialReloadFailures(MaterialActivityReloadTestCase):
    def test_missing_material_raises_and_other_keys_stay_readable(self):
        for index, (key_id, version) in enumerate(
            (("a", 1), ("a", 3), ("b", 1), ("c", 1))
        ):
            with self.subTest(material=f"{key_id}-{version}"):
                vault, root = self._build_healthy(f"missing-{index}")
                expected = self._answers(vault)
                healthy_disk = self._disk_records(root)

                self._material_path(root, key_id, version).unlink()
                self._assert_failure_window(vault, root, expected)
                # Keys already in hand stay readable, including the other
                # versions of the damaged key.
                for other_key in ("a", "b", "c"):
                    self.assertEqual(
                        vault.versions(other_key),
                        expected["keys"][other_key]["versions"],
                    )
                self.assertEqual(vault.load("a", 2), b"a-v2")
                self._assert_recovery(vault, root, expected, healthy_disk)

    def test_material_path_replaced_by_a_directory_is_unreadable(self):
        vault, root = self._build_healthy("unreadable-directory")
        expected = self._answers(vault)
        healthy_disk = self._disk_records(root)

        material = self._material_path(root, "a", 2)
        material.unlink()
        material.mkdir()
        self.addCleanup(lambda: material.is_dir() and material.rmdir())

        self._assert_failure_window(vault, root, expected)
        # The other keys (and other versions) stay readable.
        self.assertEqual(vault.load("a", 1), b"a-v1")
        self.assertEqual(vault.load("c"), b"c-v1")
        self.assertEqual(vault.active("a"), 1)  # repoint survives

        # Repair: remove the bogus directory and restore the file.
        material.rmdir()
        self._assert_recovery(vault, root, expected, healthy_disk)

    @unittest.skipUnless(
        hasattr(os, "geteuid") and os.geteuid() != 0,
        "permission bits are not enforced for root",
    )
    def test_unreadable_material_file_raises_value_error(self):
        vault, root = self._build_healthy("unreadable-permission")
        expected = self._answers(vault)
        healthy_disk = self._disk_records(root)

        material = self._material_path(root, "b", 1)
        material.chmod(0o000)
        self.addCleanup(lambda: material.exists() and material.chmod(0o600))

        self._assert_failure_window(vault, root, expected)
        # Existing keys are unaffected by the one unreadable material.
        self.assertEqual(vault.load("a"), b"a-v1")
        self.assertEqual(vault.load("c"), b"c-v1")

        material.chmod(0o600)
        self._assert_recovery(vault, root, expected, healthy_disk)

    def test_mismatching_material_raises_value_error(self):
        for index, (key_id, version, original) in enumerate(
            (("a", 1, b"a-v1"), ("a", 2, b"a-v2"), ("b", 1, None), ("c", 1, b"c-v1"))
        ):
            with self.subTest(material=f"{key_id}-{version}"):
                vault, root = self._build_healthy(f"mismatch-{index}")
                expected = self._answers(vault)
                healthy_disk = self._disk_records(root)

                self._material_path(root, key_id, version).write_bytes(
                    b"tampered bytes"
                )
                self._assert_failure_window(vault, root, expected)
                # The snapshot keeps serving the original bytes; the
                # tampered file never enters it.
                self.assertEqual(vault.versions(key_id), expected["keys"][key_id]["versions"])
                if original is not None:
                    self.assertEqual(vault.load(key_id, version), original)
                # The repointed active pointer and markers survive.
                self.assertEqual(vault.active("a"), 1)
                self.assertEqual(vault.revoked_versions("a"), [2])
                self._assert_recovery(vault, root, expected, healthy_disk)

    def test_damage_then_reload_is_read_only_across_the_whole_directory(self):
        # The failed reload must not repair, remove or create anything:
        # compare the complete record inventory (bytes and relative names)
        # before and after the failed reloads.
        vault, root = self._build_healthy("read-only-material")
        self._material_path(root, "a", 3).unlink()
        damaged = self._disk_records(root)
        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
        self.assertEqual(self._disk_records(root), damaged)
        # No stray temp file or other repair artifact appeared.
        self.assertEqual(list(root.rglob("*.tmp.*")), [])


# ---------------------------------------------------------------------------
# activity journal missing field / extra field / wrong type / broken JSON
# ---------------------------------------------------------------------------


class TestActivityJournalReloadFailures(MaterialActivityReloadTestCase):
    def _write_activations(self, root: Path, data: bytes) -> None:
        (root / ACTIVATIONS_NAME).write_bytes(data)

    def test_broken_json_makes_reload_fail_but_keeps_everything(self):
        vault, root = self._build_healthy("activity-broken-json")
        expected = self._answers(vault)
        healthy_disk = self._disk_records(root)

        self._write_activations(root, b"{not json\n")
        self._assert_failure_window(vault, root, expected)
        # The repointed pointer stays at version 1 and every key is served.
        self.assertEqual(vault.active("a"), 1)
        self.assertEqual(vault.load("a"), b"a-v1")
        self.assertEqual(vault.load("b", 1), expected["keys"]["b"]["materials"][1])
        self.assertEqual(vault.revoked_versions("a"), [2])
        self._assert_recovery(vault, root, expected, healthy_disk)

    def test_missing_field_records_are_rejected_then_recover(self):
        cases = {
            "missing_key_id": b'{"latest": 3, "version": 1}\n',
            "missing_version": b'{"key_id": "a", "latest": 3}\n',
            "missing_latest": b'{"key_id": "a", "version": 1}\n',
            "missing_two_fields": b'{"key_id": "a"}\n',
        }
        for index, (name, payload) in enumerate(cases.items()):
            with self.subTest(case=name):
                vault, root = self._build_healthy(f"activity-missing-{index}")
                expected = self._answers(vault)
                healthy_disk = self._disk_records(root)
                self._write_activations(root, payload)
                self._assert_failure_window(vault, root, expected)
                self._assert_recovery(
                    vault, root, expected, healthy_disk
                )

    def test_extra_field_records_are_rejected_then_recover(self):
        cases = {
            "one_extra_field": (
                b'{"key_id": "a", "latest": 3, "version": 1, "note": 1}\n'
            ),
            "duplicate_shape_extra": (
                b'{"key_id": "a", "latest": 3, "version": 1, '
                b'"version_again": 2}\n'
            ),
        }
        for index, (name, payload) in enumerate(cases.items()):
            with self.subTest(case=name):
                vault, root = self._build_healthy(f"activity-extra-{index}")
                expected = self._answers(vault)
                healthy_disk = self._disk_records(root)
                self._write_activations(root, payload)
                self._assert_failure_window(vault, root, expected)
                # Dropping the extra field is itself the recovery: the
                # corrected record is honoured and repoints at version 1.
                self._write_activations(root, HEALTHY_ACTIVATION)
                vault.reload()
                self.assertEqual(vault.active("a"), 1)
                self._assert_corresponds(vault, root)
                self.assertEqual(self._disk_records(root), healthy_disk)

    def test_wrongly_typed_fields_are_rejected_then_recover(self):
        cases = {
            "key_id_number": b'{"key_id": 7, "latest": 3, "version": 1}\n',
            "key_id_empty": b'{"key_id": "", "latest": 3, "version": 1}\n',
            "key_id_null": b'{"key_id": null, "latest": 3, "version": 1}\n',
            "version_string": b'{"key_id": "a", "latest": 3, "version": "1"}\n',
            "version_float": b'{"key_id": "a", "latest": 3, "version": 1.0}\n',
            "version_bool": b'{"key_id": "a", "latest": 3, "version": true}\n',
            "version_null": b'{"key_id": "a", "latest": 3, "version": null}\n',
            "latest_string": b'{"key_id": "a", "latest": "3", "version": 1}\n',
            "latest_float": b'{"key_id": "a", "latest": 3.0, "version": 1}\n',
            "latest_bool": b'{"key_id": "a", "latest": true, "version": 1}\n',
            "latest_null": b'{"key_id": "a", "latest": null, "version": 1}\n',
        }
        for index, (name, payload) in enumerate(cases.items()):
            with self.subTest(case=name):
                vault, root = self._build_healthy(f"activity-type-{index}")
                expected = self._answers(vault)
                healthy_disk = self._disk_records(root)
                self._write_activations(root, payload)
                self._assert_failure_window(vault, root, expected)
                # The bad typed record cannot move the pointer: it stays at
                # the previously validated repoint (version 1 of key a).
                self.assertEqual(vault.active("a"), 1)
                self.assertEqual(vault.load("a"), b"a-v1")
                self._assert_recovery(
                    vault, root, expected, healthy_disk
                )

    def test_good_record_then_broken_trailing_line_is_rejected(self):
        vault, root = self._build_healthy("activity-good-then-bad")
        expected = self._answers(vault)
        healthy_disk = self._disk_records(root)

        self._write_activations(root, HEALTHY_ACTIVATION + b"{broken\n")
        self._assert_failure_window(vault, root, expected)
        # The frozen snapshot still reflects the one validated good record.
        self.assertEqual(vault.active("a"), 1)

        # Removing just the bad trailing line (leaving the good one
        # byte-for-byte) recovers the vault.
        self._write_activations(root, HEALTHY_ACTIVATION)
        vault.reload()
        self.assertEqual(vault.active("a"), 1)
        self._assert_corresponds(vault, root)
        self.assertEqual(self._disk_records(root), healthy_disk)

    def test_failed_activity_reload_touches_no_other_record(self):
        vault, root = self._build_healthy("activity-read-only")
        manifest_before = (root / MANIFEST_NAME).read_bytes()
        revocations_before = (root / REVOCATIONS_NAME).read_bytes()
        materials_before = {
            path: path.read_bytes()
            for path in (root).rglob("*.bin")
        }
        self._write_activations(root, b"{garbage\n")
        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
        # Only the journal the test itself changed is different; everything
        # else -- manifest, revocation journal, materials -- is verbatim.
        self.assertEqual((root / MANIFEST_NAME).read_bytes(), manifest_before)
        self.assertEqual(
            (root / REVOCATIONS_NAME).read_bytes(), revocations_before
        )
        for path, data in materials_before.items():
            self.assertEqual(path.read_bytes(), data)
        # The failed reload does not "fix" the journal either.
        self.assertEqual(
            (root / ACTIVATIONS_NAME).read_bytes(), b"{garbage\n"
        )

    def test_repeated_queries_match_word_for_word_during_failure(self):
        vault, root = self._build_healthy("activity-word-for-word")
        self._write_activations(root, b"{garbage\n")

        def word_for_word() -> tuple:
            return (
                vault.versions("a"),
                vault.active("a"),
                tuple(vault.load("a", v) for v in vault.versions("a")),
                tuple(vault.revoked_versions("a")),
                vault.active("b"),
                vault.derivation("b", 1)["salt"],
                vault.load("c"),
                vault.manifest(),
            )

        first = word_for_word()
        for _ in range(4):
            with self.assertRaises(ValueError):
                vault.reload()
            self.assertEqual(word_for_word(), first)


# ---------------------------------------------------------------------------
# determinism across two independently built vaults
# ---------------------------------------------------------------------------


class TestReloadFailureDeterminism(MaterialActivityReloadTestCase):
    def test_same_damage_in_two_vaults_gives_same_error_and_frozen_answers(
        self,
    ):
        def damaged_outcome(name: str, damage) -> tuple:
            vault, root = self._build_healthy(name)
            frozen = self._answers(vault)
            damage(root)
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            return str(caught.exception), self._answers(vault), frozen

        def damage_material(root: Path) -> None:
            manifest = json.loads(
                (root / MANIFEST_NAME).read_bytes().decode("utf-8")
            )
            rel = manifest["keys"]["a"]["versions"][0]["file"]
            (root / rel).write_bytes(b"tampered")

        def damage_activity(root: Path) -> None:
            (root / ACTIVATIONS_NAME).write_bytes(
                b'{"key_id": "a", "version": 1}\n'  # missing latest
            )

        for label, damage in (
            ("material", damage_material),
            ("activity", damage_activity),
        ):
            first_msg, first_answers, first_frozen = damaged_outcome(
                f"det-{label}-a", damage
            )
            second_msg, second_answers, second_frozen = damaged_outcome(
                f"det-{label}-b", damage
            )
            self.assertEqual(first_msg, second_msg, label)
            self.assertEqual(first_answers, second_answers, label)
            self.assertEqual(first_frozen, second_frozen, label)
            # During the failure the frozen answers equal the pre-damage
            # answers -- nothing leaked from the corrupt disk.
            self.assertEqual(first_answers, first_frozen, label)


if __name__ == "__main__":
    unittest.main()
