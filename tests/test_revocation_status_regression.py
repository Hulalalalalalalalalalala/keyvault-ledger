"""Regression tests for the two observable results of the revocation query.

The baseline vault behaviour is already frozen; this module only pins what
``is_revoked`` answers for the status query's true and false branches and
the state that surrounds those answers:

* a version that was never revoked answers ``False``; asking again and
  again gives the identical result and neither the in-memory snapshot nor
  any disk record changes;

* a revoked version answers ``True`` and every repeated query answers
  byte-for-byte the same, while versions revoked at other points keep
  answering ``False`` and ``revoked_versions`` stays the ascending marker
  list;

* the markers survive a whole-vault ``reload()`` and a cold reopen, and
  the per-version status table is verbatim identical before and after
  each reload;

* repointing the active version (``set_active``) never moves a revocation
  marker, and a version that is revoked *after* it was pointed at keeps
  answering ``True`` without the repoint changing the status answers;

* a revoked version stays readable ``load``-by-version, returning exactly
  the bytes originally sealed, through reload and reopen.

Every case lives in its own temporary directory, goes through the shared
fixture's single teardown (lock handles returned, tree deleted) and is
independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import LOCK_NAME
from tests._fixtures import VaultFixture

# Distinct bytes per version so a byte-exact readback cannot pass by
# accident; the empty payload is included on purpose.
PAYLOADS = {
    1: b"",
    2: bytes(range(256)),
    3: b"\x00\xff\nrevoked-readback\r",
    4: "λ-material".encode("utf-8"),
}


class RevocationStatusTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.tmp_path / "vault"

    def open_vault(self, root: Path | str | None = None) -> Vault:
        """Open a vault tracked by the case's single teardown path."""
        return self.fixture.open(self.root if root is None else root)

    def seal_payloads(self, vault: Vault, key_id: str = "k") -> list[int]:
        """Seal every payload of ``PAYLOADS`` in order; return the versions."""
        versions = []
        for version in sorted(PAYLOADS):
            sealed = vault.seal(key_id, PAYLOADS[version])
            self.assertEqual(sealed, version)
            versions.append(sealed)
        return versions

    def status_table(self, vault: Vault, versions: list[int], key_id: str = "k"):
        """The per-version answers of ``is_revoked``, keyed by version."""
        return {version: vault.is_revoked(key_id, version) for version in versions}

    def disk_records(self, root: Path | None = None) -> dict:
        """All vault records on disk with bytes and mtime, lock excluded.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, so it is deliberately not part of the inventory the queries
        must leave untouched.
        """
        root = self.root if root is None else root
        records = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != LOCK_NAME:
                records[path.relative_to(root)] = (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
        return records

    def assert_disk_unchanged(self, before: dict) -> None:
        after = self.disk_records()
        self.assertEqual(sorted(after), sorted(before))
        for rel, (data, mtime) in before.items():
            self.assertEqual((self.root / rel).read_bytes(), data, str(rel))
            self.assertEqual((self.root / rel).stat().st_mtime_ns, mtime, str(rel))


class TestNeverRevokedVersionReportsFalse(RevocationStatusTestCase):
    def test_is_revoked_is_false_for_every_never_revoked_version(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        # The false branch: nothing was revoked, on this key or any other.
        for version in versions:
            self.assertIs(vault.is_revoked("k", version), False)
        # An unversioned listing agrees, for a sealed key and an unknown one.
        self.assertEqual(vault.revoked_versions("k"), [])
        self.assertEqual(vault.revoked_versions("never-sealed"), [])

    def test_repeated_false_queries_are_identical_and_change_no_state(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)

        before_disk = self.disk_records()
        before_table = self.status_table(vault, versions)
        before_manifest = vault.manifest()
        before_active = vault.active("k")

        # Repeated identical queries must give identical answers, run after
        # run, including the listings.
        runs = [
            self.status_table(vault, versions)
            for _ in range(5)
        ]
        self.assertTrue(runs)
        self.assertTrue(all(run == before_table for run in runs))
        for _ in range(5):
            self.assertEqual(vault.revoked_versions("k"), [])

        # A status query is strictly read-only: manifest and journals
        # neither grow nor shrink, byte for byte and mtime for mtime, and
        # the in-memory answers are untouched.
        self.assert_disk_unchanged(before_disk)
        self.assertEqual(vault.manifest(), before_manifest)
        self.assertEqual(vault.active("k"), before_active)
        self.assertEqual(self.status_table(vault, versions), before_table)

    def test_false_answer_does_not_depend_on_other_keys_revocations(self):
        vault = self.open_vault()
        k_versions = self.seal_payloads(vault, "k")
        other_versions = self.seal_payloads(vault, "other")
        vault.revoke("other", other_versions[0])
        # Markers on one key say nothing about a different key's versions.
        for version in k_versions:
            self.assertIs(vault.is_revoked("k", version), False)
        self.assertEqual(vault.revoked_versions("k"), [])
        self.assertEqual(vault.revoked_versions("other"), [other_versions[0]])


class TestRevokedVersionReportsTrue(RevocationStatusTestCase):
    def test_is_revoked_is_true_for_a_revoked_version(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        vault.revoke("k", versions[1])
        self.assertIs(vault.is_revoked("k", versions[1]), True)
        # The other versions keep answering false.
        for version in versions:
            if version != versions[1]:
                self.assertIs(vault.is_revoked("k", version), False)
        self.assertEqual(vault.revoked_versions("k"), [versions[1]])

    def test_repeated_true_queries_are_identical(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        vault.revoke("k", versions[0])
        vault.revoke("k", versions[2])
        expected = self.status_table(vault, versions)
        self.assertEqual(
            expected,
            {versions[0]: True, versions[1]: False, versions[2]: True, versions[3]: False},
        )
        # Asking again and again gives the completely identical result.
        for _ in range(5):
            self.assertEqual(self.status_table(vault, versions), expected)
            self.assertEqual(vault.revoked_versions("k"), [versions[0], versions[2]])

    def test_true_query_is_read_only_against_disk_and_snapshot(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        vault.revoke("k", versions[3])
        before_disk = self.disk_records()
        before_table = self.status_table(vault, versions)
        for _ in range(5):
            self.assertIs(vault.is_revoked("k", versions[3]), True)
        self.assert_disk_unchanged(before_disk)
        self.assertEqual(self.status_table(vault, versions), before_table)


class TestMarkersSurviveFullReload(RevocationStatusTestCase):
    def test_per_version_status_is_verbatim_identical_after_reload(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        vault.revoke("k", versions[0])
        vault.revoke("k", versions[2])
        before = self.status_table(vault, versions)
        before_listing = vault.revoked_versions("k")

        # A whole-vault reload keeps the marker journal and re-derives the
        # exact same per-version answers.
        vault.reload()
        self.assertEqual(self.status_table(vault, versions), before)
        self.assertEqual(vault.revoked_versions("k"), before_listing)
        vault.reload()
        self.assertEqual(self.status_table(vault, versions), before)
        self.assertEqual(vault.revoked_versions("k"), before_listing)

    def test_per_version_status_is_identical_after_cold_reopen(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        vault.revoke("k", versions[1])
        vault.revoke("k", versions[3])
        before = self.status_table(vault, versions)
        before_listing = vault.revoked_versions("k")

        # A second, independent handle reads the journals from disk and
        # lands on the verbatim same answers; reloading that handle and the
        # original one again changes nothing.
        reopened = self.open_vault()
        self.assertEqual(self.status_table(reopened, versions), before)
        self.assertEqual(reopened.revoked_versions("k"), before_listing)
        reopened.reload()
        self.assertEqual(self.status_table(reopened, versions), before)
        self.assertEqual(self.status_table(vault, versions), before)
        self.assertEqual(vault.revoked_versions("k"), before_listing)

    def test_status_identical_after_reload_on_a_second_disk_built_the_same_way(self):
        # Repeating the identical input in an independently built vault must
        # produce the identical status table before and after a full reload.
        tables = []
        for name in ("reload-copy-a", "reload-copy-b"):
            root = self.fixture.path(name)
            vault = self.open_vault(root)
            versions = self.seal_payloads(vault)
            vault.revoke("k", versions[0])
            vault.revoke("k", versions[3])
            table_before = self.status_table(vault, versions)
            vault.reload()
            reopened = self.open_vault(root)
            table_after = self.status_table(reopened, versions)
            self.assertEqual(table_after, table_before)
            tables.append(table_after)
        self.assertEqual(tables[0], tables[1])


class TestActiveRepointDoesNotMoveMarkers(RevocationStatusTestCase):
    def test_set_active_leaves_existing_markers_in_place(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        vault.revoke("k", versions[1])
        before = self.status_table(vault, versions)

        # Repointing at a non-revoked historical version moves only the
        # active pointer; the marker table must not change at all.
        vault.set_active("k", versions[0])
        self.assertEqual(vault.active("k"), versions[0])
        self.assertEqual(self.status_table(vault, versions), before)
        self.assertEqual(vault.revoked_versions("k"), [versions[1]])
        vault.reload()
        self.assertEqual(self.status_table(vault, versions), before)
        self.assertEqual(vault.active("k"), versions[0])
        self.assertEqual(self.open_vault().active("k"), versions[0])

    def test_version_revoked_after_being_pointed_at_reports_true(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        # Point at a version first, then revoke it afterwards: revocation
        # never moves the pointer, and the later marker does not disturb the
        # status answers of any other version.
        vault.set_active("k", versions[0])
        before_revoke = self.status_table(vault, versions)
        vault.revoke("k", versions[0])

        expected = dict(before_revoke)
        expected[versions[0]] = True
        self.assertEqual(self.status_table(vault, versions), expected)
        self.assertIs(vault.is_revoked("k", versions[0]), True)
        # The active pointer stays exactly where the repoint put it.
        self.assertEqual(vault.active("k"), versions[0])
        # And the whole picture survives reload and cold reopen.
        vault.reload()
        self.assertEqual(vault.active("k"), versions[0])
        self.assertEqual(self.status_table(vault, versions), expected)
        reopened = self.open_vault()
        self.assertEqual(reopened.active("k"), versions[0])
        self.assertEqual(self.status_table(reopened, versions), expected)

    def test_sealing_after_repoint_adds_no_marker_and_keeps_old_ones(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        vault.revoke("k", versions[1])
        vault.set_active("k", versions[0])
        # A later seal supersedes the repoint and becomes active, but it
        # cannot add, remove or shift a revocation marker.
        new_version = vault.seal("k", b"fresh-after-repoint")
        self.assertEqual(vault.active("k"), new_version)
        self.assertEqual(vault.revoked_versions("k"), [versions[1]])
        self.assertIs(vault.is_revoked("k", versions[1]), True)
        self.assertIs(vault.is_revoked("k", new_version), False)
        vault.reload()
        self.assertEqual(vault.revoked_versions("k"), [versions[1]])
        self.assertIs(vault.is_revoked("k", new_version), False)


class TestRevokedVersionStaysReadable(RevocationStatusTestCase):
    def test_revoked_version_reads_back_byte_for_byte(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        for version in versions:
            vault.revoke("k", version)
        # Every revoked version is still loadable by version number with
        # exactly the bytes originally sealed.
        for version in versions:
            self.assertIs(type(vault.load("k", version)), bytes)
            self.assertEqual(vault.load("k", version), PAYLOADS[version])
        # The versions list and active pointer are untouched as well.
        self.assertEqual(vault.versions("k"), versions)
        self.assertEqual(vault.active("k"), versions[-1])

    def test_revoked_bytes_identical_after_reload_and_reopen(self):
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        vault.revoke("k", versions[0])
        vault.revoke("k", versions[2])

        vault.reload()
        for version in (versions[0], versions[2]):
            self.assertEqual(vault.load("k", version), PAYLOADS[version])
            self.assertIs(vault.is_revoked("k", version), True)

        reopened = self.open_vault()
        reopened.reload()
        for version in versions:
            self.assertEqual(reopened.load("k", version), PAYLOADS[version])
        # A non-revoked version read through the same handle matches too.
        self.assertEqual(reopened.load("k", versions[1]), PAYLOADS[versions[1]])

    def test_status_and_material_stay_stable_across_repeated_runs(self):
        # The whole revoke/query/read sequence is deterministic: running it
        # three times against one directory yields the identical answers and
        # never changes a sealed byte.
        vault = self.open_vault()
        versions = self.seal_payloads(vault)
        vault.revoke("k", versions[2])

        outcomes = []
        for _ in range(3):
            outcomes.append(
                (
                    self.status_table(vault, versions),
                    vault.revoked_versions("k"),
                    {v: vault.load("k", v) for v in versions},
                )
            )
            vault.reload()
        self.assertTrue(outcomes)
        self.assertTrue(all(outcome == outcomes[0] for outcome in outcomes))
        self.assertEqual(
            outcomes[0][2], {version: PAYLOADS[version] for version in versions}
        )


if __name__ == "__main__":
    unittest.main()
