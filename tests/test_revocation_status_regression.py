"""Regression tests pinning both answers of the revocation status query.

``is_revoked(key_id, version) -> bool`` is the read-only revocation-status
entry; this module freezes its two observable answers and the state around
them without touching any product code:

* a version that was never revoked answers ``False``; asking never changes
  any state -- the in-memory snapshot and the on-disk manifest, journals and
  materials are byte-for-byte and mtime-for-mtime untouched, and asking again
  returns the identical answer;

* a revoked version answers ``True`` and every version that was not revoked
  keeps answering ``False``; repeating the same queries in the same order
  gives the identical result sequence;

* the markers survive a whole ``reload()`` and a cold reopen: per-version
  status is, version for version, verbatim what it was before the reload,
  and reload itself writes nothing;

* repointing the active version (``set_active``) never moves a revocation
  marker, and a version that is revoked *after* it was pointed at stays a
  normal answer for the status query -- revocation never moves the active
  pointer either;

* a revoked version stays readable by version, byte for byte identical to
  what was sealed, derived material included; revoking the active version
  leaves the active pointer and the unversioned read where they were.

The exception vocabulary of the query (empty id, non-integer version,
unknown key/version) is pinned by ``test_query_error_contract`` and
``test_load_version_validation`` and is deliberately not redefined here.
Every case works in its own temporary directory, uses the standard library
only, returns its lock handle through the shared fixture and is independent
of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import unittest

from keyvault_ledger import Vault
from keyvault_ledger.vault import LOCK_NAME
from tests._fixtures import VaultFixture


class RevocationStatusTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.root

    def open_vault(self, root=None) -> Vault:
        # Tracked so the single fixture cleanup returns the lock handle and
        # removes the temporary tree however the case ends.
        return self.fixture.open(root)

    def _status_map(self, vault: Vault, key_id: str) -> dict[int, bool]:
        """The query's answer for every version of one key, keyed by version."""
        return {
            version: vault.is_revoked(key_id, version)
            for version in vault.versions(key_id)
        }

    def _disk_state(self) -> dict:
        """Every persisted record as (bytes, mtime); the lock file is excluded.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, exactly as in the reload regression suite.
        """
        files = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != LOCK_NAME:
                files[path.relative_to(self.root)] = (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
        return files


# ---------------------------------------------------------------------------
# the false branch: a version that was never revoked answers False
# ---------------------------------------------------------------------------


class TestNeverRevokedVersionReportsFalse(RevocationStatusTestCase):
    def test_every_version_of_a_never_revoked_key_answers_false(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        derived = vault.derive_seal("k", b"pw", b"salty", 100, 16)
        vault.seal("other", b"m")

        # Plain versions, the derived version, the active version (queried
        # explicitly) and a second key all answer a genuine False.
        for version in (1, 2, derived):
            self.assertIs(vault.is_revoked("k", version), False)
        self.assertIs(vault.is_revoked("other", 1), False)
        # The list view agrees with the per-version answers.
        self.assertEqual(vault.revoked_versions("k"), [])
        self.assertEqual(vault.revoked_versions("other"), [])

    def test_asking_is_stable_and_changes_no_state(self):
        vault = self.open_vault()
        payloads = [b"", bytes(range(256)), b"\n\xff suffix", "λ-key".encode()]
        for payload in payloads:
            vault.seal("k", payload)
        vault.derive_seal("d", b"pw", b"salt", 200, 20)

        before_files = self._disk_state()
        before_manifest = vault.manifest()

        def all_answers() -> dict:
            return {
                key_id: self._status_map(vault, key_id)
                for key_id in ("k", "d")
            }

        # A burst of status queries: every answer is False, every run is
        # identical, and none of them is a write.
        first = all_answers()
        for _ in range(5):
            self.assertEqual(all_answers(), first)
        self.assertEqual(first, {
            "k": {1: False, 2: False, 3: False, 4: False},
            "d": {1: False},
        })

        # No state moved: the active pointers, version lists, revoked lists
        # and readable bytes are exactly what was sealed.
        self.assertEqual(vault.active("k"), 4)
        self.assertEqual(vault.active("d"), 1)
        self.assertEqual(vault.revoked_versions("k"), [])
        for index, payload in enumerate(payloads, start=1):
            self.assertEqual(vault.load("k", index), payload)

        # Disk inventory: same files, same bytes, untouched mtimes.
        self.assertEqual(sorted(self._disk_state()), sorted(before_files))
        for rel, (data, mtime) in before_files.items():
            path = self.root / rel
            self.assertEqual(path.read_bytes(), data, str(rel))
            self.assertEqual(path.stat().st_mtime_ns, mtime, str(rel))
        self.assertEqual(vault.manifest(), before_manifest)

        # A whole reload leaves every false answer just as false.
        vault.reload()
        self.assertEqual(all_answers(), first)


# ---------------------------------------------------------------------------
# the true branch: a revoked version answers True, again and again
# ---------------------------------------------------------------------------


class TestRevokedVersionReportsTrue(RevocationStatusTestCase):
    def test_revoked_versions_answer_true_and_the_rest_false(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        # Revoke out of order on purpose: the marker is per version and the
        # ascending list view must not depend on revocation order.
        vault.revoke("k", 3)
        vault.revoke("k", 1)

        self.assertIs(vault.is_revoked("k", 1), True)
        self.assertIs(vault.is_revoked("k", 2), False)
        self.assertIs(vault.is_revoked("k", 3), True)
        self.assertEqual(self._status_map(vault, "k"), {1: True, 2: False, 3: True})
        self.assertEqual(vault.revoked_versions("k"), [1, 3])

    def test_repeated_queries_after_revoke_are_identical(self):
        vault = self.open_vault()
        for payload in (b"a", b"b", b"c", b"d"):
            vault.seal("k", payload)
        vault.revoke("k", 2)
        vault.revoke("k", 4)

        def answer_sequence() -> tuple:
            # Query in a fixed interleaving that exercises both branches:
            # false, true, false, true, and the revoked-list view.
            return (
                vault.is_revoked("k", 1),
                vault.is_revoked("k", 2),
                vault.is_revoked("k", 3),
                vault.is_revoked("k", 4),
                tuple(vault.revoked_versions("k")),
            )

        expected = (False, True, False, True, (2, 4))
        for _ in range(6):
            self.assertEqual(answer_sequence(), expected)

        # Querying itself appends nothing and rewrites nothing.
        files = self._disk_state()
        for _ in range(6):
            answer_sequence()
        for rel, (data, mtime) in files.items():
            path = self.root / rel
            self.assertEqual(path.read_bytes(), data, str(rel))
            self.assertEqual(path.stat().st_mtime_ns, mtime, str(rel))


# ---------------------------------------------------------------------------
# markers survive a whole reload, version for version verbatim
# ---------------------------------------------------------------------------


class TestRevocationStatusSurvivesFullReload(RevocationStatusTestCase):
    def _build_two_key_state(self) -> Vault:
        vault = self.open_vault()
        # Plain and derived versions interleaved, markers on a non-active
        # and an active version, across two keys and with one key left
        # completely unrevoked.
        vault.seal("alpha", b"a1")
        vault.seal("alpha", b"a2")
        vault.revoke("alpha", 1)
        vault.derive_seal("alpha", b"pw-a", b"salt-a", 500, 24)

        vault.seal("bravo", b"b1")
        vault.derive_seal("bravo", b"pw-b", b"salt-b", 700, 32)
        vault.seal("bravo", b"b3")
        vault.revoke("bravo", 3)
        vault.revoke("bravo", 2)

        vault.seal("charlie", b"c1")
        vault.seal("charlie", b"c2")
        return vault

    def _all_statuses(self, vault: Vault) -> dict:
        return {
            key_id: self._status_map(vault, key_id)
            for key_id in ("alpha", "bravo", "charlie")
        }

    def test_per_version_status_is_verbatim_before_and_after_reload(self):
        vault = self._build_two_key_state()
        before = self._all_statuses(vault)
        before_lists = {
            key_id: vault.revoked_versions(key_id)
            for key_id in ("alpha", "bravo", "charlie")
        }
        files_before = self._disk_state()

        # Repeated whole-vault reloads on the live handle change nothing.
        for _ in range(3):
            vault.reload()
            self.assertEqual(self._all_statuses(vault), before)

        # Reload is strictly read-only: no record grew, shrank or moved.
        self.assertEqual(sorted(self._disk_state()), sorted(files_before))
        for rel, (data, mtime) in files_before.items():
            path = self.root / rel
            self.assertEqual(path.read_bytes(), data, str(rel))
            self.assertEqual(path.stat().st_mtime_ns, mtime, str(rel))

        # A cold opener on the same directory derives the identical
        # per-version answers from the append-only journal.
        reopened = self.open_vault(self.root)
        self.assertEqual(self._all_statuses(reopened), before)
        reopened_lists = {
            key_id: reopened.revoked_versions(key_id)
            for key_id in ("alpha", "bravo", "charlie")
        }
        self.assertEqual(reopened_lists, before_lists)

        # And a second fresh handle after one more reload still agrees
        # verbatim -- including the boolean identity of each answer.
        reopened.reload()
        for key_id, statuses in before.items():
            for version, status in statuses.items():
                self.assertIs(
                    reopened.is_revoked(key_id, version),
                    status,
                    msg=f"{key_id} {version}",
                )

    def test_marker_survives_reload_when_it_is_the_active_version(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 2)  # revoke the current active version
        self.assertEqual(vault.active("k"), 2)
        before = self._status_map(vault, "k")

        vault.reload()
        self.assertEqual(self._status_map(vault, "k"), before)
        self.assertIs(vault.is_revoked("k", 2), True)
        self.assertIs(vault.is_revoked("k", 1), False)
        # Revocation is a marker only: the active pointer did not move.
        self.assertEqual(vault.active("k"), 2)

        reopened = self.open_vault(self.root)
        self.assertEqual(self._status_map(reopened, "k"), before)
        self.assertEqual(reopened.active("k"), 2)


# ---------------------------------------------------------------------------
# repointing the active version never moves a revocation marker
# ---------------------------------------------------------------------------


class TestRepointingActiveDoesNotMoveRevocation(RevocationStatusTestCase):
    def test_repoint_and_later_revoke_leave_status_queries_stable(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        vault.revoke("k", 3)

        # Repoint at an unrevoked historical version: the marker on the
        # former newest version stays exactly where it was.
        self.assertIsNone(vault.set_active("k", 2))
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(self._status_map(vault, "k"), {1: False, 2: False, 3: True})
        self.assertEqual(vault.revoked_versions("k"), [3])

        # The pointed-at version is revoked *after* it was pointed at.  That
        # is not an error, moves neither the pointer nor the older marker,
        # and the status query simply reports the new fact.
        vault.revoke("k", 2)
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"v2")  # revoked stays readable
        self.assertEqual(self._status_map(vault, "k"), {1: False, 2: True, 3: True})
        self.assertEqual(vault.revoked_versions("k"), [2, 3])

        before = self._status_map(vault, "k")
        vault.reload()
        self.assertEqual(self._status_map(vault, "k"), before)
        self.assertEqual(vault.active("k"), 2)

        # A cold reopen replays the journal and the repoint identically.
        reopened = self.open_vault(self.root)
        self.assertEqual(self._status_map(reopened, "k"), before)
        self.assertEqual(reopened.active("k"), 2)

        # Sealing a new version afterwards makes it active again; neither
        # marker moves.
        vault.seal("k", b"v4")
        self.assertEqual(vault.active("k"), 4)
        self.assertEqual(
            self._status_map(vault, "k"),
            {1: False, 2: True, 3: True, 4: False},
        )
        vault.reload()
        reopened_v4 = self.open_vault(self.root)
        self.assertEqual(reopened_v4.active("k"), 4)
        self.assertEqual(
            self._status_map(reopened_v4, "k"),
            {1: False, 2: True, 3: True, 4: False},
        )


# ---------------------------------------------------------------------------
# revoked versions keep reading back byte for byte
# ---------------------------------------------------------------------------


class TestRevokedMaterialReadsBackByteForByte(RevocationStatusTestCase):
    def test_every_revoked_version_loads_exactly_what_was_sealed(self):
        vault = self.open_vault()
        payloads = [
            b"",
            bytes(range(256)),
            b"\x00\r\n\xff plain tail",
            "ünïcode-κλειδί".encode("utf-8"),
        ]
        for payload in payloads:
            vault.seal("k", payload)
        password, salt, iterations, length = b"passphrase", b"\x01salty\xff", 800, 40
        derived_version = vault.derive_seal(
            "d", password, salt, iterations, length
        )
        expected_derived = hashlib.pbkdf2_hmac(
            "sha256", password, salt, iterations, dklen=length
        )

        # Revoke every stored version of one key and the derived version.
        for version in range(1, len(payloads) + 1):
            vault.revoke("k", version)
        vault.revoke("d", derived_version)

        # All marked true, yet every versioned read returns the exact bytes
        # written, as a genuine bytes object.
        for index, payload in enumerate(payloads, start=1):
            self.assertIs(vault.is_revoked("k", index), True)
            self.assertIs(type(vault.load("k", index)), bytes)
            self.assertEqual(vault.load("k", index), payload)
        self.assertIs(vault.is_revoked("d", derived_version), True)
        self.assertEqual(vault.load("d", derived_version), expected_derived)
        self.assertEqual(
            vault.derivation("d", derived_version),
            {"salt": salt, "iterations": iterations, "length": length},
        )

        # The version lists and active pointers never changed.
        self.assertEqual(vault.versions("k"), [1, 2, 3, 4])
        self.assertEqual(vault.versions("d"), [derived_version])
        self.assertEqual(vault.active("k"), 4)
        self.assertEqual(vault.active("d"), derived_version)

        before = {
            "k": [vault.load("k", i) for i in range(1, 5)],
            "d": vault.load("d", derived_version),
        }

        # Byte-exact readback survives a full reload and a cold reopen.
        vault.reload()
        for index, payload in enumerate(payloads, start=1):
            self.assertEqual(vault.load("k", index), payload)
        self.assertEqual(vault.load("d", derived_version), expected_derived)

        reopened = self.open_vault(self.root)
        after = {
            "k": [reopened.load("k", i) for i in range(1, 5)],
            "d": reopened.load("d", derived_version),
        }
        self.assertEqual(after, before)
        self.assertEqual(after["d"], expected_derived)


if __name__ == "__main__":
    unittest.main()
