"""Regression tests: the active-version query now shares the empty-id rule.

Baseline drift being pinned: every read query that takes a key id rejected
an empty id with ``ValueError`` at the entry, except ``active``, which used
to treat the empty id as an unknown key and answer ``KeyError``.  The
active-version query now runs the same entry validation as the other reads,
so an empty key id raises ``ValueError`` there too.  This module nails the
aligned contract down end to end:

* the empty id raises ``ValueError`` at ``active`` and at every other read
  entry -- ``load``, ``derivation``, ``is_revoked`` and
  ``revoked_versions`` -- each entry genuinely triggered, on an empty vault
  and on a populated one, with the shared message, deterministically (the
  same battery run twice raises the same types in the same order);

* non-empty behaviour is completely unchanged: the active version of a key
  is its most recently sealed version, a repointed key answers the version
  it was pointed back at, sealing new material moves the pointer to the new
  version, and none of this is affected by reopening the vault or by a full
  reload; unknown keys and versions keep their ``KeyError`` and non-integer
  versions their ``TypeError``;

* every rejected call is strictly read-only: the on-disk manifest,
  materials and journals neither gain nor lose a byte, the in-memory
  snapshot answers word for word what it answered before, and material
  already sealed under the same key reads back byte for byte;

* the reload-failure/recovery round trip keeps its comparison point: while
  an out-of-band corruption makes ``reload()`` fail, the frozen snapshot
  keeps answering (and the empty-id gate still fires first); once the
  damage is removed a full ``reload()`` succeeds and the snapshot matches
  the disk records one to one, as does a fresh opener.

Everything happens inside temporary directories, uses the standard library
only and is independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
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
        return self.fixture.open() if root is None else self.fixture.open(root)

    def _disk_records(self, root: Path) -> dict[str, bytes]:
        """Every vault record on disk (the lock file carries no key data)."""
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }


class TestEmptyIdIsValueErrorAtEveryReadEntry(ActiveEmptyIdTestCase):
    def test_empty_id_raises_value_error_at_active_and_each_read(self):
        vault = self.open_vault()
        # Each entry is genuinely triggered, on an empty vault first and on
        # a populated one after: existing state must not change the
        # classification.
        for populated in (False, True):
            with self.subTest(populated=populated):
                with self.assertRaises(ValueError, msg="active"):
                    vault.active("")
                with self.assertRaises(ValueError, msg="load unversioned"):
                    vault.load("")
                with self.assertRaises(ValueError, msg="load versioned"):
                    vault.load("", 1)
                with self.assertRaises(ValueError, msg="derivation unversioned"):
                    vault.derivation("")
                with self.assertRaises(ValueError, msg="derivation versioned"):
                    vault.derivation("", 1)
                with self.assertRaises(ValueError, msg="is_revoked"):
                    vault.is_revoked("", 1)
                with self.assertRaises(ValueError, msg="revoked_versions"):
                    vault.revoked_versions("")
            vault.seal("k", b"m")

    def test_empty_id_message_is_the_shared_message_at_active(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError) as caught:
            vault.active("")
        self.assertEqual(str(caught.exception), "key_id must not be empty")

    def test_empty_id_battery_is_deterministic(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        calls = (
            lambda: vault.active(""),
            lambda: vault.load(""),
            lambda: vault.load("", 1),
            lambda: vault.derivation(""),
            lambda: vault.derivation("", 1),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
        )

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

        expected = [ValueError] * len(calls)
        self.assertEqual(run(), expected)
        self.assertEqual(run(), expected)


class TestNonEmptyActiveSemanticsUnchanged(ActiveEmptyIdTestCase):
    def test_active_is_the_most_recently_sealed_version(self):
        vault = self.open_vault()
        self.assertEqual(vault.seal("k", b"v1"), 1)
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.seal("k", b"v2"), 2)
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"v2")

    def test_repoint_then_new_seal_moves_the_pointer(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        # A repointed key answers the version it was pointed back at.
        vault.set_active("k", 1)
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"v1")
        # Sealing new material lands the pointer on the new version again.
        self.assertEqual(vault.seal("k", b"v4"), 4)
        self.assertEqual(vault.active("k"), 4)
        self.assertEqual(vault.load("k"), b"v4")

    def test_repoint_survives_reload_and_reopen(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.set_active("k", 1)
        vault.reload()
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.load("k"), b"v1")
        reopened = self.open_vault(self.root)
        self.assertEqual(reopened.active("k"), 1)
        self.assertEqual(reopened.load("k"), b"v1")

    def test_unknown_key_and_bad_version_types_keep_their_exceptions(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.active("never-sealed")
        with self.assertRaises(KeyError):
            vault.load("never-sealed")
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed")
        with self.assertRaises(KeyError):
            vault.is_revoked("never-sealed", 1)
        self.assertEqual(vault.revoked_versions("never-sealed"), [])
        for bad in (1.0, True, "1", (1,)):
            with self.assertRaises(TypeError, msg=f"load {bad!r}"):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("k", bad)
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("k", bad)


class TestRejectedEmptyIdCallsAreReadOnly(ActiveEmptyIdTestCase):
    def test_disk_and_snapshot_untouched_and_material_byte_exact(self):
        vault = self.open_vault()
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        vault.set_active("k", 1)
        vault.revoke("k", 2)

        before_records = self._disk_records(self.root)
        snapshot_before = {
            "active-k": vault.active("k"),
            "active-d": vault.active("d"),
            "versions-k": vault.versions("k"),
            "revoked-k": vault.revoked_versions("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "derivation-d": vault.derivation("d", 1),
            "manifest": vault.manifest(),
        }

        for call in (
            lambda: vault.active(""),
            lambda: vault.load(""),
            lambda: vault.load("", 1),
            lambda: vault.derivation(""),
            lambda: vault.derivation("", 1),
            lambda: vault.is_revoked("", 1),
            lambda: vault.revoked_versions(""),
        ):
            with self.assertRaises(ValueError):
                call()

        # Manifest, materials and both journals neither gained nor lost a
        # byte; the in-memory snapshot answers word for word what it did.
        self.assertEqual(self._disk_records(self.root), before_records)
        snapshot_after = {
            "active-k": vault.active("k"),
            "active-d": vault.active("d"),
            "versions-k": vault.versions("k"),
            "revoked-k": vault.revoked_versions("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "derivation-d": vault.derivation("d", 1),
            "manifest": vault.manifest(),
        }
        self.assertEqual(snapshot_after, snapshot_before)
        # Material sealed under the same key reads back byte for byte.
        self.assertEqual(vault.load("k", 1), b"k-v1")
        self.assertEqual(vault.load("k", 2), b"k-v2")


class TestReloadFailureRecoveryRoundTrip(ActiveEmptyIdTestCase):
    def _answers(self, vault: Vault) -> dict:
        return {
            "active": vault.active("k"),
            "versions": vault.versions("k"),
            "revoked": vault.revoked_versions("k"),
            "materials": {v: vault.load("k", v) for v in vault.versions("k")},
            "manifest": vault.manifest(),
        }

    def _assert_snapshot_matches_disk(self, vault: Vault) -> None:
        """Re-derive every answer straight from the records on disk."""
        manifest = json.loads(
            (self.root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )
        self.assertEqual(vault.manifest(), manifest)
        entry = manifest["keys"]["k"]
        versions = [record["version"] for record in entry["versions"]]
        self.assertEqual(vault.versions("k"), versions)
        for record in entry["versions"]:
            data = (self.root / record["file"]).read_bytes()
            self.assertEqual(vault.load("k", record["version"]), data)
            self.assertEqual(
                hashlib.sha256(data).hexdigest(), record["sha256"]
            )
        # The one repoint record is still bound (no seal since), so the
        # active pointer sits at the repointed version.
        lines = (self.root / ACTIVATIONS_NAME).read_text("utf-8").splitlines()
        record = json.loads(lines[-1])
        self.assertEqual(record["latest"], versions[-1])
        self.assertEqual(vault.active("k"), record["version"])
        revoked = {
            json.loads(line)["version"]
            for line in (self.root / REVOCATIONS_NAME)
            .read_text("utf-8")
            .splitlines()
        }
        self.assertEqual(set(vault.revoked_versions("k")), revoked)

    def test_failed_reload_freezes_then_recovery_corresponds_one_to_one(self):
        vault = self.open_vault()
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        vault.seal("k", b"k-v3")
        vault.set_active("k", 1)
        vault.revoke("k", 2)
        frozen = self._answers(vault)
        healthy_disk = self._disk_records(self.root)

        # Damage the activation journal out of band: reload validation
        # fails, the snapshot stays frozen, and the empty-id gate at
        # ``active`` still fires before any state is consulted.
        activations = self.root / ACTIVATIONS_NAME
        activations.write_bytes(b"{broken json\n")
        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
            self.assertEqual(self._answers(vault), frozen)
            with self.assertRaises(ValueError):
                vault.active("")
        # The failed reloads neither grew nor shrank any record on disk.
        damaged_disk = self._disk_records(self.root)
        self.assertEqual(
            damaged_disk, {**healthy_disk, "activations.jsonl": b"{broken json\n"}
        )

        # Recovery: restore the healthy record, reload, and the snapshot
        # matches the disk records one to one -- for this handle and for a
        # fresh opener.
        activations.write_bytes(healthy_disk["activations.jsonl"])
        vault.reload()
        self.assertEqual(self._answers(vault), frozen)
        self._assert_snapshot_matches_disk(vault)
        self.assertEqual(self._disk_records(self.root), healthy_disk)
        reopened = self.open_vault(self.root)
        self.assertEqual(self._answers(reopened), frozen)
        self._assert_snapshot_matches_disk(reopened)
        # The aligned empty-id rule holds after the round trip too.
        with self.assertRaises(ValueError):
            reopened.active("")


if __name__ == "__main__":
    unittest.main()
