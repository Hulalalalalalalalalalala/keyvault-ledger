"""Explicit regressions for the whole reload failure -> recovery round trip.

The baseline vault already implements sealing, reading, revocation, active
repointing and whole-vault reload, the double invariance on a failed reload
already has coverage, and the three README subcommands keep their existing
semantics.  This module adds *no* product behaviour, public interface or
CLI behaviour.  It adds the one layer that was still only checked
implicitly: after a damaged vault is repaired and reloaded, the recovered
in-memory snapshot and a **brand-new instance** opened on the same
directory must read exactly the same state -- item by item, byte for byte.

The cases hand-damage the disk records of a temporary vault and drive the
whole round trip:

* every failure family named in the contract makes a whole ``reload()``
  raise exactly ``ValueError`` -- a missing material, a corrupt or missing
  manifest, a broken persisted ``derivation`` record (bad declared length,
  corrupt salt), an activity-journal record missing a field / carrying a
  broken trailing line, and a corrupt or duplicate revocation record;

* during the failure window the in-memory snapshot is frozen: the keys
  already in hand stay readable, every stored version reads back exactly
  the bytes originally written (derived material byte-for-byte equal to a
  fresh PBKDF2 run), repeated queries return the identical answers, the
  per-version revocation flags and derivation records stay pinned, the
  disk records neither gain nor lose a byte, another failed reload ends
  with the same ``ValueError``, and a cold opener raises that same
  ``ValueError``;

* the entry-point vocabulary is unchanged while the window is open: an
  empty key id raises ``ValueError``, a non-genuine-integer version raises
  ``TypeError`` and an unknown key/version raises ``KeyError``;

* once the bad record is removed or corrected, one full ``reload()``
  succeeds: the recovered answers are exactly the answers frozen in the
  failure window (recovery moves nothing), the on-disk records are
  byte-for-byte the healthy ones, and -- the direct point of this module
  -- a freshly opened instance reports, per key, the identical version
  list, active version, revocation markers (plus per-version status),
  derivation records and material bytes, the manifest and the unknown-key
  answers included; every material byte also equals the raw material file
  on disk.  A second fresh instance reads identically and independently,
  and running the identical damage/repair input a second time reproduces
  the identical exception and the identical recovered state.

Everything happens inside per-case temporary directories, uses the
standard library only, spawns no thread or process of its own and is
independent of execution order.  The shared fixture drains, returns and
deletes everything on its single teardown path::

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
    _dump_manifest,
)
from tests._fixtures import VaultFixture

# Exact bytes every version was created with.  Derived versions carry the
# PBKDF2 output a caller with the passphrase would re-derive; the passphrase
# itself is never persisted.
PLAIN_MATERIALS = {
    ("alpha", 1): b"alpha-one",
    ("alpha", 2): b"alpha-two",
    ("alpha", 3): b"alpha-three",
    ("charlie", 1): b"charlie-one",
    ("charlie", 3): b"charlie-three",
}
DERIVED_PARAMETERS = {
    ("bravo", 1): (b"bravo-passphrase-1", b"bravo-salt-1", 1000, 32),
    ("bravo", 2): (b"bravo-passphrase-2", b"bravo-salt-2", 2000, 24),
    ("charlie", 2): (b"charlie-passphrase", b"charlie-salt", 500, 20),
}
KEY_IDS = ("alpha", "bravo", "charlie")


def _derive(key_id: str, version: int) -> bytes:
    password, salt, iterations, length = DERIVED_PARAMETERS[(key_id, version)]
    return hashlib.pbkdf2_hmac(
        "sha256", password, salt, iterations, dklen=length
    )


def _original_material(key_id: str, version: int) -> bytes:
    sealed = PLAIN_MATERIALS.get((key_id, version))
    return _derive(key_id, version) if sealed is None else sealed


class _RoundTripSupport:
    """Shared construction, state capture and assertions (not a TestCase).

    Concrete cases multiply-inherit this with ``unittest.TestCase`` so the
    round-trip tests and the standalone correspondence tests each run
    exactly once rather than being inherited a second time.
    """

    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        # Every scenario gets its own vault subdirectory, so cases never
        # share disk state and execution order cannot matter.
        self.root = self.fixture.tmp_path / "vaults"

    def open_vault(self, root: Path) -> Vault:
        """Open a vault through the case's single teardown path."""
        return self.fixture.open(root)

    # ------------------------------------------------------------------
    # fixture construction: both material kinds, markers and repoints
    # ------------------------------------------------------------------

    def _build_healthy(self, name: str) -> tuple[Vault, Path]:
        """Build the shared rich vault and return ``(handle, root)``.

        ``alpha`` has three plain versions, is repointed at version 1
        (bound to newest 3) and has version 2 revoked; ``bravo`` has two
        derived versions; ``charlie`` interleaves plain/derived/plain,
        revokes version 1 and repoints at the derived version 2 (bound to
        newest 3).  Every record kind is therefore at stake at once.
        """
        root = self.root / name
        vault = self.open_vault(root)

        vault.seal("alpha", PLAIN_MATERIALS[("alpha", 1)])
        vault.seal("alpha", PLAIN_MATERIALS[("alpha", 2)])
        vault.seal("alpha", PLAIN_MATERIALS[("alpha", 3)])
        vault.set_active("alpha", 1)  # activations.jsonl: bound latest=3
        vault.revoke("alpha", 2)

        password, salt, iterations, length = DERIVED_PARAMETERS[("bravo", 1)]
        vault.derive_seal("bravo", password, salt, iterations, length)
        password, salt, iterations, length = DERIVED_PARAMETERS[("bravo", 2)]
        vault.derive_seal("bravo", password, salt, iterations, length)

        vault.seal("charlie", PLAIN_MATERIALS[("charlie", 1)])
        password, salt, iterations, length = DERIVED_PARAMETERS[
            ("charlie", 2)
        ]
        vault.derive_seal("charlie", password, salt, iterations, length)
        vault.seal("charlie", PLAIN_MATERIALS[("charlie", 3)])
        vault.revoke("charlie", 1)
        vault.set_active("charlie", 2)  # points at derived v2, latest=3

        return vault, root

    # ------------------------------------------------------------------
    # observable state and disk inventory
    # ------------------------------------------------------------------

    def _answers(self, vault: Vault) -> dict:
        """Every state a reader can query, in a comparable structure."""
        keys = {}
        for key_id in KEY_IDS:
            versions = vault.versions(key_id)
            keys[key_id] = {
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
        return {
            "keys": keys,
            "manifest": vault.manifest(),
            "unknown_versions": vault.versions("never-sealed"),
            "unknown_revoked": vault.revoked_versions("never-sealed"),
        }

    def _word_for_word(self, vault: Vault) -> tuple:
        """A flat tuple re-queried while the failure window is open."""
        answers = []
        for key_id in KEY_IDS:
            versions = vault.versions(key_id)
            answers.extend(
                (
                    key_id,
                    tuple(versions),
                    vault.active(key_id),
                    tuple(vault.revoked_versions(key_id)),
                    tuple(
                        (v, vault.is_revoked(key_id, v)) for v in versions
                    ),
                    tuple(vault.load(key_id, v) for v in versions),
                    tuple(
                        (v, vault.derivation(key_id, v)) for v in versions
                    ),
                )
            )
        answers.append(vault.manifest())
        return tuple(answers)

    def _disk_records(self, root: Path) -> dict[str, bytes]:
        """All vault records on disk keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device carrying no key
        data, so it is excluded from the byte-for-byte inventory.
        """
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def _material_file(
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

    def _edit_manifest(self, root: Path, mutate) -> None:
        path = root / MANIFEST_NAME
        manifest = json.loads(path.read_bytes().decode("utf-8"))
        mutate(manifest)
        # Serialize exactly like the product does, so a benign edit would
        # not introduce a formatting difference.
        path.write_bytes(_dump_manifest(manifest))

    @staticmethod
    def _record(manifest: dict, key_id: str, version: int) -> dict:
        return next(
            record
            for record in manifest["keys"][key_id]["versions"]
            if record["version"] == version
        )

    # ------------------------------------------------------------------
    # the frozen failure window
    # ------------------------------------------------------------------

    def _assert_frozen_readback(self, vault: Vault) -> None:
        """Keys in hand stay readable; bytes equal what was sealed."""
        for key_id in KEY_IDS:
            for version in vault.versions(key_id):
                material = vault.load(key_id, version)
                self.assertIs(type(material), bytes)
                self.assertEqual(
                    material, _original_material(key_id, version)
                )
                # Asking twice yields the same object bytes.
                self.assertEqual(
                    vault.load(key_id, version), material
                )
        # The repointed active pointers and the markers survive untouched.
        self.assertEqual(vault.active("alpha"), 1)
        self.assertEqual(vault.active("charlie"), 2)
        self.assertEqual(vault.active("bravo"), 2)
        self.assertTrue(vault.is_revoked("alpha", 2))
        self.assertFalse(vault.is_revoked("alpha", 1))
        self.assertTrue(vault.is_revoked("charlie", 1))
        self.assertEqual(vault.revoked_versions("alpha"), [2])
        self.assertEqual(vault.revoked_versions("charlie"), [1])
        self.assertEqual(vault.revoked_versions("bravo"), [])

    def _assert_entry_contract_unchanged(self, vault: Vault) -> None:
        """The documented ValueError/TypeError/KeyError vocabulary holds."""
        # Empty id -> ValueError, checked ahead of the version type.
        for call in (
            lambda: vault.load(""),
            lambda: vault.load("", 1),
            lambda: vault.load("", 1.0),
            lambda: vault.active(""),
            lambda: vault.is_revoked("", 1),
            lambda: vault.is_revoked("", 1.0),
            lambda: vault.revoked_versions(""),
            lambda: vault.derivation(""),
            lambda: vault.derivation("", 1),
            lambda: vault.derivation("", 1.0),
        ):
            with self.assertRaises(ValueError, msg=call):
                call()

        # Non-genuine-integer version -> TypeError (bools/floats do not
        # count, even 1.0 == 1).
        for call in (
            lambda: vault.load("alpha", 1.0),
            lambda: vault.load("alpha", True),
            lambda: vault.derivation("alpha", 1.0),
            lambda: vault.derivation("bravo", False),
            lambda: vault.is_revoked("alpha", 1.0),
            lambda: vault.is_revoked("charlie", True),
        ):
            with self.assertRaises(TypeError, msg=call):
                call()

        # Unknown key / nonexistent genuine-int version -> KeyError.
        for call in (
            lambda: vault.load("never-sealed"),
            lambda: vault.load("never-sealed", 1),
            lambda: vault.load("alpha", 99),
            lambda: vault.active("never-sealed"),
            lambda: vault.is_revoked("never-sealed", 1),
            lambda: vault.is_revoked("alpha", 99),
            lambda: vault.derivation("never-sealed"),
            lambda: vault.derivation("alpha", 99),
        ):
            with self.assertRaises(KeyError, msg=call):
                call()

        # versions alone has no id gate and answers [].
        self.assertEqual(vault.versions(""), [])
        self.assertEqual(vault.versions("never-sealed"), [])

    def _assert_failure_window(
        self, vault: Vault, root: Path, frozen: dict, damaged: dict
    ) -> str:
        """Pin the frozen window; return the complaint message."""
        message = None
        first_words = None
        for _ in range(3):
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            if message is None:
                message = str(caught.exception)
            else:
                self.assertEqual(str(caught.exception), message)
            # Snapshot untouched word for word after every failed reload.
            self.assertEqual(self._answers(vault), frozen)
            words = self._word_for_word(vault)
            if first_words is None:
                first_words = words
            else:
                self.assertEqual(words, first_words)
            self._assert_frozen_readback(vault)
            self._assert_entry_contract_unchanged(vault)

        # Re-triggering the whole reload with the damage still in place
        # ends with the same ValueError again.
        with self.assertRaises(ValueError) as caught:
            vault.reload()
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(self._answers(vault), frozen)

        # A cold opener rejects the same state with the same complaint.
        with self.assertRaises(ValueError) as caught:
            Vault(root)
        self.assertEqual(str(caught.exception), message)

        # Disk records neither gained nor lost a byte.
        self.assertEqual(self._disk_records(root), damaged)
        return message

    # ------------------------------------------------------------------
    # recovery and the direct snapshot/new-instance comparison
    # ------------------------------------------------------------------

    def _assert_materials_match_raw_disk(
        self, vault: Vault, root: Path
    ) -> None:
        """Every served byte equals the raw material file on disk."""
        manifest = json.loads(
            (root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )
        for key_id, entry in manifest["keys"].items():
            for record in entry["versions"]:
                version = record["version"]
                raw = (root / record["file"]).read_bytes()
                served = vault.load(key_id, version)
                self.assertEqual(served, raw)
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), record["sha256"]
                )

    def _assert_new_instance_matches_recovered_snapshot(
        self, live: Vault, root: Path
    ) -> Vault:
        """The direct point: a fresh instance equals the recovered snapshot.

        The assertion is written item by item (not only as one big
        structure equality): version list, active version, revocation
        markers with per-version status, derivation records and material
        bytes per key, the persisted manifest and unknown-key answers, and
        every material byte against the raw material file on disk.
        """
        fresh = self.open_vault(root)

        # Whole-state equality first...
        self.assertEqual(self._answers(fresh), self._answers(live))

        # ...then the explicit, item-by-item comparison.
        for key_id in KEY_IDS:
            live_versions = live.versions(key_id)
            fresh_versions = fresh.versions(key_id)
            self.assertEqual(fresh_versions, live_versions, key_id)
            self.assertEqual(fresh.active(key_id), live.active(key_id), key_id)
            self.assertEqual(
                fresh.revoked_versions(key_id),
                live.revoked_versions(key_id),
                key_id,
            )
            for version in live_versions:
                self.assertEqual(
                    fresh.is_revoked(key_id, version),
                    live.is_revoked(key_id, version),
                    (key_id, version),
                )
                self.assertEqual(
                    fresh.derivation(key_id, version),
                    live.derivation(key_id, version),
                    (key_id, version),
                )
                live_bytes = live.load(key_id, version)
                fresh_bytes = fresh.load(key_id, version)
                self.assertIs(type(fresh_bytes), bytes)
                self.assertEqual(fresh_bytes, live_bytes, (key_id, version))
                self.assertEqual(
                    fresh_bytes, _original_material(key_id, version)
                )

        self.assertEqual(fresh.manifest(), live.manifest())
        self.assertEqual(
            fresh.versions("never-sealed"), live.versions("never-sealed")
        )
        self.assertEqual(
            fresh.revoked_versions("never-sealed"),
            live.revoked_versions("never-sealed"),
        )
        self._assert_materials_match_raw_disk(fresh, root)
        self._assert_materials_match_raw_disk(live, root)
        return fresh

    def _run_round_trip(self, name: str, damage, repair) -> None:
        """Drive one full damage -> frozen window -> repair round trip.

        The identical input is run twice to pin determinism and order
        independence; the second pass must end with the same complaint and
        the same recovered state as the first.
        """
        vault, root = self._build_healthy(name)
        frozen = self._answers(vault)
        healthy_disk = self._disk_records(root)

        message = None
        for cycle in range(2):
            damage(root)
            damaged_disk = self._disk_records(root)
            message = self._assert_failure_window(
                vault, root, frozen, damaged_disk
            )

            # Undo the bad bytes (remove or correct): one full reload
            # restores normal service.
            repair(root, healthy_disk)
            vault.reload()

            # Recovery moves none of the frozen answers; it only adds the
            # post-recovery comparison point.
            self.assertEqual(self._answers(vault), frozen)

            # The direct assertion: a brand-new instance reads exactly the
            # recovered snapshot, and a second fresh instance agrees with
            # the first, independently.
            fresh = self._assert_new_instance_matches_recovered_snapshot(
                vault, root
            )
            another = self.open_vault(root)
            self.assertEqual(self._answers(another), self._answers(fresh))
            self.assertEqual(self._answers(another), frozen)

            # The repaired directory is byte-for-byte the healthy one:
            # no record gained, none lost.
            self.assertEqual(self._disk_records(root), healthy_disk)

        # The complaint seen in the second identical cycle is the same.
        # (message holds the final cycle's value; compare once more with a
        # fresh damage to make the point explicit.)
        damage(root)
        with self.assertRaises(ValueError) as caught:
            vault.reload()
        self.assertEqual(str(caught.exception), message)
        repair(root, healthy_disk)
        vault.reload()
        self.assertEqual(self._answers(vault), frozen)
        final = self.open_vault(root)
        self.assertEqual(self._answers(final), frozen)
        self.assertEqual(self._disk_records(root), healthy_disk)


class TestReloadFailureRecoveryRoundTrip(_RoundTripSupport, unittest.TestCase):
    """One full failure -> frozen window -> recovery round trip per family."""

    # Damage catalogue: material, manifest, derivation records, journals.

    def test_missing_material_round_trip_plain_and_derived(self):
        scenarios = [
            ("missing-alpha-1", "alpha", 1),
            ("missing-bravo-1", "bravo", 1),
            ("missing-charlie-2", "charlie", 2),
        ]
        for label, key_id, version in scenarios:
            with self.subTest(case=label):
                def damage(root, key_id=key_id, version=version):
                    self._material_file(root, key_id, version).unlink()

                def repair(root, healthy, key_id=key_id, version=version):
                    # Correct the missing record by restoring its exact
                    # original bytes.
                    path = self._material_file(root, key_id, version)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(_original_material(key_id, version))
                    # Everything else must already match the healthy bytes.
                    self.assertEqual(self._disk_records(root), healthy)

                self._run_round_trip(label, damage, repair)

    def test_tampered_material_round_trip(self):
        def damage(root):
            self._material_file(root, "charlie", 3).write_bytes(
                b"tampered bytes"
            )

        def repair(root, healthy):
            path = self._material_file(root, "charlie", 3)
            path.write_bytes(_original_material("charlie", 3))
            self.assertEqual(self._disk_records(root), healthy)

        self._run_round_trip("tampered-material", damage, repair)

    def test_corrupt_and_missing_manifest_round_trip(self):
        def corrupt(root):
            (root / MANIFEST_NAME).write_bytes(b"{not json\n")

        def restore_manifest(root, healthy):
            (root / MANIFEST_NAME).write_bytes(healthy[MANIFEST_NAME])
            self.assertEqual(self._disk_records(root), healthy)

        self._run_round_trip("manifest-corrupt", corrupt, restore_manifest)

        def remove(root):
            (root / MANIFEST_NAME).unlink()

        self._run_round_trip("manifest-missing", remove, restore_manifest)

    def test_manifest_unsupported_format_round_trip(self):
        def damage(root):
            self._edit_manifest(root, lambda manifest: manifest.__setitem__(
                "format", 999
            ))

        def repair(root, healthy):
            (root / MANIFEST_NAME).write_bytes(healthy[MANIFEST_NAME])
            self.assertEqual(self._disk_records(root), healthy)

        self._run_round_trip("manifest-format", damage, repair)

    def test_broken_derivation_records_round_trip(self):
        def set_length(manifest, value):
            self._record(manifest, "charlie", 2)["derivation"][
                "length"
            ] = value

        def set_salt(manifest, value):
            self._record(manifest, "charlie", 2)["derivation"][
                "salt"
            ] = value

        cases = [
            (
                "length-too-small",
                lambda root: self._edit_manifest(
                    root, lambda m: set_length(m, 16)
                ),
            ),
            (
                "length-too-large",
                lambda root: self._edit_manifest(
                    root, lambda m: set_length(m, 48)
                ),
            ),
            (
                "iterations-zero",
                lambda root: self._edit_manifest(
                    root,
                    lambda m: self._record(m, "charlie", 2)["derivation"]
                    .__setitem__("iterations", 0),
                ),
            ),
            (
                "iterations-float",
                lambda root: self._edit_manifest(
                    root,
                    lambda m: self._record(m, "charlie", 2)["derivation"]
                    .__setitem__("iterations", 1.5),
                ),
            ),
            (
                "salt-corrupt",
                lambda root: self._edit_manifest(
                    root, lambda m: set_salt(m, "!!!not base64!!!")
                ),
            ),
            (
                "salt-missing",
                lambda root: self._edit_manifest(
                    root,
                    lambda m: self._record(m, "charlie", 2)["derivation"]
                    .pop("salt"),
                ),
            ),
        ]
        for label, damage in cases:
            with self.subTest(case=label):
                def repair(root, healthy):
                    # Correct the one bad manifest record: the healthy
                    # manifest bytes are put back verbatim.
                    (root / MANIFEST_NAME).write_bytes(
                        healthy[MANIFEST_NAME]
                    )
                    self.assertEqual(self._disk_records(root), healthy)

                self._run_round_trip(f"derivation-{label}", damage, repair)

    def test_activity_journal_round_trip(self):
        healthy_alpha = (
            b'{"key_id": "alpha", "latest": 3, "version": 1}\n'
        )

        def missing_field(root):
            # The activity record named in the contract: a field missing.
            (root / ACTIVATIONS_NAME).write_bytes(
                b'{"key_id": "alpha", "version": 1}\n'
            )

        def broken_json(root):
            (root / ACTIVATIONS_NAME).write_bytes(b"{not json\n")

        def good_then_bad_trailing(root):
            # Keep the good records, append one broken trailing line.
            with (root / ACTIVATIONS_NAME).open("ab") as fh:
                fh.write(b"{broken\n")

        def repair_restore(root, healthy):
            (root / ACTIVATIONS_NAME).write_bytes(
                healthy[ACTIVATIONS_NAME]
            )
            self.assertEqual(self._disk_records(root), healthy)

        def repair_drop_trailing(root, healthy):
            # Remove precisely the bad trailing record; the good records
            # it followed must survive byte-for-byte.
            path = root / ACTIVATIONS_NAME
            lines = path.read_bytes().splitlines(keepends=True)
            self.assertEqual(lines[-1], b"{broken\n")
            path.write_bytes(b"".join(lines[:-1]))
            # The surviving good alpha line is untouched...
            self.assertIn(healthy_alpha, path.read_bytes())
            # ...and the journal is back to the healthy bytes exactly.
            self.assertEqual(self._disk_records(root), healthy)

        self._run_round_trip(
            "activity-missing-field", missing_field, repair_restore
        )
        self._run_round_trip(
            "activity-broken-json", broken_json, repair_restore
        )
        self._run_round_trip(
            "activity-trailing-line",
            good_then_bad_trailing,
            repair_drop_trailing,
        )

    def test_revocation_journal_round_trip(self):
        def broken_json(root):
            (root / REVOCATIONS_NAME).write_bytes(b"{not json\n")

        def duplicate_trailing(root):
            # Append a duplicate of the existing alpha v2 revocation;
            # whole-vault validation rejects a repeated revocation.
            with (root / REVOCATIONS_NAME).open("ab") as fh:
                fh.write(b'{"key_id": "alpha", "version": 2}\n')

        def repair_restore(root, healthy):
            (root / REVOCATIONS_NAME).write_bytes(
                healthy[REVOCATIONS_NAME]
            )
            self.assertEqual(self._disk_records(root), healthy)

        def repair_drop_trailing(root, healthy):
            # Remove precisely the duplicated trailing record; the
            # surviving journal must be byte-for-byte the healthy one.
            path = root / REVOCATIONS_NAME
            lines = path.read_bytes().splitlines(keepends=True)
            self.assertEqual(
                lines[-1], b'{"key_id": "alpha", "version": 2}\n'
            )
            path.write_bytes(b"".join(lines[:-1]))
            self.assertEqual(self._disk_records(root), healthy)

        self._run_round_trip(
            "revocation-broken-json", broken_json, repair_restore
        )
        self._run_round_trip(
            "revocation-duplicate",
            duplicate_trailing,
            repair_drop_trailing,
        )


# ---------------------------------------------------------------------------
# the recovered correspondence, as a standalone direct assertion
# ---------------------------------------------------------------------------


class TestRecoveredSnapshotOneToOne(_RoundTripSupport, unittest.TestCase):
    def test_fresh_instance_after_recovery_equals_recovered_snapshot(self):
        vault, root = self._build_healthy("one-to-one")
        frozen = self._answers(vault)
        healthy_disk = self._disk_records(root)

        # Damage: activity record missing a field -- one of the named
        # failure families.
        (root / ACTIVATIONS_NAME).write_bytes(
            b'{"key_id": "alpha", "latest": 3}\n'
        )
        damaged_disk = self._disk_records(root)

        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
        self.assertEqual(self._answers(vault), frozen)
        self.assertEqual(self._disk_records(root), damaged_disk)

        # Correct the record, one full reload restores normal service.
        (root / ACTIVATIONS_NAME).write_bytes(
            healthy_disk[ACTIVATIONS_NAME]
        )
        vault.reload()
        recovered = self._answers(vault)
        # Recovery moved none of the answers frozen during the window.
        self.assertEqual(recovered, frozen)

        # The direct regression point: a brand-new instance reads a state
        # completely identical to the recovered snapshot, item by item.
        fresh = self.open_vault(root)
        self._assert_new_instance_matches_recovered_snapshot(vault, root)
        self.assertEqual(self._answers(fresh), recovered)

        # The two instances stay independent and identical afterwards:
        # closing one changes neither the other's answers nor the disk.
        fresh.close()
        fresh.close()  # repeated release is a harmless no-op
        self.assertEqual(self._answers(vault), recovered)
        another = self.open_vault(root)
        self.assertEqual(self._answers(another), recovered)
        self.assertEqual(self._disk_records(root), healthy_disk)

    def test_identical_round_trip_in_two_vaults_is_deterministic(self):
        def one_outcome(name: str):
            vault, root = self._build_healthy(name)
            frozen = self._answers(vault)
            healthy_disk = self._disk_records(root)

            (root / ACTIVATIONS_NAME).write_bytes(
                b'{"key_id": "alpha", "latest": 3}\n'
            )
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            message = str(caught.exception)
            frozen_answers = self._answers(vault)

            (root / ACTIVATIONS_NAME).write_bytes(
                healthy_disk[ACTIVATIONS_NAME]
            )
            vault.reload()
            recovered_answers = self._answers(vault)
            fresh = self.open_vault(root)
            fresh_answers = self._answers(fresh)
            return (
                message,
                frozen_answers,
                recovered_answers,
                fresh_answers,
                self._disk_records(root),
            )

        first = one_outcome("determinism-a")
        second = one_outcome("determinism-b")

        # Same complaint, same frozen answers, same recovered answers, and
        # the new-instance comparison equal in both independently built
        # vaults; even the on-disk records are byte-for-byte the same.
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[4], second[4])
        # The frozen answers are the recovered answers: recovery moved
        # nothing, only added the one-to-one comparison point.
        self.assertEqual(first[1], first[2])
        self.assertEqual(first[2], first[3])


if __name__ == "__main__":
    unittest.main()
