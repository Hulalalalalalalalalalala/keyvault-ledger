"""Explicit regressions for the whole reload failure -> recovery round trip.

The baseline vault already implements sealing, reading, revocation,
repointing and whole-vault reload, and the double invariance (snapshot
frozen / disk untouched) already has coverage elsewhere.  This module adds
only the layer that was still missing -- a direct, item-by-item comparison
between the recovered snapshot and the disk records once the damage is
undone.  No product code, public interface or CLI behaviour is touched.

The fixture vault is built with every record kind at once: three plain
versions of one key (repointed at a historical version, another version
revoked), two derived versions of a second key, and two plain versions of
a third key (one revoked).  Each scenario then damages the records in a
temporary vault directory by hand, triggers a whole ``reload()`` and
observes the exception together with both states:

* missing material, a corrupt manifest, a hand-broken derivation record or
  an activity-journal line missing a field all make a whole ``reload()``
  raise exactly ``ValueError``;

* during the failure window the in-memory snapshot is frozen word for
  word: the keys already in hand stay readable, the read-back bytes are
  byte-for-byte what was sealed/derived, per-version revocation statuses,
  derivation parameters and the active version are verbatim, the records
  on disk neither gain nor lose a byte, repeated queries give the same
  answers, triggering ``reload()`` again ends with the same ``ValueError``,
  and a brand-new opener rejects that state with the same complaint;

* the entry vocabulary is unchanged throughout -- empty id raises
  ``ValueError``, a non-genuine-integer version raises ``TypeError`` and an
  unknown key/version raises ``KeyError`` -- on a healthy vault and during
  the failure window alike;

* once the bad record is removed or corrected, another full ``reload()``
  restores normal service without moving any answer the failure window
  froze; the direct correspondence point is then asserted explicitly: a
  brand-new instance (one straight from its constructor, and one after an
  extra reload) reads a state exactly equal to the recovered snapshot --
  version lists, active version, revocation markers and per-version
  revocation status, derivation records and every material byte -- and
  both instances agree one to one with every record file on disk.

The same input is repeatable with identical results regardless of order,
everything lives under a temporary directory, and the shared single
teardown drains workers, returns every handle and deletes the tree::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import base64
import hashlib
import json
import unittest
from pathlib import Path

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    ACTIVATIONS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
    _dump_manifest,
)
from tests._fixtures import VaultFixture

# What was originally written, keyed by (key id, version).  During the
# failure window every one of these bytes must still read back exactly.
ALPHA_MATERIALS = {
    1: b"alpha-one",
    2: b"alpha-two",
    3: b"alpha-three",
}
BRAVO_DERIVATIONS = {
    1: (b"passphrase-bravo", b"salt-bravo", 1000, 32),
    2: (b"passphrase-two", b"salt-two", 2500, 24),
}
CHARLIE_MATERIALS = {1: b"charlie-one", 2: b"charlie-two"}
KEY_IDS = ("alpha", "bravo", "charlie")


def bravo_material(version: int) -> bytes:
    password, salt, iterations, length = BRAVO_DERIVATIONS[version]
    return hashlib.pbkdf2_hmac(
        "sha256", password, salt, iterations, dklen=length
    )


ORIGINAL_BYTES = {
    **{("alpha", v): data for v, data in ALPHA_MATERIALS.items()},
    **{("bravo", v): bravo_material(v) for v in (1, 2)},
    **{("charlie", v): data for v, data in CHARLIE_MATERIALS.items()},
}


class RoundtripCorrespondenceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        # Every scenario gets its own vault subdirectory, so execution
        # order can never matter.
        self.roots = self.fixture.tmp_path / "vaults"

    def open_vault(self, root: Path) -> Vault:
        """Open a vault through the case's single teardown path."""
        return self.fixture.open(root)

    # ------------------------------------------------------------------
    # fixture construction: every record kind in one healthy vault
    # ------------------------------------------------------------------

    def _build_healthy(self, name: str) -> tuple[Vault, Path]:
        """Build the rich healthy vault and return ``(handle, root)``.

        ``alpha``: three plain versions, repointed at version 1 (bound to
        newest sealed 3), version 2 revoked; ``bravo``: two derived
        versions; ``charlie``: two plain versions, version 1 revoked.
        """
        root = self.roots / name
        vault = self.open_vault(root)
        vault.seal("alpha", ALPHA_MATERIALS[1])
        vault.seal("alpha", ALPHA_MATERIALS[2])
        vault.seal("alpha", ALPHA_MATERIALS[3])
        vault.set_active("alpha", 1)  # activations.jsonl, bound latest=3
        vault.revoke("alpha", 2)
        password, salt, iterations, length = BRAVO_DERIVATIONS[1]
        vault.derive_seal("bravo", password, salt, iterations, length)
        password, salt, iterations, length = BRAVO_DERIVATIONS[2]
        vault.derive_seal("bravo", password, salt, iterations, length)
        vault.seal("charlie", CHARLIE_MATERIALS[1])
        vault.seal("charlie", CHARLIE_MATERIALS[2])
        vault.revoke("charlie", 1)
        return vault, root

    # ------------------------------------------------------------------
    # observable state and raw disk records
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

    def _disk_records(self, root: Path) -> dict[str, bytes]:
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
        for relative in self._disk_records(root).keys() - healthy.keys():
            (root / relative).unlink()
        for relative, data in healthy.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)

    def _edit_manifest(self, root: Path, mutate) -> None:
        path = root / MANIFEST_NAME
        manifest = json.loads(path.read_bytes().decode("utf-8"))
        mutate(manifest)
        path.write_bytes(_dump_manifest(manifest))

    @staticmethod
    def _manifest_record(manifest: dict, key_id: str, version: int) -> dict:
        return next(
            record
            for record in manifest["keys"][key_id]["versions"]
            if record["version"] == version
        )

    def _material_file(
        self, root: Path, key_id: str, version: int
    ) -> Path:
        manifest = json.loads(
            (root / MANIFEST_NAME).read_bytes().decode("utf-8")
        )
        return root / self._manifest_record(manifest, key_id, version)["file"]

    # ------------------------------------------------------------------
    # the four out-of-band damage shapes, each with a surgical repair
    # ------------------------------------------------------------------

    DAMAGE_KINDS = (
        "missing_material",
        "manifest_corrupt",
        "derivation_record_bad",
        "activity_missing_field",
    )

    def _apply_damage(self, root: Path, healthy: dict[str, bytes], name: str) -> None:
        """Damage exactly the one record named by ``name`` out of band."""
        if name == "missing_material":
            # The material disappears; its manifest record and digest stay.
            self._material_file(root, "alpha", 3).unlink()
        elif name == "manifest_corrupt":
            (root / MANIFEST_NAME).write_bytes(b"{not json\n")
        elif name == "derivation_record_bad":
            # bravo version 1 stores 32 derived bytes; claiming length 7 is
            # a self-inconsistent persisted derivation record.
            self._edit_manifest(
                root,
                lambda m: self._manifest_record(m, "bravo", 1)[
                    "derivation"
                ].__setitem__("length", 7),
            )
        elif name == "activity_missing_field":
            # The healthy journal keeps its one good record; a second line
            # missing its ``latest`` field is appended.
            with (root / ACTIVATIONS_NAME).open("ab") as handle:
                handle.write(b'{"key_id": "alpha", "version": 1}\n')
        else:  # pragma: no cover - guards the test table itself
            raise AssertionError(f"unknown damage kind {name!r}")

    def _apply_repair(self, root: Path, healthy: dict[str, bytes], name: str) -> None:
        """Remove/correct just the offending bytes for ``name``.

        The repaired directory is byte-identical to the captured healthy
        state.
        """
        if name == "missing_material":
            path = self._material_file(root, "alpha", 3)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(ALPHA_MATERIALS[3])
        elif name == "manifest_corrupt":
            (root / MANIFEST_NAME).write_bytes(healthy[MANIFEST_NAME])
        elif name == "derivation_record_bad":
            self._edit_manifest(
                root,
                lambda m: self._manifest_record(m, "bravo", 1)[
                    "derivation"
                ].__setitem__("length", 32),
            )
        elif name == "activity_missing_field":
            (root / ACTIVATIONS_NAME).write_bytes(healthy[ACTIVATIONS_NAME])
        else:  # pragma: no cover - guards the test table itself
            raise AssertionError(f"unknown damage kind {name!r}")

    # ------------------------------------------------------------------
    # the frozen failure window
    # ------------------------------------------------------------------

    def _assert_failure_window(
        self, vault: Vault, root: Path, frozen: dict
    ) -> str:
        """Drive the failure window; return the first error message.

        The damage is already on disk when this is called.  Three live
        reloads fail with one identical ``ValueError``, a brand-new opener
        fails with the same complaint, every answer stays byte-for-byte
        frozen (with the originally written bytes still served), and disk
        records neither gain nor lose anything.
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
            # Snapshot frozen word for word.
            self.assertEqual(self._answers(vault), frozen)

        # Triggering the whole reload once more ends the same way.
        with self.assertRaises(ValueError) as caught:
            vault.reload()
        self.assertEqual(str(caught.exception), message)

        # A brand-new opener on the same directory rejects that state with
        # the very same ValueError (its constructor releases its own lock
        # handle on failure, so nothing is left open).
        with self.assertRaises(ValueError) as caught:
            Vault(root)
        self.assertEqual(str(caught.exception), message)

        # Keys already in hand stay readable; read-back is byte-for-byte
        # what was sealed or derived.
        for (key_id, version), original in ORIGINAL_BYTES.items():
            self.assertIs(type(vault.load(key_id, version)), bytes)
            self.assertEqual(vault.load(key_id, version), original)
        # The repointed active pointer and every marker survive.
        self.assertEqual(vault.active("alpha"), 1)
        self.assertEqual(vault.active("bravo"), 2)
        self.assertEqual(vault.active("charlie"), 2)
        self.assertEqual(vault.revoked_versions("alpha"), [2])
        self.assertEqual(vault.revoked_versions("charlie"), [1])
        self.assertTrue(vault.is_revoked("alpha", 2))
        self.assertFalse(vault.is_revoked("alpha", 1))

        # Repeated queries while the failure persists give identical
        # answers, failed reloads in between included.
        first_word = self._word_for_word(vault)
        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
            self.assertEqual(self._word_for_word(vault), first_word)

        # Disk records: same names, same bytes; no repair artifact.
        self.assertEqual(self._disk_records(root), damaged_disk)
        self.assertEqual(list(root.rglob("*.tmp.*")), [])
        return message

    def _word_for_word(self, vault: Vault) -> tuple:
        """A flat tuple of every repeated query's answer."""
        words = []
        for key_id in KEY_IDS:
            versions = vault.versions(key_id)
            words.extend(
                (
                    ("versions", key_id, tuple(versions)),
                    ("active", key_id, vault.active(key_id)),
                    (
                        "revoked",
                        key_id,
                        tuple(vault.revoked_versions(key_id)),
                    ),
                    (
                        "statuses",
                        key_id,
                        tuple(
                            (v, vault.is_revoked(key_id, v)) for v in versions
                        ),
                    ),
                    (
                        "materials",
                        key_id,
                        tuple((v, vault.load(key_id, v)) for v in versions),
                    ),
                    (
                        "derivations",
                        key_id,
                        tuple(
                            (v, vault.derivation(key_id, v)) for v in versions
                        ),
                    ),
                )
            )
        words.append(("manifest", vault.manifest()))
        words.append(("unknown-versions", vault.versions("never-sealed")))
        words.append(
            ("unknown-revoked", vault.revoked_versions("never-sealed"))
        )
        return tuple(words)

    # ------------------------------------------------------------------
    # independent snapshot <-> disk correspondence
    # ------------------------------------------------------------------

    def _assert_instance_matches_disk_records(
        self, vault: Vault, root: Path
    ) -> None:
        """Re-derive every answer straight from the raw disk records.

        No private vault state is consulted: the manifest and both
        journals are parsed independently and must explain every public
        answer exactly, with the material inventory matching one to one.
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
                # The instance serves the exact bytes persisted on disk...
                self.assertEqual(vault.load(key_id, version), data)
                # ...and those bytes match the manifest digest.
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
            for version in versions:
                self.assertEqual(
                    vault.is_revoked(key_id, version),
                    version in revoked.get(key_id, set()),
                )

        # Material inventory one to one: every stored file is declared by
        # exactly one manifest record and vice versa.
        declared = {
            record["file"]
            for entry in manifest["keys"].values()
            for record in entry["versions"]
        }
        on_disk = {
            str(path.relative_to(root))
            for path in (root / MATERIALS_DIR).rglob("*.bin")
        }
        self.assertEqual(on_disk, declared)

        # Unknown-key query shapes.
        self.assertEqual(vault.versions("never-sealed"), [])
        self.assertEqual(vault.revoked_versions("never-sealed"), [])

    def _assert_keywise_identical(self, left: dict, right: dict) -> None:
        """Direct item-by-item equality of two full answer structures."""
        self.assertEqual(set(left["keys"]), set(right["keys"]))
        for key_id in left["keys"]:
            a, b = left["keys"][key_id], right["keys"][key_id]
            # Version lists, active version, revocation markers and
            # per-version statuses line up individually.
            self.assertEqual(a["versions"], b["versions"], key_id)
            self.assertEqual(a["active"], b["active"], key_id)
            self.assertEqual(a["revoked"], b["revoked"], key_id)
            self.assertEqual(a["is_revoked"], b["is_revoked"], key_id)
            self.assertEqual(a["derivations"], b["derivations"], key_id)
            self.assertEqual(set(a["materials"]), set(b["materials"]), key_id)
            for version in a["versions"]:
                ma, mb = a["materials"][version], b["materials"][version]
                self.assertIs(type(ma), bytes, key_id)
                self.assertIs(type(mb), bytes, key_id)
                self.assertEqual(ma, mb, (key_id, version))
        self.assertEqual(left["manifest"], right["manifest"])
        self.assertEqual(left["unknown_versions"], right["unknown_versions"])
        self.assertEqual(left["unknown_revoked"], right["unknown_revoked"])

    def _assert_recovered_one_to_one(
        self, vault: Vault, root: Path, frozen: dict, healthy: dict
    ) -> None:
        """Recovery correspondence: fresh instance == recovered == disk.

        Runs after the surgical repair.  One full reload must succeed, move
        none of the frozen answers, and then a brand-new instance -- both
        straight from its constructor and after an extra reload -- must
        read exactly the recovered state, and both instances must match
        every disk record one to one.
        """
        vault.reload()  # must not raise
        recovered = self._answers(vault)
        # Recovery adds no answer and moves none frozen during the window.
        self.assertEqual(recovered, frozen)

        # The direct point this module exists to pin: a brand-new instance
        # (constructor validation included, no reload call first) reads a
        # state completely identical to the recovered snapshot.
        fresh = self.open_vault(root)
        fresh_answers = self._answers(fresh)
        self.assertEqual(fresh_answers, recovered)
        self._assert_keywise_identical(fresh_answers, recovered)
        self._assert_instance_matches_disk_records(fresh, root)
        self._assert_instance_matches_disk_records(vault, root)

        # The same after the new instance performs its own full reload.
        fresh.reload()
        self.assertEqual(self._answers(fresh), recovered)
        self._assert_keywise_identical(self._answers(fresh), recovered)
        vault.reload()
        self.assertEqual(self._answers(vault), recovered)

        # The repair put the directory back byte-for-byte.
        self.assertEqual(self._disk_records(root), healthy)

    # ------------------------------------------------------------------
    # the unchanged entry vocabulary
    # ------------------------------------------------------------------

    def _entry_contract_signature(self, vault: Vault) -> tuple:
        """Classify every read-entry exception in one fixed order.

        Read queries consult only the held snapshot plus entry validation,
        so the signature must come out identically on a healthy vault and
        while the failure window persists.  Entry-level rejects for the two
        library-only writes (empty id, non-int version) are included too:
        they fire before any disk read.
        """
        V, T, K = ValueError, TypeError, KeyError

        def classify(expected, call):
            try:
                result = call()
            except Exception as exc:  # type is the recorded result
                if expected is None:
                    raise
                self.assertIsInstance(
                    exc,
                    expected,
                    msg=f"{call!r} raised {type(exc).__name__}, expected "
                    f"{expected.__name__}",
                )
                return type(exc).__name__
            if expected is not None:
                self.fail(
                    f"{call!r} did not raise {expected.__name__}"
                )
            return ("OK", result)

        return tuple(
            (
                label,
                classify(
                    expected,
                    lambda call=call: call(),
                ),
            )
            for label, expected, call in (
                ("load empty id", V, lambda: vault.load("", 1)),
                ("load empty id before type", V, lambda: vault.load("", 1.0)),
                ("load float version", T, lambda: vault.load("alpha", 1.0)),
                ("load bool version", T, lambda: vault.load("alpha", True)),
                ("load string version", T, lambda: vault.load("alpha", "1")),
                ("load unknown key", K, lambda: vault.load("ghost")),
                ("load unknown version", K, lambda: vault.load("alpha", 99)),
                (
                    "derivation empty id (active sentinel)",
                    V,
                    lambda: vault.derivation(""),
                ),
                (
                    "derivation float version",
                    T,
                    lambda: vault.derivation("bravo", 2.0),
                ),
                ("derivation unknown key", K, lambda: vault.derivation("ghost")),
                (
                    "derivation unknown version",
                    K,
                    lambda: vault.derivation("bravo", 99),
                ),
                ("active empty id", V, lambda: vault.active("")),
                ("active unknown key", K, lambda: vault.active("ghost")),
                ("is_revoked empty id", V, lambda: vault.is_revoked("", 1)),
                (
                    "is_revoked float version",
                    T,
                    lambda: vault.is_revoked("alpha", 1.0),
                ),
                (
                    "is_revoked unknown key",
                    K,
                    lambda: vault.is_revoked("ghost", 1),
                ),
                (
                    "revoked_versions empty id",
                    V,
                    lambda: vault.revoked_versions(""),
                ),
                ("revoke empty id", V, lambda: vault.revoke("", 1)),
                ("revoke bool version", T, lambda: vault.revoke("alpha", True)),
                ("set_active empty id", V, lambda: vault.set_active("", 1)),
                (
                    "set_active float version",
                    T,
                    lambda: vault.set_active("alpha", 1.0),
                ),
                ("versions empty id", None, lambda: vault.versions("")),
                ("versions unknown key", None, lambda: vault.versions("ghost")),
                (
                    "revoked_versions unknown key",
                    None,
                    lambda: vault.revoked_versions("ghost"),
                ),
            )
        )


# ---------------------------------------------------------------------------
# every damage kind: ValueError, frozen window, then one-to-one recovery
# ---------------------------------------------------------------------------


class TestReloadFailureWindowAndRecovery(RoundtripCorrespondenceTestCase):
    def test_each_damage_freezes_then_recovers_one_to_one(self):
        for index, name in enumerate(self.DAMAGE_KINDS):
            with self.subTest(case=name):
                vault, root = self._build_healthy(f"roundtrip-{index}")
                frozen = self._answers(vault)
                healthy_disk = self._disk_records(root)

                self._apply_damage(root, healthy_disk, name)
                self._assert_failure_window(vault, root, frozen)

                # Remove/correct just the bad record; normal service comes
                # back with the snapshot matching disk one to one.
                self._apply_repair(root, healthy_disk, name)
                self._assert_recovered_one_to_one(
                    vault, root, frozen, healthy_disk
                )

    def test_second_new_opener_after_recovery_agrees_as_well(self):
        # The explicit correspondence point on its own: after one damaged
        # reload is repaired, two independently opened instances and the
        # recovered handle must all read the exact same state.
        vault, root = self._build_healthy("two-openers")
        frozen = self._answers(vault)
        healthy_disk = self._disk_records(root)

        (root / MANIFEST_NAME).write_bytes(b"{not json\n")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(self._answers(vault), frozen)

        (root / MANIFEST_NAME).write_bytes(healthy_disk[MANIFEST_NAME])
        vault.reload()
        recovered = self._answers(vault)

        first = self.open_vault(root)
        second = self.open_vault(root)
        self.assertEqual(self._answers(first), recovered)
        self.assertEqual(self._answers(second), recovered)
        self._assert_keywise_identical(self._answers(first), recovered)
        self._assert_keywise_identical(self._answers(second), recovered)
        self._assert_instance_matches_disk_records(first, root)
        self._assert_instance_matches_disk_records(second, root)
        self.assertEqual(self._disk_records(root), healthy_disk)


# ---------------------------------------------------------------------------
# frozen answers and disk records are verbatim across repeated failures
# ---------------------------------------------------------------------------


class TestFrozenStateIsVerbatim(RoundtripCorrespondenceTestCase):
    def test_answers_and_disk_are_verbatim_across_repeated_failures(self):
        vault, root = self._build_healthy("verbatim")
        frozen = self._answers(vault)
        healthy_disk = self._disk_records(root)

        # A missing material: the most direct "records on disk changed,
        # snapshot did not" shape.
        self._material_file(root, "charlie", 2).unlink()
        damaged_disk = self._disk_records(root)
        self.assertNotEqual(damaged_disk, healthy_disk)

        first_words = None
        for _ in range(5):
            with self.assertRaises(ValueError):
                vault.reload()
            words = self._word_for_word(vault)
            if first_words is None:
                first_words = words
            else:
                self.assertEqual(words, first_words)
            self.assertEqual(self._answers(vault), frozen)

        # The originally written bytes are all still served from the
        # frozen snapshot, damaged key included.
        for (key_id, version), original in ORIGINAL_BYTES.items():
            self.assertEqual(vault.load(key_id, version), original)

        # A new opener fails too and changes nothing.
        with self.assertRaises(ValueError):
            Vault(root)
        self.assertEqual(self._disk_records(root), damaged_disk)

        # Repair: recovery adds only the correspondence point, moving
        # nothing frozen above.
        path = self._material_file(root, "charlie", 2)
        path.write_bytes(CHARLIE_MATERIALS[2])
        self._assert_recovered_one_to_one(vault, root, frozen, healthy_disk)

    def test_per_version_revocation_and_derivation_answers_stay_verbatim(self):
        vault, root = self._build_healthy("per-version-verbatim")
        frozen = self._answers(vault)
        healthy_disk = self._disk_records(root)

        # Corrupt the activity journal with a line missing a field.
        with (root / ACTIVATIONS_NAME).open("ab") as handle:
            handle.write(b'{"key_id": "bravo", "version": 1}\n')

        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()

        # Per-version revocation statuses, version by version.
        for key_id in KEY_IDS:
            for version in vault.versions(key_id):
                self.assertEqual(
                    vault.is_revoked(key_id, version),
                    frozen["keys"][key_id]["is_revoked"][version],
                )
                self.assertEqual(
                    vault.derivation(key_id, version),
                    frozen["keys"][key_id]["derivations"][version],
                )
        self.assertEqual(vault.active("alpha"), 1)
        self.assertEqual(vault.revoked_versions("alpha"), [2])
        self.assertEqual(vault.revoked_versions("charlie"), [1])

        (root / ACTIVATIONS_NAME).write_bytes(healthy_disk[ACTIVATIONS_NAME])
        self._assert_recovered_one_to_one(vault, root, frozen, healthy_disk)


# ---------------------------------------------------------------------------
# ValueError / TypeError / KeyError vocabulary is unchanged during failure
# ---------------------------------------------------------------------------


class TestEntryContractUnchangedDuringFailure(RoundtripCorrespondenceTestCase):
    def test_entry_exceptions_are_identical_healthy_and_frozen(self):
        vault, root = self._build_healthy("entry-contract")
        frozen = self._answers(vault)
        healthy_disk = self._disk_records(root)

        healthy_signature = self._entry_contract_signature(vault)

        # Damage the persisted derivation record out of band.
        self._edit_manifest(
            root,
            lambda m: self._manifest_record(m, "bravo", 1)[
                "derivation"
            ].__setitem__("length", 7),
        )
        with self.assertRaises(ValueError) as caught:
            vault.reload()
        message = str(caught.exception)

        # The same fixed classification while the failure persists.
        frozen_signature = self._entry_contract_signature(vault)
        self.assertEqual(frozen_signature, healthy_signature)
        self.assertEqual(self._answers(vault), frozen)

        # Representative direct assertions of the three vocabularies.
        with self.assertRaises(ValueError):
            vault.load("", 1)
        with self.assertRaises(TypeError):
            vault.load("alpha", 1.0)
        with self.assertRaises(KeyError):
            vault.load("ghost")
        with self.assertRaises(KeyError):
            vault.load("alpha", 99)

        # Triggering reload again ends with the same ValueError, and none of
        # the rejected entry calls wrote anything.
        with self.assertRaises(ValueError) as again:
            vault.reload()
        self.assertEqual(str(again.exception), message)

        # Correct the record and recover: same frozen answers, one-to-one
        # with disk via a brand-new instance.
        self._edit_manifest(
            root,
            lambda m: self._manifest_record(m, "bravo", 1)[
                "derivation"
            ].__setitem__("length", 32),
        )
        self._assert_recovered_one_to_one(vault, root, frozen, healthy_disk)

        # A healthy vault after recovery has the very same vocabulary.
        self.assertEqual(
            self._entry_contract_signature(vault), healthy_signature
        )


# ---------------------------------------------------------------------------
# repeatability: identical input twice, and two independent vaults
# ---------------------------------------------------------------------------


class TestRoundtripDeterminism(RoundtripCorrespondenceTestCase):
    def test_same_damage_twice_gives_same_error_and_same_recovery(self):
        for index, name in enumerate(self.DAMAGE_KINDS):
            with self.subTest(case=name):
                vault, root = self._build_healthy(f"repeat-{index}")
                frozen = self._answers(vault)
                healthy_disk = self._disk_records(root)

                messages = []
                for _ in range(2):
                    self._apply_damage(root, healthy_disk, name)
                    with self.assertRaises(ValueError) as caught:
                        vault.reload()
                    messages.append(str(caught.exception))
                    self.assertEqual(self._answers(vault), frozen)
                    self._apply_repair(root, healthy_disk, name)
                    vault.reload()
                    self.assertEqual(self._answers(vault), frozen)
                    self._assert_instance_matches_disk_records(vault, root)

                self.assertEqual(messages[0], messages[1])
                fresh = self.open_vault(root)
                self.assertEqual(self._answers(fresh), frozen)
                self._assert_keywise_identical(self._answers(fresh), frozen)
                self.assertEqual(self._disk_records(root), healthy_disk)

    def test_same_damage_in_two_vaults_gives_same_everything(self):
        outcomes = []
        for name in ("det-a", "det-b"):
            vault, root = self._build_healthy(name)
            frozen = self._answers(vault)
            healthy_disk = self._disk_records(root)

            # Corrupt manifest bytes.
            (root / MANIFEST_NAME).write_bytes(b"{not json\n")
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            message = str(caught.exception)
            self.assertEqual(self._answers(vault), frozen)

            # Repair and recover.
            (root / MANIFEST_NAME).write_bytes(healthy_disk[MANIFEST_NAME])
            vault.reload()
            recovered = self._answers(vault)
            self._assert_instance_matches_disk_records(vault, root)
            fresh = self.open_vault(root)
            fresh_answers = self._answers(fresh)
            self._assert_keywise_identical(fresh_answers, recovered)
            outcomes.append(
                (message, frozen, recovered, fresh_answers, self._disk_records(root))
            )

        first, second = outcomes
        self.assertEqual(first[0], second[0])  # same complaint
        self.assertEqual(first[1], second[1])  # same frozen state
        self.assertEqual(first[2], second[2])  # same recovered state
        self.assertEqual(first[3], second[3])  # same brand-new-instance state


# ---------------------------------------------------------------------------
# tail hygiene lives next door
# ---------------------------------------------------------------------------
#
# The "no unclosed-handle ResourceWarning in the tail" check spawns a fresh
# interpreter over this module and so lives in its own file
# (test_reload_roundtrip_resources.py): kept here it would be discovered by
# that very subprocess and would spawn itself again.  The child runs this
# module by name, not by discovery, so there is exactly one level.


if __name__ == "__main__":
    unittest.main()
