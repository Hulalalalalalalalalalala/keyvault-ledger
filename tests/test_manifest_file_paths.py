"""Regression tests for the cross-platform spelling of manifest file paths.

Every manifest record points at its material file with a path relative to
the vault root, written with forward slashes as the only separator -- the
same spelling on every platform, even ones whose native separator is the
backslash.  The record carries no drive letter, no absolute path and no
directory outside the vault, so the same record resolves to the same
material file wherever the vault directory is opened, and the bytes read
back are byte-for-byte the sealed ones.

The baseline vault capabilities (version numbers, the active pointer,
revocation markers, derivation records and their exception vocabulary) are
already implemented and are intentionally not touched here.  These cases
only freeze the path-spelling contract:

* every record the vault writes -- for plain and derived seals alike, for
  ASCII and non-ASCII key ids -- uses the single canonical form: relative,
  slash-separated, free of backslashes, drive letters, absolute roots and
  ``.``/``..``/empty segments, and the raw manifest bytes carry no
  backslash at all;

* following the recorded path from the vault root always lands on the
  material file, and the bytes served equal the file byte for byte;
  ``reload()`` and a full reopen parse every record to the same material
  file with the same result as before the reload;

* a hand-edited record whose path spelling is illegal -- backslash
  separators, a drive letter, an absolute path (even one pointing inside
  the vault), a leading slash, ``.``/``..``/empty segments, a trailing
  slash, a non-string value -- or whose canonical path names a file that
  does not exist makes the whole ``reload()`` raise exactly ``ValueError``
  on every platform; while the failure persists the in-memory snapshot is
  frozen, the keys in hand stay readable and the disk records neither gain
  nor lose a byte, and a cold opener raises the same ``ValueError``;

* correcting the bad record and reloading again restores the vault, with
  the snapshot matching the disk records one to one;

* a vault root whose path contains spaces or non-ASCII characters seals,
  reads, reloads and reopens exactly like an ordinary root -- the persisted
  records are byte-for-byte identical to the ones an ordinary root gets
  for the same operations.

Everything happens inside per-case temporary directories, uses the
standard library only and is independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path, PurePosixPath

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    ACTIVATIONS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    MATERIALS_DIR,
    REVOCATIONS_NAME,
    _dump_manifest,
    _encode_key_id,
)
from tests._fixtures import VaultFixture

# Exact inputs of the shared healthy vault: two plain versions on "alpha"
# (repointed at 1, version 2 revoked), two derived versions on "bravo",
# and one plain version on a key id with a space and non-ASCII characters
# (its encoded on-disk form must still be slash-free).
ALPHA_MATERIALS = {1: b"alpha-one", 2: b"alpha-two"}
BRAVO_PARAMETERS = {
    1: (b"bravo-passphrase-1", b"bravo-salt-1", 1000, 32),
    2: (b"bravo-passphrase-2", b"bravo-salt-2", 2000, 24),
}
UNICODE_KEY_ID = "金 庫"
UNICODE_MATERIAL = b"unicode-key-material"

KEY_IDS = ("alpha", "bravo", UNICODE_KEY_ID)


def _derived_material(version: int) -> bytes:
    password, salt, iterations, length = BRAVO_PARAMETERS[version]
    return hashlib.pbkdf2_hmac(
        "sha256", password, salt, iterations, dklen=length
    )


class ManifestFilePathTestCase(unittest.TestCase):
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

    def _build_healthy(self, root: Path) -> Vault:
        """Build the shared healthy vault at ``root`` deterministically."""
        vault = self._open(root)
        vault.seal("alpha", ALPHA_MATERIALS[1])
        vault.seal("alpha", ALPHA_MATERIALS[2])
        vault.set_active("alpha", 1)  # bound to latest=2
        vault.revoke("alpha", 2)
        for version in (1, 2):
            password, salt, iterations, length = BRAVO_PARAMETERS[version]
            vault.derive_seal("bravo", password, salt, iterations, length)
        vault.seal(UNICODE_KEY_ID, UNICODE_MATERIAL)
        return vault

    # ------------------------------------------------------------------
    # observable-state and disk captures
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
                "materials": {v: vault.load(key_id, v) for v in versions},
                "derivations": {
                    v: vault.derivation(key_id, v) for v in versions
                },
            }
        return {"keys": keys, "manifest": vault.manifest()}

    def _disk_bytes(self, root: Path) -> dict[str, bytes]:
        """All vault records on disk keyed by relative path.

        ``vault.lock`` is only a mutual-exclusion device and carries no
        key data, so it is deliberately excluded.
        """
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }

    def _manifest(self, root: Path) -> dict:
        return json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))

    def _edit_manifest(self, root: Path, mutate) -> None:
        path = root / MANIFEST_NAME
        manifest = json.loads(path.read_bytes().decode("utf-8"))
        mutate(manifest)
        # Serialize exactly like the product does, so the only difference
        # a repair introduces is the corrected record itself.
        path.write_bytes(_dump_manifest(manifest))

    # ------------------------------------------------------------------
    # the canonical-form and resolution assertions
    # ------------------------------------------------------------------

    def _assert_canonical_file_fields(self, vault: Vault, root: Path) -> None:
        """Every record uses the one portable spelling and resolves."""
        raw = (root / MANIFEST_NAME).read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        for key_id, entry in manifest["keys"].items():
            for record in entry["versions"]:
                rel = record["file"]
                with self.subTest(key=key_id, version=record["version"]):
                    self.assertIsInstance(rel, str)
                    # Relative, slash-separated, nothing platform-specific.
                    self.assertNotIn("\\", rel)
                    self.assertNotIn(":", rel)
                    self.assertFalse(PurePosixPath(rel).is_absolute())
                    segments = rel.split("/")
                    self.assertTrue(
                        all(
                            segment not in ("", ".", "..")
                            for segment in segments
                        ),
                        rel,
                    )
                    # The shape the vault itself writes.
                    self.assertEqual(segments[0], MATERIALS_DIR)
                    self.assertEqual(
                        segments[1], _encode_key_id(key_id)
                    )
                    self.assertEqual(
                        segments[2], f"{record['version']}.bin"
                    )
                    self.assertEqual(len(segments), 3)
                    # Following the path from the vault root lands on the
                    # material file, inside the vault, and the bytes served
                    # for this version are the file's bytes exactly.
                    resolved = (root / rel).resolve()
                    self.assertTrue(
                        resolved.is_relative_to(root.resolve())
                    )
                    self.assertTrue(resolved.is_file())
                    data = resolved.read_bytes()
                    self.assertEqual(
                        vault.load(key_id, record["version"]), data
                    )
                    self.assertEqual(
                        hashlib.sha256(data).hexdigest(), record["sha256"]
                    )

    def _assert_snapshot_corresponds_to_disk(
        self, vault: Vault, root: Path
    ) -> None:
        """Re-derive every observable answer straight from the records."""
        manifest = self._manifest(root)
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

    # ------------------------------------------------------------------
    # the failure window and the recovery afterwards
    # ------------------------------------------------------------------

    def _assert_failure_then_recovery(
        self, vault: Vault, root: Path, expected: dict, corrupt
    ) -> None:
        """Drive one full corruption window and the recovery afterwards."""
        healthy_disk = self._disk_bytes(root)

        corrupt()
        failing_disk = self._disk_bytes(root)

        # Three failed reloads: the complaint is deterministic and every
        # observable answer stays frozen, keys in hand included.
        message = None
        for _ in range(3):
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            if message is None:
                message = str(caught.exception)
            else:
                self.assertEqual(str(caught.exception), message)
            self.assertEqual(self._answers(vault), expected)

        # A cold opener rejects the very same state with the same
        # complaint.
        with self.assertRaises(ValueError) as caught:
            Vault(root)
        self.assertEqual(str(caught.exception), message)

        # The failed reloads/open neither added nor removed a disk record.
        self.assertEqual(self._disk_bytes(root), failing_disk)

        # Correct the bad record: one successful reload restores the full
        # snapshot<->disk correspondence, for the live handle and for a
        # fresh opener alike.
        self._restore_manifest(root, healthy_disk)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self._assert_snapshot_corresponds_to_disk(vault, root)
        reopened = self._open(root)
        self.assertEqual(self._answers(reopened), expected)
        self._assert_snapshot_corresponds_to_disk(reopened, root)
        self.assertEqual(self._disk_bytes(root), healthy_disk)

        # Repeating the identical failing input gives the identical
        # result, then recovery works a second time as well.
        corrupt()
        with self.assertRaises(ValueError) as caught:
            vault.reload()
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(self._answers(vault), expected)
        self._restore_manifest(root, healthy_disk)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self._assert_snapshot_corresponds_to_disk(vault, root)

    def _restore_manifest(self, root: Path, healthy: dict[str, bytes]) -> None:
        """Put the healthy manifest bytes back verbatim."""
        (root / MANIFEST_NAME).write_bytes(healthy[MANIFEST_NAME])


# ---------------------------------------------------------------------------
# the written form: one canonical spelling on every platform
# ---------------------------------------------------------------------------


class TestWrittenFileFieldForm(ManifestFilePathTestCase):
    def test_written_records_use_the_canonical_portable_spelling(self):
        root = self.root / "canonical"
        vault = self._build_healthy(root)
        self._assert_canonical_file_fields(vault, root)

    def test_raw_manifest_bytes_carry_no_backslash(self):
        # With ASCII-only key ids nothing in the persisted manifest needs
        # a JSON escape, so a backslash can never appear in the raw bytes
        # at all -- the file fields are slash-separated even on platforms
        # whose native separator is the backslash.
        root = self.root / "raw-bytes"
        vault = self._open(root)
        vault.seal("alpha", ALPHA_MATERIALS[1])
        vault.seal("alpha", ALPHA_MATERIALS[2])
        password, salt, iterations, length = BRAVO_PARAMETERS[1]
        vault.derive_seal("bravo", password, salt, iterations, length)
        raw = (root / MANIFEST_NAME).read_bytes()
        self.assertNotIn(b"\\", raw)
        vault.reload()
        self.assertNotIn(b"\\", (root / MANIFEST_NAME).read_bytes())

    def test_reload_and_reopen_parse_records_to_the_same_materials(self):
        root = self.root / "reparse"
        vault = self._build_healthy(root)
        before = self._answers(vault)
        # Per-record resolution before the reload: record path -> bytes.
        resolution_before = {
            record["file"]: (root / record["file"]).read_bytes()
            for entry in self._manifest(root)["keys"].values()
            for record in entry["versions"]
        }

        vault.reload()
        self.assertEqual(self._answers(vault), before)
        self._assert_canonical_file_fields(vault, root)
        self._assert_snapshot_corresponds_to_disk(vault, root)

        # A full reopen parses every record to the same material file and
        # the same bytes as before the reload.
        reopened = self._open(root)
        self.assertEqual(self._answers(reopened), before)
        resolution_after = {
            record["file"]: (root / record["file"]).read_bytes()
            for entry in self._manifest(root)["keys"].values()
            for record in entry["versions"]
        }
        self.assertEqual(resolution_after, resolution_before)
        reopened.reload()
        self.assertEqual(self._answers(reopened), before)

    def test_repeated_identical_inputs_persist_identical_records(self):
        # The same deterministic operations in two independent vault
        # directories persist byte-for-byte identical manifests.
        outcomes = []
        for name in ("repeat-a", "repeat-b"):
            root = self.root / name
            vault = self._build_healthy(root)
            outcomes.append(
                (
                    (root / MANIFEST_NAME).read_bytes(),
                    self._answers(vault),
                )
            )
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        self.assertEqual(outcomes[0][1], outcomes[1][1])


# ---------------------------------------------------------------------------
# illegal path spellings and dangling pointers are rejected on reload
# ---------------------------------------------------------------------------


class TestIllegalFileFieldRejected(ManifestFilePathTestCase):
    def _corrupt_file_field(self, root: Path, value) -> None:
        """Point the "alpha" version-1 record at ``value`` out of band."""

        def mutate(manifest):
            record = next(
                record
                for record in manifest["keys"]["alpha"]["versions"]
                if record["version"] == 1
            )
            record["file"] = value

        self._edit_manifest(root, mutate)

    def test_illegal_path_spellings_are_rejected_then_recover(self):
        good_rel = (
            f"{MATERIALS_DIR}/{_encode_key_id('alpha')}/1.bin"
        )
        # Each case computes the illegal value from the vault root, so the
        # absolute-path cases genuinely point inside the vault -- even
        # those are rejected, because the record must stay relative.
        cases = {
            "empty_string": lambda root: "",
            "backslash_separators": lambda root: good_rel.replace("/", "\\"),
            "drive_letter_slashes": lambda root: f"C:/{good_rel}",
            "drive_letter_backslashes": (
                lambda root: "C:\\" + good_rel.replace("/", "\\")
            ),
            "posix_absolute_inside_vault": (
                lambda root: str((root / good_rel).resolve())
            ),
            "leading_slash": lambda root: f"/{good_rel}",
            "parent_escape_outside_vault": lambda root: f"../{good_rel}",
            "parent_segment_resolving_inside": (
                lambda root: f"{MATERIALS_DIR}/x/../{good_rel.split('/', 1)[1]}"
            ),
            "dot_segment": (
                lambda root: f"{MATERIALS_DIR}/./{good_rel.split('/', 1)[1]}"
            ),
            "empty_segment": (
                lambda root: f"{MATERIALS_DIR}//{good_rel.split('/', 1)[1]}"
            ),
            "trailing_slash": lambda root: f"{good_rel}/",
            "non_string_number": lambda root: 123,
            "non_string_null": lambda root: None,
            "non_string_list": lambda root: [good_rel],
        }
        for index, (name, make_bad) in enumerate(cases.items()):
            with self.subTest(case=name):
                root = self.root / f"illegal-{index}"
                vault = self._build_healthy(root)
                expected = self._answers(vault)
                self._assert_failure_then_recovery(
                    vault,
                    root,
                    expected,
                    lambda: self._corrupt_file_field(root, make_bad(root)),
                )

    def test_canonical_path_to_a_missing_file_is_rejected(self):
        # The spelling is legal but names no file inside the vault: the
        # whole reload still fails and recovers once the record is fixed.
        root = self.root / "dangling"
        vault = self._build_healthy(root)
        expected = self._answers(vault)
        dangling = f"{MATERIALS_DIR}/{_encode_key_id('alpha')}/99.bin"
        self.assertFalse((root / dangling).exists())
        self._assert_failure_then_recovery(
            vault, root, expected,
            lambda: self._corrupt_file_field(root, dangling),
        )

    def test_backslash_spelling_rejected_even_with_matching_bytes(self):
        # Plant a file whose on-disk name literally matches the backslash
        # spelling where the platform allows it: the record is still
        # rejected, because the backslash form itself is illegal -- the
        # same answer a backslash-native platform gives.
        root = self.root / "backslash-with-file"
        vault = self._build_healthy(root)
        expected = self._answers(vault)
        good_rel = f"{MATERIALS_DIR}/{_encode_key_id('alpha')}/1.bin"
        bad_rel = good_rel.replace("/", "\\")
        # Plant the stray file only where the backslash spelling names a
        # file distinct from the real material (on a backslash-native
        # platform the two spellings resolve to the same file, and the
        # record is rejected for its form alone).
        planted_path = root / bad_rel
        planted = False
        if planted_path.resolve() != (root / good_rel).resolve():
            try:
                planted_path.write_bytes(ALPHA_MATERIALS[1])
                planted = True
            except OSError:
                # A platform where a backslash cannot appear in a file
                # name: the record is rejected all the same.
                pass
        self._corrupt_file_field(root, bad_rel)
        failing_disk = self._disk_bytes(root)
        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
            self.assertEqual(self._answers(vault), expected)
        self.assertEqual(self._disk_bytes(root), failing_disk)
        # Remove the planted stray file again and correct the record, so
        # the recovery comparison sees exactly the healthy tree.
        if planted:
            planted_path.unlink()
        self._corrupt_file_field(root, good_rel)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self._assert_snapshot_corresponds_to_disk(vault, root)


# ---------------------------------------------------------------------------
# vault roots containing spaces or non-ASCII characters
# ---------------------------------------------------------------------------


class TestVaultRootWithSpacesAndNonAscii(ManifestFilePathTestCase):
    def test_unusual_root_behaves_like_an_ordinary_root(self):
        plain_root = self.root / "ordinary"
        fancy_root = self.root / "vault 金庫 目录 root"

        plain = self._build_healthy(plain_root)
        fancy = self._build_healthy(fancy_root)

        # The persisted records carry no trace of the root path: the
        # manifests are byte-for-byte identical, and every observable
        # answer matches.
        self.assertEqual(
            (plain_root / MANIFEST_NAME).read_bytes(),
            (fancy_root / MANIFEST_NAME).read_bytes(),
        )
        self.assertEqual(self._answers(fancy), self._answers(plain))
        self._assert_canonical_file_fields(fancy, fancy_root)

        # Reload and reopen on the unusual root parse every record to the
        # same material file, with the answers unchanged.
        before = self._answers(fancy)
        fancy.reload()
        self.assertEqual(self._answers(fancy), before)
        self._assert_snapshot_corresponds_to_disk(fancy, fancy_root)
        reopened = self._open(fancy_root)
        self.assertEqual(self._answers(reopened), before)
        self._assert_snapshot_corresponds_to_disk(reopened, fancy_root)

        # Sealing, revoking and repointing keep working on the unusual
        # root, and the new records keep the canonical spelling.
        fancy.seal("alpha", b"alpha-three")
        fancy.set_active("alpha", 1)
        fancy.revoke("alpha", 1)
        fancy.reload()
        self.assertEqual(fancy.load("alpha", 3), b"alpha-three")
        self.assertEqual(fancy.active("alpha"), 1)
        self.assertTrue(fancy.is_revoked("alpha", 1))
        self._assert_canonical_file_fields(fancy, fancy_root)
        self._assert_snapshot_corresponds_to_disk(fancy, fancy_root)

    def test_unusual_root_results_are_repeatable(self):
        outcomes = []
        for name in ("金庫 a", "金庫 b"):
            root = self.root / name
            vault = self._build_healthy(root)
            vault.reload()
            outcomes.append(
                (
                    (root / MANIFEST_NAME).read_bytes(),
                    self._answers(vault),
                )
            )
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        self.assertEqual(outcomes[0][1], outcomes[1][1])


if __name__ == "__main__":
    unittest.main()
