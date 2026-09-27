"""Regression tests for the unified cross-platform material path writing.

Every manifest record points at its material file with one and the same
path writing on every platform: a path relative to the vault root, joined
with forward slashes -- never a backslash, never a drive letter or UNC
share, never an absolute path, never a ``.``/``..`` or empty segment, and
never a directory name from outside the vault.  The baseline vault
capabilities (version numbers, active pointer, revocation markers,
derivation records and their exception vocabulary, the three CLI
subcommands) are already implemented and are intentionally not touched
here.  These cases only freeze the path convention and its enforcement:

* every record a vault writes -- plain or derived seal, ordinary or
  unusual key id, ordinary or space/non-ASCII vault root -- lands in the
  manifest in exactly that one form, parses to the same parts under
  POSIX and Windows path rules, and resolves under the vault root to the
  exact bytes that were sealed;

* a full ``reload()`` and a cold reopen parse every record back to the
  same material file, byte for byte, and the persisted manifest is
  bit-identical before and after;

* a hand-edited record whose ``file`` breaks the convention (backslash
  separators, a drive letter, a UNC share, an absolute path -- even one
  pointing inside the vault -- a ``..`` escape, a ``.``/empty segment, a
  trailing slash, an empty or non-string value) or that points at a file
  that does not exist makes the whole reload raise exactly ``ValueError``;
  while the failure persists the in-memory snapshot is frozen, the disk
  records neither gain nor lose a byte, and a cold opener raises the same
  ``ValueError``; correcting the record and reloading again restores the
  vault, with the snapshot matching the disk records one to one;

* repeating the identical build input produces byte-identical manifests,
  independently of the vault root's spelling and of execution order.

Everything happens inside per-case temporary directories, uses the
standard library only and is independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath

from keyvault_ledger import Vault
from keyvault_ledger.vault import (
    LOCK_NAME,
    MANIFEST_NAME,
    MATERIALS_DIR,
    _dump_manifest,
    _encode_key_id,
)
from tests._fixtures import VaultFixture

# Key ids exercising every encoding branch of the material path: plain
# ASCII, a space, non-ASCII, a slash, a dot-only id and an id long enough
# to switch the encoding to its hashed form.
KEY_IDS = ("plain", "sp ace", "ünïcode-键", "a/b", "..", "x" * 200)

PLAIN_MATERIAL = b"plain-material-\x00\xff"
DERIVED = (b"path-convention-passphrase", b"path-convention-salt", 1500, 40)


class _ManifestPathSupport:
    """Shared vault construction and assertions (not a TestCase)."""

    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        # Each scenario gets its own vault subdirectory, so cases never
        # share disk state and execution order cannot matter.
        self.root = self.fixture.tmp_path / "vaults"

    def _open(self, root: Path) -> Vault:
        """Open a vault through the case's single teardown path."""
        return self.fixture.open(root)

    def _build_healthy(self, root: Path) -> Vault:
        """Seal plain and derived versions of every key id at ``root``."""
        vault = self._open(root)
        for key_id in KEY_IDS:
            vault.seal(key_id, PLAIN_MATERIAL + key_id.encode("utf-8"))
            password, salt, iterations, length = DERIVED
            vault.derive_seal(
                key_id,
                password + key_id.encode("utf-8"),
                salt,
                iterations,
                length,
            )
        return vault

    @staticmethod
    def _disk_manifest(root: Path) -> dict:
        return json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))

    def _records(self, root: Path) -> dict[tuple[str, int], dict]:
        """Every manifest version record keyed by ``(key_id, version)``."""
        manifest = self._disk_manifest(root)
        return {
            (key_id, record["version"]): record
            for key_id, entry in manifest["keys"].items()
            for record in entry["versions"]
        }

    def _answers(self, vault: Vault) -> dict:
        """Every state a reader can query, in a comparable structure."""
        keys = {}
        for key_id in KEY_IDS:
            versions = vault.versions(key_id)
            keys[key_id] = {
                "versions": versions,
                "active": vault.active(key_id),
                "materials": {v: vault.load(key_id, v) for v in versions},
                "derivations": {v: vault.derivation(key_id, v) for v in versions},
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

    def _assert_unified_writing(
        self, root: Path, key_id: str, version: int, record: dict
    ) -> None:
        """One record's ``file`` follows the one cross-platform writing."""
        rel = record["file"]
        # Exactly the form the vault writes: root-relative, slash-joined.
        self.assertIsInstance(rel, str)
        self.assertEqual(
            rel,
            f"{MATERIALS_DIR}/{_encode_key_id(key_id)}/{version}.bin",
        )
        # No backslash, no drive letter, no absolute spelling -- the path
        # parses to the very same parts under POSIX and Windows rules.
        self.assertNotIn("\\", rel)
        self.assertFalse(PurePosixPath(rel).is_absolute())
        self.assertEqual(PureWindowsPath(rel).drive, "")
        self.assertFalse(PureWindowsPath(rel).is_absolute())
        self.assertEqual(
            PureWindowsPath(rel).parts, PurePosixPath(rel).parts
        )
        segments = rel.split("/")
        self.assertTrue(segments)
        self.assertNotIn("", segments)
        self.assertNotIn(".", segments)
        self.assertNotIn("..", segments)
        # Nothing from outside the vault leaks into the record: no root
        # prefix, no absolute spelling of the vault directory.
        self.assertNotIn(str(root), rel)
        self.assertFalse(rel.startswith("/"))
        # Following the record from the vault root lands on the material
        # file itself, inside the vault.
        material_path = root / rel
        self.assertTrue(material_path.is_file())
        self.assertTrue(
            material_path.resolve().is_relative_to(root.resolve())
        )


class TestUnifiedPathWriting(_ManifestPathSupport, unittest.TestCase):
    def test_every_record_is_written_root_relative_with_forward_slashes(self):
        root = self.root / "unified"
        vault = self._build_healthy(root)
        records = self._records(root)
        # Two versions (one plain, one derived) per key id.
        self.assertEqual(len(records), 2 * len(KEY_IDS))
        for (key_id, version), record in records.items():
            with self.subTest(key_id=key_id, version=version):
                self._assert_unified_writing(root, key_id, version, record)
                # The bytes found by following the record are exactly the
                # bytes the vault serves for that version.
                self.assertEqual(
                    (root / record["file"]).read_bytes(),
                    vault.load(key_id, version),
                )
                self.assertEqual(
                    hashlib.sha256(
                        (root / record["file"]).read_bytes()
                    ).hexdigest(),
                    record["sha256"],
                )

    def test_raw_manifest_text_carries_no_platform_separator(self):
        root = self.root / "raw-text"
        self._build_healthy(root)
        text = (root / MANIFEST_NAME).read_bytes().decode("utf-8")
        # The persisted JSON never spells a material path with a
        # backslash or with the vault root baked in.  (Key ids may carry
        # JSON \uXXXX escapes; the convention governs the file fields.)
        file_lines = [
            line for line in text.splitlines() if '"file":' in line
        ]
        self.assertEqual(len(file_lines), 2 * len(KEY_IDS))
        for line in file_lines:
            self.assertNotIn("\\", line)
            self.assertNotIn(str(root), line)
        self.assertNotIn(str(root), text)
        for key_id in KEY_IDS:
            encoded = _encode_key_id(key_id)
            self.assertIn(f'"file": "{MATERIALS_DIR}/{encoded}/1.bin"', text)
            self.assertIn(f'"file": "{MATERIALS_DIR}/{encoded}/2.bin"', text)

    def test_reload_and_reopen_parse_records_to_the_same_materials(self):
        root = self.root / "roundtrip"
        vault = self._build_healthy(root)
        before = self._answers(vault)
        manifest_bytes = (root / MANIFEST_NAME).read_bytes()
        # Map every record to the bytes it points at, straight from disk.
        pointed = {
            sel: (root / record["file"]).read_bytes()
            for sel, record in self._records(root).items()
        }

        vault.reload()
        self.assertEqual(self._answers(vault), before)
        # Reloading is read-only: the manifest is byte-for-byte untouched.
        self.assertEqual((root / MANIFEST_NAME).read_bytes(), manifest_bytes)

        reopened = self._open(root)
        self.assertEqual(self._answers(reopened), before)
        reopened.reload()
        self.assertEqual(self._answers(reopened), before)
        # Every record still parses to the same file with the same bytes.
        for sel, record in self._records(root).items():
            key_id, version = sel
            self.assertEqual((root / record["file"]).read_bytes(), pointed[sel])
            self.assertEqual(reopened.load(key_id, version), pointed[sel])
        self.assertEqual((root / MANIFEST_NAME).read_bytes(), manifest_bytes)

    def test_identical_inputs_produce_byte_identical_manifests(self):
        first_root = self.root / "determinism-a"
        second_root = self.root / "determinism-b"
        self._build_healthy(first_root)
        self._build_healthy(second_root)
        self.assertEqual(
            (first_root / MANIFEST_NAME).read_bytes(),
            (second_root / MANIFEST_NAME).read_bytes(),
        )

    def test_root_with_spaces_and_non_ascii_behaves_identically(self):
        plain_root = self.root / "ordinary"
        fancy_root = self.root / "vau lt ünïcode 空格 root"
        plain_vault = self._build_healthy(plain_root)
        fancy_vault = self._build_healthy(fancy_root)

        # Sealing, reading and reloading under the decorated root produce
        # exactly the results the ordinary root produced -- including the
        # persisted manifest bytes, since records never name the root.
        self.assertEqual(
            self._answers(fancy_vault), self._answers(plain_vault)
        )
        self.assertEqual(
            (fancy_root / MANIFEST_NAME).read_bytes(),
            (plain_root / MANIFEST_NAME).read_bytes(),
        )
        fancy_vault.reload()
        self.assertEqual(
            self._answers(fancy_vault), self._answers(plain_vault)
        )
        reopened = self._open(fancy_root)
        self.assertEqual(self._answers(reopened), self._answers(plain_vault))
        for (key_id, version), record in self._records(fancy_root).items():
            with self.subTest(key_id=key_id, version=version):
                self._assert_unified_writing(
                    fancy_root, key_id, version, record
                )
                self.assertEqual(
                    (fancy_root / record["file"]).read_bytes(),
                    reopened.load(key_id, version),
                )


class TestIllegalPathWritingRejected(_ManifestPathSupport, unittest.TestCase):
    """A record whose ``file`` breaks the convention fails the reload."""

    def _edit_manifest(self, root: Path, mutate) -> None:
        path = root / MANIFEST_NAME
        manifest = json.loads(path.read_bytes().decode("utf-8"))
        mutate(manifest)
        path.write_bytes(_dump_manifest(manifest))

    def _set_file(self, root: Path, key_id: str, version: int, value) -> None:
        def mutate(manifest):
            record = next(
                record
                for record in manifest["keys"][key_id]["versions"]
                if record["version"] == version
            )
            record["file"] = value

        self._edit_manifest(root, mutate)

    def _assert_rejected_then_recovers(self, name: str, make_bad) -> None:
        root = self.root / name
        vault = self._build_healthy(root)
        expected = self._answers(vault)
        healthy_disk = self._disk_bytes(root)

        good_rel = self._records(root)[("plain", 1)]["file"]
        bad = make_bad(good_rel, root)
        self._set_file(root, "plain", 1, bad)
        failing_disk = self._disk_bytes(root)

        # The failure is deterministic: same ValueError on every reload,
        # and the in-memory snapshot stays frozen throughout.
        message = None
        for _ in range(3):
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            if message is None:
                message = str(caught.exception)
            else:
                self.assertEqual(str(caught.exception), message)
            self.assertEqual(self._answers(vault), expected)

        # A cold opener rejects the very same state identically.
        with self.assertRaises(ValueError) as caught:
            Vault(root)
        self.assertEqual(str(caught.exception), message)

        # The failed reloads neither added nor removed a disk record.
        self.assertEqual(self._disk_bytes(root), failing_disk)

        # Correcting the record restores the vault completely: one full
        # reload and the snapshot matches the disk records one to one.
        self._set_file(root, "plain", 1, good_rel)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self.assertEqual(self._disk_bytes(root), healthy_disk)
        reopened = self._open(root)
        self.assertEqual(self._answers(reopened), expected)
        for sel, record in self._records(root).items():
            key_id, version = sel
            self.assertEqual(
                (root / record["file"]).read_bytes(),
                reopened.load(key_id, version),
            )

    def test_backslash_separators_are_rejected(self):
        self._assert_rejected_then_recovers(
            "backslash", lambda good, root: good.replace("/", "\\")
        )

    def test_absolute_path_pointing_inside_vault_is_rejected(self):
        # Even though the file exists and the digest would match, an
        # absolute spelling is not the unified writing.
        self._assert_rejected_then_recovers(
            "absolute-inside", lambda good, root: str(root / good)
        )

    def test_absolute_posix_path_is_rejected(self):
        self._assert_rejected_then_recovers(
            "absolute-posix", lambda good, root: "/" + good
        )

    def test_drive_letter_spellings_are_rejected(self):
        self._assert_rejected_then_recovers(
            "drive-slash", lambda good, root: "C:/" + good
        )
        self._assert_rejected_then_recovers(
            "drive-backslash",
            lambda good, root: "C:\\" + good.replace("/", "\\"),
        )
        self._assert_rejected_then_recovers(
            "drive-relative", lambda good, root: "C:" + good
        )

    def test_unc_share_is_rejected(self):
        self._assert_rejected_then_recovers(
            "unc-share", lambda good, root: "//server/share/" + good
        )

    def test_parent_segment_escaping_vault_is_rejected(self):
        self._assert_rejected_then_recovers(
            "parent-escape", lambda good, root: "../" + good
        )

    def test_parent_segment_staying_inside_is_rejected(self):
        self._assert_rejected_then_recovers(
            "parent-inside",
            lambda good, root: f"{MATERIALS_DIR}/../" + good,
        )

    def test_dot_and_empty_segments_are_rejected(self):
        self._assert_rejected_then_recovers(
            "dot-segment",
            lambda good, root: f"{MATERIALS_DIR}/./"
            + good[len(MATERIALS_DIR) + 1:],
        )
        self._assert_rejected_then_recovers(
            "empty-segment",
            lambda good, root: good.replace("/", "//", 1),
        )
        self._assert_rejected_then_recovers(
            "trailing-slash", lambda good, root: good + "/"
        )

    def test_empty_and_non_string_file_are_rejected(self):
        self._assert_rejected_then_recovers("empty", lambda good, root: "")
        self._assert_rejected_then_recovers("null", lambda good, root: None)
        self._assert_rejected_then_recovers("number", lambda good, root: 7)
        self._assert_rejected_then_recovers(
            "list", lambda good, root: [good]
        )

    def test_record_pointing_at_nonexistent_file_is_rejected(self):
        self._assert_rejected_then_recovers(
            "nonexistent",
            lambda good, root: f"{MATERIALS_DIR}/ghost/9.bin",
        )

    def test_outside_directory_name_is_rejected(self):
        # A record naming a sibling directory of the vault root mixes an
        # outside directory name into the vault's records.
        self._assert_rejected_then_recovers(
            "outside-directory",
            lambda good, root: "../other-vault/" + good,
        )


if __name__ == "__main__":
    unittest.main()
