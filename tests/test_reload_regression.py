"""Regression tests pinning the observable result of a whole-vault reload
after the derived-material records or the activity journals are corrupted.

The baseline vault capabilities, exception vocabulary and CLI entry points
are already implemented and are intentionally not touched here.  These cases
only freeze observable behaviour:

* every hand-edited corruption of a persisted ``derivation`` record (salt
  missing/tampered, declared length not matching the stored material,
  non-positive or non-integer iterations/length, a non-object/emptied
  derivation object) or of either append-only journal (a malformed record,
  a record pointing at a key/version that never existed, an activation whose
  target version is above the newest sealed version it is bound to) makes a
  whole-vault ``reload()`` raise exactly ``ValueError``;

* while such a failure persists the in-memory snapshot is frozen: the keys
  already in hand stay readable and the answers for versions, active
  version, revocation markers and derivation parameters are byte-for-byte
  identical before and after every failed reload -- a cold open on the same
  directory raises the same ``ValueError``;

* a failed reload is strictly read-only: the on-disk records neither grow
  nor shrink, byte for byte;

* once the corruption is undone a reload succeeds again and the snapshot
  corresponds to the disk records exactly, including the point that follows
  one successful reload (every stored byte equals its material file and its
  manifest digest, journals imply the revoked set and active pointer);

* the passphrase still never reaches disk and material read back is
  byte-for-byte equal to a fresh PBKDF2 derivation with the queried
  parameters.

Everything happens inside a temporary directory, uses the standard library
only and is independent of execution order::

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
    REVOCATIONS_NAME,
    _dump_manifest,
)

from tests.vaultcase import VaultFixtureCase

PLAIN_MATERIALS = {1: b"plain-one", 2: b"plain-two"}
DRV_PARAMETERS = {
    1: (b"passphrase-alpha", b"salt-alpha", 1000, 32),
    2: (b"passphrase-bravo", b"salt-bravo", 2500, 48),
}
MIX_DERIVED = (b"passphrase-charlie", b"salt-charlie", 500, 24)
MIX_PLAIN = {1: b"mix-one", 3: b"mix-three"}


class ReloadRegressionTestCase(VaultFixtureCase):
    def setUp(self) -> None:
        super().setUp()
        # Each corruption scenario gets its own vault subdirectory, so cases
        # never share disk state and execution order cannot matter.
        self.root = Path(self._tmp.name) / "vaults"

    def _open(self, root: Path) -> Vault:
        return self.open_vault(root)

    # ------------------------------------------------------------------
    # fixture construction
    # ------------------------------------------------------------------

    def _build_healthy(self, name: str) -> tuple[Vault, Path]:
        """Build a rich, healthy vault and return ``(handle, root)``.

        The state exercises every record kind at once: plain and derived
        seals interleaved on one key, a key with only derived versions,
        revocation markers and two repoints (one of them at a derived
        version), each bound to the newest sealed version of its key.
        """
        root = self.root / name
        vault = self._open(root)

        vault.seal("plain", PLAIN_MATERIALS[1])
        vault.seal("plain", PLAIN_MATERIALS[2])
        vault.set_active("plain", 1)  # bound to latest=2

        password, salt, iterations, length = DRV_PARAMETERS[1]
        vault.derive_seal("drv", password, salt, iterations, length)
        password, salt, iterations, length = DRV_PARAMETERS[2]
        vault.derive_seal("drv", password, salt, iterations, length)
        vault.revoke("drv", 1)

        vault.seal("mix", MIX_PLAIN[1])
        password, salt, iterations, length = MIX_DERIVED
        vault.derive_seal("mix", password, salt, iterations, length)
        vault.seal("mix", MIX_PLAIN[3])
        vault.revoke("mix", 1)
        vault.set_active("mix", 2)  # points at the derived version, latest=3

        return vault, root

    # ------------------------------------------------------------------
    # observable-state snapshots
    # ------------------------------------------------------------------

    def _answers(self, vault: Vault) -> dict:
        """Every state a reader can query, in a comparable plain structure."""
        keys = {}
        for key_id in ("plain", "drv", "mix"):
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

    def _assert_snapshot_corresponds_to_disk(
        self, vault: Vault, root: Path
    ) -> None:
        """Re-derive every observable answer straight from the disk records.

        This never consults the vault's private state: it parses the manifest
        and both journals itself and requires the public snapshot to match
        exactly.  It is the post-successful-reload correspondence point.
        """
        manifest = json.loads((root / MANIFEST_NAME).read_bytes().decode("utf-8"))
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
                # The snapshot serves the exact bytes persisted on disk...
                self.assertEqual(vault.load(key_id, version), data)
                # ...and those bytes match the manifest digest and the
                # declared derived length.
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
    # the core failure/recovery contract
    # ------------------------------------------------------------------

    def _assert_failure_then_recovery(
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
        self._assert_snapshot_corresponds_to_disk(vault, root)
        reopened = self._open(root)
        self.assertEqual(self._answers(reopened), expected)
        self._assert_snapshot_corresponds_to_disk(reopened, root)
        self.assertEqual(self._disk_bytes(root), healthy_disk)

        # Repeating the identical failing input gives the identical result,
        # then recovery works a second time as well.
        corrupt()
        with self.assertRaises(ValueError) as caught:
            vault.reload()
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(self._answers(vault), expected)
        self._restore_disk(root, healthy_disk)
        vault.reload()
        self.assertEqual(self._answers(vault), expected)
        self._assert_snapshot_corresponds_to_disk(vault, root)

    # ------------------------------------------------------------------
    # manifest editing helpers
    # ------------------------------------------------------------------

    def _edit_manifest(self, root: Path, mutate) -> None:
        path = root / MANIFEST_NAME
        manifest = json.loads(path.read_bytes().decode("utf-8"))
        mutate(manifest)
        path.write_bytes(_dump_manifest(manifest))

    @staticmethod
    def _record(manifest: dict, key_id: str, version: int) -> dict:
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
        rel = self._record(manifest, key_id, version)["file"]
        return root / rel


# ---------------------------------------------------------------------------
# corrupted persisted derivation records
# ---------------------------------------------------------------------------


class TestDerivedRecordReloadFailures(ReloadRegressionTestCase):
    def test_every_derivation_corruption_is_rejected_then_recovers(self):
        # Each entry is (case name, mutator applied out of band to disk).
        # The mutator targets the derivation record of "mix" version 2,
        # which is derived and also the repointed active version.
        def tampered_salt(mutate_field):
            def apply(root):
                self._edit_manifest(
                    root,
                    lambda m: mutate_field(
                        self._record(m, "mix", 2)["derivation"]
                    ),
                )

            return apply

        def bad_parameters(field, value):
            return tampered_salt(lambda d: d.__setitem__(field, value))

        def derivation_object(value):
            def apply(root):
                self._edit_manifest(
                    root,
                    lambda m: self._record(m, "mix", 2).__setitem__(
                        "derivation", value
                    ),
                )

            return apply

        cases = [
            ("salt_deleted", tampered_salt(lambda d: d.pop("salt"))),
            ("salt_empty", bad_parameters("salt", "")),
            ("salt_non_string", bad_parameters("salt", 123)),
            ("salt_null", bad_parameters("salt", None)),
            ("salt_not_base64", bad_parameters("salt", "!!!not-base64!!!")),
            ("salt_undershot_padding", bad_parameters("salt", "abcde")),
            ("length_smaller_than_material", bad_parameters("length", 16)),
            ("length_larger_than_material", bad_parameters("length", 48)),
            ("iterations_zero", bad_parameters("iterations", 0)),
            ("iterations_negative", bad_parameters("iterations", -1000)),
            ("length_zero", bad_parameters("length", 0)),
            ("length_negative", bad_parameters("length", -1)),
            ("iterations_float", bad_parameters("iterations", 1.5)),
            ("length_float", bad_parameters("length", 24.0)),
            ("iterations_string", bad_parameters("iterations", "1000")),
            ("length_string", bad_parameters("length", "24")),
            ("iterations_bool_true", bad_parameters("iterations", True)),
            ("iterations_bool_false", bad_parameters("iterations", False)),
            ("length_bool_true", bad_parameters("length", True)),
            ("length_null", bad_parameters("length", None)),
            ("iterations_list", bad_parameters("iterations", [1000])),
            ("derivation_object_empty", derivation_object({})),
            ("derivation_object_string", derivation_object("x")),
            ("derivation_object_number", derivation_object(123)),
            ("derivation_object_list", derivation_object(["salt"])),
        ]

        for index, (name, corrupt) in enumerate(cases):
            with self.subTest(case=name):
                vault, case_root = self._build_healthy(f"derived-{index}")
                expected = self._answers(vault)
                self._assert_failure_then_recovery(
                    vault, case_root, expected, lambda: corrupt(case_root)
                )

    def test_salt_corruption_on_other_derived_key_is_rejected(self):
        # The salt check must not depend on the derived version being the
        # active one or living on a particular key.
        vault, root = self._build_healthy("derived-other-key")
        expected = self._answers(vault)

        def corrupt():
            self._edit_manifest(
                root,
                lambda m: self._record(m, "drv", 1)["derivation"].update(
                    salt="not base64"
                ),
            )

        self._assert_failure_then_recovery(vault, root, expected, corrupt)

    def test_missing_material_file_is_rejected_for_every_record_kind(self):
        for index, (key_id, version, name) in enumerate(
            (("mix", 2, "derived"), ("mix", 3, "plain"), ("drv", 1, "derived"))
        ):
            with self.subTest(material=name):
                vault, root = self._build_healthy(f"missing-material-{index}")
                expected = self._answers(vault)

                def corrupt(key_id=key_id, version=version):
                    self._material_file(root, key_id, version).unlink()

                self._assert_failure_then_recovery(
                    vault, root, expected, corrupt
                )

    def test_tampered_derived_material_is_rejected(self):
        vault, root = self._build_healthy("tampered-derived-material")
        expected = self._answers(vault)

        def corrupt():
            self._material_file(root, "mix", 2).write_bytes(b"tampered bytes")

        self._assert_failure_then_recovery(vault, root, expected, corrupt)

    def test_emptied_derivation_record_fails_on_cold_open_as_well(self):
        # The "whole derivation record cleared" shape ({}) is rejected not
        # only by a live reload but also when a fresh handle opens the vault.
        vault, root = self._build_healthy("emptied-record-cold")
        self._edit_manifest(
            root,
            lambda m: self._record(m, "mix", 2).__setitem__(
                "derivation", {}
            ),
        )
        with self.assertRaises(ValueError):
            Vault(root)
        # While the failure persists the existing handle keeps its snapshot.
        expected = self._answers(vault)
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(self._answers(vault), expected)


# ---------------------------------------------------------------------------
# corrupted activation (activity) journal
# ---------------------------------------------------------------------------


class TestActivationJournalReloadFailures(ReloadRegressionTestCase):
    def _write_activations(self, root: Path, data: bytes) -> None:
        (root / ACTIVATIONS_NAME).write_bytes(data)

    def test_malformed_activity_records_are_rejected_then_recover(self):
        bad_payloads = {
            "not_json": b"{not json\n",
            "not_object": b"[1, 2]\n",
            "empty_record": b"\n",
            "invalid_utf8": b"\xff\xfe\n",
            "missing_key_id": b'{"version": 1, "latest": 1}\n',
            "missing_version": b'{"key_id": "mix", "latest": 3}\n',
            "missing_latest": b'{"key_id": "mix", "version": 2}\n',
            "empty_key_id": b'{"key_id": "", "version": 1, "latest": 1}\n',
            "string_version": b'{"key_id": "mix", "version": "2", "latest": 3}\n',
            "float_version": b'{"key_id": "mix", "version": 2.0, "latest": 3}\n',
            "bool_version": b'{"key_id": "mix", "version": true, "latest": 3}\n',
            "string_latest": b'{"key_id": "mix", "version": 2, "latest": "3"}\n',
            "null_latest": b'{"key_id": "mix", "version": 2, "latest": null}\n',
            "unknown_key": b'{"key_id": "ghost", "version": 1, "latest": 1}\n',
            "unknown_version": b'{"key_id": "mix", "version": 99, "latest": 99}\n',
            "version_above_bound_latest": b'{"key_id": "mix", "version": 3, "latest": 2}\n',
            "bound_latest_never_sealed": b'{"key_id": "mix", "version": 1, "latest": 9}\n',
            "bound_latest_zero": b'{"key_id": "mix", "version": 1, "latest": 0}\n',
            "good_then_bad": (
                b'{"key_id": "mix", "version": 2, "latest": 3}\n'
                b"{not json\n"
            ),
        }

        for index, (name, payload) in enumerate(bad_payloads.items()):
            with self.subTest(case=name):
                vault, root = self._build_healthy(f"activation-{index}")
                expected = self._answers(vault)
                self._assert_failure_then_recovery(
                    vault,
                    root,
                    expected,
                    lambda payload=payload, root=root: self._write_activations(
                        root, payload
                    ),
                )

    def test_activity_record_for_unknown_key_rejected_on_cold_open(self):
        vault, root = self._build_healthy("activation-cold")
        self._write_activations(
            root, b'{"key_id": "ghost", "version": 1, "latest": 1}\n'
        )
        with self.assertRaises(ValueError):
            Vault(root)
        expected = self._answers(vault)
        with self.assertRaises(ValueError):
            vault.reload()
        # The live handle's repointed state survives untouched.
        self.assertEqual(vault.active("mix"), 2)
        self.assertEqual(vault.active("plain"), 1)
        self.assertEqual(self._answers(vault), expected)

    def test_failed_activity_reload_does_not_move_active_pointer(self):
        vault, root = self._build_healthy("activation-active-kept")
        self._write_activations(
            root, b'{"key_id": "plain", "version": 2, "latest": 9}\n'
        )
        with self.assertRaises(ValueError):
            vault.reload()
        # The corrupt, never-validated record cannot repoint anything: the
        # previously validated state (plain repointed at 1) is unchanged.
        self.assertEqual(vault.active("plain"), 1)
        self.assertEqual(vault.load("plain"), PLAIN_MATERIALS[1])
        self.assertEqual(vault.active("mix"), 2)


# ---------------------------------------------------------------------------
# corrupted revocation journal
# ---------------------------------------------------------------------------


class TestRevocationJournalReloadFailures(ReloadRegressionTestCase):
    def _write_revocations(self, root: Path, data: bytes) -> None:
        (root / REVOCATIONS_NAME).write_bytes(data)

    def test_malformed_revocation_records_are_rejected_then_recover(self):
        bad_payloads = {
            "not_json": b"{not json\n",
            "not_object": b"[1, 2]\n",
            "empty_record": b"\n",
            "invalid_utf8": b"\xff\xfe\n",
            "missing_key_id": b'{"version": 1}\n',
            "missing_version": b'{"key_id": "mix"}\n',
            "empty_key_id": b'{"key_id": "", "version": 1}\n',
            "string_version": b'{"key_id": "mix", "version": "1"}\n',
            "float_version": b'{"key_id": "mix", "version": 2.0}\n',
            "null_version": b'{"key_id": "mix", "version": null}\n',
            "unknown_key": b'{"key_id": "ghost", "version": 1}\n',
            "unknown_version": b'{"key_id": "mix", "version": 99}\n',
            "version_zero": b'{"key_id": "mix", "version": 0}\n',
            "negative_version": b'{"key_id": "mix", "version": -1}\n',
            "duplicate_record": (
                b'{"key_id": "mix", "version": 1}\n'
                b'{"key_id": "mix", "version": 1}\n'
            ),
            "good_then_bad": (
                b'{"key_id": "mix", "version": 1}\n'
                b'{"key_id": "mix"}\n'
            ),
        }

        for index, (name, payload) in enumerate(bad_payloads.items()):
            with self.subTest(case=name):
                vault, root = self._build_healthy(f"revocation-{index}")
                expected = self._answers(vault)

                def corrupt(payload=payload, root=root):
                    self._write_revocations(root, payload)

                self._assert_failure_then_recovery(
                    vault, root, expected, corrupt
                )

    def test_corrupt_revocation_journal_keeps_markers_unchanged(self):
        vault, root = self._build_healthy("revocation-markers-kept")
        expected = self._answers(vault)
        self._write_revocations(root, b"{garbage\n")
        for _ in range(3):
            with self.assertRaises(ValueError):
                vault.reload()
        self.assertTrue(vault.is_revoked("mix", 1))
        self.assertTrue(vault.is_revoked("drv", 1))
        self.assertEqual(vault.revoked_versions("mix"), [1])
        self.assertEqual(vault.revoked_versions("drv"), [1])
        self.assertFalse(vault.is_revoked("mix", 2))
        self.assertEqual(self._answers(vault), expected)
        # The corrupted bytes are neither repaired nor removed by reload.
        self.assertEqual((root / REVOCATIONS_NAME).read_bytes(), b"{garbage\n")


# ---------------------------------------------------------------------------
# successful reload: snapshot/disk correspondence and passphrase handling
# ---------------------------------------------------------------------------


class TestSuccessfulReloadCorrespondence(ReloadRegressionTestCase):
    def test_successful_reload_leaves_snapshot_corresponding_to_disk(self):
        vault, root = self._build_healthy("healthy-correspondence")
        vault.reload()
        self._assert_snapshot_corresponds_to_disk(vault, root)
        # A second successful reload and a fresh opener land on the same
        # fully corresponding state.
        vault.reload()
        self._assert_snapshot_corresponds_to_disk(vault, root)
        reopened = self._open(root)
        reopened.reload()
        self._assert_snapshot_corresponds_to_disk(reopened, root)
        self.assertEqual(self._answers(reopened), self._answers(vault))

    def test_readback_equals_fresh_rederivation_after_reload(self):
        root = self.root / "readback"
        vault = self._open(root)
        password = b"a-passphrase-nobody-should-ever-see"
        salt = b"\x00\x01 rederive salt \xff"
        iterations, length = 3210, 40
        version = vault.derive_seal(
            "k", password, salt, iterations, length
        )

        vault.reload()
        params = vault.derivation("k")
        rederived = hashlib.pbkdf2_hmac(
            "sha256",
            password,
            params["salt"],
            params["iterations"],
            dklen=params["length"],
        )
        self.assertEqual(params["salt"], salt)
        self.assertEqual(params["iterations"], iterations)
        self.assertEqual(params["length"], length)
        self.assertEqual(vault.load("k"), rederived)
        self.assertEqual(vault.load("k", version), rederived)
        self.assertEqual(len(vault.load("k", version)), length)

        # Reopening and reloading must not change a single byte.
        reopened = self._open(root)
        reopened.reload()
        self.assertEqual(reopened.load("k", version), rederived)
        self.assertEqual(
            reopened.derivation("k", version),
            {"salt": salt, "iterations": iterations, "length": length},
        )

        # The passphrase never reaches disk: none of the persisted records
        # (manifest, journals, materials) contains it.
        for path in root.rglob("*"):
            if path.is_file() and path.name != LOCK_NAME:
                self.assertNotIn(password, path.read_bytes())

    def test_readback_stays_byte_exact_through_corruption_cycle(self):
        vault, root = self._build_healthy("readback-through-cycle")
        password, salt, iterations, length = MIX_DERIVED
        expected_material = hashlib.pbkdf2_hmac(
            "sha256", password, salt, iterations, dklen=length
        )
        healthy_disk = self._disk_bytes(root)
        expected = self._answers(vault)

        # Corrupt the activity journal, fail reload, then undo exactly that
        # corruption: the derived material reads back identically and the
        # recovered snapshot re-derives to the same bytes.
        (root / ACTIVATIONS_NAME).write_bytes(b"{broken\n")
        with self.assertRaises(ValueError):
            vault.reload()
        self.assertEqual(vault.load("mix", 2), expected_material)

        self._restore_disk(root, healthy_disk)
        vault.reload()
        self.assertEqual(vault.load("mix", 2), expected_material)
        self.assertEqual(self._answers(vault), expected)
        self.assertEqual(self._disk_bytes(root), healthy_disk)
        self._assert_snapshot_corresponds_to_disk(vault, root)
        params = vault.derivation("mix", 2)
        self.assertEqual(
            hashlib.pbkdf2_hmac(
                "sha256",
                password,
                params["salt"],
                params["iterations"],
                dklen=params["length"],
            ),
            expected_material,
        )

    def test_identical_corruption_in_two_vaults_gives_identical_results(self):
        # Repeating the same hand-edited input against independently built
        # vaults must produce the same exception and the same frozen state.
        payload = b'{"key_id": "mix", "version": 99, "latest": 99}\n'
        outcomes = []
        for name in ("determinism-a", "determinism-b"):
            vault, root = self._build_healthy(name)
            expected = self._answers(vault)
            (root / ACTIVATIONS_NAME).write_bytes(payload)
            with self.assertRaises(ValueError) as caught:
                vault.reload()
            outcomes.append((str(caught.exception), self._answers(vault)))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        # The two healthy vaults were built from the same deterministic
        # operations, so their frozen answers are identical too.
        self.assertEqual(outcomes[0][1], outcomes[1][1])


if __name__ == "__main__":
    unittest.main()
