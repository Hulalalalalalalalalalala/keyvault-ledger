"""Regression tests for failed whole-vault reloads.

These cases pin the observable result of corrupting a *derivation record*
(``manifest.json``) or an *activity-log record* (``activations.jsonl`` and,
for revocation, ``revocations.jsonl``) and then triggering one full
:meth:`Vault.reload`.

The contract nailed down here, end to end:

* every kind of corruption named below makes ``reload()`` raise
  ``ValueError`` — never another exception type;
* while the corruption is on disk the in-memory snapshot is the one built
  before the tampering: versions, the active version, revocation markers,
  derivation parameters and the readable materials are all unchanged, so
  keys already in hand stay readable exactly as before;
* a failed reload never writes: the manifest, both journals and every
  material file stay byte-for-byte (and the file set stays) as they were the
  instant the failing reload began — the corruption is neither repaired nor
  truncated;
* the queries run before and after the failure return verbatim-identical
  results;
* once the tampering is undone, a reload succeeds again and the freshly
  validated snapshot corresponds exactly to the restored disk records;
* a successful reload leaves a checkpoint at which the in-memory snapshot
  and the disk records agree completely.

Only the standard library is used, every case reads and writes inside a
fresh temporary directory, and the same prepared input always produces the
same result.  Runnable with the rest of the suite::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
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

PASSWORD = b"correct horse battery staple"
SALT = b"\x00\x01 salty bytes \xff"
ITERATIONS = 1000
LENGTH = 32


class ReloadCorruptionCase(unittest.TestCase):
    """One throwaway vault root per test plus observable-state probes.

    The probes describe only what a holder of the vault can query (the
    in-memory snapshot) and the exact bytes sitting on disk; nothing reaches
    into private vault structures.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "vault"

    # ------------------------------------------------------------------
    # paths and disk manipulation helpers
    # ------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def revocations_path(self) -> Path:
        return self.root / REVOCATIONS_NAME

    @property
    def activations_path(self) -> Path:
        return self.root / ACTIVATIONS_NAME

    def disk_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_bytes().decode("utf-8"))

    def material_path(self, key_id: str = "k", version: int = 1) -> Path:
        for record in self.disk_manifest()["keys"][key_id]["versions"]:
            if record["version"] == version:
                return self.root / record["file"]
        raise KeyError((key_id, version))

    def alter_manifest(self, mutate) -> None:
        """Load the manifest, apply ``mutate(manifest)`` and write it back."""
        manifest = self.disk_manifest()
        mutate(manifest)
        self.manifest_path.write_bytes(_dump_manifest(manifest))

    def derivation_record(self, key_id: str = "k", version: int = 1) -> dict:
        for record in self.disk_manifest()["keys"][key_id]["versions"]:
            if record["version"] == version:
                return record["derivation"]
        raise KeyError((key_id, version))

    def alter_derivation(self, mutate, key_id: str = "k", version: int = 1) -> None:
        def _mutate(manifest: dict) -> None:
            for record in manifest["keys"][key_id]["versions"]:
                if record["version"] == version:
                    mutate(record["derivation"])
                    return
            raise KeyError((key_id, version))

        self.alter_manifest(_mutate)

    def write_activations(self, lines: bytes | list[bytes]) -> None:
        data = lines if isinstance(lines, bytes) else b"".join(lines)
        self.activations_path.write_bytes(data)

    def write_revocations(self, lines: bytes | list[bytes]) -> None:
        data = lines if isinstance(lines, bytes) else b"".join(lines)
        self.revocations_path.write_bytes(data)

    def activation_line(
        self, key_id: str = "k", version: int = 1, latest: int = 3
    ) -> bytes:
        return (
            json.dumps({"key_id": key_id, "version": version, "latest": latest})
            + "\n"
        ).encode("utf-8")

    def revocation_line(self, key_id: str = "k", version: int = 1) -> bytes:
        return (json.dumps({"key_id": key_id, "version": version}) + "\n").encode(
            "utf-8"
        )

    # ------------------------------------------------------------------
    # observable-state views
    # ------------------------------------------------------------------

    def snapshot_view(self, vault: Vault, root: Path | None = None) -> dict:
        """Everything a caller can observe through the public interface.

        The failure contract is that this whole structure is verbatim the
        healthy snapshot before the failing reload and after it.
        """
        view: dict = {}
        for key_id in vault.manifest()["keys"]:
            versions = vault.versions(key_id)
            view[key_id] = {
                "active": vault.active(key_id),
                "versions": versions,
                "materials": {v: vault.load(key_id, v) for v in versions},
                "revoked": vault.revoked_versions(key_id),
                "revocation_markers": {
                    v: vault.is_revoked(key_id, v) for v in versions
                },
                "derivations": {v: vault.derivation(key_id, v) for v in versions},
                "derivation_active": vault.derivation(key_id),
            }
        return view

    def disk_view(self, root: Path | None = None) -> dict[str, bytes]:
        """Every byte under the vault root, keyed by its relative path.

        ``vault.lock`` is only a mutual-exclusion device and carries no key
        data, so it is deliberately excluded.
        """
        base = self.root if root is None else root
        files: dict[str, bytes] = {}
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name == LOCK_NAME:
                continue
            files[str(path.relative_to(base))] = path.read_bytes()
        return files

    def assert_reload_raises_value_error(self, vault: Vault) -> ValueError:
        """Reload must raise ValueError (exact type, not merely a subclass)."""
        with self.assertRaises(ValueError) as caught:
            vault.reload()
        self.assertIs(type(caught.exception), ValueError)
        return caught.exception

    def assert_failed_reload_preserves_both(
        self, vault: Vault, healthy_snapshot: dict
    ) -> ValueError:
        """Reload with the (already tampered) disk, then check both states.

        Baselines differ on purpose:

        * ``healthy_snapshot`` was captured *before* tampering and is what the
          in-memory snapshot must still equal after the failed reload — the
          holder keeps serving the old, complete keyring;
        * the disk baseline is captured *here*, the instant before reload, so
          it contains the tampered bytes.  After the failed reload the disk
          must be byte-identical to that: the reload neither repairs nor
          truncates nor adds anything.
        """
        disk_at_reload = self.disk_view()
        error = self.assert_reload_raises_value_error(vault)
        # Snapshot invariant: same answers, byte for byte, as before tamper.
        self.assertEqual(self.snapshot_view(vault), healthy_snapshot)
        # Disk invariant: no record added, removed or rewritten by reload.
        self.assertEqual(self.disk_view(), disk_at_reload)
        return error


# ----------------------------------------------------------------------
# derivation-record corruption
# ----------------------------------------------------------------------


class TestDerivationReloadCorruption(ReloadCorruptionCase):
    def prepare(
        self, *, repoint: bool = False, revoke: bool = False
    ) -> tuple[Vault, int]:
        """Derived v1, plain v2, derived active v3, plus a second key.

        The mix proves a single bad derivation record fails the *whole*
        reload while the healthy records around it stay served untouched.
        """
        vault = Vault(self.root)
        v1 = vault.derive_seal("k", PASSWORD, SALT, ITERATIONS, LENGTH)
        vault.seal("k", b"plain-material")
        v3 = vault.derive_seal("k", PASSWORD, b"second-salt", 500, 16)
        vault.seal("other", b"other-key-material")
        if repoint:
            vault.set_active("k", v1)
        # Revoking the repointed version *after* the repoint is allowed and
        # leaves the pointer where it was (revoke never moves active).
        if revoke:
            vault.revoke("k", v1)
        return vault, v3

    def test_tampered_salt_is_rejected(self):
        vault, _ = self.prepare()
        original = self.derivation_record()["salt"]
        healthy = self.snapshot_view(vault)

        for tampered in (
            "!" + original[1:],                  # undecodable base64
            original[:-1],                       # truncated base64
            "!!!not-base64!!!",                  # garbage
            base64.b64encode(b"").decode(),      # decodes to empty bytes
        ):
            self.alter_derivation(lambda d, value=tampered: d.update(salt=value))
            self.assert_failed_reload_preserves_both(vault, healthy)
            self.alter_derivation(lambda d: d.update(salt=original))

        # Boundary: a valid base64 salt that merely names *different* bytes is
        # not detectable from disk (the passphrase never lands there), so
        # reload accepts it.  Pin that this is the full checkable rejection
        # set and the record is otherwise self-consistent.
        self.alter_derivation(
            lambda d: d.update(salt=base64.b64encode(b"a-different-salt").decode())
        )
        vault.reload()

    def test_missing_salt_is_rejected(self):
        vault, _ = self.prepare()
        healthy = self.snapshot_view(vault)
        for mutate in (
            lambda d: d.pop("salt"),
            lambda d: d.update(salt=""),
            lambda d: d.update(salt=123),
            lambda d: d.update(salt=None),
            lambda d: d.update(salt=["AAEC"]),
        ):
            self.alter_derivation(mutate)
            self.assert_failed_reload_preserves_both(vault, healthy)

    def test_declared_length_mismatching_material_is_rejected(self):
        vault, active_version = self.prepare()
        healthy = self.snapshot_view(vault)
        for bad_length in (16, 48, LENGTH - 1, LENGTH + 1):
            self.alter_derivation(lambda d, n=bad_length: d.update(length=n))
            self.assert_failed_reload_preserves_both(vault, healthy)
        # The live snapshot keeps the true parameters: v3 (16 bytes) active,
        # v1 still 32, byte for byte.
        self.assertEqual(vault.derivation("k")["length"], 16)
        self.assertEqual(vault.derivation("k", 1)["length"], LENGTH)
        self.assertEqual(active_version, 3)

    def test_non_positive_iterations_or_length_is_rejected(self):
        vault, _ = self.prepare()
        healthy = self.snapshot_view(vault)
        for bad in (0, -1, -1000):
            self.alter_derivation(lambda d, n=bad: d.update(iterations=n))
            self.assert_failed_reload_preserves_both(vault, healthy)
        for bad in (0, -1, -1000):
            self.alter_derivation(lambda d, n=bad: d.update(length=n))
            self.assert_failed_reload_preserves_both(vault, healthy)
        self.assertEqual(vault.derivation("k", 1)["iterations"], ITERATIONS)

    def test_corrupt_derivation_parameters_are_rejected(self):
        vault, _ = self.prepare()
        healthy = self.snapshot_view(vault)
        bad_values = (True, False, 1.5, "1000", None, [1000])
        for bad in bad_values:
            self.alter_derivation(lambda d, n=bad: d.update(iterations=n))
            self.assert_failed_reload_preserves_both(vault, healthy)
        for bad in bad_values:
            self.alter_derivation(lambda d, n=bad: d.update(length=n))
            self.assert_failed_reload_preserves_both(vault, healthy)

    def test_non_object_derivation_block_is_rejected(self):
        vault, _ = self.prepare()
        healthy = self.snapshot_view(vault)
        for bad in ("x", 123, ["salt"]):
            def _mutate(manifest: dict, value=bad) -> None:
                manifest["keys"]["k"]["versions"][0]["derivation"] = value

            self.alter_manifest(_mutate)
            self.assert_failed_reload_preserves_both(vault, healthy)

    def test_emptied_derivation_block_is_rejected(self):
        # "整份派生记录清空": an emptied derivation object carries no salt, so
        # reload rejects it with ValueError like any damaged record and keeps
        # both states.
        vault, _ = self.prepare()
        healthy = self.snapshot_view(vault)

        def _empty(manifest: dict) -> None:
            manifest["keys"]["k"]["versions"][0]["derivation"] = {}

        self.alter_manifest(_empty)
        self.assert_failed_reload_preserves_both(vault, healthy)

    def test_removed_or_null_derivation_block_is_accepted_as_plain(self):
        # Boundary next to the emptied-object case: dropping the block
        # entirely (or nulling it) leaves the version indistinguishable from
        # a directly-sealed one — the passphrase never reaches disk and
        # nothing self-inconsistent remains — so reload accepts it and reports
        # an empty derivation record.  Pin the exact boundary.
        for mutate in (
            lambda record: record.pop("derivation"),
            lambda record: record.update(derivation=None),
        ):
            vault, _ = self.prepare()

            def _mutate(manifest: dict) -> None:
                mutate(manifest["keys"]["k"]["versions"][0])

            self.alter_manifest(_mutate)
            vault.reload()  # accepted
            self.assertEqual(vault.derivation("k", 1), {})
            # The material itself is unaffected and still served byte for byte.
            self.assertEqual(
                vault.load("k", 1),
                hashlib.pbkdf2_hmac(
                    "sha256", PASSWORD, SALT, ITERATIONS, dklen=LENGTH
                ),
            )

    def test_other_record_failure_keeps_removed_lookalike_block_in_memory(self):
        # Before a successful reload observes a wiped block, the open handle
        # keeps the original parameters.  Combine the wipe on v1 with a hard
        # error on v3 (iterations == 0): the whole reload fails, so even the
        # wiped block never takes effect in the live snapshot.
        vault, _ = self.prepare()
        healthy = self.snapshot_view(vault)

        def _damage(manifest: dict) -> None:
            manifest["keys"]["k"]["versions"][0].pop("derivation")
            manifest["keys"]["k"]["versions"][2]["derivation"]["iterations"] = 0

        self.alter_manifest(_damage)
        self.assert_failed_reload_preserves_both(vault, healthy)
        self.assertEqual(
            vault.derivation("k", 1),
            {"salt": SALT, "iterations": ITERATIONS, "length": LENGTH},
        )

    def test_missing_material_of_derived_version_is_rejected(self):
        # "让素材文件缺失": the material backing a derived version is gone.
        vault, _ = self.prepare()
        healthy = self.snapshot_view(vault)
        self.material_path("k", 1).unlink()
        self.assert_failed_reload_preserves_both(vault, healthy)
        # The derived bytes already in hand remain readable byte for byte.
        expected = hashlib.pbkdf2_hmac(
            "sha256", PASSWORD, SALT, ITERATIONS, dklen=LENGTH
        )
        self.assertEqual(vault.load("k", 1), expected)

    def test_failure_is_independent_of_revoke_and_repoint_state(self):
        # The derivation rejection holds with a revocation marker and a bound
        # repoint already applied, and leaves both exactly as they were.
        vault, _ = self.prepare(repoint=True, revoke=True)
        healthy = self.snapshot_view(vault)
        self.alter_derivation(lambda d: d.update(iterations=0))
        self.assert_failed_reload_preserves_both(vault, healthy)
        self.assertEqual(vault.active("k"), 1)
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertEqual(vault.revoked_versions("k"), [1])

    def test_readback_matches_rederivation_after_successful_reload(self):
        # The passphrase never lands on disk; after a healthy reload the
        # read-back material equals a same-parameter re-derivation exactly.
        vault, _ = self.prepare()
        vault.reload()
        for version, password, salt, iterations, length in (
            (1, PASSWORD, SALT, ITERATIONS, LENGTH),
            (3, PASSWORD, b"second-salt", 500, 16),
        ):
            record = vault.derivation("k", version)
            rederived = hashlib.pbkdf2_hmac(
                "sha256",
                password,
                record["salt"],
                record["iterations"],
                dklen=record["length"],
            )
            self.assertEqual(vault.load("k", version), rederived)
            self.assertEqual(record["salt"], salt)
            self.assertEqual(record["iterations"], iterations)
            self.assertEqual(record["length"], length)


# ----------------------------------------------------------------------
# activity-log corruption (activations + revocations journals)
# ----------------------------------------------------------------------


class TestActivationLogReloadCorruption(ReloadCorruptionCase):
    def prepare(self, *, revoked: bool = False) -> tuple[Vault, dict]:
        vault = Vault(self.root)
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.seal("k", b"v3")
        vault.seal("other", b"o1")
        vault.set_active("k", 1)  # binds the repoint to latest == 3
        if revoked:
            vault.revoke("k", 2)
        return vault, self.snapshot_view(vault)

    def test_corrupt_activation_record_is_rejected(self):
        vault, healthy = self.prepare()
        for payload in (
            b"{not json\n",
            b"[1, 2, 3]\n",
            b'{"key_id": "", "version": 1, "latest": 3}\n',
            b'{"key_id": "k", "version": "1", "latest": 3}\n',
            b'{"key_id": "k", "version": 1, "latest": "3"}\n',
            b'{"key_id": "k", "version": 1}\n',
            b'{"key_id": "k", "latest": 3}\n',
            b'{"version": 1, "latest": 3}\n',
            b"\n",
            b"\xff\xfe\n",
        ):
            self.write_activations(payload)
            self.assert_failed_reload_preserves_both(vault, healthy)

    def test_record_pointing_at_never_existing_key_is_rejected(self):
        vault, healthy = self.prepare()
        self.write_activations(self.activation_line(key_id="ghost"))
        self.assert_failed_reload_preserves_both(vault, healthy)
        # The repointed live snapshot survives; the unknown key stays absent.
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(vault.versions("ghost"), [])

    def test_record_pointing_at_never_existing_version_is_rejected(self):
        vault, healthy = self.prepare()
        for payload in (
            self.activation_line(version=5, latest=5),
            self.activation_line(version=0, latest=3),
            self.activation_line(version=-1, latest=3),
            self.activation_line(version=1, latest=9),
            self.activation_line(version=4, latest=4),
        ):
            self.write_activations(payload)
            self.assert_failed_reload_preserves_both(vault, healthy)

    def test_record_above_bound_latest_sealed_version_is_rejected(self):
        # 版本号超出该密钥绑定时的最新封存版本: a target above the bound newest
        # sealed version (version > latest), or latest beyond what was sealed,
        # is a corrupt record rejected on the same terms.
        vault, healthy = self.prepare()
        for payload in (
            self.activation_line(version=2, latest=1),
            self.activation_line(version=3, latest=2),
            self.activation_line(version=1, latest=4),
        ):
            self.write_activations(payload)
            self.assert_failed_reload_preserves_both(vault, healthy)

    def test_every_line_validates_even_when_an_earlier_one_cannot_win(self):
        vault, healthy = self.prepare()
        # A well-formed record followed by a corrupt one still fails the whole
        # reload; the good line never masks the bad one, and the active pointer
        # from the previously-loaded journal is untouched.
        self.write_activations(
            self.activation_line(version=2, latest=3)
            + self.activation_line(version=99, latest=3)
        )
        self.assert_failed_reload_preserves_both(vault, healthy)
        self.assertEqual(vault.active("k"), 1)

    def test_corrupt_revocation_record_is_rejected(self):
        # revocations.jsonl is the other append-only activity log.
        vault, healthy = self.prepare(revoked=True)
        for payload in (
            b"{not json\n",
            b'{"key_id": "", "version": 1}\n',
            b'{"key_id": "k", "version": "1"}\n',
            b'{"key_id": "k"}\n',
            b'{"version": 1}\n',
            b"\n",
            b"\xff\xfe\n",
        ):
            self.write_revocations(payload)
            self.assert_failed_reload_preserves_both(vault, healthy)

    def test_revocation_of_never_existing_key_or_version_is_rejected(self):
        vault, healthy = self.prepare(revoked=True)
        for payload in (
            self.revocation_line(key_id="ghost", version=1),
            self.revocation_line(key_id="k", version=9),
            self.revocation_line(key_id="other", version=2),
        ):
            # Keep the one valid revocation (k@2), then append the bad line.
            self.write_revocations([self.revocation_line("k", 2), payload])
            self.assert_failed_reload_preserves_both(vault, healthy)

    def test_duplicate_revocation_record_is_rejected(self):
        vault, healthy = self.prepare(revoked=True)
        line = self.revocation_line("k", 2)
        self.write_revocations([line, line])
        self.assert_failed_reload_preserves_both(vault, healthy)

    def test_failed_reload_repairs_nothing_on_disk(self):
        vault, healthy = self.prepare()
        garbage = b"{garbage\n"
        self.write_activations(garbage)
        error = self.assert_failed_reload_preserves_both(vault, healthy)
        self.assertTrue(str(error))
        # The corrupt bytes are left exactly as hand-written.
        self.assertEqual(self.activations_path.read_bytes(), garbage)


# ----------------------------------------------------------------------
# recovery after the tampering is undone
# ----------------------------------------------------------------------


class TestReloadRecovery(ReloadCorruptionCase):
    def _restore_disk(self, healthy_disk: dict, root: Path | None = None) -> None:
        base = self.root if root is None else root
        for rel_path, data in healthy_disk.items():
            path = base / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def test_derivation_corruption_is_recoverable(self):
        vault = Vault(self.root)
        version = vault.derive_seal("k", PASSWORD, SALT, ITERATIONS, LENGTH)
        vault.seal("k", b"plain")
        healthy_snapshot = self.snapshot_view(vault)
        healthy_disk = self.disk_view()

        self.alter_derivation(lambda d: d.update(iterations=0))
        self.assert_reload_raises_value_error(vault)
        self._restore_disk(healthy_disk)

        vault.reload()  # healthy again
        self.assertEqual(self.snapshot_view(vault), healthy_snapshot)
        self.assertEqual(self.disk_view(), healthy_disk)
        # A brand-new handle validates the same bytes and reaches the same state.
        reopened = Vault(self.root)
        self.assertEqual(self.snapshot_view(reopened), healthy_snapshot)
        self.assertEqual(reopened.active("k"), version + 1)
        self.assertEqual(
            reopened.derivation("k", 1),
            {"salt": SALT, "iterations": ITERATIONS, "length": LENGTH},
        )

    def test_missing_material_is_recoverable(self):
        vault = Vault(self.root)
        vault.derive_seal("k", PASSWORD, SALT, ITERATIONS, LENGTH)
        healthy_snapshot = self.snapshot_view(vault)
        healthy_disk = self.disk_view()

        self.material_path("k", 1).unlink()
        self.assert_reload_raises_value_error(vault)
        self._restore_disk(healthy_disk)

        vault.reload()
        self.assertEqual(self.snapshot_view(vault), healthy_snapshot)
        self.assertEqual(self.disk_view(), healthy_disk)
        self.assertEqual(Vault(self.root).load("k"), vault.load("k"))

    def test_activation_corruption_is_recoverable(self):
        vault = Vault(self.root)
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.set_active("k", 1)
        healthy_snapshot = self.snapshot_view(vault)
        healthy_disk = self.disk_view()

        self.write_activations(b"{corrupt\n")
        self.assert_reload_raises_value_error(vault)
        self._restore_disk(healthy_disk)

        vault.reload()
        self.assertEqual(vault.active("k"), 1)
        self.assertEqual(self.snapshot_view(vault), healthy_snapshot)
        self.assertEqual(self.disk_view(), healthy_disk)
        self.assertEqual(Vault(self.root).active("k"), 1)

    def test_revocation_corruption_is_recoverable(self):
        vault = Vault(self.root)
        vault.seal("k", b"v1")
        vault.revoke("k", 1)
        healthy_snapshot = self.snapshot_view(vault)
        healthy_disk = self.disk_view()

        self.write_revocations(b"{corrupt\n")
        self.assert_reload_raises_value_error(vault)
        self._restore_disk(healthy_disk)

        vault.reload()
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertEqual(self.snapshot_view(vault), healthy_snapshot)
        self.assertEqual(self.disk_view(), healthy_disk)


# ----------------------------------------------------------------------
# successful-reload checkpoints and determinism
# ----------------------------------------------------------------------


class TestReloadCheckpoints(ReloadCorruptionCase):
    def test_successful_reload_leaves_snapshot_and_disk_in_full_correspondence(
        self,
    ):
        vault = Vault(self.root)
        vault.derive_seal("k", PASSWORD, SALT, ITERATIONS, LENGTH)
        vault.seal("k", b"plain")
        vault.derive_seal("k", PASSWORD, b"salt-b", 250, 20)
        vault.seal("other", b"o-material")
        vault.revoke("k", 1)
        vault.set_active("k", 2)

        vault.reload()

        # Checkpoint: re-validating the very same disk bytes must produce the
        # identical snapshot and must not change a single byte on disk.
        disk_before = self.disk_view()
        snapshot_before = self.snapshot_view(vault)
        vault.reload()
        vault.reload()
        self.assertEqual(self.disk_view(), disk_before)
        self.assertEqual(self.snapshot_view(vault), snapshot_before)

        # A fresh handle built from those bytes observes the same state, so
        # the in-memory snapshot and the on-disk records fully correspond.
        self.assertEqual(self.snapshot_view(Vault(self.root)), snapshot_before)

        # And that state is exactly the one the operations above produced.
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.revoked_versions("k"), [1])
        self.assertEqual(
            vault.derivation("k", 1),
            {"salt": SALT, "iterations": ITERATIONS, "length": LENGTH},
        )
        self.assertEqual(vault.derivation("k", 2), {})
        self.assertEqual(vault.load("other"), b"o-material")

    def test_healthy_reload_never_writes_and_creates_no_journal(self):
        vault = Vault(self.root)
        vault.derive_seal("k", PASSWORD, SALT, ITERATIONS, LENGTH)
        # Opening and sealing never create the activations journal.
        self.assertFalse(self.activations_path.exists())
        before = self.disk_view()
        vault.reload()
        self.assertEqual(self.disk_view(), before)
        self.assertFalse(self.activations_path.exists())


class TestFailedReloadDeterminism(ReloadCorruptionCase):
    """Same corrupted input -> same exception, same preserved two states."""

    def _build_identical_vault(self, root: Path) -> Vault:
        vault = Vault(root)
        vault.derive_seal("k", PASSWORD, SALT, ITERATIONS, LENGTH)
        vault.seal("k", b"plain")
        vault.seal("other", b"o")
        vault.set_active("k", 1)
        vault.reload()
        return vault

    def _run_case(self, name: str, corrupt) -> tuple[str, dict, dict]:
        root = Path(self._tmp.name) / name
        vault = self._build_identical_vault(root)
        healthy_snapshot = self.snapshot_view(vault, root)
        corrupt(root)
        tampered_disk = self.disk_view(root)
        try:
            vault.reload()
        except ValueError as exc:
            error = f"ValueError: {exc}"
        else:
            raise AssertionError("reload unexpectedly succeeded")
        # The live snapshot stays healthy; the disk stays as tampered.
        self.assertEqual(self.snapshot_view(vault, root), healthy_snapshot)
        self.assertEqual(self.disk_view(root), tampered_disk)
        return error, healthy_snapshot, tampered_disk

    def test_repeated_identical_corruption_gives_identical_results(self):
        def bad_derivation(root: Path) -> None:
            path = root / MANIFEST_NAME
            manifest = json.loads(path.read_bytes())
            manifest["keys"]["k"]["versions"][0]["derivation"]["iterations"] = 0
            path.write_bytes(_dump_manifest(manifest))

        def bad_activation(root: Path) -> None:
            (root / ACTIVATIONS_NAME).write_bytes(b"{garbage\n")

        def missing_material(root: Path) -> None:
            manifest = json.loads((root / MANIFEST_NAME).read_bytes())
            rel = manifest["keys"]["k"]["versions"][0]["file"]
            (root / rel).unlink()

        for name, corrupt in (
            ("derivation", bad_derivation),
            ("activation", bad_activation),
            ("material", missing_material),
        ):
            first = self._run_case(f"{name}-a", corrupt)
            second = self._run_case(f"{name}-b", corrupt)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
