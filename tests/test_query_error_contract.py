"""Regression tests for the error contract of the two read-only queries.

The derivation-parameter query (``derivation``) and the revocation-status
query (``is_revoked``) share one entry-point rule with the material read;
this module pins that rule for the two query entries specifically:

* an empty key id raises ``ValueError`` and that check runs first, ahead of
  the version type check -- including when the derivation query omits the
  version altogether (``version=None``, the active-version sentinel);
* a non-genuine-integer version raises ``TypeError`` -- floats and bools do
  not count, and ``1.0`` comparing equal to ``1`` must not admit it; passing
  ``None`` explicitly to ``is_revoked`` is the same illegal version;
* unknown key / nonexistent version with a genuine int stays ``KeyError``;
  a directly sealed version answers the derivation query with an empty
  record ``{}`` rather than an error;
* every rejected call is strictly read-only: the persisted manifest,
  journals and material bytes neither grow nor shrink, and the in-memory
  snapshot is unchanged;
* material sealed before the rejected calls keeps reading back byte for
  byte.

Every case works in its own temporary directory, uses the standard library
only, returns its lock handle through the shared fixture and is independent
of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import unittest

from keyvault_ledger import Vault
from tests._fixtures import VaultFixture


# An int subclass is not a *genuine* int: the entry check uses ``type(x) is
# int``, so it is rejected exactly like a float or bool.
class _IntLike(int):
    pass


class QueryContractTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.root

    def open_vault(self) -> Vault:
        # Tracked so the single fixture cleanup returns the lock handle and
        # removes the temporary tree however the case ends.
        return self.fixture.open()


# Values that are never genuine integers, even when they compare equal to
# one (1.0 == 1 and True == 1).
NON_INTEGER_VERSIONS = (
    1.0,          # numerically equal to version 1 -- still not an int
    2.0,
    2.5,
    -1.0,
    True,         # bool is an int subclass and equals 1
    False,        # equals 0, which is not even a sealed version
    "1",
    (1,),         # a container, never an int
    _IntLike(1),  # an int subclass, not a genuine int
    object(),
)


class TestEmptyKeyIdReportedAsValueError(QueryContractTestCase):
    def test_empty_id_with_float_or_bool_version_is_value_error(self):
        vault = self.open_vault()
        # The identifier is judged first, before any state is read: a
        # float- or bool-valued version on an empty id is an identifier
        # problem (ValueError), never TypeError.
        for bad in (1.0, 2.5, True, False, "1", (1,), _IntLike(1)):
            with self.assertRaises(ValueError, msg=f"derivation {bad!r}"):
                vault.derivation("", bad)
            with self.assertRaises(ValueError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("", bad)

    def test_derivation_without_version_empty_id_still_value_error(self):
        vault = self.open_vault()
        # Omitting the version is the documented active-version sentinel;
        # the empty id is still checked first and fails with exactly the
        # same ValueError type as an explicit version would.
        with self.assertRaises(ValueError):
            vault.derivation("")
        with self.assertRaises(ValueError):
            vault.derivation("", None)
        with self.assertRaises(ValueError):
            vault.derivation("", version=None)

    def test_empty_id_precedence_identical_on_both_queries(self):
        vault = self.open_vault()
        # Whatever the version argument looks like, both queries classify an
        # empty id the same way; checking the version type first would be
        # the wrong order.  Neither half needs a sealed key: the empty-id
        # ValueError and the non-empty-id TypeError both fire at the entry,
        # before any key/version lookup.
        versions = (1.0, True, "x", None, 1, -3)
        for version in versions:
            with self.assertRaises(ValueError, msg=f"derivation {version!r}"):
                vault.derivation("", version)
            with self.assertRaises(ValueError, msg=f"is_revoked {version!r}"):
                vault.is_revoked("", version)
        # A non-empty id with the very same arguments surfaces the version
        # type problem instead, proving the ordering rather than the value.
        with self.assertRaises(TypeError):
            vault.derivation("k", 1.0)
        with self.assertRaises(TypeError):
            vault.is_revoked("k", 1.0)


class TestNonIntegerVersionReportedAsTypeError(QueryContractTestCase):
    def test_non_integer_versions_raise_type_error_on_both_queries(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("k", b"pw", b"salt", 100, 16)
        for bad in NON_INTEGER_VERSIONS:
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("k", bad)
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("k", bad)

    def test_float_equal_to_real_version_cannot_pass(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        # The core regression: 1.0 == 1 must not read as version 1 just
        # because the numbers happen to compare equal.
        self.assertTrue(1.0 == 1)
        with self.assertRaises(TypeError):
            vault.derivation("k", 1.0)
        with self.assertRaises(TypeError):
            vault.is_revoked("k", 1.0)
        with self.assertRaises(TypeError):
            vault.derivation("k", True)
        with self.assertRaises(TypeError):
            vault.is_revoked("k", True)
        # ...while the genuine integers reach the record and answer normally.
        self.assertEqual(vault.derivation("k", 1), {})
        self.assertFalse(vault.is_revoked("k", 1))

    def test_is_revoked_none_version_is_an_illegal_version(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        # Unlike derivation/load, is_revoked has no active-version sentinel:
        # an explicit None is a non-integer version and fails at the entry.
        with self.assertRaises(TypeError):
            vault.is_revoked("k", None)
        with self.assertRaises(TypeError):
            vault.is_revoked("never-sealed", None)
        # Empty id still wins over that type check.
        with self.assertRaises(ValueError):
            vault.is_revoked("", None)

    def test_non_int_version_rejected_before_unknown_key_lookup(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        # Entry validation fires before existence on both queries.
        with self.assertRaises(TypeError):
            vault.derivation("never-sealed", 1.0)
        with self.assertRaises(TypeError):
            vault.is_revoked("never-sealed", True)

    def test_error_message_is_the_shared_version_message(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        with self.assertRaises(TypeError) as caught:
            vault.derivation("k", 1.0)
        self.assertEqual(str(caught.exception), "version must be an int")
        with self.assertRaises(TypeError) as caught:
            vault.is_revoked("k", None)
        self.assertEqual(str(caught.exception), "version must be an int")


class TestGenuineIntLookupSemantics(QueryContractTestCase):
    def test_unknown_key_or_missing_version_is_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        # Revocation status: unknown key (explicit genuine-int version) and
        # versions that were never sealed are unchanged KeyError behaviour.
        with self.assertRaises(KeyError):
            vault.is_revoked("never-sealed", 1)
        for missing in (0, 2, 99, -1):
            with self.assertRaises(KeyError, msg=f"is_revoked {missing}"):
                vault.is_revoked("k", missing)
        # Derivation parameters: same rule, including the unversioned
        # active-version lookup of an unknown key.
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed")
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed", 1)
        for missing in (0, 3, 99):
            with self.assertRaises(KeyError, msg=f"derivation {missing}"):
                vault.derivation("d", missing)
            with self.assertRaises(KeyError, msg=f"derivation plain {missing}"):
                vault.derivation("k", missing)

    def test_directly_sealed_version_returns_empty_record(self):
        vault = self.open_vault()
        version = vault.seal("k", b"sealed-bytes")
        # A direct seal answers the derivation query with {} -- by explicit
        # version, through the active-version default, and again after the
        # rejected calls below cannot change that.
        self.assertEqual(vault.derivation("k", version), {})
        self.assertEqual(vault.derivation("k"), {})
        self.assertEqual(vault.derivation("k", None), {})

    def test_derived_record_is_returned_for_a_derived_version(self):
        vault = self.open_vault()
        version = vault.derive_seal("k", b"pw", b"salty", 1000, 24)
        self.assertEqual(
            vault.derivation("k", version),
            {"salt": b"salty", "iterations": 1000, "length": 24},
        )
        self.assertEqual(vault.derivation("k"), vault.derivation("k", version))
        # Sealing a direct version afterwards leaves the historical derived
        # record intact while the active one is now a direct seal.
        vault.seal("k", b"plain")
        self.assertEqual(vault.derivation("k", version)["iterations"], 1000)
        self.assertEqual(vault.derivation("k"), {})


class TestRejectedQueriesAreReadOnly(QueryContractTestCase):
    def _disk_bytes(self) -> dict:
        files = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                files[path.relative_to(self.root)] = (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
        return files

    def test_rejected_queries_touch_neither_disk_nor_snapshot(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)

        before_files = self._disk_bytes()
        before_manifest = vault.manifest()
        snapshot_before = {
            "versions-k": vault.versions("k"),
            "versions-d": vault.versions("d"),
            "active-k": vault.active("k"),
            "active-d": vault.active("d"),
            "revoked-k": vault.revoked_versions("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "derivation-d": vault.derivation("d", 1),
            "derivation-k": vault.derivation("k", 1),
        }

        rejected = (
            lambda: vault.derivation("", 1.0),
            lambda: vault.derivation("", True),
            lambda: vault.derivation(""),
            lambda: vault.derivation("", None),
            lambda: vault.derivation("k", 1.0),
            lambda: vault.derivation("k", True),
            lambda: vault.derivation("k", "1"),
            lambda: vault.derivation("never-sealed", 1.0),
            lambda: vault.derivation("never-sealed", 1),
            lambda: vault.derivation("k", 99),
            lambda: vault.is_revoked("", 1.0),
            lambda: vault.is_revoked("", None),
            lambda: vault.is_revoked("k", 1.0),
            lambda: vault.is_revoked("k", None),
            lambda: vault.is_revoked("k", True),
            lambda: vault.is_revoked("never-sealed", 1.0),
            lambda: vault.is_revoked("never-sealed", 1),
            lambda: vault.is_revoked("k", 99),
        )
        for call in rejected:
            with self.assertRaises((ValueError, TypeError, KeyError)):
                call()

        # Disk inventory: same files, same bytes, untouched mtimes -- the
        # manifest, both journals (when present) and every material record
        # neither grow nor shrink.
        self.assertEqual(sorted(self._disk_bytes()), sorted(before_files))
        for rel, (data, mtime) in before_files.items():
            path = self.root / rel
            self.assertEqual(path.read_bytes(), data, str(rel))
            self.assertEqual(path.stat().st_mtime_ns, mtime, str(rel))

        # The persisted manifest view is byte-for-byte the same object graph.
        self.assertEqual(vault.manifest(), before_manifest)

        # In-memory snapshot: versions, active pointers, markers and reads
        # are exactly what they were before the rejected calls.
        snapshot_after = {
            "versions-k": vault.versions("k"),
            "versions-d": vault.versions("d"),
            "active-k": vault.active("k"),
            "active-d": vault.active("d"),
            "revoked-k": vault.revoked_versions("k"),
            "load-k1": vault.load("k", 1),
            "load-k2": vault.load("k", 2),
            "derivation-d": vault.derivation("d", 1),
            "derivation-k": vault.derivation("k", 1),
        }
        self.assertEqual(snapshot_after, snapshot_before)

        # A full reload stays healthy and answers identically afterwards.
        vault.reload()
        self.assertEqual(vault.load("k", 1), snapshot_before["load-k1"])
        self.assertEqual(vault.load("k", 2), snapshot_before["load-k2"])
        self.assertFalse(vault.is_revoked("k", 1))
        self.assertEqual(vault.derivation("d", 1)["iterations"], 100)

    def test_existing_material_remains_readable_byte_for_byte(self):
        vault = self.open_vault()
        payloads = [b"", bytes(range(256)), b"\x00\xff\nsuffix", "λ-key".encode()]
        for payload in payloads:
            vault.seal("k", payload)
        password, salt, iterations, length = b"pw", b"salty", 500, 32
        derived_version = vault.derive_seal(
            "d", password, salt, iterations, length
        )
        expected_derived = hashlib.pbkdf2_hmac(
            "sha256", password, salt, iterations, dklen=length
        )

        # Burn a sequence of rejected queries against both entries.
        for call in (
            lambda: vault.derivation("", 1.0),
            lambda: vault.derivation("k", True),
            lambda: vault.derivation("ghost", 1),
            lambda: vault.is_revoked("", None),
            lambda: vault.is_revoked("k", 1.0),
            lambda: vault.is_revoked("ghost", 1),
        ):
            with self.assertRaises((ValueError, TypeError, KeyError)):
                call()

        # Same key, same bytes: every stored version reads back exactly the
        # bytes that were written, derived material included.
        for index, payload in enumerate(payloads, start=1):
            self.assertIs(type(vault.load("k", index)), bytes)
            self.assertEqual(vault.load("k", index), payload)
        self.assertEqual(vault.load("d", derived_version), expected_derived)
        self.assertEqual(vault.derivation("d", derived_version)["salt"], salt)


class TestRejectedQueriesAreDeterministic(QueryContractTestCase):
    def test_repeated_bad_calls_raise_the_same_types_in_order(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        calls = (
            lambda: vault.derivation("", 1.0),
            lambda: vault.derivation(""),
            lambda: vault.derivation("k", 1.0),
            lambda: vault.derivation("k", None),
            lambda: vault.derivation("ghost", 1),
            lambda: vault.derivation("k", 99),
            lambda: vault.is_revoked("", 1.0),
            lambda: vault.is_revoked("", None),
            lambda: vault.is_revoked("k", 1.0),
            lambda: vault.is_revoked("k", None),
            lambda: vault.is_revoked("ghost", 1),
            lambda: vault.is_revoked("k", 99),
        )

        _OK = "no exception"

        def run() -> list:
            seen = []
            for call in calls:
                try:
                    call()
                except Exception as exc:  # noqa: BLE001 - type is the result
                    seen.append(type(exc))
                else:
                    seen.append(_OK)
            return seen

        expected = [
            ValueError,   # derivation empty id + float
            ValueError,   # derivation empty id, version omitted
            TypeError,    # derivation float version
            _OK,          # derivation(None sentinel) resolves active version
            KeyError,     # derivation unknown key
            KeyError,     # derivation missing version
            ValueError,   # is_revoked empty id + float
            ValueError,   # is_revoked empty id + None
            TypeError,    # is_revoked float version
            TypeError,    # is_revoked None version
            KeyError,     # is_revoked unknown key
            KeyError,     # is_revoked missing version
        ]
        first, second, third = run(), run(), run()
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(third, expected)


if __name__ == "__main__":
    unittest.main()
