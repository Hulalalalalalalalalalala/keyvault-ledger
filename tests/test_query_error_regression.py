"""Regression tests for the error contract of the two per-version queries.

``derivation`` (the derived-parameter query) and ``is_revoked`` (the
revocation status query) share one entry-point rule, and this module pins
it for both entries together:

* an empty key id raises ``ValueError`` and that check runs first, ahead of
  the version type check -- an empty id paired with a float or bool version
  is still an identifier problem, on both queries and whether or not the
  version was passed explicitly;
* once the id is a non-empty string, a version that is not a genuine
  ``int`` raises ``TypeError`` -- bools and floats do not count, and a float
  numerically equal to an existing version (``1.0`` vs ``1``) must not slip
  through just because it compares equal; ``None`` is the documented active
  sentinel of ``derivation`` only, so ``is_revoked("k", None)`` is a
  ``TypeError``;
* only then does existence matter: an unknown key or a genuine-int version
  that was never sealed raises ``KeyError`` on both queries, while a
  directly sealed version answers ``derivation`` with an empty record
  ``{}`` and no error;
* every rejected call is strictly read-only: the persisted manifest,
  material bytes and journals neither grow nor shrink, and the in-memory
  snapshot is unchanged;
* legal reads come back byte-for-byte identical for every stored version.

The write-side vocabulary (duplicate revocation, repointing, ...) is
covered by the main suite and is intentionally not redefined here.  Every
case works in its own temporary directory, uses the standard library only
and is independent of execution order::

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


class QueryValidationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.root

    def open_vault(self) -> Vault:
        # Tracked so the single fixture cleanup returns the lock handle and
        # removes the temporary tree however the case ends.
        return self.fixture.open()


# Values that are never genuine ints, including ones that compare equal to
# real sealed versions.
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


class TestQueryExceptionOrder(QueryValidationTestCase):
    """Empty id -> ValueError beats the version type check on both reads."""

    def test_empty_id_with_float_or_bool_version_is_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        for bad in (1.0, 2.5, -1.0, True, False, "1", _IntLike(1)):
            with self.assertRaises(ValueError, msg=f"derivation {bad!r}"):
                vault.derivation("", bad)
            with self.assertRaises(ValueError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("", bad)

    def test_derivation_without_version_empty_id_matches_explicit_version(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        # Omitting the version must surface exactly the same empty-id
        # ValueError as passing an explicit (even illegal) version.
        with self.assertRaises(ValueError) as omitted:
            vault.derivation("")
        with self.assertRaises(ValueError) as explicit_int:
            vault.derivation("", 1)
        with self.assertRaises(ValueError) as explicit_float:
            vault.derivation("", 1.0)
        self.assertIs(type(omitted.exception), type(explicit_int.exception))
        self.assertIs(type(omitted.exception), type(explicit_float.exception))
        self.assertEqual(
            str(omitted.exception), str(explicit_int.exception)
        )

    def test_empty_id_precedes_none_version_on_is_revoked(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        # Checking the version type first would turn this into TypeError;
        # the id check wins on both queries.
        with self.assertRaises(ValueError):
            vault.is_revoked("", None)
        with self.assertRaises(ValueError):
            vault.derivation("", None)

    def test_empty_id_value_error_is_not_type_or_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        for query in (
            lambda: vault.derivation(""),
            lambda: vault.derivation("", 1.0),
            lambda: vault.is_revoked("", 1),
            lambda: vault.is_revoked("", None),
        ):
            with self.assertRaises(ValueError) as caught:
                query()
            self.assertNotIsInstance(caught.exception, TypeError)
            self.assertNotIsInstance(caught.exception, KeyError)


class TestQueryNonIntegerVersion(QueryValidationTestCase):
    """A non-genuine-integer version is TypeError on both query entries."""

    def test_non_integer_version_raises_type_error_on_both_queries(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.seal("k", b"second")
        vault.derive_seal("k", b"pw", b"salt", 100, 16)
        for bad in NON_INTEGER_VERSIONS:
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("k", bad)
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("k", bad)

    def test_float_equal_to_a_real_version_cannot_pass(self):
        # The core regression: 1.0 == 1 must not read version 1 through
        # either query just because the values compare equal.
        vault = self.open_vault()
        vault.seal("k", b"v1")
        self.assertTrue(1.0 == 1)  # the equality that must not admit it
        with self.assertRaises(TypeError):
            vault.derivation("k", 1.0)
        with self.assertRaises(TypeError):
            vault.is_revoked("k", 1.0)
        with self.assertRaises(TypeError):
            vault.derivation("k", True)
        with self.assertRaises(TypeError):
            vault.is_revoked("k", True)
        # ...while the genuine integer answers both entries normally.
        self.assertEqual(vault.derivation("k", 1), {})
        self.assertFalse(vault.is_revoked("k", 1))

    def test_non_int_version_rejected_before_unknown_key_lookup(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        # Entry validation fires before existence on both reads: a bad
        # version type on an unknown key is TypeError, not KeyError.
        for bad in (1.0, 2.5, True, False, "1", _IntLike(1)):
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("never-sealed", bad)
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("never-sealed", bad)

    def test_is_revoked_none_version_is_an_illegal_version(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        # is_revoked has no active-version sentinel: None is a non-int
        # version there, on known and unknown (but non-empty) keys.
        with self.assertRaises(TypeError):
            vault.is_revoked("k", None)
        with self.assertRaises(TypeError):
            vault.is_revoked("never-sealed", None)
        # None stays the legal active sentinel of the derivation query.
        self.assertEqual(vault.derivation("k", None), {})
        self.assertEqual(
            vault.derivation("d", None),
            {"salt": b"salt", "iterations": 100, "length": 16},
        )

    def test_error_message_is_the_shared_version_message(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        with self.assertRaises(TypeError) as derivation_error:
            vault.derivation("k", 1.0)
        with self.assertRaises(TypeError) as revoked_error:
            vault.is_revoked("k", 1.0)
        self.assertEqual(str(derivation_error.exception), "version must be an int")
        self.assertEqual(str(revoked_error.exception), "version must be an int")


class TestQueryExistenceAndRecords(QueryValidationTestCase):
    """After entry validation, KeyError governs unknown keys/versions."""

    def test_unknown_key_or_missing_version_raises_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)

        with self.assertRaises(KeyError):
            vault.derivation("ghost")
        with self.assertRaises(KeyError):
            vault.derivation("ghost", 1)
        with self.assertRaises(KeyError):
            vault.is_revoked("ghost", 1)

        for missing in (0, 2, 3, 99, -1):
            with self.assertRaises(KeyError, msg=f"derivation k {missing}"):
                vault.derivation("k", missing)
            with self.assertRaises(KeyError, msg=f"is_revoked k {missing}"):
                vault.is_revoked("k", missing)
        # Version 2 exists only on "k"; the derived key has just version 1.
        with self.assertRaises(KeyError):
            vault.derivation("d", 2)
        with self.assertRaises(KeyError):
            vault.is_revoked("d", 2)

    def test_directly_sealed_version_returns_empty_derivation_record(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("k", b"pw", b"salty", 1000, 24)
        params = {"salt": b"salty", "iterations": 1000, "length": 24}
        # The directly sealed version answers with an empty record; the
        # derived one reports its parameters explicitly and through the
        # active sentinel (the derived version is the newest seal).
        self.assertEqual(vault.derivation("k", 1), {})
        self.assertEqual(vault.derivation("k", 2), params)
        self.assertEqual(vault.derivation("k"), params)
        self.assertEqual(vault.derivation("k", None), params)
        # And both answers survive a reload.
        vault.reload()
        self.assertEqual(vault.derivation("k", 1), {})
        self.assertEqual(vault.derivation("k", 2), params)

    def test_is_revoked_status_unchanged_by_failed_queries(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertFalse(vault.is_revoked("k", 2))


class TestRejectedQueriesAreReadOnly(QueryValidationTestCase):
    def _snapshot_disk(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def test_rejected_queries_touch_neither_disk_nor_snapshot(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        vault.revoke("k", 1)

        files_before = self._snapshot_disk()
        mtimes_before = {
            rel: (self.root / rel).stat().st_mtime_ns
            for rel in files_before
        }
        manifest_before = vault.manifest()

        rejected = (
            lambda: vault.derivation("", 1.0),
            lambda: vault.derivation("", True),
            lambda: vault.derivation("", None),
            lambda: vault.derivation("", 1),
            lambda: vault.is_revoked("", 1.0),
            lambda: vault.is_revoked("", None),
            lambda: vault.derivation("k", 1.0),
            lambda: vault.derivation("k", True),
            lambda: vault.derivation("k", "1"),
            lambda: vault.is_revoked("k", 2.5),
            lambda: vault.is_revoked("k", None),
            lambda: vault.derivation("never-sealed", 1.0),
            lambda: vault.is_revoked("never-sealed", 1.0),
            lambda: vault.derivation("ghost", 1),
            lambda: vault.is_revoked("ghost", 1),
            lambda: vault.derivation("k", 99),
            lambda: vault.is_revoked("k", 99),
        )
        for call in rejected:
            with self.assertRaises((ValueError, TypeError, KeyError)):
                call()

        # Disk: no file added, removed, rewritten or even retouched.
        self.assertEqual(sorted(self._snapshot_disk()), sorted(files_before))
        for rel, data in files_before.items():
            path = self.root / rel
            self.assertEqual(path.read_bytes(), data, rel)
            self.assertEqual(path.stat().st_mtime_ns, mtimes_before[rel], rel)

        # In-memory snapshot: manifest, version lists, active pointer,
        # revocation markers and derivation records are all unchanged.
        self.assertEqual(vault.manifest(), manifest_before)
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.versions("d"), [1])
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.revoked_versions("k"), [1])
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertFalse(vault.is_revoked("k", 2))
        self.assertEqual(vault.derivation("k", 1), {})
        self.assertEqual(
            vault.derivation("d", 1),
            {"salt": b"salt", "iterations": 100, "length": 16},
        )

        # A full reload stays healthy and sees exactly the same state.
        vault.reload()
        self.assertTrue(vault.is_revoked("k", 1))
        self.assertEqual(vault.derivation("k", 1), {})
        self.assertEqual(
            vault.derivation("d", 1),
            {"salt": b"salt", "iterations": 100, "length": 16},
        )

    def test_existing_material_still_reads_back_byte_for_byte(self):
        vault = self.open_vault()
        payloads = [b"", bytes(range(256)), b"\x00\xff\n\r suffix",
                    "ünïcode-α".encode("utf-8")]
        for payload in payloads:
            vault.seal("k", payload)
        password, salt, iterations, length = b"pw", b"salty", 1000, 32
        derived_version = vault.derive_seal(
            "d", password, salt, iterations, length
        )
        expected = hashlib.pbkdf2_hmac(
            "sha256", password, salt, iterations, dklen=length
        )

        # Bang at both queries with every rejection shape.
        for bad in NON_INTEGER_VERSIONS:
            with self.assertRaises(TypeError):
                vault.derivation("k", bad)
            with self.assertRaises(TypeError):
                vault.is_revoked("k", bad)
        with self.assertRaises(ValueError):
            vault.derivation("")
        with self.assertRaises(KeyError):
            vault.derivation("ghost")

        # Every byte sealed is still returned exactly as written.
        for index, payload in enumerate(payloads, start=1):
            self.assertIs(type(vault.load("k", index)), bytes)
            self.assertEqual(vault.load("k", index), payload)
        self.assertEqual(vault.load("k"), payloads[-1])
        self.assertEqual(vault.load("d", derived_version), expected)


class TestRejectedQueriesAreDeterministic(QueryValidationTestCase):
    def test_repeating_the_same_inputs_gives_the_same_exception_types(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)

        calls = (
            ("derivation empty", lambda: vault.derivation("")),
            ("derivation empty+float", lambda: vault.derivation("", 1.0)),
            ("is_revoked empty+None", lambda: vault.is_revoked("", None)),
            ("derivation float", lambda: vault.derivation("k", 1.0)),
            ("is_revoked None", lambda: vault.is_revoked("k", None)),
            ("is_revoked bool", lambda: vault.is_revoked("k", True)),
            ("derivation unknown+float",
             lambda: vault.derivation("ghost", 1.0)),
            ("derivation unknown key", lambda: vault.derivation("ghost", 1)),
            ("is_revoked unknown key", lambda: vault.is_revoked("ghost", 1)),
            ("derivation missing version", lambda: vault.derivation("k", 99)),
            ("is_revoked missing version", lambda: vault.is_revoked("k", 99)),
        )

        def run() -> list[type[BaseException]]:
            seen = []
            for _label, call in calls:
                try:
                    call()
                except Exception as exc:  # noqa: BLE001 - type is the result
                    seen.append(type(exc))
                else:
                    seen.append(type(None))
            return seen

        first = run()
        second = run()
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            [
                ValueError,   # derivation empty id, version omitted
                ValueError,   # empty id beats the float
                ValueError,   # empty id beats None on is_revoked
                TypeError,    # valid id + float version
                TypeError,    # is_revoked has no None sentinel
                TypeError,    # bool is not a genuine int
                TypeError,    # version type beats unknown-key lookup
                KeyError,     # genuine int, unknown key
                KeyError,
                KeyError,     # genuine int, version never sealed
                KeyError,
            ],
        )


if __name__ == "__main__":
    unittest.main()
