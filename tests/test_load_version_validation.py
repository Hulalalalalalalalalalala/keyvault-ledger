"""Regression tests for version-number validation at the ``load`` entry.

``load`` is the per-version material read; the derivation query
(``derivation``) and the revocation status query (``is_revoked``) already
shared the same entry-point rule, which this module pins for all three
reads together:

* an empty key id raises ``ValueError`` and that check runs first, ahead of
  the version type check;
* a non-genuine-integer version raises ``TypeError`` -- bools and floats do
  not count, and a float numerically equal to an existing version (``1.0``
  vs ``1``) must not slip through just because it compares equal;
* only once the id is a non-empty string and the version is a genuine int
  (or ``None``, the documented "active version" sentinel of ``load`` and
  ``derivation``) does key/version existence matter: an unknown key or a
  version that was never sealed then raises ``KeyError``;
* every rejected call is strictly read-only: the persisted manifest and
  material bytes neither grow nor shrink, and the in-memory snapshot is
  unchanged;
* legal reads come back byte-for-byte identical for every stored version.

The revocation/activation conflict vocabulary (duplicate revocation,
repointing at the active version, ...) is covered by the main suite and is
intentionally not redefined here.  Every case works in its own temporary
directory, uses the standard library only and is independent of execution
order::

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


class LoadValidationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.root

    def open_vault(self) -> Vault:
        # Tracked so the single fixture cleanup returns the lock handle and
        # removes the temporary tree however the case ends.
        return self.fixture.open()


class TestLoadVersionValidation(LoadValidationTestCase):
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

    def test_non_integer_version_raises_type_error_even_when_equal(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        for bad in self.NON_INTEGER_VERSIONS:
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.load("k", bad)

    def test_float_equal_to_a_real_version_cannot_read_it(self):
        # The core regression: 1.0 == 1 used to hit the version dict and
        # return version 1's material.  It must be rejected at the entry.
        vault = self.open_vault()
        vault.seal("k", b"v1")
        self.assertTrue(1.0 == 1)  # the equality that used to admit it
        with self.assertRaises(TypeError):
            vault.load("k", 1.0)
        with self.assertRaises(TypeError):
            vault.load("k", True)
        # ...while the genuine integer keeps reading the exact bytes.
        self.assertEqual(vault.load("k", 1), b"v1")

    def test_none_version_still_resolves_the_active_version(self):
        vault = self.open_vault()
        vault.seal("k", b"old")
        vault.seal("k", b"new")
        self.assertEqual(vault.load("k", None), b"new")
        self.assertEqual(vault.load("k"), b"new")

    def test_empty_key_id_value_error_precedes_version_type(self):
        vault = self.open_vault()
        # The id is checked first, before any state is read: ValueError wins
        # no matter what the version looks like, including a bad type and
        # the None sentinel.
        for version in (1.0, True, "1", (1,), None, 1):
            with self.assertRaises(ValueError, msg=repr(version)):
                vault.load("", version)

    def test_non_int_version_rejected_before_unknown_key_lookup(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        # Entry validation fires before existence, matching the other
        # per-version reads: a bad version type on an unknown key is a
        # TypeError, not a KeyError.
        with self.assertRaises(TypeError):
            vault.load("never-sealed", 1.0)
        with self.assertRaises(TypeError):
            vault.load("never-sealed", True)

    def test_genuine_int_unknown_key_or_version_still_raises_key_error(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(KeyError):
            vault.load("never-sealed")
        with self.assertRaises(KeyError):
            vault.load("never-sealed", 1)
        for bad_version in (0, 2, -1, 99):
            with self.assertRaises(KeyError, msg=repr(bad_version)):
                vault.load("k", bad_version)

    def test_error_message_is_the_shared_version_message(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(TypeError) as caught:
            vault.load("k", 1.0)
        self.assertEqual(str(caught.exception), "version must be an int")


class TestRejectedLoadIsReadOnly(LoadValidationTestCase):
    def test_rejected_loads_touch_neither_disk_nor_snapshot(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)

        manifest = self.root / "manifest.json"
        manifest_bytes = manifest.read_bytes()
        manifest_mtime = manifest.stat().st_mtime_ns
        material_bytes = {
            path: path.read_bytes()
            for path in (self.root / "materials").rglob("*.bin")
        }

        rejected = (
            lambda: vault.load("", 1.0),
            lambda: vault.load("", None),
            lambda: vault.load("k", 1.0),
            lambda: vault.load("k", True),
            lambda: vault.load("k", 2.5),
            lambda: vault.load("k", "1"),
            lambda: vault.load("never-sealed", 1.0),
            lambda: vault.load("never-sealed", 1),
            lambda: vault.load("k", 99),
        )
        for call in rejected:
            with self.assertRaises((ValueError, TypeError, KeyError)):
                call()

        # Disk: the manifest and every material file are byte-identical and
        # were not rewritten.
        self.assertEqual(manifest.read_bytes(), manifest_bytes)
        self.assertEqual(manifest.stat().st_mtime_ns, manifest_mtime)
        for path, before in material_bytes.items():
            self.assertEqual(path.read_bytes(), before)

        # In-memory snapshot: versions, active pointer and readback are
        # exactly what they were before the rejected calls.
        self.assertEqual(vault.versions("k"), [1, 2])
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"v2")
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.load("k", 2), b"v2")

        # A full reload stays healthy and reads the same bytes.
        vault.reload()
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertEqual(vault.load("k", 2), b"v2")

    def test_repeated_bad_calls_are_deterministic(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        first, second = [], []
        for call in (
            lambda: vault.load("", 1.0),
            lambda: vault.load("k", 1.0),
            lambda: vault.load("k", 99),
            lambda: vault.load("ghost", 1),
        ):
            for sink in (first, second):
                try:
                    call()
                except Exception as exc:  # noqa: BLE001 - type is the result
                    sink.append(type(exc))
        self.assertEqual(first, second)
        self.assertEqual(
            first, [ValueError, TypeError, KeyError, KeyError]
        )


class TestLegalReadbackByteExact(LoadValidationTestCase):
    def test_every_version_reads_back_byte_for_byte(self):
        vault = self.open_vault()
        payloads = [
            b"",
            bytes(range(256)),
            b"\x00\xff\n\r plain suffix",
            "ünïcode-α".encode("utf-8"),
        ]
        for payload in payloads:
            vault.seal("k", payload)
        for index, payload in enumerate(payloads, start=1):
            self.assertIs(type(vault.load("k", index)), bytes)
            self.assertEqual(vault.load("k", index), payload)
        # The active (unversioned) read is the last sealed payload.
        self.assertEqual(vault.load("k"), payloads[-1])

    def test_derived_version_reads_back_as_pbkdf2_output(self):
        vault = self.open_vault()
        password, salt, iterations, length = b"pw", b"salty", 1000, 32
        version = vault.derive_seal("k", password, salt, iterations, length)
        expected = hashlib.pbkdf2_hmac(
            "sha256", password, salt, iterations, dklen=length
        )
        self.assertEqual(vault.load("k", version), expected)
        # Repeated reads return independent equal bytes, never mutated.
        first = vault.load("k", version)
        second = vault.load("k", version)
        self.assertEqual(first, second)
        self.assertEqual(first, expected)

    def test_separate_keys_and_versions_do_not_interfere(self):
        vault = self.open_vault()
        vault.seal("a", b"a-one")
        vault.seal("b", b"b-one")
        vault.seal("a", b"a-two")
        self.assertEqual(vault.load("a", 1), b"a-one")
        self.assertEqual(vault.load("a", 2), b"a-two")
        self.assertEqual(vault.load("b", 1), b"b-one")


class TestPerVersionQueryTypeMatrix(LoadValidationTestCase):
    """Pin the exception type for the three per-version reads together."""

    # Values that are never genuine ints.
    BAD_VERSIONS = (1.0, 2.5, True, False, "1", (1,), _IntLike(1))

    def test_load_derivation_is_revoked_reject_non_int_versions(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("k", b"pw", b"salt", 100, 16)

        for bad in self.BAD_VERSIONS:
            with self.assertRaises(TypeError, msg=f"load {bad!r}"):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("k", bad)
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("k", bad)

    def test_empty_id_value_error_precedes_version_for_every_read(self):
        vault = self.open_vault()
        # A bad version must not turn the empty-id ValueError into TypeError.
        # Every call dies at the entry on the empty id, so no key has to
        # exist for any of these assertions.
        for bad in (1.0, True, "1"):
            with self.assertRaises(ValueError, msg=f"load {bad!r}"):
                vault.load("", bad)
            with self.assertRaises(ValueError, msg=f"derivation {bad!r}"):
                vault.derivation("", bad)
            with self.assertRaises(ValueError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("", bad)

    def test_unknown_key_or_version_is_key_error_for_genuine_ints(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)

        with self.assertRaises(KeyError):
            vault.load("ghost")
        with self.assertRaises(KeyError):
            vault.derivation("ghost")
        with self.assertRaises(KeyError):
            vault.is_revoked("ghost", 1)

        for bad_version in (0, 3, 99):
            with self.assertRaises(KeyError, msg=f"load {bad_version}"):
                vault.load("k", bad_version)
            with self.assertRaises(KeyError, msg=f"derivation {bad_version}"):
                vault.derivation("k", bad_version)
            with self.assertRaises(KeyError, msg=f"is_revoked {bad_version}"):
                vault.is_revoked("k", bad_version)

    def test_active_sentinel_and_normal_reads_keep_working(self):
        vault = self.open_vault()
        vault.seal("k", b"plain")
        vault.derive_seal("d", b"pw", b"salt", 100, 16)
        # None means "active version" for load and derivation only;
        # is_revoked requires an explicit version and rejects None.
        self.assertEqual(vault.load("k", None), b"plain")
        self.assertEqual(vault.derivation("k", None), {})
        self.assertEqual(
            vault.derivation("d", None),
            {"salt": b"salt", "iterations": 100, "length": 16},
        )
        with self.assertRaises(TypeError):
            vault.is_revoked("k", None)
        self.assertFalse(vault.is_revoked("k", 1))
        self.assertTrue(vault.derivation("d", 1))


if __name__ == "__main__":
    unittest.main()
