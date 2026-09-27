"""Explicit regression tests pinning the entry-point exception contract.

The baseline vault already implements every entry point; this module adds
no product behaviour.  It nails down, one case per rule, *which* exception
type each bad input raises and the order in which the checks fire:

* an empty key id raises ``ValueError`` at every entry that takes a key
  id -- ``seal``, ``derive_seal``, ``load``, ``derivation``, ``active``,
  ``revoke``, ``is_revoked``, ``revoked_versions`` and ``set_active`` --
  and that check runs first, ahead of the version type check, the material
  type check and the passphrase/salt/parameter checks (the one read with
  no id gate, ``versions``, treats the empty id like an unknown key and
  returns an empty list, which is pinned as baseline behaviour);

* a version that is not a genuine integer raises ``TypeError`` at
  ``load``, ``derivation``, ``is_revoked``, ``revoke`` and
  ``set_active``: floats and bools do not count (an ``int`` subclass does
  not either), and a float numerically equal to an existing version is
  rejected just the same; the type check precedes the key/version
  existence lookup;

* with a genuine integer, an unknown key or nonexistent version raises
  ``KeyError`` with one consistent vocabulary across the material read,
  the derivation query, the revocation query and repointing;

* material that is not bytes-like raises ``TypeError`` at ``seal``; a
  passphrase or salt that is not a genuine ``bytes`` value (so neither
  ``bytearray`` nor ``memoryview``) raises ``TypeError`` at
  ``derive_seal``; an empty salt raises ``ValueError``; an iteration count
  or length that is not a genuine integer raises ``TypeError`` and one that
  is not positive raises ``ValueError``;

* revoking the same version twice and repointing at the version that is
  already active (or at a revoked version) raises ``ValueError``; a failed
  call appends not even half a record -- in particular the very first,
  failed repoint never creates ``activations.jsonl``;

* the revoked-version listing answers an unknown key with an empty list
  (only the empty id is a ``ValueError``);

* the whole matrix is deterministic: the same calls in the same order
  raise the same exception types run after run, the in-memory snapshot is
  unchanged and the on-disk manifest, journals and materials neither gain
  nor lose a byte.

Everything happens inside temporary directories, uses the standard
library only and is independent of execution order::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

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


# An int subclass is not a *genuine* int: the entry check uses
# ``type(x) is int``, so it is rejected exactly like a float or bool.
class _IntLike(int):
    pass


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


class ExceptionContractTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VaultFixture(self)
        self.tmp_path = self.fixture.tmp_path
        self.root = self.fixture.root

    def open_vault(self) -> Vault:
        # Tracked so the single fixture cleanup returns the lock handle and
        # removes the temporary tree however the case ends.
        return self.fixture.open()

    def _rich_vault(self) -> Vault:
        """A vault carrying state every conflict case can lean on.

        Key ``k`` has three versions, version 1 revoked and the active
        pointer at the newest version 3; key ``d`` has one derived version;
        key ``plain`` has two versions and is repointed at version 1.
        """
        vault = self.open_vault()
        vault.seal("k", b"k-v1")
        vault.seal("k", b"k-v2")
        vault.seal("k", b"k-v3")
        vault.revoke("k", 1)
        vault.derive_seal("d", b"pw", b"salty", 100, 16)
        vault.seal("plain", b"p1")
        vault.seal("plain", b"p2")
        vault.set_active("plain", 1)
        return vault

    def _disk_records(self) -> dict[Path, bytes]:
        return {
            path: path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file() and path.name != LOCK_NAME
        }


class TestEmptyKeyIdIsValueErrorAndCheckedFirst(ExceptionContractTestCase):
    def test_empty_id_is_value_error_at_every_entry(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.seal("", b"m")
        with self.assertRaises(ValueError):
            vault.derive_seal("", b"pw", b"salt", 1, 1)
        with self.assertRaises(ValueError):
            vault.load("", 1)
        with self.assertRaises(ValueError):
            vault.derivation("", 1)
        with self.assertRaises(ValueError):
            vault.revoke("", 1)
        with self.assertRaises(ValueError):
            vault.is_revoked("", 1)
        with self.assertRaises(ValueError):
            vault.revoked_versions("")
        with self.assertRaises(ValueError):
            vault.set_active("", 1)
        with self.assertRaises(ValueError):
            vault.active("")
        # ``versions`` is the one read with no empty-id gate: the empty id
        # is an unknown key there and answers [], exactly like an unknown
        # non-empty id (pinned below).

    def test_versions_has_no_entry_validation_and_treats_empty_id_as_unknown(self):
        # ``versions`` performs no key-id validation: an empty (or any
        # unknown) id is simply absent from the snapshot and so answers an
        # empty list, exactly like an unknown non-empty id.  This is the
        # baseline behaviour, pinned as-is.
        vault = self.open_vault()
        vault.seal("k", b"m")
        self.assertEqual(vault.versions(""), [])
        self.assertEqual(vault.versions("never-sealed"), [])

    def test_active_empty_id_is_value_error_like_the_other_reads(self):
        # ``active`` shares the entry validation of every other read: the
        # empty id fails with ValueError at the entry before the snapshot is
        # consulted, while an unknown non-empty id stays a KeyError.
        vault = self.open_vault()
        vault.seal("k", b"m")
        with self.assertRaises(ValueError):
            vault.active("")
        with self.assertRaises(KeyError):
            vault.active("never-sealed")

    def test_empty_id_value_error_precedes_version_type_check(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        # Whatever the version argument looks like -- float, bool, string,
        # the None sentinel, a genuine int -- the empty id is judged first.
        for bad_version in (1.0, 2.5, True, False, "1", (1,), None, 1, 0):
            with self.assertRaises(ValueError, msg=f"load {bad_version!r}"):
                vault.load("", bad_version)
            with self.assertRaises(ValueError, msg=f"deriv {bad_version!r}"):
                vault.derivation("", bad_version)
            with self.assertRaises(ValueError, msg=f"revoked {bad_version!r}"):
                vault.is_revoked("", bad_version)
            with self.assertRaises(ValueError, msg=f"revoke {bad_version!r}"):
                vault.revoke("", bad_version)
            with self.assertRaises(ValueError, msg=f"repoint {bad_version!r}"):
                vault.set_active("", bad_version)

    def test_empty_id_value_error_precedes_material_and_derivation_types(self):
        vault = self.open_vault()
        # The id fails before the material type is even looked at.
        with self.assertRaises(ValueError):
            vault.seal("", "not-bytes")
        with self.assertRaises(ValueError):
            vault.seal("", None)
        # The id fails before the passphrase/salt/parameter checks.
        with self.assertRaises(ValueError):
            vault.derive_seal("", "not-bytes-pw", "", True, 0)
        with self.assertRaises(ValueError):
            vault.derive_seal("", None, None, None, None)

    def test_empty_id_message_is_the_shared_message(self):
        vault = self.open_vault()
        for call in (
            lambda: vault.seal("", b"m"),
            lambda: vault.load("", 1),
            lambda: vault.revoke("", 1),
            lambda: vault.set_active("", 1),
            lambda: vault.derive_seal("", b"pw", b"salt", 1, 1),
            lambda: vault.active(""),
        ):
            with self.assertRaises(ValueError) as caught:
                call()
            self.assertEqual(str(caught.exception), "key_id must not be empty")


class TestNonIntegerVersionIsTypeError(ExceptionContractTestCase):
    def test_all_versioned_entries_reject_non_integer_versions(self):
        vault = self._rich_vault()
        for bad in NON_INTEGER_VERSIONS:
            with self.assertRaises(TypeError, msg=f"load {bad!r}"):
                vault.load("k", bad)
            with self.assertRaises(TypeError, msg=f"derivation {bad!r}"):
                vault.derivation("k", bad)
            with self.assertRaises(TypeError, msg=f"is_revoked {bad!r}"):
                vault.is_revoked("k", bad)
            with self.assertRaises(TypeError, msg=f"revoke {bad!r}"):
                vault.revoke("k", bad)
            with self.assertRaises(TypeError, msg=f"set_active {bad!r}"):
                vault.set_active("k", bad)

    def test_float_equal_to_a_real_version_is_still_rejected(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        # The equality that would otherwise admit the read must not matter.
        self.assertTrue(1.0 == 1)
        self.assertTrue(True == 1)
        for entry in ("load", "derivation", "is_revoked", "revoke", "set_active"):
            with self.assertRaises(TypeError, msg=entry):
                getattr(vault, entry)("k", 1.0)
            with self.assertRaises(TypeError, msg=f"{entry}-bool"):
                getattr(vault, entry)("k", True)
        # Genuine ints reach the records and answer normally.
        self.assertEqual(vault.load("k", 1), b"v1")
        self.assertFalse(vault.is_revoked("k", 1))
        self.assertEqual(vault.derivation("k", 1), {})

    def test_version_type_check_precedes_key_existence(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        # A bad version type on an unknown key is a TypeError, never the
        # KeyError the unknown key alone would raise.
        for entry in ("load", "derivation", "is_revoked", "revoke", "set_active"):
            with self.assertRaises(TypeError, msg=f"{entry} unknown"):
                getattr(vault, entry)("never-sealed", 1.0)
            with self.assertRaises(TypeError, msg=f"{entry} bool unknown"):
                getattr(vault, entry)("never-sealed", True)

    def test_shared_version_type_message(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        for entry in ("load", "derivation", "is_revoked", "revoke", "set_active"):
            with self.assertRaises(TypeError) as caught:
                getattr(vault, entry)("k", 1.0)
            self.assertEqual(str(caught.exception), "version must be an int")


class TestUnknownKeyOrVersionIsKeyError(ExceptionContractTestCase):
    def test_unknown_key_raises_key_error_at_every_lookup(self):
        vault = self._rich_vault()
        with self.assertRaises(KeyError):
            vault.load("never-sealed")
        with self.assertRaises(KeyError):
            vault.load("never-sealed", 1)
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed")
        with self.assertRaises(KeyError):
            vault.derivation("never-sealed", 1)
        with self.assertRaises(KeyError):
            vault.active("never-sealed")
        with self.assertRaises(KeyError):
            vault.is_revoked("never-sealed", 1)
        with self.assertRaises(KeyError):
            vault.revoke("never-sealed", 1)
        with self.assertRaises(KeyError):
            vault.set_active("never-sealed", 1)

    def test_missing_version_raises_key_error_with_one_consistent_rule(self):
        vault = self._rich_vault()
        # ``k`` holds versions 1..3, ``d`` holds one derived version.
        for missing in (0, 4, 99, -1):
            with self.assertRaises(KeyError, msg=f"load {missing}"):
                vault.load("k", missing)
            with self.assertRaises(KeyError, msg=f"deriv {missing}"):
                vault.derivation("k", missing)
            with self.assertRaises(KeyError, msg=f"is_revoked {missing}"):
                vault.is_revoked("k", missing)
            with self.assertRaises(KeyError, msg=f"revoke {missing}"):
                vault.revoke("k", missing)
            with self.assertRaises(KeyError, msg=f"set_active {missing}"):
                vault.set_active("k", missing)
        for missing in (0, 2, 99):
            with self.assertRaises(KeyError):
                vault.derivation("d", missing)

    def test_genuine_int_is_required_before_the_key_error_path(self):
        vault = self._rich_vault()
        # Sanity pinning the boundary: genuine int -> KeyError, equal-value
        # float -> TypeError, at each of the four consistent entries.
        for entry in ("load", "derivation", "is_revoked"):
            with self.assertRaises(KeyError):
                getattr(vault, entry)("ghost", 1)
            with self.assertRaises(TypeError):
                getattr(vault, entry)("ghost", 1.0)


class TestMaterialPasswordSaltAndParameterTypes(ExceptionContractTestCase):
    def test_seal_material_must_be_bytes_like(self):
        vault = self.open_vault()
        for bad in ("text", 1, 1.5, None, [b"x"], {"k": b"v"}, object()):
            with self.assertRaises(TypeError, msg=repr(bad)):
                vault.seal("k", bad)
        # Genuine bytes-like buffers are admitted and read back exact.
        self.assertEqual(vault.seal("k", bytearray(b"array")), 1)
        self.assertEqual(vault.seal("k", memoryview(b"view")), 2)
        self.assertEqual(vault.load("k", 1), b"array")
        self.assertEqual(vault.load("k", 2), b"view")

    def test_password_and_salt_must_be_genuine_bytes(self):
        vault = self.open_vault()
        bad_values = ("text", 1, None, [b"x"], bytearray(b"x"), memoryview(b"x"))
        for bad in bad_values:
            with self.assertRaises(TypeError, msg=f"password {bad!r}"):
                vault.derive_seal("k", bad, b"salt", 1, 1)
            with self.assertRaises(TypeError, msg=f"salt {bad!r}"):
                vault.derive_seal("k", b"pw", bad, 1, 1)
        # Genuine bytes pass the type gate.
        self.assertEqual(vault.derive_seal("k", b"pw", b"salt", 1, 1), 1)

    def test_empty_salt_is_value_error_distinct_from_type_error(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"", 1, 1)
        # A non-bytes empty-ish salt is still the type error first.
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", "", 1, 1)
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", bytearray(), 1, 1)

    def test_iterations_and_length_non_integer_are_type_errors(self):
        vault = self.open_vault()
        for bad in (True, False, 1.0, 2.5, "1", None, (1,), _IntLike(1)):
            with self.assertRaises(TypeError, msg=f"iterations {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(TypeError, msg=f"length {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)

    def test_iterations_and_length_non_positive_are_value_errors(self):
        vault = self.open_vault()
        for bad in (0, -1, -1000):
            with self.assertRaises(ValueError, msg=f"iterations {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", bad, 1)
            with self.assertRaises(ValueError, msg=f"length {bad!r}"):
                vault.derive_seal("k", b"pw", b"salt", 1, bad)

    def test_parameter_check_order_is_pinned(self):
        vault = self.open_vault()
        # Password is judged before salt: a non-bytes password with an
        # empty salt is the password TypeError, not the salt ValueError.
        with self.assertRaises(TypeError):
            vault.derive_seal("k", "pw", b"", 1, 1)
        # Iterations is judged before length: bad iterations type wins over
        # a bad length value, and bad iterations value wins over a bad
        # length type.
        with self.assertRaises(TypeError):
            vault.derive_seal("k", b"pw", b"salt", True, 0)
        with self.assertRaises(ValueError):
            vault.derive_seal("k", b"pw", b"salt", 0, True)

    def test_parameter_error_messages(self):
        vault = self.open_vault()
        with self.assertRaises(TypeError) as caught:
            vault.derive_seal("k", "pw", b"salt", 1, 1)
        self.assertEqual(str(caught.exception), "password must be bytes")
        with self.assertRaises(TypeError) as caught:
            vault.derive_seal("k", b"pw", 1, 1, 1)
        self.assertEqual(str(caught.exception), "salt must be bytes")
        with self.assertRaises(ValueError) as caught:
            vault.derive_seal("k", b"pw", b"", 1, 1)
        self.assertEqual(str(caught.exception), "salt must not be empty")
        with self.assertRaises(TypeError) as caught:
            vault.derive_seal("k", b"pw", b"salt", True, 1)
        self.assertEqual(str(caught.exception), "iterations must be an int")
        with self.assertRaises(ValueError) as caught:
            vault.derive_seal("k", b"pw", b"salt", 0, 1)
        self.assertEqual(str(caught.exception), "iterations must be at least 1")
        with self.assertRaises(TypeError) as caught:
            vault.derive_seal("k", b"pw", b"salt", 1, False)
        self.assertEqual(str(caught.exception), "length must be an int")
        with self.assertRaises(ValueError) as caught:
            vault.derive_seal("k", b"pw", b"salt", 1, 0)
        self.assertEqual(str(caught.exception), "length must be at least 1")


class TestConflictValueErrorsLeaveNoRecord(ExceptionContractTestCase):
    def test_duplicate_revocation_is_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        # Repeating again is the same ValueError; a distinct version works.
        with self.assertRaises(ValueError):
            vault.revoke("k", 1)
        vault.revoke("k", 2)
        self.assertEqual(vault.revoked_versions("k"), [1, 2])

    def test_repointing_at_current_active_version_is_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        # The newest sealed version is active.
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)
        vault.set_active("k", 1)
        # Now version 1 is active: repointing at it is the same error.
        with self.assertRaises(ValueError):
            vault.set_active("k", 1)
        # Repointing away and back is allowed; only a no-op repoint fails.
        vault.set_active("k", 2)
        vault.set_active("k", 1)
        self.assertEqual(vault.active("k"), 1)

    def test_repointing_at_revoked_version_is_value_error(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        with self.assertRaises(ValueError):
            vault.set_active("k", 1)
        # The pointer stayed at the unrevoked active version 2.
        self.assertEqual(vault.active("k"), 2)
        self.assertEqual(vault.load("k"), b"v2")
        # A non-integer version at that revoked target is still the earlier
        # TypeError, not the revoked-target ValueError.
        with self.assertRaises(TypeError):
            vault.set_active("k", 1.0)

    def test_failed_first_repoint_never_creates_the_journal(self):
        vault = self.open_vault()
        vault.seal("k", b"v1")
        vault.seal("k", b"v2")
        vault.revoke("k", 1)
        journal = self.root / ACTIVATIONS_NAME
        self.assertFalse(journal.exists())
        # Every way the first repoint can fail must leave no journal file.
        for call in (
            lambda: vault.set_active("never-sealed", 1),
            lambda: vault.set_active("k", 99),
            lambda: vault.set_active("k", 1),      # revoked
            lambda: vault.set_active("k", 2),      # already active
            lambda: vault.set_active("", 1),
            lambda: vault.set_active("k", 1.0),
            lambda: vault.set_active("k", True),
        ):
            with self.assertRaises((ValueError, KeyError, TypeError)):
                call()
            self.assertFalse(
                journal.exists(),
                "a failed first repoint created activations.jsonl",
            )
        # The failed calls moved no pointer either; another no-op repoint
        # fails the same way and still creates nothing.
        self.assertEqual(vault.active("k"), 2)
        with self.assertRaises(ValueError):
            vault.set_active("k", 2)
        self.assertFalse(journal.exists())

        # The first *successful* repoint then creates exactly one record:
        # seal version 3 so an unrevoked historical target exists, and
        # repoint at it (bound to the new newest sealed version 3).
        self.assertEqual(vault.seal("k", b"v3"), 3)
        vault.set_active("k", 2)
        self.assertEqual(vault.active("k"), 2)
        self.assertTrue(journal.exists())
        self.assertEqual(
            journal.read_text("utf-8").splitlines(),
            ['{"key_id": "k", "latest": 3, "version": 2}'],
        )

    def test_failed_revoke_and_repoint_append_not_half_a_record(self):
        vault = self._rich_vault()
        revocations = self.root / REVOCATIONS_NAME
        activations = self.root / ACTIVATIONS_NAME
        revocations_before = revocations.read_bytes()
        activations_before = activations.read_bytes()
        manifest_before = (self.root / MANIFEST_NAME).read_bytes()

        for call in (
            lambda: vault.revoke("never-sealed", 1),
            lambda: vault.revoke("k", 99),
            lambda: vault.revoke("k", 1),       # already revoked
            lambda: vault.revoke("", 1),
            lambda: vault.revoke("k", 1.0),
            lambda: vault.set_active("never-sealed", 1),
            lambda: vault.set_active("k", 99),
            lambda: vault.set_active("k", 1),   # revoked
            lambda: vault.set_active("plain", 1),  # already active there
            lambda: vault.set_active("", 1),
            lambda: vault.set_active("k", 1.0),
        ):
            with self.assertRaises((ValueError, KeyError, TypeError)):
                call()

        # Neither journal gained a byte and the manifest is untouched: the
        # failed calls left not even half a record.
        self.assertEqual(revocations.read_bytes(), revocations_before)
        self.assertEqual(activations.read_bytes(), activations_before)
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), manifest_before)
        # Journal content still parses into exactly the whole records that
        # existed before -- no truncated trailing line.
        for line in revocations.read_text("utf-8").splitlines():
            self.assertTrue(line.strip())
        for line in activations.read_text("utf-8").splitlines():
            self.assertTrue(line.strip())


class TestRevokedVersionsListing(ExceptionContractTestCase):
    def test_unknown_key_answers_empty_list(self):
        vault = self.open_vault()
        vault.seal("k", b"m")
        vault.revoke("k", 1)
        self.assertEqual(vault.revoked_versions("never-sealed"), [])
        self.assertEqual(vault.revoked_versions("other-never-sealed"), [])
        # A key that exists but was never revoked is also empty.
        vault.seal("plain", b"m")
        self.assertEqual(vault.revoked_versions("plain"), [])
        # The known revoked listing is unaffected by the empty answers.
        self.assertEqual(vault.revoked_versions("k"), [1])

    def test_only_empty_id_is_an_error_for_the_listing(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.revoked_versions("")


class TestExceptionMatrixIsDeterministicAndReadOnly(ExceptionContractTestCase):
    def test_repeated_bad_calls_raise_the_same_types_in_order(self):
        vault = self._rich_vault()
        calls = (
            lambda: vault.seal("", "x"),
            lambda: vault.seal("k", "not-bytes"),
            lambda: vault.derive_seal("", "pw", b"", True, 0),
            lambda: vault.derive_seal("k", "pw", b"salt", 1, 1),
            lambda: vault.derive_seal("k", b"pw", b"", 1, 1),
            lambda: vault.derive_seal("k", b"pw", b"salt", True, 1),
            lambda: vault.derive_seal("k", b"pw", b"salt", 1, 0),
            lambda: vault.load("", 1.0),
            lambda: vault.load("k", 1.0),
            lambda: vault.load("ghost", 1),
            lambda: vault.load("k", 99),
            lambda: vault.derivation("", None),
            lambda: vault.derivation("ghost"),
            lambda: vault.is_revoked("", None),
            lambda: vault.is_revoked("k", None),
            lambda: vault.is_revoked("ghost", 1),
            lambda: vault.revoke("k", 1),       # duplicate
            lambda: vault.revoke("ghost", 1),
            lambda: vault.revoke("k", 1.0),
            lambda: vault.set_active("k", 1),   # revoked
            lambda: vault.set_active("plain", 1),  # already active
            lambda: vault.set_active("ghost", 1),
            lambda: vault.set_active("k", 99),
            lambda: vault.set_active("k", 1.0),
            lambda: vault.revoked_versions(""),
            lambda: vault.active(""),
        )
        expected = [            ValueError,   # seal empty id (precedes material type)
            TypeError,    # seal non-bytes material
            ValueError,   # derive_seal empty id
            TypeError,    # password not bytes
            ValueError,   # empty salt
            TypeError,    # iterations bool
            ValueError,   # length zero
            ValueError,   # load empty id
            TypeError,    # load float version
            KeyError,     # load unknown key
            KeyError,     # load missing version
            ValueError,   # derivation empty id (None sentinel or not)
            KeyError,     # derivation unknown key
            ValueError,   # is_revoked empty id
            TypeError,    # is_revoked None version
            KeyError,     # is_revoked unknown key
            ValueError,   # duplicate revocation
            KeyError,     # revoke unknown key
            TypeError,    # revoke float version
            ValueError,   # repoint at revoked version
            ValueError,   # repoint at active version
            KeyError,     # repoint unknown key
            KeyError,     # repoint missing version
            TypeError,    # repoint float version
            ValueError,   # revoked listing empty id
            ValueError,   # active empty id, same gate as every other read
        ]

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

        first, second, third = run(), run(), run()
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(third, expected)

    def test_rejected_calls_touch_neither_disk_nor_snapshot(self):
        vault = self._rich_vault()
        before_records = self._disk_records()
        snapshot_before = {
            "k-versions": vault.versions("k"),
            "k-active": vault.active("k"),
            "k-revoked": vault.revoked_versions("k"),
            "plain-versions": vault.versions("plain"),
            "plain-active": vault.active("plain"),
            "plain-revoked": vault.revoked_versions("plain"),
            "d-versions": vault.versions("d"),
            "unknown-listing": vault.revoked_versions("ghost"),
            "k-v1": vault.load("k", 1),
            "k-v2": vault.load("k", 2),
            "k-v3": vault.load("k", 3),
            "plain-v1": vault.load("plain", 1),
            "derivation-d": vault.derivation("d", 1),
            "manifest": vault.manifest(),
        }

        for call in (
            lambda: vault.seal("", "x"),
            lambda: vault.seal("k", 123),
            lambda: vault.derive_seal("", "pw", b"", True, 0),
            lambda: vault.derive_seal("k", bytearray(b"pw"), b"salt", 1, 1),
            lambda: vault.derive_seal("k", b"pw", memoryview(b"s"), 1, 1),
            lambda: vault.derive_seal("k", b"pw", b"salt", 1.0, 1),
            lambda: vault.derive_seal("k", b"pw", b"salt", 1, -1),
            lambda: vault.load("", 1),
            lambda: vault.load("k", True),
            lambda: vault.load("ghost", 1),
            lambda: vault.derivation("ghost", 1.0),
            lambda: vault.is_revoked("ghost", 1),
            lambda: vault.revoke("k", 1),
            lambda: vault.revoke("ghost", 1),
            lambda: vault.set_active("k", 1),
            lambda: vault.set_active("ghost", 1),
            lambda: vault.set_active("plain", 1),
        ):
            with self.assertRaises((ValueError, TypeError, KeyError)):
                call()

        # Same files, same bytes -- the manifest, both journals and the
        # materials neither gained nor lost a byte.
        self.assertEqual(self._disk_records(), before_records)

        # The in-memory snapshot answers word for word what it did before.
        snapshot_after = {
            "k-versions": vault.versions("k"),
            "k-active": vault.active("k"),
            "k-revoked": vault.revoked_versions("k"),
            "plain-versions": vault.versions("plain"),
            "plain-active": vault.active("plain"),
            "plain-revoked": vault.revoked_versions("plain"),
            "d-versions": vault.versions("d"),
            "unknown-listing": vault.revoked_versions("ghost"),
            "k-v1": vault.load("k", 1),
            "k-v2": vault.load("k", 2),
            "k-v3": vault.load("k", 3),
            "plain-v1": vault.load("plain", 1),
            "derivation-d": vault.derivation("d", 1),
            "manifest": vault.manifest(),
        }
        self.assertEqual(snapshot_after, snapshot_before)

        # A healthy reload after the storm keeps every answer identical.
        vault.reload()
        self.assertEqual(vault.versions("k"), [1, 2, 3])
        self.assertEqual(vault.revoked_versions("k"), [1])
        self.assertEqual(vault.active("plain"), 1)
        self.assertEqual(vault.load("k", 3), b"k-v3")


if __name__ == "__main__":
    unittest.main()
