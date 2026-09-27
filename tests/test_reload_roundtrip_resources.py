"""Tail-hygiene regression for the reload failure -> recovery round trip.

The round-trip correspondence cases themselves live in
``test_reload_roundtrip_correspondence``.  This module pins the closing
requirement of the task: once everything is drained the tail carries no
unclosed-resource machinery text -- no process left running, no thread
undrained, no lock handle left open (a failed cold opener included).

The check is kept in its own file deliberately: it runs the sibling
module in a *fresh interpreter* with ``ResourceWarning`` promoted to an
error.  Living inside the module under test would make that subprocess
discover and spawn itself.  The child runs the module by name, so there
is exactly one level, and it uses the shared scrubbed environment::

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest

from tests._fixtures import REPO_ROOT, cli_env

# Signatures CPython itself appends when a resource (here the vault lock
# file handle) reaches shutdown without being closed, or a worker is left
# running/draining.
_WARNING_RE = re.compile(
    r"resourcewarning|unclosed|exception ignored in|unraisablehook|"
    r"still running|dangling",
    re.IGNORECASE,
)


class TestRoundtripTailHasNoResourceWarnings(unittest.TestCase):
    def test_roundtrip_module_is_clean_under_error_resource_warning(self):
        process = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::ResourceWarning",
                "-m",
                "unittest",
                "tests.test_reload_roundtrip_correspondence",
            ],
            cwd=REPO_ROOT,
            env=cli_env(),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            process.returncode,
            0,
            msg=(
                "round-trip module failed under error::ResourceWarning\n"
                f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            ),
        )
        # The tail carries neither a promoted warning nor the machinery
        # text CPython appends for a leaked handle or running worker.
        self.assertFalse(
            _WARNING_RE.search(process.stderr),
            f"warning machinery leaked into stderr: {process.stderr!r}",
        )
        self.assertIn("OK", process.stderr)

    def test_roundtrip_module_is_clean_under_always_resource_warning(self):
        # Under the non-erroring "always" policy a leaked handle would not
        # fail the run but would still print a ResourceWarning line; the
        # tail must stay empty of it.
        process = subprocess.run(
            [
                sys.executable,
                "-W",
                "always::ResourceWarning",
                "-m",
                "unittest",
                "tests.test_reload_roundtrip_correspondence",
            ],
            cwd=REPO_ROOT,
            env=cli_env(),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            process.returncode,
            0,
            msg=(
                "round-trip module failed under always::ResourceWarning\n"
                f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            ),
        )
        self.assertFalse(
            _WARNING_RE.search(process.stderr),
            f"warning machinery leaked into stderr: {process.stderr!r}",
        )
        self.assertIn("OK", process.stderr)


if __name__ == "__main__":
    unittest.main()
