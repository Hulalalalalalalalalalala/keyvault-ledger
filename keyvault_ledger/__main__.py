"""Command line interface for keyvault_ledger.

    python3 -m keyvault_ledger --root ./vault versions
    python3 -m keyvault_ledger --root ./vault seal <key-id> --material-file <path>
    python3 -m keyvault_ledger --root ./vault reload
"""

from __future__ import annotations

import argparse
import sys

from .vault import Vault


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="keyvault_ledger")
    parser.add_argument("--root", required=True, help="vault directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("versions", help="list keys, versions and active version")

    seal = sub.add_parser("seal", help="seal key material as a new version")
    seal.add_argument("key_id")
    seal.add_argument(
        "--material-file", required=True, help="file holding the key material"
    )

    sub.add_parser("reload", help="re-read and validate the whole keyring")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    vault: Vault | None = None
    try:
        vault = Vault(args.root)
        if args.command == "versions":
            manifest = vault.manifest()
            for key_id in sorted(manifest["keys"]):
                entry = manifest["keys"][key_id]
                versions = ",".join(
                    str(record["version"]) for record in entry["versions"]
                )
                print(f"{key_id}\tactive={entry['active']}\tversions={versions}")
            return 0
        if args.command == "seal":
            with open(args.material_file, "rb") as fh:
                material = fh.read()
            print(vault.seal(args.key_id, material))
            return 0
        if args.command == "reload":
            vault.reload()
            print("reloaded")
            return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        # Return the lock file handle before the process exits instead of
        # leaving it for interpreter shutdown to close: repeated release is
        # an error-free no-op and the next operation reacquires the lock.
        if vault is not None:
            vault.close()
    return 2  # unreachable: argparse enforces a known command


if __name__ == "__main__":
    sys.exit(main())
