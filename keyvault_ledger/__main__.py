"""Command line interface for keyvault-ledger.

    python3 -m keyvault_ledger --root ./vault versions
    python3 -m keyvault_ledger --root ./vault seal <key-id> --material-file <path>
    python3 -m keyvault_ledger --root ./vault reload
"""

from __future__ import annotations

import argparse
import sys

from . import Vault


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyvault_ledger",
        description="Local key vault with an append-only version manifest.",
    )
    parser.add_argument("--root", required=True, help="vault directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("versions", help="list every key's versions and active version")

    seal = sub.add_parser("seal", help="seal key material as a new version")
    seal.add_argument("key_id", help="key identifier")
    seal.add_argument(
        "--material-file", required=True, help="file holding the key material"
    )

    sub.add_parser("reload", help="re-read and validate the whole keyring")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    vault = Vault(args.root)

    if args.command == "versions":
        keys = vault.manifest()["keys"]
        if not keys:
            print("(no keys)")
        for key_id in sorted(keys):
            entry = keys[key_id]
            versions = ",".join(str(r["version"]) for r in entry["versions"])
            print(f"{key_id}: active={entry['active']} versions={versions}")
        return 0

    if args.command == "seal":
        with open(args.material_file, "rb") as fh:
            material = fh.read()
        version = vault.seal(args.key_id, material)
        print(f"{args.key_id}: sealed version {version}")
        return 0

    if args.command == "reload":
        vault.reload()
        count = len(vault.manifest()["keys"])
        print(f"reloaded: {count} key(s)")
        return 0

    return 2  # unreachable: subparsers are required


if __name__ == "__main__":
    sys.exit(main())
