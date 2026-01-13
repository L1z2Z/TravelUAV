#!/usr/bin/env python3
"""Temporary cleanup script for TravelUAV extracted dataset.

Requirements:
1) Delete all `.extract_done` files under the given root directory.
2) Flatten duplicated nested folders:
   Turn   <root>/<name>/<name>/...  into  <root>/<name>/...

This script is intentionally conservative:
- It will NOT overwrite existing files/directories when flattening.
- If a name conflict is detected, it raises an error and stops.

Usage:
  python3 scripts/fix_dataset_extracted.py --root /home/liz/data/TravelUAV_data/dataset_extracted

Optional:
  --dry-run   Print actions without making changes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def remove_extract_done_files(root: Path, *, dry_run: bool) -> int:
    count = 0
    for marker in root.rglob(".extract_done"):
        if not marker.is_file():
            continue
        count += 1
        if dry_run:
            print(f"[dry-run] delete {marker}")
        else:
            try:
                marker.unlink()
                print(f"[delete] {marker}")
            except OSError as exc:
                raise RuntimeError(f"Failed to delete {marker}: {exc}") from exc
    return count


def flatten_duplicate_nested_dirs(root: Path, *, dry_run: bool) -> int:
    """Flatten <root>/<name>/<name>/... into <root>/<name>/..."""

    changed = 0
    for outer in sorted([p for p in root.iterdir() if p.is_dir()]):
        inner = outer / outer.name
        if not inner.exists() or not inner.is_dir():
            continue

        print(f"[flatten] {inner} -> {outer}")

        # Move each child of inner into outer.
        for child in sorted(inner.iterdir()):
            dest = outer / child.name
            if dest.exists():
                raise RuntimeError(
                    f"Name conflict while flattening: destination already exists: {dest}\n"
                    f"Refusing to overwrite. Resolve manually then re-run."
                )

            changed += 1
            if dry_run:
                print(f"  [dry-run] move {child} -> {dest}")
            else:
                shutil.move(str(child), str(dest))
                print(f"  [move] {child} -> {dest}")

        # Remove now-empty inner folder.
        if dry_run:
            print(f"  [dry-run] rmdir {inner}")
        else:
            try:
                inner.rmdir()
                print(f"  [rmdir] {inner}")
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to remove inner dir (not empty or permission issue): {inner}: {exc}"
                ) from exc

    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/liz/data/TravelUAV_data/dataset_extracted",
        help="Dataset extracted root directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without making changes",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    print(f"Root: {root}")
    print(f"Dry-run: {args.dry_run}")

    deleted = remove_extract_done_files(root, dry_run=args.dry_run)
    moved = flatten_duplicate_nested_dirs(root, dry_run=args.dry_run)

    print("\nDone")
    print(f"- .extract_done deleted: {deleted}")
    print(f"- items moved from nested duplicate dirs: {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
