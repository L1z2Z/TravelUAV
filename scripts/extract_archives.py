#!/usr/bin/env python3
"""Extract .zip archives under a root directory.

Supports split ZIP archives where parts are named like:
  name.z01, name.z02, ... name.zip

Usage:
  python3 scripts/extract_archives.py --root /home/liz/TravelUAV/dataset --out ~/data/TravelUAV_data/dataset_extracted --tool 7z --clean
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence


def _find_bsdtar_executable() -> str | None:
    bsdtar = shutil.which("bsdtar")
    if bsdtar and os.access(bsdtar, os.X_OK):
        return bsdtar

    for candidate in (
        Path.home() / "miniconda3" / "bin" / "bsdtar",
        Path.home() / "anaconda3" / "bin" / "bsdtar",
    ):
        if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return None


def _find_7z_executable() -> str | None:
    for name in ("7z", "7za"):
        exe = shutil.which(name)
        if exe and os.access(exe, os.X_OK):
            return exe

    for candidate in (
        Path.home() / "miniconda3" / "envs" / "llamauav" / "bin" / "7z",
        Path.home() / "miniconda3" / "envs" / "llamauav" / "bin" / "7za",
        Path.home() / "miniconda3" / "bin" / "7z",
        Path.home() / "miniconda3" / "bin" / "7za",
    ):
        if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return None


def _which_tool(preferred: str | None) -> tuple[str, list[str]]:
    """Return (tool_name, base_args).

    Prefers 7z (or 7za) because it handles split ZIP reliably.
    Falls back to unzip for normal .zip files.
    """

    if preferred:
        # Accept either a known tool name or an absolute/relative path to the executable.
        preferred_path = Path(preferred).expanduser()
        if preferred_path.exists() and preferred_path.is_file() and os.access(preferred_path, os.X_OK):
            exe = str(preferred_path)
            name = preferred_path.name.lower()
            if name in {"7z", "7za"}:
                # Keep output quiet; errors will still be captured.
                return name, [exe, "x", "-y", "-bso0", "-bsp0"]
            if name == "bsdtar":
                return name, [exe, "-xf"]
            if name == "unzip":
                return name, [exe, "-o"]
            raise ValueError("--tool path must point to one of: 7z, 7za, bsdtar, unzip")

        preferred = preferred.lower()
        if preferred not in {"7z", "7za", "bsdtar", "unzip"}:
            raise ValueError("--tool must be one of: 7z, 7za, bsdtar, unzip (or a path to one of them)")

        resolved = shutil.which(preferred)
        if resolved is None:
            raise RuntimeError(f"Requested tool '{preferred}' not found in PATH")

        if preferred in {"7z", "7za"}:
            return preferred, [resolved, "x", "-y", "-bso0", "-bsp0"]
        if preferred == "bsdtar":
            return preferred, [resolved, "-xf"]
        return preferred, [resolved, "-o"]

    for candidate in ("7z", "7za"):
        if shutil.which(candidate):
            return candidate, [candidate, "x", "-y", "-bso0", "-bsp0"]

    # libarchive's bsdtar handles many zip variants better than the very old unzip 6.00
    bsdtar = _find_bsdtar_executable()
    if bsdtar:
        return "bsdtar", [bsdtar, "-xf"]

    if shutil.which("unzip"):
        return "unzip", ["unzip", "-o"]

    raise RuntimeError("No extractor found. Install '7z' (p7zip), 'bsdtar' (libarchive), or 'unzip'.")


def _iter_zip_files(root: Path, *, exclude_dirs: Sequence[Path]) -> Iterable[Path]:
    """Yield .zip files under root, skipping excluded directories.

    Using os.walk lets us prune large trees (e.g., already-extracted outputs).
    """

    exclude_set = {p.resolve() for p in exclude_dirs if p is not None}
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath).resolve()

        # Prune excluded subtrees.
        pruned: list[str] = []
        for d in list(dirnames):
            candidate = (current / d).resolve()
            if candidate in exclude_set or d in {".cache", "__MACOSX"}:
                pruned.append(d)
        for d in pruned:
            dirnames.remove(d)

        for name in filenames:
            if not name.endswith(".zip"):
                continue
            if name.startswith("__MACOSX"):
                continue
            yield (Path(dirpath) / name)


def _has_split_parts(zip_path: Path) -> bool:
    base = zip_path.with_suffix("")  # removes .zip
    return (base.with_suffix(".z01")).exists()


def _extract_one(
    zip_path: Path,
    out_dir: Path,
    tool: str,
    base_args: list[str],
    *,
    keep: bool,
    clean: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".extract_done"
    if marker.exists():
        print(f"[skip] {zip_path} -> {out_dir} (already done)")
        return

    if clean and out_dir.exists():
        # Keep the directory itself, but remove previous partial contents.
        for child in out_dir.iterdir():
            if child.name == marker.name:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass

    split = _has_split_parts(zip_path)
    archive_to_extract: Path = zip_path

    # Only 7z/7za can reliably read split ZIP parts directly.
    # For other tools (bsdtar/unzip), merge parts into a temporary single .zip first.
    tmpdir_ctx = None
    if split and tool not in {"7z", "7za"}:
        if shutil.which("zip") is None:
            raise RuntimeError("Split zip detected but 'zip' command is not available to merge parts. Install p7zip (7z) or zip.")

        tmpdir_ctx = tempfile.TemporaryDirectory(prefix="merge_zip_")
        tmpdir = Path(tmpdir_ctx.__enter__())
        merged_zip = tmpdir / zip_path.name

        merge_cmd = ["zip", "-s", "0", str(zip_path), "--out", str(merged_zip)]
        merge_proc = subprocess.run(
            merge_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if merge_proc.returncode != 0:
            out = merge_proc.stdout or ""
            tail = "\n".join(out.splitlines()[-80:])
            # ensure tempdir cleanup
            tmpdir_ctx.__exit__(None, None, None)
            raise RuntimeError(
                f"Failed to merge split zip before extraction: {zip_path}\n"
                f"Command: {' '.join(merge_cmd)}\n"
                f"--- output (tail) ---\n{tail}"
            )

        archive_to_extract = merged_zip

    if tool in {"7z", "7za"}:
        cmd: list[str] = base_args + [f"-o{str(out_dir)}", str(zip_path)]
    elif tool == "bsdtar":
        cmd = base_args + [str(archive_to_extract), "-C", str(out_dir)]
    else:
        cmd = base_args + [str(archive_to_extract), "-d", str(out_dir)]

    split_note = " (split)" if _has_split_parts(zip_path) else ""
    print(f"[extract]{split_note} {zip_path} -> {out_dir}")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # 7z returns 0 = OK, 1 = warnings (files locked, some errors but partial success).
    if tool in {"7z", "7za"} and proc.returncode in {0, 1}:
        if proc.returncode == 1:
            print(f"[warn] 7z reported warnings for {zip_path}")
            marker.write_text(f"ok_with_warnings\nsource={zip_path}\n")
        else:
            marker.write_text(f"ok\nsource={zip_path}\n")

        if tmpdir_ctx is not None:
            tmpdir_ctx.__exit__(None, None, None)

        if not keep:
            base = zip_path.with_suffix("")
            to_delete = [zip_path]
            for part in sorted(zip_path.parent.glob(base.name + ".z[0-9][0-9]")):
                to_delete.append(part)
            for p in to_delete:
                try:
                    p.unlink()
                except OSError:
                    pass
        return

    if proc.returncode != 0:
        # Provide the tail of output for debugging.
        out = proc.stdout or ""

        # If unzip hits its classic overlapped/zip-bomb false positive on large archives,
        # automatically retry with bsdtar if available.
        if tool == "unzip":
            lower = out.lower()
            if (
                "overlapped components" in lower
                or "possible zip bomb" in lower
                or "invalid compressed data" in lower
            ):
                bsdtar_exe = _find_bsdtar_executable()
                if bsdtar_exe:
                    retry_cmd = [bsdtar_exe, "-xf", str(archive_to_extract), "-C", str(out_dir)]
                    print(f"[retry] unzip failed; trying bsdtar for {zip_path}")
                    retry = subprocess.run(retry_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    if retry.returncode == 0:
                        marker.write_text(f"ok\nsource={zip_path}\n")
                        if tmpdir_ctx is not None:
                            tmpdir_ctx.__exit__(None, None, None)
                        return

        # If libarchive (bsdtar) fails to decompress, try 7z as a last resort.
        if tool == "bsdtar":
            seven_zip = _find_7z_executable()
            if seven_zip:
                retry_cmd = [seven_zip, "x", "-y", f"-o{str(out_dir)}", str(zip_path)]
                print(f"[retry] bsdtar failed; trying 7z for {zip_path}")
                retry = subprocess.run(retry_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if retry.returncode == 0:
                    marker.write_text(f"ok\nsource={zip_path}\n")
                    if tmpdir_ctx is not None:
                        tmpdir_ctx.__exit__(None, None, None)
                    return

        tail = "\n".join(out.splitlines()[-80:])
        if tmpdir_ctx is not None:
            tmpdir_ctx.__exit__(None, None, None)
        raise RuntimeError(f"Extraction failed for {zip_path}\nCommand: {' '.join(cmd)}\n--- output (tail) ---\n{tail}")

    marker.write_text(f"ok\nsource={zip_path}\n")

    if tmpdir_ctx is not None:
        tmpdir_ctx.__exit__(None, None, None)

    if not keep:
        # If it's a split zip, remove all parts name.z01, name.z02..., name.zip
        base = zip_path.with_suffix("")
        to_delete = [zip_path]
        for part in sorted(zip_path.parent.glob(base.name + ".z[0-9][0-9]")):
            to_delete.append(part)
        for p in to_delete:
            try:
                p.unlink()
            except OSError:
                pass


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Extract .zip and split .z01 archives under a directory")
    parser.add_argument("--root", default=str(Path.cwd() / "new_env"), help="Root directory to scan (default: ./new_env)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output base directory (default: <root>/extracted). Each archive extracts into <out>/<archive_name>/",
    )
    parser.add_argument("--tool", default=None, help="Force a tool: 7z, 7za, bsdtar, or unzip (default: auto)")
    parser.add_argument("--keep", action="store_true", help="Keep archive files after successful extraction")
    parser.add_argument("--clean", action="store_true", help="Delete partial output dir contents before extracting")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be extracted")

    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 2

    out_base = Path(args.out).expanduser().resolve() if args.out else (root / "extracted")

    tool, base_args = _which_tool(args.tool)
    print(f"Using tool: {tool}")
    print(f"Scanning: {root}")

    exclude_dirs = []
    # Avoid scanning the extraction output (often huge) if it is inside the root.
    try:
        out_base.relative_to(root)
        exclude_dirs.append(out_base)
    except ValueError:
        pass

    zip_files = sorted(_iter_zip_files(root, exclude_dirs=exclude_dirs))
    if not zip_files:
        print("No .zip files found.")
        return 0

    failures: list[tuple[Path, str]] = []
    for zip_path in zip_files:
        # Make a stable output directory name: archive filename without .zip
        archive_name = zip_path.stem
        out_dir = out_base / archive_name

        if args.dry_run:
            split_note = " (split)" if _has_split_parts(zip_path) else ""
            print(f"[dry-run]{split_note} {zip_path} -> {out_dir}")
            continue

        try:
            _extract_one(zip_path, out_dir, tool, base_args, keep=args.keep, clean=args.clean)
        except Exception as exc:  # noqa: BLE001
            failures.append((zip_path, str(exc)))
            print(f"[error] {zip_path}: {exc}", file=sys.stderr)

    if failures:
        print("\nSome archives failed:")
        for zip_path, msg in failures:
            print(f"- {zip_path}: {msg}")
        return 1

    print("\nAll archives extracted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
