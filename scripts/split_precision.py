"""Split results_bundle into fp16 and 4bit trees via symlinks (Linux/Kaggle + Windows).

Usage (from repo root, where results_bundle/ lives):
    python scripts/split_precision.py
    python scripts/split_precision.py --src /kaggle/working/results
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def link_dir(src: Path, dest: Path) -> bool:
    if dest.exists() or dest.is_symlink():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    # 1) symlink (works on Linux/Kaggle; needs admin or Developer Mode on Windows)
    try:
        dest.symlink_to(src, target_is_directory=True)
        return dest.exists() or dest.is_symlink()
    except OSError:
        pass
    # 2) Windows junction
    if os.name == "nt":
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dest), str(src)],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and dest.exists():
            return True
    # 3) last resort: copy tree (slow, but always works)
    try:
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return True
    except OSError as exc:
        print(f"FAILED to link/copy {src.name}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default=None,
        help="results root (default: ./results_bundle or ./results)",
    )
    p.add_argument("--fp16-out", default="results_fp16")
    p.add_argument("--bit4-out", default="results_4bit")
    args = p.parse_args()

    root = Path.cwd()
    if args.src:
        src = Path(args.src)
    elif (root / "results_bundle").is_dir():
        src = root / "results_bundle"
    elif (root / "results").is_dir():
        src = root / "results"
    else:
        print(
            "ERROR: no results_bundle/ or results/ in cwd. "
            "cd to the repo root, or pass --src /path/to/results",
            file=sys.stderr,
        )
        return 1

    if not src.is_dir():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 1

    manifests = list(src.rglob("manifest.json"))
    if not manifests:
        print(f"ERROR: no manifest.json under {src}", file=sys.stderr)
        return 1

    fp16 = Path(args.fp16_out)
    bit4 = Path(args.bit4_out)
    # wipe stale empty dirs from a failed prior run
    for d in (fp16, bit4):
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    fp16.mkdir(exist_ok=True)
    bit4.mkdir(exist_ok=True)

    n16 = n4 = fail = 0
    for manifest in manifests:
        st = json.loads(manifest.read_text(encoding="utf-8"))
        is4 = bool(st.get("config", {}).get("model", {}).get("load_in_4bit"))
        dest_root = bit4 if is4 else fp16
        dest = dest_root / manifest.parent.name
        ok = link_dir(manifest.parent, dest)
        if not ok:
            fail += 1
            continue
        if is4:
            n4 += 1
        else:
            n16 += 1

    print(f"src={src}")
    print(f"manifests_found={len(manifests)}")
    print(f"fp16={n16} 4bit={n4} failed={fail}")
    print(f"fp16_dirs={len(list(fp16.iterdir()))} 4bit_dirs={len(list(bit4.iterdir()))}")
    if n16 == 0:
        print(
            "ERROR: fp16 tree is empty. Analysis will fail. "
            "Check --src path and that graded.jsonl exists in run folders.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
