"""Download models and datasets for offline use, and package them for Kaggle.

Why this exists
---------------
Kaggle sessions may have internet disabled, and even when it is enabled,
re-downloading a 7B model every session wastes minutes of a 30 GPU-hour weekly
budget. The workflow is:

1. Run this script once **in a session with internet enabled** (or locally):

       python scripts/prefetch_assets.py --model Qwen/Qwen2.5-7B-Instruct \\
           --datasets gsm8k math500 --out /kaggle/working/assets

2. Zip `assets/` and upload it as a Kaggle Dataset (or use the "Save Version"
   output), then attach it as an input in later sessions.

3. In offline sessions, point the config at the mounted copy:

       python -m src.runner --config configs/gsm8k_cot_zeroshot.yaml \\
           --set model.local_path=/kaggle/input/my-assets/models/Qwen2.5-7B-Instruct \\
           --set data.local_dir=/kaggle/input/my-assets/datasets/gsm8k

   and export `HF_HUB_OFFLINE=1` so nothing silently reaches for the network.

Datasets are saved BOTH as the raw HF cache (via `snapshot_download`-style
`load_dataset` + `save_to_disk`) and as harness-schema JSONL, because the JSONL
is small, human-readable, diffable, and loads without the `datasets` package.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DataConfig  # noqa: E402
from src.utils import human_time, setup_logging, write_json_atomic  # noqa: E402

log = logging.getLogger("prefetch")

#: Files we never need for inference; skipping them saves a lot of disk/time.
MODEL_IGNORE_PATTERNS = (
    "*.msgpack",
    "*.h5",
    "*.onnx",
    "*.onnx_data",
    "*.tflite",
    "*.pth",
    "*.bin",  # prefer safetensors; see --allow-bin
    "original/*",
    "*.gguf",
)


def dir_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / 1024**3


def prefetch_model(
    repo_id: str,
    out_root: Path,
    revision: Optional[str] = None,
    allow_bin: bool = False,
    token: Optional[str] = None,
) -> Path:
    """Snapshot a model repo into `out_root/models/<name>`.

    Uses `local_dir` so the result is a plain directory that can be handed to
    `from_pretrained` directly, rather than the symlinked HF cache layout (Kaggle
    Dataset packaging does not preserve symlinks).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from exc

    dest = out_root / "models" / repo_id.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    ignore = [p for p in MODEL_IGNORE_PATTERNS if not (allow_bin and p == "*.bin")]
    log.info("downloading model %s -> %s", repo_id, dest)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        ignore_patterns=ignore,
        token=token or os.environ.get("HF_TOKEN"),
    )
    log.info("model %s ready (%.2f GB)", repo_id, dir_size_gb(dest))
    return dest


def prefetch_dataset(
    name: str,
    out_root: Path,
    split: Optional[str] = None,
    subset: Optional[str] = None,
    subsample: Optional[int] = None,
    subsample_seed: int = 1234,
    save_arrow: bool = False,
) -> Dict[str, Any]:
    """Materialise one registered dataset as harness-schema JSONL.

    Saving the *harness schema* rather than the raw dataset means the offline
    path exercises exactly the same `Example` objects as the online path, so an
    offline session cannot silently differ from an online one.
    """
    from src.datasets_ import (
        cache_dir_for,
        load_dataset_examples,
        save_examples_jsonl,
    )

    dest = cache_dir_for(name, str(out_root / "datasets"))
    dest.mkdir(parents=True, exist_ok=True)
    cfg = DataConfig(
        name=name,
        split=split or DataConfig().split,
        subset=subset,
        subsample=subsample,
        subsample_seed=subsample_seed,
    )
    # Prefetch the FULL split; subsampling stays a run-time decision so one
    # cached copy serves every subsample size and seed.
    cfg.subsample = None
    examples = load_dataset_examples(cfg)
    jsonl_path = dest / "examples.jsonl"
    save_examples_jsonl(examples, jsonl_path)

    info = {
        "name": name,
        "split": cfg.split,
        "subset": subset,
        "n_examples": len(examples),
        "jsonl": str(jsonl_path),
        "answer_types": sorted({e.answer_type for e in examples}),
    }

    if save_arrow:
        try:
            import datasets as hfds

            raw = hfds.load_dataset(name if "/" in name else name, subset, split=cfg.split)
            raw.save_to_disk(str(dest / "arrow"))
            info["arrow"] = str(dest / "arrow")
        except Exception as exc:  # noqa: BLE001 - arrow copy is a nice-to-have
            log.warning("could not save arrow copy for %s: %s", name, exc)

    write_json_atomic(dest / "info.json", info)
    log.info("dataset %s ready: %d examples -> %s", name, len(examples), jsonl_path)
    return info


def make_zip(source: Path, zip_path: Path) -> Path:
    """Zip the asset tree for download from /kaggle/working."""
    zip_path = zip_path.with_suffix("")
    out = shutil.make_archive(str(zip_path), "zip", root_dir=str(source))
    log.info("wrote %s (%.2f GB)", out, Path(out).stat().st_size / 1024**3)
    return Path(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/prefetch_assets.py",
        description=(
            "Download models/datasets for offline Kaggle runs and package them "
            "as a Kaggle Dataset."
        ),
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="HF model repo id, repeatable",
    )
    p.add_argument("--revision", default=None, help="pin a model revision (recommended)")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        metavar="NAME",
        help="registered dataset names (see `python -m src.runner --list`)",
    )
    p.add_argument("--split", default=None, help="override the split for all datasets")
    p.add_argument("--subset", default=None, help="dataset config/subset name")
    p.add_argument(
        "--out",
        default="assets",
        help="output root (use /kaggle/working/assets on Kaggle)",
    )
    p.add_argument(
        "--allow-bin",
        action="store_true",
        help="also download pytorch_model.bin shards (default: safetensors only)",
    )
    p.add_argument(
        "--save-arrow",
        action="store_true",
        help="additionally save the raw HF arrow dataset (larger)",
    )
    p.add_argument("--zip", action="store_true", help="zip the asset tree when done")
    p.add_argument("--token", default=None, help="HF token (or set HF_TOKEN)")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if not args.model and not args.datasets:
        log.error("nothing to do: pass --model and/or --datasets")
        return 2

    # Prefetching inherently needs the network; a stale offline flag from a
    # previous cell would make every download fail with a confusing error.
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.pop(var, None):
            log.warning("unset %s for this download run", var)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"models": [], "datasets": []}

    for repo in args.model:
        dest = prefetch_model(
            repo,
            out_root,
            revision=args.revision,
            allow_bin=args.allow_bin,
            token=args.token,
        )
        manifest["models"].append(
            {
                "repo_id": repo,
                "revision": args.revision,
                "path": str(dest),
                "size_gb": round(dir_size_gb(dest), 3),
            }
        )

    for name in args.datasets:
        try:
            manifest["datasets"].append(
                prefetch_dataset(
                    name,
                    out_root,
                    split=args.split,
                    subset=args.subset,
                    save_arrow=args.save_arrow,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not abort the rest
            log.error("failed to prefetch dataset %s: %s", name, exc)
            manifest["datasets"].append({"name": name, "error": str(exc)})

    write_json_atomic(out_root / "assets_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print(f"\nasset tree: {out_root.resolve()}  ({dir_size_gb(out_root):.2f} GB)")
    print(
        "\nNext session (offline): attach this as a Kaggle Dataset input, then run\n"
        "  export HF_HUB_OFFLINE=1\n"
        "  python -m src.runner --config configs/<cell>.yaml \\\n"
        "      --set model.local_path=/kaggle/input/<dataset-slug>/models/<model-dir> \\\n"
        "      --set data.local_dir=/kaggle/input/<dataset-slug>/datasets/<name>"
    )

    if args.zip:
        make_zip(out_root, Path(str(out_root) + "_bundle"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
