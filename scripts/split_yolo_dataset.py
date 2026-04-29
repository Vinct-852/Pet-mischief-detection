#!/usr/bin/env python3
"""
Pool images + YOLO labels from a prepared tree (e.g. coco_filtered/images/{train,val}) and
split into train / val / test with a fixed random seed (default 80% / 10% / 10%).

Writes a fresh YOLO layout under --output, ``data.yaml`` (train/val/test), and
``dataset_manifest.json`` with byte sizes and per-class instance counts.

Does not modify the source directory (copies only).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DATASET_SOURCE_FOLDER_NAME = "coco_filtered"
DATASET_OUTPUT_FOLDER_NAME = "curated_yolo"


def parse_data_yaml_names(data_yaml: Path) -> list[str]:
    """Minimal parser for ``names:`` block (no PyYAML dependency)."""
    text = data_yaml.read_text()
    names: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            m = re.match(r"\s*(\d+)\s*:\s*(\S+)", line)
            if m:
                names[int(m.group(1))] = m.group(2)
            elif stripped and not line.startswith(" ") and not line.startswith("\t"):
                break
    if not names:
        raise ValueError(f"No names parsed from {data_yaml}")
    return [names[i] for i in sorted(names)]


def collect_pairs(
    source: Path,
    pool_subdirs: tuple[str, ...],
) -> list[tuple[Path, Path]]:
    """Return list of (image_path, label_path) with matching stems."""
    pairs: list[tuple[Path, Path]] = []
    for sub in pool_subdirs:
        img_root = source / "images" / sub
        lbl_root = source / "labels" / sub
        if not img_root.is_dir():
            continue
        for img in sorted(img_root.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = img.stem
            lbl = lbl_root / f"{stem}.txt"
            if not lbl.is_file():
                print(f"warning: no label for {img}", file=sys.stderr)
                continue
            pairs.append((img, lbl))
    if not pairs:
        raise FileNotFoundError(
            f"No image+label pairs under {source}/images/{{{','.join(pool_subdirs)}}}"
        )
    return pairs


def count_instances(label_paths: list[Path]) -> Counter[int]:
    c: Counter[int] = Counter()
    for p in label_paths:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            c[int(line.split()[0])] += 1
    return c


def split_indices(
    n: int,
    seed: int,
    train_pct: int,
    val_pct: int,
    test_pct: int,
) -> tuple[list[int], list[int], list[int]]:
    if train_pct + val_pct + test_pct != 100:
        raise ValueError("train + val + test percentages must sum to 100")
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = (n * train_pct) // 100
    n_val = (n * val_pct) // 100
    n_test = n - n_train - n_val
    train_i = idx[:n_train]
    val_i = idx[n_train : n_train + n_val]
    test_i = idx[n_train + n_val :]
    return train_i, val_i, test_i


def total_size(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.is_file())


def write_data_yaml(
    output: Path,
    path_field: str,
    names: list[str],
    include_test: bool,
) -> None:
    lines = [
        f"path: {path_field}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.extend(["", f"nc: {len(names)}", "names:"])
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    (output / "data.yaml").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=Path(DATASET_SOURCE_FOLDER_NAME),
        help="YOLO-style root with images/{train,val,...} and labels/{...}.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(DATASET_OUTPUT_FOLDER_NAME),
        help="Destination root (created).",
    )
    ap.add_argument(
        "--pool-splits",
        default="train,val",
        help="Comma-separated names under images/ and labels/ to merge before splitting.",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    ap.add_argument("--train-pct", type=int, default=80, help="Training %% (default 80).")
    ap.add_argument("--val-pct", type=int, default=10, help="Validation %% (default 10).")
    ap.add_argument("--test-pct", type=int, default=10, help="Test %% (default 10).")
    ap.add_argument(
        "--names-yaml",
        type=Path,
        default=None,
        help="Defaults to --source/data.yaml for class names.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing --output/images and --output/labels.",
    )
    args = ap.parse_args()

    pool_subdirs = tuple(s.strip() for s in args.pool_splits.split(",") if s.strip())
    names_yaml = args.names_yaml or (args.source / "data.yaml")
    if not names_yaml.is_file():
        raise FileNotFoundError(f"Need data.yaml at {names_yaml} (or pass --names-yaml)")
    class_names = parse_data_yaml_names(names_yaml)

    pairs = collect_pairs(args.source, pool_subdirs)
    n = len(pairs)
    train_i, val_i, test_i = split_indices(
        n, args.seed, args.train_pct, args.val_pct, args.test_pct
    )

    print(f"Pooled {n} image+label pairs from {pool_subdirs}")
    print(
        f"Split (seed={args.seed}): train={len(train_i)} val={len(val_i)} test={len(test_i)}"
    )

    out_images = args.output / "images"
    out_labels = args.output / "labels"
    if out_images.exists() or out_labels.exists():
        if not args.force:
            raise SystemExit(
                f"Refusing to write: {out_images} or {out_labels} exists. Use --force."
            )
        shutil.rmtree(out_images, ignore_errors=True)
        shutil.rmtree(out_labels, ignore_errors=True)

    for split in ("train", "val", "test"):
        (out_images / split).mkdir(parents=True, exist_ok=True)
        (out_labels / split).mkdir(parents=True, exist_ok=True)

    def copy_split(indices: list[int], split: str) -> tuple[list[Path], list[Path]]:
        imgs: list[Path] = []
        lbls: list[Path] = []
        for i in indices:
            src_img, src_lbl = pairs[i]
            dst_img = out_images / split / src_img.name
            dst_lbl = out_labels / split / src_lbl.name
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
            imgs.append(dst_img)
            lbls.append(dst_lbl)
        return imgs, lbls

    train_imgs, train_lbls = copy_split(train_i, "train")
    val_imgs, val_lbls = copy_split(val_i, "val")
    test_imgs, test_lbls = copy_split(test_i, "test")

    try:
        path_field = args.output.resolve().relative_to(Path.cwd().resolve())
        path_str = path_field.as_posix()
    except ValueError:
        path_str = args.output.as_posix()
    if not path_str.startswith((".", "/")):
        path_str = "./" + path_str

    write_data_yaml(args.output, path_str, class_names, include_test=True)

    def split_stats(imgs: list[Path], lbls: list[Path]) -> dict:
        inst = count_instances(lbls)
        per_class = {class_names[k]: inst.get(k, 0) for k in range(len(class_names))}
        return {
            "images": len(imgs),
            "image_bytes": total_size(imgs),
            "label_bytes": total_size(lbls),
            "instances_total": int(sum(inst.values())),
            "instances_per_class": per_class,
        }

    manifest = {
        "schema": "pet_mischief_dataset_manifest_v1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_dir": str(args.source.resolve()),
        "output_dir": str(args.output.resolve()),
        "pool_splits": list(pool_subdirs),
        "reproducibility": {
            "seed": args.seed,
            "split_percent": {
                "train": args.train_pct,
                "val": args.val_pct,
                "test": args.test_pct,
            },
        },
        "class_names_ordered": class_names,
        "yolo_label_format": (
            "One line per object: class_id cx cy w h (normalized 0..1); "
            "class_id matches names order in data.yaml."
        ),
        "splits": {
            "train": split_stats(train_imgs, train_lbls),
            "val": split_stats(val_imgs, val_lbls),
            "test": split_stats(test_imgs, test_lbls),
        },
        "totals": {
            "images": n,
            "image_bytes": total_size([p for p, _ in pairs]),
        },
    }
    manifest_path = args.output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output / 'data.yaml'}")
    print(f"Wrote {manifest_path}")

    # Human-readable summary on stdout
    for split_name in ("train", "val", "test"):
        s = manifest["splits"][split_name]
        print(
            f"\n[{split_name}] images={s['images']} | "
            f"image_mb={s['image_bytes']/1024/1024:.2f} | "
            f"instances={s['instances_total']}"
        )
        for cls, cnt in s["instances_per_class"].items():
            print(f"  {cls:12s} {cnt:>6}")


if __name__ == "__main__":
    main()
