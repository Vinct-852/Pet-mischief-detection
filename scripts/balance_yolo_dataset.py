#!/usr/bin/env python3
"""
Balance a YOLO dataset by downsampling excess images with a fixed seed.

Default behavior:
- Reads class names from source data.yaml.
- Balances the train/val splits independently.
- Uses a primary-class strategy for multi-label images:
  each image is assigned to one primary class (the rarest class present in that split),
  then every class bucket is downsampled to the smallest bucket size.
- Copies selected files into a new output dataset directory.
- Leaves the source dataset unchanged.

This produces equal image counts per class in the primary-class view.
Raw per-class instance totals may still differ because images can contain multiple classes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_data_yaml_names(data_yaml: Path) -> list[str]:
    """Minimal parser for names block without requiring PyYAML."""
    text = data_yaml.read_text(encoding="utf-8")
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
        raise ValueError(f"No class names parsed from {data_yaml}")
    return [names[i] for i in sorted(names)]


def collect_split_pairs(source: Path, split: str) -> list[tuple[Path, Path]]:
    """Collect image/label path pairs for one split."""
    img_root = source / "images" / split
    lbl_root = source / "labels" / split
    if not img_root.is_dir() or not lbl_root.is_dir():
        return []

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(img_root.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = lbl_root / f"{img.stem}.txt"
        if not lbl.is_file():
            print(f"warning: no label for {img}", file=sys.stderr)
            continue
        pairs.append((img, lbl))
    return pairs


def read_label_classes(label_path: Path, nc: int) -> set[int]:
    """Return set of class ids present in one label file."""
    present: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cid = int(line.split()[0])
        if cid < 0 or cid >= nc:
            raise ValueError(f"class id {cid} out of range in {label_path}")
        present.add(cid)
    return present


def count_instances(label_paths: list[Path]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for path in label_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            counts[int(line.split()[0])] += 1
    return counts


def image_class_presence(label_paths: list[Path], nc: int) -> Counter[int]:
    """Count images containing each class (presence, not instances)."""
    counts: Counter[int] = Counter()
    for path in label_paths:
        for cid in read_label_classes(path, nc):
            counts[cid] += 1
    return counts


def select_balanced_primary(
    pairs: list[tuple[Path, Path]],
    nc: int,
    seed: int,
    require_all_classes: bool,
) -> tuple[list[tuple[Path, Path]], dict[int, int], dict[int, int], int]:
    """Select a balanced subset using primary-class downsampling."""
    if not pairs:
        return [], {}, {}, 0

    # Build class presence map per image.
    per_image_classes: dict[int, set[int]] = {}
    for i, (_, lbl) in enumerate(pairs):
        classes = read_label_classes(lbl, nc)
        if not classes:
            continue
        per_image_classes[i] = classes

    if not per_image_classes:
        raise ValueError("No labeled images found in selected split")

    # Presence count determines rarity for primary assignment.
    present_counts: dict[int, int] = {cid: 0 for cid in range(nc)}
    for cls_set in per_image_classes.values():
        for cid in cls_set:
            present_counts[cid] += 1

    if require_all_classes:
        missing = [cid for cid, c in present_counts.items() if c == 0]
        if missing:
            raise ValueError(
                "Cannot balance because some classes are missing: "
                + ", ".join(str(m) for m in missing)
            )

    # Assign each image to one primary class (rarest present class).
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, cls_set in per_image_classes.items():
        primary = min(cls_set, key=lambda cid: (present_counts[cid], cid))
        buckets[primary].append(i)

    non_empty = {cid: ids for cid, ids in buckets.items() if ids}
    if not non_empty:
        raise ValueError("No non-empty class buckets were formed")

    min_bucket = min(len(ids) for ids in non_empty.values())
    rng = random.Random(seed)

    chosen_idx: set[int] = set()
    before_primary = {cid: len(non_empty.get(cid, [])) for cid in range(nc)}
    after_primary = {cid: 0 for cid in range(nc)}

    for cid in range(nc):
        ids = list(non_empty.get(cid, []))
        rng.shuffle(ids)
        kept = ids[:min_bucket]
        chosen_idx.update(kept)
        after_primary[cid] = len(kept)

    selected_pairs = [pairs[i] for i in sorted(chosen_idx)]
    return selected_pairs, before_primary, after_primary, min_bucket


def copy_pairs(pairs: list[tuple[Path, Path]], out_root: Path, split: str) -> tuple[list[Path], list[Path]]:
    out_img = out_root / "images" / split
    out_lbl = out_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    imgs: list[Path] = []
    lbls: list[Path] = []
    for src_img, src_lbl in pairs:
        dst_img = out_img / src_img.name
        dst_lbl = out_lbl / src_lbl.name
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)
        imgs.append(dst_img)
        lbls.append(dst_lbl)
    return imgs, lbls


def format_path_for_yaml(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        out = rel.as_posix()
    except ValueError:
        out = path.as_posix()
    if not out.startswith((".", "/")):
        out = "./" + out
    return out


def write_data_yaml(out_root: Path, names: list[str], include_test: bool) -> None:
    lines = [
        f"path: {format_path_for_yaml(out_root)}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.extend(["", f"nc: {len(names)}", "names:"])
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=Path("curated_yolo_with_negatives"),
        help="Source YOLO dataset root.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("curated_yolo_with_negatives_balanced"),
        help="Output YOLO dataset root.",
    )
    ap.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated splits to balance independently (default: train,val).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument(
        "--names-yaml",
        type=Path,
        default=None,
        help="Path to YAML containing names block (default: --source/data.yaml).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output images/labels if present.",
    )
    ap.add_argument(
        "--allow-missing-classes",
        action="store_true",
        help="Allow balancing over only classes present in each split.",
    )
    ap.add_argument(
        "--copy-other-splits",
        action="store_true",
        help="Copy non-balanced splits as-is (for example test).",
    )
    args = ap.parse_args()

    names_yaml = args.names_yaml or (args.source / "data.yaml")
    if not names_yaml.is_file():
        raise FileNotFoundError(f"Missing names yaml: {names_yaml}")
    class_names = parse_data_yaml_names(names_yaml)
    nc = len(class_names)

    selected_splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if not selected_splits:
        raise ValueError("--splits cannot be empty")

    out_images = args.output / "images"
    out_labels = args.output / "labels"
    if out_images.exists() or out_labels.exists():
        if not args.force:
            raise SystemExit(
                f"Refusing to write: {out_images} or {out_labels} exists. Use --force."
            )
        shutil.rmtree(out_images, ignore_errors=True)
        shutil.rmtree(out_labels, ignore_errors=True)

    args.output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "schema": "balanced_yolo_dataset_manifest_v1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_dir": str(args.source.resolve()),
        "output_dir": str(args.output.resolve()),
        "strategy": "primary_class_downsampling",
        "seed": args.seed,
        "balanced_splits": list(selected_splits),
        "class_names_ordered": class_names,
        "splits": {},
    }

    written_splits: set[str] = set()

    for split_idx, split in enumerate(selected_splits):
        pairs = collect_split_pairs(args.source, split)
        if not pairs:
            print(f"warning: split {split} not found or empty; skipping")
            continue

        before_labels = [lbl for _, lbl in pairs]
        before_instances = count_instances(before_labels)
        before_presence = image_class_presence(before_labels, nc)

        selected_pairs, before_primary, after_primary, min_bucket = select_balanced_primary(
            pairs=pairs,
            nc=nc,
            seed=args.seed + split_idx,
            require_all_classes=not args.allow_missing_classes,
        )
        out_imgs, out_lbls = copy_pairs(selected_pairs, args.output, split)
        written_splits.add(split)

        after_instances = count_instances(out_lbls)
        after_presence = image_class_presence(out_lbls, nc)

        split_manifest = {
            "before": {
                "images": len(pairs),
                "instances_total": int(sum(before_instances.values())),
                "instances_per_class": {
                    class_names[cid]: int(before_instances.get(cid, 0)) for cid in range(nc)
                },
                "image_presence_per_class": {
                    class_names[cid]: int(before_presence.get(cid, 0)) for cid in range(nc)
                },
            },
            "after": {
                "images": len(selected_pairs),
                "instances_total": int(sum(after_instances.values())),
                "instances_per_class": {
                    class_names[cid]: int(after_instances.get(cid, 0)) for cid in range(nc)
                },
                "image_presence_per_class": {
                    class_names[cid]: int(after_presence.get(cid, 0)) for cid in range(nc)
                },
            },
            "primary_bucket_before": {
                class_names[cid]: int(before_primary.get(cid, 0)) for cid in range(nc)
            },
            "primary_bucket_after": {
                class_names[cid]: int(after_primary.get(cid, 0)) for cid in range(nc)
            },
            "min_bucket_size": int(min_bucket),
            "dropped_images": int(len(pairs) - len(selected_pairs)),
        }
        manifest["splits"][split] = split_manifest

        print(
            f"[{split}] before={len(pairs)} after={len(selected_pairs)} "
            f"dropped={len(pairs) - len(selected_pairs)} min_bucket={min_bucket}"
        )
        print(f"[{split}] primary buckets after balancing:")
        for cid, name in enumerate(class_names):
            print(f"  {name:12s}: {after_primary.get(cid, 0):>6}")

    if args.copy_other_splits:
        src_images_root = args.source / "images"
        if src_images_root.is_dir():
            for split_dir in sorted(src_images_root.iterdir()):
                if not split_dir.is_dir():
                    continue
                split = split_dir.name
                if split in written_splits:
                    continue
                pairs = collect_split_pairs(args.source, split)
                if not pairs:
                    continue
                copy_pairs(pairs, args.output, split)
                written_splits.add(split)
                print(f"[{split}] copied without balancing: {len(pairs)} images")

    include_test = (args.output / "images" / "test").is_dir()
    write_data_yaml(args.output, class_names, include_test=include_test)

    manifest_path = args.output / "balance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output / 'data.yaml'}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
