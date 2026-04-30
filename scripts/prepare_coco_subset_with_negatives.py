#!/usr/bin/env python3
"""
Build a COCO subset for YOLO without downloading full train2017.zip / val2017.zip.

1. Fetch only annotations_trainval2017.zip (small compared to image zips).
2. Filter instances JSON with default behavior for larger data:
   keep images that contain at least one non-cat target object
   (cup/laptop/keyboard/vase/plant/scissors), regardless of cat presence.
   This includes:
   - cat + object images
   - object-only images (no cat)
3. Download only the image files that appear in the filtered annotations.

Optional mode `--any-target-class` keeps the full union (including cat-only images).
COCO names the class "potted plant"; we expose it as "plant" in data.yaml and labels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# COCO uses these exact strings in instances_*.json
COCO_NAME_TO_EXPORT = {
    "cat": "cat",
    "cup": "cup",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "vase": "vase",
    "potted plant": "plant",
    "scissors": "scissors",
}

SPLITS = {
    "train": {
        "ann_file": "instances_train2017.json",
        "img_subdir": "train2017",
    },
    "val": {
        "ann_file": "instances_val2017.json",
        "img_subdir": "val2017",
    },
}

ANNOTATIONS_ZIP_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_IMAGE_BASE = "http://images.cocodataset.org"


def _download(url: str, dest: Path, description: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {description or url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - trusted COCO URL


def ensure_annotations(annotations_dir: Path) -> Path:
    """Return folder containing ``instances_train2017.json``."""
    train_json = annotations_dir / "instances_train2017.json"
    if train_json.is_file():
        return annotations_dir

    extract_parent = annotations_dir.parent
    extract_parent.mkdir(parents=True, exist_ok=True)
    zip_path = extract_parent / "annotations_trainval2017.zip"
    if not zip_path.is_file():
        _download(ANNOTATIONS_ZIP_URL, zip_path, "annotations_trainval2017.zip")

    print(f"Extracting {zip_path} -> {extract_parent}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_parent)

    if not train_json.is_file():
        raise FileNotFoundError(
            f"After unzip, expected {train_json}. "
            "Use --annotations-dir pointing at the folder that contains instances_train2017.json."
        )
    return annotations_dir


def _image_ids_with_non_cat_object(
    annotations: list[dict],
    target_cats: dict[int, str],
) -> tuple[set[int], int, int, int]:
    """Keep images that have at least one non-cat target object."""
    cat_coco_ids = [cid for cid, name in target_cats.items() if name == "cat"]
    if len(cat_coco_ids) != 1:
        raise RuntimeError("Expected exactly one 'cat' category in target set")
    cat_id = cat_coco_ids[0]
    non_cat_ids = {cid for cid in target_cats if cid != cat_id}

    by_image: dict[int, set[int]] = defaultdict(set)
    for ann in annotations:
        cid = ann["category_id"]
        if cid in target_cats:
            by_image[ann["image_id"]].add(cid)

    keep: set[int] = set()
    with_cat_and_object = 0
    object_without_cat = 0
    cat_only = 0

    for img_id, present in by_image.items():
        has_cat = cat_id in present
        has_non_cat = bool(present & non_cat_ids)
        if has_non_cat:
            keep.add(img_id)
            if has_cat:
                with_cat_and_object += 1
            else:
                object_without_cat += 1
        elif has_cat:
            cat_only += 1

    return keep, with_cat_and_object, object_without_cat, cat_only


def _balanced_image_ids_by_primary_class(
    annotations: list[dict],
    kept_ids: set[int],
    target_cats: dict[int, str],
    seed: int,
) -> tuple[set[int], int]:
    """Downsample to equal image counts per class using a primary-class assignment.

    Notes:
    - Images can have multiple classes, so exact equality by raw presence is not always feasible.
    - We assign each image to one "primary" class, then downsample each class bucket
      to the smallest bucket size so class image counts are exactly equal in this view.
    """
    image_to_cats: dict[int, set[int]] = defaultdict(set)
    for ann in annotations:
        img_id = ann["image_id"]
        cid = ann["category_id"]
        if img_id in kept_ids and cid in target_cats:
            image_to_cats[img_id].add(cid)

    class_image_counts: dict[int, int] = {cid: 0 for cid in target_cats}
    for present in image_to_cats.values():
        for cid in present:
            class_image_counts[cid] += 1

    primary_buckets: dict[int, list[int]] = {cid: [] for cid in target_cats}
    for img_id, present in image_to_cats.items():
        if not present:
            continue
        # Prefer the rarest class present in each image as the primary class.
        primary = min(present, key=lambda cid: (class_image_counts[cid], cid))
        primary_buckets[primary].append(img_id)

    min_bucket = min(len(v) for v in primary_buckets.values())
    rng = random.Random(seed)
    balanced_ids: set[int] = set()
    for cid, ids in primary_buckets.items():
        rng.shuffle(ids)
        take = ids[:min_bucket]
        balanced_ids.update(take)
        print(
            f"  primary-class bucket '{target_cats[cid]}': original={len(ids)} kept={len(take)}"
        )

    return balanced_ids, min_bucket


def filter_split(
    split_name: str,
    ann_path: Path,
    img_subdir: str,
    output_dir: Path,
    any_target_class: bool,
    balanced: bool,
    balanced_seed: int,
) -> tuple[list[dict], list[str]]:
    _ = img_subdir  # retained for signature parity with the original script
    with ann_path.open() as f:
        coco = json.load(f)

    target_cats: dict[int, str] = {}
    for c in coco["categories"]:
        name = c["name"]
        if name in COCO_NAME_TO_EXPORT:
            target_cats[c["id"]] = name

    if len(target_cats) != len(COCO_NAME_TO_EXPORT):
        found = {c["name"] for c in coco["categories"] if c["name"] in COCO_NAME_TO_EXPORT}
        missing = set(COCO_NAME_TO_EXPORT) - found
        raise RuntimeError(f"Missing categories in {ann_path}: {missing}")

    id_remap = {
        old_id: new_id for new_id, old_id in enumerate(sorted(target_cats.keys()))
    }
    export_names = [
        COCO_NAME_TO_EXPORT[target_cats[oid]] for oid in sorted(target_cats.keys())
    ]
    print(f"[{split_name}] Class order (by COCO id): {export_names}")

    if any_target_class:
        kept_ids = {
            a["image_id"]
            for a in coco["annotations"]
            if a["category_id"] in target_cats
        }
        print(f"[{split_name}] image filter: any target class (includes cat-only)")
    else:
        kept_ids, cat_plus_obj, obj_only, cat_only_dropped = _image_ids_with_non_cat_object(
            coco["annotations"], target_cats
        )
        print(
            f"[{split_name}] image filter: at least one non-cat target object "
            f"(cat optional). kept cat+object={cat_plus_obj}, "
            f"object-without-cat={obj_only}, dropped cat-only={cat_only_dropped}"
        )

    if balanced:
        print(f"[{split_name}] balancing enabled: downsampling to equal class image counts")
        kept_ids, per_class_kept = _balanced_image_ids_by_primary_class(
            coco["annotations"], kept_ids, target_cats, balanced_seed
        )
        print(
            f"[{split_name}] balanced subset selected with {per_class_kept} images per class "
            f"(primary-class view), total images={len(kept_ids)}"
        )

    filtered_anns = [
        a
        for a in coco["annotations"]
        if a["category_id"] in target_cats and a["image_id"] in kept_ids
    ]
    filtered_images = [img for img in coco["images"] if img["id"] in kept_ids]
    print(f"[{split_name}] images: {len(filtered_images)}, annotations: {len(filtered_anns)}")

    for ann in filtered_anns:
        ann["category_id"] = id_remap[ann["category_id"]]

    out_ann_dir = output_dir / "annotations"
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_ann_dir / f"instances_{split_name}.json"

    new_categories = [
        {
            "id": new_id,
            "name": COCO_NAME_TO_EXPORT[target_cats[old_id]],
            "supercategory": "object",
        }
        for old_id, new_id in id_remap.items()
    ]
    with out_json.open("w") as f:
        json.dump(
            {
                "info": coco.get("info", {}),
                "licenses": coco.get("licenses", []),
                "categories": new_categories,
                "images": filtered_images,
                "annotations": filtered_anns,
            },
            f,
        )
    print(f"[{split_name}] wrote {out_json}")
    return filtered_images, export_names


def _fetch_one(url: str, dest: Path, retries: int) -> tuple[str, bool, str]:
    if dest.is_file():
        return (url, True, "skipped existing")
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)  # noqa: S310
            return (url, True, "ok")
        except (urllib.error.URLError, OSError) as e:
            last_err = str(e)
            if attempt == retries:
                return (url, False, last_err)
    return (url, False, last_err)


def download_images(
    split_name: str,
    img_subdir: str,
    filtered_images: list[dict],
    output_dir: Path,
    workers: int,
    retries: int,
) -> None:
    out_img_dir = output_dir / "images" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    base = f"{COCO_IMAGE_BASE}/{img_subdir}"

    tasks = []
    for img_meta in filtered_images:
        name = img_meta["file_name"]
        url = f"{base}/{name}"
        dest = out_img_dir / name
        tasks.append((url, dest))

    failed: list[str] = []
    done = 0
    total = len(tasks)
    print(f"[{split_name}] downloading {total} images ({workers} workers)...")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_fetch_one, url, dest, retries): (url, dest) for url, dest in tasks
        }
        for fut in as_completed(futures):
            url, ok, msg = fut.result()
            done += 1
            if not ok:
                failed.append(f"{url} ({msg})")
            if done % 500 == 0 or done == total:
                print(f"[{split_name}]  {done}/{total} files processed")

    if failed:
        print(f"[{split_name}] {len(failed)} download failures (showing up to 10):", file=sys.stderr)
        for line in failed[:10]:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[{split_name}] images -> {out_img_dir}")


def coco_to_yolo(ann_json: Path, out_label_dir: Path) -> None:
    with ann_json.open() as f:
        data = json.load(f)

    img_info = {img["id"]: img for img in data["images"]}
    ann_by_image: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    out_label_dir.mkdir(parents=True, exist_ok=True)

    for img_id, anns in ann_by_image.items():
        meta = img_info[img_id]
        w, h = meta["width"], meta["height"]
        stem = Path(meta["file_name"]).stem
        lines = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (out_label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(output_dir: Path, names: list[str]) -> None:
    try:
        path_field = output_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        path_field = output_dir
    path_str = path_field.as_posix()
    if not path_str.startswith((".", "/")):
        path_str = "./" + path_str
    lines = [
        f"path: {path_str}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"wrote {output_dir / 'data.yaml'}")


def verify_labels(labels_root: Path, class_names: list[str]) -> None:
    for split in ["train", "val"]:
        label_dir = labels_root / split
        if not label_dir.is_dir():
            continue
        counts: Counter[int] = Counter()
        for txt in label_dir.glob("*.txt"):
            for line in txt.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                counts[int(line.split()[0])] += 1
        print(f"\n[{split}] instances per class:")
        for cid, name in enumerate(class_names):
            print(f"  {name:12s}: {counts.get(cid, 0):>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Directory containing instances_train2017.json (default: data/annotations/).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("coco_filtered_with_negatives"),
        help="Filtered dataset root (images, labels, annotations, data.yaml).",
    )
    ap.add_argument(
        "--skip-image-download",
        action="store_true",
        help="Only build filtered JSON + YOLO labels + yaml (no HTTP image fetch).",
    )
    ap.add_argument("--workers", type=int, default=8, help="Parallel image downloads.")
    ap.add_argument("--retries", type=int, default=3, help="Per-file download retries.")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Print per-split class counts from label txt files.",
    )
    ap.add_argument(
        "--any-target-class",
        action="store_true",
        help=(
            "Keep every image that contains any target class (includes cat-only images). "
            "Default keeps images with at least one non-cat target object, with or without cat."
        ),
    )
    ap.add_argument(
        "--balanced",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Set to 1 to extract a class-balanced subset (equal images per class via "
            "primary-class downsampling). Set to 0 to keep imbalanced data (default)."
        ),
    )
    ap.add_argument(
        "--balanced-seed",
        type=int,
        default=42,
        help="Random seed for balanced downsampling.",
    )
    args = ap.parse_args()

    ann_root = ensure_annotations(args.annotations_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names_ref: list[str] | None = None
    for split, cfg in SPLITS.items():
        ann_file = ann_root / cfg["ann_file"]
        filtered_images, export_names = filter_split(
            split,
            ann_file,
            cfg["img_subdir"],
            args.output_dir,
            any_target_class=args.any_target_class,
            balanced=bool(args.balanced),
            balanced_seed=args.balanced_seed,
        )
        names_ref = export_names

        if not args.skip_image_download:
            download_images(
                split,
                cfg["img_subdir"],
                filtered_images,
                args.output_dir,
                workers=args.workers,
                retries=args.retries,
            )

        coco_to_yolo(
            args.output_dir / "annotations" / f"instances_{split}.json",
            args.output_dir / "labels" / split,
        )

    if names_ref is not None:
        write_data_yaml(args.output_dir, names_ref)

    if args.verify:
        verify_labels(args.output_dir / "labels", names_ref or [])

    print("\nDone. Classes (remapped ids):", names_ref)


if __name__ == "__main__":
    main()
