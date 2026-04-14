# Curated detection dataset

This document describes how the YOLO-format dataset is built, which preprocessing applies, and how to reproduce the **train / validation / test** split. **Exact image counts and disk sizes** depend on your run; after building, see `curated_yolo/dataset_manifest.json` (machine-readable) and the console summary from `split_yolo_dataset.py`.

## Layout (after full pipeline)

```text
coco_filtered/                    # Stage 1: COCO subset + COCO’s train/val folders
├── images/train
├── images/val
├── labels/train
├── labels/val
├── annotations/instances_{train,val}.json
└── data.yaml

curated_yolo/                     # Stage 2: reproducible 80/10/10 split for training
├── images/{train,val,test}
├── labels/{train,val,test}
├── data.yaml                     # train / val / test paths + class names
└── dataset_manifest.json         # sizes, class distribution, seed, percentages
```

## Pre-processing (stage 1: `scripts/prepare_coco_subset.py`)

1. **Annotations only from COCO** — Downloads `annotations_trainval2017.zip` (not full image zips).
2. **Class filter** — Seven COCO classes: `cat`, `cup`, `laptop`, `keyboard`, `vase`, `potted plant` (exported as **`plant`**), `scissors`. Category IDs are remapped to **0–6** in sorted original COCO category id order.
3. **Image selection (default)** — Keeps images where a **cat** co-occurs with **at least one** other target class (cat + cup, cat + laptop, …). Use `--any-target-class` for the union of all seven classes without requiring cat.
4. **Images** — Downloads only the JPEGs referenced by the filtered annotations from COCO’s CDN.
5. **YOLO labels** — For each kept image, writes one `.txt` per image: one line per box, `class_id cx cy w h` in normalized coordinates (YOLO detection format).

## Split (stage 2: `scripts/split_yolo_dataset.py`)

1. **Pools** all `(image, label)` pairs from `coco_filtered/images/{train,val}` and matching `labels/`.
2. **Shuffles** the list with a fixed **`--seed`** (default **42**).
3. **Splits** **80% / 10% / 10%** into `train` / `val` / `test` using integer percentages (remainder goes to test so that counts sum exactly).
4. **Copies** files into `curated_yolo/` (source tree is unchanged).
5. **Writes** `data.yaml` with `train`, `val`, and **`test`** keys, and **`dataset_manifest.json`** with per-split image counts, **image bytes**, and **instances per class**.

Re-running with the same `--source`, `--seed`, and percentages yields the same assignment.

## End-to-end commands

```bash
# 1) Build coco_filtered/ (YOLO labels + optional image download)
python3 scripts/prepare_coco_subset.py --verify

# 2) Build curated_yolo/ with reproducible 80/10/10 split
python3 scripts/split_yolo_dataset.py --force
```

Custom split (example: 70/15/15, seed 123):

```bash
python3 scripts/split_yolo_dataset.py --seed 123 --train-pct 70 --val-pct 15 --test-pct 15 --force
```

## Format for training (YOLO)

- **Images:** RGB files under `images/<split>/`.
- **Labels:** Same basename as the image, extension `.txt`, UTF-8 text.
- **Each line:** `class_id center_x center_y width height` with values in **\[0, 1\]** relative to image width/height.
- **`data.yaml`:** `path`, `train`, `val`, `test`, `nc`, `names` — compatible with Ultralytics YOLO-style configs.

Conversion from COCO JSON to YOLO lines is implemented in `prepare_coco_subset.py` (`coco_to_yolo`). The split step does not alter label contents.

## Where to read sizes and class distribution

| Artifact | Content |
|----------|---------|
| `curated_yolo/dataset_manifest.json` | Per-split `image_bytes`, `instances_per_class`, `seed`, percentages |
| Terminal output of `split_yolo_dataset.py` | Short table of counts per split |
| `python3 scripts/prepare_coco_subset.py --verify` | Class counts under **COCO’s** train/val folders before re-split |

After you run the split, paste or summarize `dataset_manifest.json` in reports if you need fixed numbers in prose; the JSON is the source of truth for that run.
