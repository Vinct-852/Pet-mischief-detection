# Curated Detection Dataset

This document is the technical specification for dataset construction in this project. It covers business intent, architecture, implementation logic, and reproducibility controls for:

- `coco_filtered/` (stage 1: class-filtered COCO subset in YOLO format)
- `curated_yolo/` (stage 2: reproducible train/val/test split)
- `curated_yolo_balanced/` (stage 2.5: optional train-only balancing subset)

Exact counts depend on configuration and source availability. For any specific run, use `dataset_manifest.json` as source of truth.

## 1) Business Logic and Product Intent

The model target is pet-mischief detection in realistic household scenes. The business requirement is not generic COCO detection, but robust detection of cats and common risky/valuable objects around them.

### Detection classes

The pipeline uses seven classes:

- `cat`
- `cup`
- `plant` (from COCO `potted plant`)
- `laptop`
- `keyboard`
- `vase`
- `scissors`

### Why this class set

- Represents frequent home/desk contexts where pet interference matters.
- Mixes high-frequency and rare classes to evaluate robustness, not only easy classes.
- Preserves multi-object context required by real-world mischief events.

### Image inclusion policy

Configurable by `ANY_TARGET_CLASS`:

- `True` (current default): include images containing any target class (max coverage).
- `False`: include images where `cat` co-occurs with at least one other target class (event-focused subset).

This is the primary business-rule switch that controls how broad versus behavior-specific the data pool is.

## 2) Technology Stack and Runtime

- **Language:** Python 3.11+
- **Data source:** COCO 2017 annotations + image CDN
- **Model training/eval:** Ultralytics YOLOv8
- **Label format:** YOLO detection text format (`class_id cx cy w h`)
- **Execution mode:** notebook-driven (`proj_v3.ipynb`) with deterministic seed-based split

## 3) Pipeline Architecture

```text
COCO annotations zip
        |
        v
Stage 1 (Task 1): filter classes + select images + COCO->YOLO conversion
        |
        v
coco_filtered/
  images/{train,val}
  labels/{train,val}
  annotations/instances_{train,val}.json
  data.yaml
        |
        v
Stage 2 (Task 2): reproducible shuffle + 80/10/10 split
        |
        v
curated_yolo/
  images/{train,val,test}
  labels/{train,val,test}
  data.yaml
  dataset_manifest.json
        |
        v
Optional Stage 2.5: class-balanced train subset creation
        |
        v
curated_yolo_balanced/
  images/{train,val,test}
  labels/{train,val,test}
  data.yaml
  dataset_manifest.json
```

## 4) Design Decisions

- **Two-stage construction** separates expensive COCO filtering from flexible split experiments.
- **Seeded split** makes experiments reproducible and comparable over time.
- **Manifest output** gives machine-readable evidence for audit/reporting.
- **Optional balancing stage** allows fairness/performance tuning without mutating the canonical curated dataset.
- **Val/test kept stable by default in balancing stage** to avoid evaluation leakage and preserve comparability.

## 5) Stage 1 Implementation (COCO subset preparation)

### Inputs

- `annotations_trainval2017.zip`
- Class mapping config (`TARGET_CLASSES`)
- Selection mode (`ANY_TARGET_CLASS`)

### Core logic

1. Load COCO JSON (`instances_train2017.json`, `instances_val2017.json`).
2. Keep only configured category IDs.
3. Build kept image IDs:
   - Any target class (`ANY_TARGET_CLASS=True`), or
   - Cat + at least one other target class (`False` mode).
4. Filter annotations/images to kept IDs.
5. Remap category IDs to contiguous model IDs `0..6`.
6. Download only referenced images.
7. Convert COCO bbox to YOLO normalized format and write labels.

### Output contract

- One label file per image.
- Label and image basenames must match.
- IDs in label files must align with `data.yaml` names order.

## 6) Stage 2 Implementation (reproducible split)

### Inputs

- `coco_filtered/images/{train,val}`
- `coco_filtered/labels/{train,val}`
- `SEED`, `TRAIN_PCT`, `VAL_PCT`, `TEST_PCT`

### Core logic

1. Collect valid `(image, label)` pairs.
2. Shuffle pair indices with deterministic seed.
3. Compute integer cut points for train/val.
4. Assign remainder to test to keep total exact.
5. Copy files into `curated_yolo`.
6. Write:
   - `data.yaml`
   - `dataset_manifest.json` (per-split image counts and class-instance counts)

### Important implementation behavior

- No per-class cap is applied in stage 2.
- Class distribution is inherited from filtered COCO pool + random split.
- Re-running with same inputs/config yields same split assignment.

## 7) Stage 2.5 Implementation (balanced subset)

This optional stage addresses class imbalance in train data.

### Inputs

- `curated_yolo/` train split
- `BALANCE_MODE`:
  - `min`: target is minority class instance count
  - `fixed`: target is `TARGET_PER_CLASS`

### Selection algorithm

- Parse class-instance counts per label file.
- Greedy multi-label selection:
  - Compute class deficits vs target.
  - Select image that best reduces deficits.
  - Repeat until no deficit can be improved.
- Optional trim pass removes redundant images while keeping target coverage.

### Output strategy

- Balanced `train` subset.
- By default copy original `val/test` unchanged.
- Emit separate `curated_yolo_balanced/data.yaml` and `dataset_manifest.json`.

## 8) Reproducibility and Determinism

- `SEED` controls split order.
- `BALANCED_SEED` controls subset selection tie-breaking.
- Stable outputs depend on:
  - unchanged source files
  - identical config values
  - same directory contents

For experiment logging, store:

- `dataset_manifest.json`
- training command/config
- model artifact path (`weights/best.pt`)

## 9) Performance and Operational Notes

- Dataset build can be slow due to:
  - network-bound image download (stage 1)
  - large-scale file copy (stage 2/2.5)
  - label parsing + greedy selection complexity (stage 2.5)
- On Windows, `copy2` adds metadata overhead versus plain `copy`.
- For rapid iteration, disable val/test copy in stage 2.5 if not required.

## 10) Dataset Format Contract (YOLO)

- Images: `images/<split>/...`
- Labels: `labels/<split>/<same_stem>.txt`
- Label line: `class_id center_x center_y width height`
- Coordinates normalized to `[0,1]`
- `data.yaml` keys:
  - `path`
  - `train`
  - `val`
  - `test`
  - `nc`
  - `names`

## 11) Validation and Sanity Checklist

Before training:

1. `data.yaml` paths resolve correctly.
2. Each image has a corresponding label file.
3. Class IDs in labels are within `[0, nc-1]`.
4. `dataset_manifest.json` class names match training config.
5. Train/val/test counts are non-zero and align with expected split ratios.

## 12) Run Artifacts to Reference

- `curated_yolo/dataset_manifest.json` - canonical split stats
- `curated_yolo_balanced/dataset_manifest.json` - balanced split stats
- training `runs/detect/.../results.csv` - epoch metrics trend
- validation `runs/detect/val*/` - confusion matrix and PR/recall curves
