# Pet-mischief-detection

## COCO subset (`scripts/prepare_coco_subset.py`)

This script: ensures annotations → filters seven target classes → optional parallel per-image download → YOLO `.txt` labels → `data.yaml`. It avoids downloading full `train2017.zip` / `val2017.zip` by fetching only `annotations_trainval2017.zip` and then downloading individual image files from the COCO CDN.

**Image selection (default):** keeps only images where **cat** appears together with **at least one other** target class (e.g. cat+cup, cat+laptop, cat+vase—any pairwise combo with cat). This matches a pet-vs-objects use case better than “any image containing any class.” Use `--any-target-class` to revert to the older behavior: keep every image that contains *any* of the seven classes (large union).

Class IDs are **0–6** in **sorted original COCO category id** order (see console output: `Class order (by COCO id): …`). That order may differ from a hand-written static `data.yaml`; always align training configs with the remapped IDs this script prints and writes.

COCO names the plant class **`potted plant`**; the script exposes it as **`plant`** in `data.yaml` and labels.

### Full pipeline (annotations + images + labels + yaml)

```bash
cd /path/to/Pet-mischief-detection
python3 scripts/prepare_coco_subset.py --verify
```

Optional: union of all target classes (no cat co-occurrence requirement):

```bash
python3 scripts/prepare_coco_subset.py --any-target-class --verify
```

### JSON + YOLO labels + yaml only (no image download)

Use this if you already have images locally or will sync them another way:

```bash
python3 scripts/prepare_coco_subset.py --skip-image-download --verify
```
