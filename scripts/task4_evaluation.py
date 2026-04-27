#!/usr/bin/env python3

import json
import glob
import collections
import shutil
import subprocess
from pathlib import Path
import torch
from ultralytics import YOLO

# ============ Settings ============
PROJECT_ROOT = Path(__file__).parent.parent 
# fall back to pretrained small model if missing
MODEL_PATH = PROJECT_ROOT / "runs/detect/train/weights/best.pt"
FALLBACK_MODEL = "yolov8n.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_YAML = PROJECT_ROOT / "curated_yolo/data.yaml"
TEST_IMAGES = PROJECT_ROOT / "curated_yolo/images/test"
PREDICT_DIR = PROJECT_ROOT / "runs/predict/pred_test"
PREDICT_LABELS = PREDICT_DIR / "labels"
MISCHIEF_OUT = PROJECT_ROOT / "mischief_pred_out"
EXAMPLES_DIR = MISCHIEF_OUT / "examples"

CLASS_NAMES = ["cat", "cup", "plant", "laptop", "keyboard", "vase", "scissors"]

# ============ 1. Model Prediction ============
print("=" * 60)
print("Step 1: Running model prediction on test set")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"Model not found at {MODEL_PATH}; falling back to pretrained {FALLBACK_MODEL}")
    model = YOLO(FALLBACK_MODEL)
else:
    model = YOLO(str(MODEL_PATH))

model.predict(
    source=str(TEST_IMAGES),
    save=True,
    save_txt=True,
    project=str(PROJECT_ROOT / "runs/predict"),
    name="pred_test",
    exist_ok=True,
    device=DEVICE,
    workers=0,
)

print(f" Predictions saved to {PREDICT_LABELS}")

# ============ 2. Running mischief system ============
print("\n" + "=" * 60)
print("Step 2: Running mischief detection on predictions")
print("=" * 60)

try:
    subprocess.run([
        "python", str(PROJECT_ROOT / "scripts/mischief.py"),
        "--images-dir", str(TEST_IMAGES),
        "--labels-dir", str(PREDICT_LABELS),
        "--out", str(MISCHIEF_OUT),
        "--draw"
    ], check=True)
except subprocess.CalledProcessError as e:
    print("mischief.py failed:", e)
    raise

print(f" Mischief outputs saved to {MISCHIEF_OUT}")

# ============ 3. Statistical analysis ============
print("\n" + "=" * 60)
print("Step 3: Statistical analysis of mischief scores")
print("=" * 60)

json_files = list(MISCHIEF_OUT.glob("*.mischief.json"))
print(f"Found {len(json_files)} JSON files")

# initialize empty lists so later code is safe even if no files are found
scores = []
warnings = []
for f in json_files:
    data = json.load(open(f))
    scores.append(data.get("mischief_score", 0))
    warnings.append(data.get("top_warning", ""))
    
    print(f"\n Score statistics:")
    print(f"   Min: {min(scores)}")
    print(f"   Median: {sorted(scores)[len(scores)//2]}")
    print(f"   Max: {max(scores)}")
    print(f"   Mean: {sum(scores)/len(scores):.2f}")
    
    print(f"\n Top warnings:")
    for warn, count in collections.Counter(warnings).most_common(5):
        print(f"   {warn}: {count}")

# ============ 4. Copying example images ============
print("\n" + "=" * 60)
print("Step 4: Copying example images for report")
print("=" * 60)

EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

high, low, medium = [], [], []
for j in json_files:
    data = json.load(open(j))
    score = data["mischief_score"]
    if score >= 50:
        high.append(j)
    elif score <= 10:
        low.append(j)
    else:
        medium.append(j)

for category, lst, limit in [("high", high, 3), ("medium", medium, 3), ("low", low, 3)]:
    cat_dir = EXAMPLES_DIR / category
    cat_dir.mkdir(exist_ok=True)
    for j in lst[:limit]:
        img = MISCHIEF_OUT / (j.stem + ".annotated.jpg")
        if img.exists():
            shutil.copy2(img, cat_dir / img.name)
            print(f"   Copied {img.name} to {category}/")

print(f"Example images saved to {EXAMPLES_DIR}")

# ============ 5. mAP evaluation ============
print("\n" + "=" * 60)
print("Step 5: mAP evaluation (Task 4A)")
print("=" * 60)


results = None
try:
    results = model.val(data=str(DATA_YAML), split="test")
except Exception as e:
    print("model.val failed:", e)

print(f"\n Primary metrics:")
print(f"   mAP@0.5:     {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
print(f"   mAP@0.5:0.95: {results.box.map:.4f} ({results.box.map*100:.2f}%)")

if results is not None:
    print(f"\n Per-class AP@0.5:")
    ap_list = list(results.box.ap)
    for i, name in enumerate(CLASS_NAMES):
        if i < len(ap_list):
            print(f"   {name:12s}: {ap_list[i]:.4f} ({ap_list[i]*100:.2f}%)")
        else:
            print(f"   {name:12s}: N/A")

    best_idx = int(results.box.ap.argmax())
    worst_idx = int(results.box.ap.argmin())
    print(f"\n Best performing class: {CLASS_NAMES[best_idx]} ({results.box.ap[best_idx]*100:.1f}%)")
    print(f" Worst performing class: {CLASS_NAMES[worst_idx]} ({results.box.ap[worst_idx]*100:.1f}%)")
else:
    print("Skipping per-class AP: validation failed or returned no results")

# ============ 6. save results in JSON ============
summary = {
    "n_images": len(json_files),
    "score_stats": {
        "min": min(scores) if scores else 0,
        "median": sorted(scores)[len(scores)//2] if scores else 0,
        "max": max(scores) if scores else 0,
        "mean": sum(scores)/len(scores) if scores else 0,
    },
    "mAP": {
        "mAP@0.5": float(results.box.map50) if results is not None else 0.0,
        "mAP@0.5:0.95": float(results.box.map) if results is not None else 0.0,
    },
    "per_class_AP": {name: float(results.box.ap[i]) for i, name in enumerate(CLASS_NAMES) if results is not None and i < len(results.box.ap)},
}

summary_path = MISCHIEF_OUT / "evaluation_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n Saved evaluation summary to {summary_path}")

print("\n" + "=" * 60)
print("Task 4A & 4B complete!")
print(f"   - Predictions: {PREDICT_DIR}")
print(f"   - Mischief outputs: {MISCHIEF_OUT}")
print(f"   - Example images: {EXAMPLES_DIR}")
print("=" * 60)