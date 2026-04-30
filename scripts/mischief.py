#!/usr/bin/env python3
"""
Compute mischief scores from YOLO-format detections and produce annotated images/JSON.

Usage:
  python3 scripts/mischief.py --image images/example.jpg --labels labels/example.txt --out out
  python3 scripts/mischief.py --images-dir curated_yolo/images --labels-dir curated_yolo/labels --out out

The script implements the project's rules for classes:
 0 cat, 1 cup, 2 plant, 3 laptop, 4 keyboard, 5 vase, 6 scissors

Rules:
 - Cat + Cup: IoU>0 or center_distance < 0.12*diag -> High (weight 2)
 - Cat + Plant/Vase: IoU>0 or center_distance < 0.15*diag -> High (weight 2)
 - Cat + Laptop: IoU>0 or center_distance < 0.18*diag -> Medium-High (weight 1.5)
 - Cat + Keyboard: center_distance < 0.18*diag -> Medium (weight 1)
 - Cat + Scissors: IoU>0 or center_distance < 0.10*diag -> High (weight 2)

Safety rule: if there are no non-cat detections (only cat(s) or no detections) -> safe/low risk.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = [
    "cat",
    "cup",
    "plant",
    "laptop",
    "keyboard",
    "vase",
    "scissors",
]


@dataclass
class Det:
    cid: int
    conf: Optional[float]
    cx: float
    cy: float
    w: float
    h: float
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1,y1,x2,y2 pixel coords


def read_yolo_labels(path: Path) -> List[Det]:
    if not path.is_file():
        return []
    out: List[Det] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        vals = [float(p) for p in parts[1:5]]
        conf = float(parts[5]) if len(parts) >= 6 else None
        out.append(Det(cid=cid, conf=conf, cx=vals[0], cy=vals[1], w=vals[2], h=vals[3]))
    return out


def to_pixels(det: Det, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    cx = det.cx * img_w
    cy = det.cy * img_h
    bw = det.w * img_w
    bh = det.h * img_h
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return (max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2))


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


RULES = {
    "cup": {"thresh": 0.12, "weight": 2.0, "msg": "Mischief Alert! Your cat is plotting against your drink!"},
    "plant": {"thresh": 0.15, "weight": 2.0, "msg": "High Risk: cat near fragile object (plant)."},
    "vase": {"thresh": 0.15, "weight": 2.0, "msg": "High Risk: cat near fragile object (vase)."},
    "laptop": {"thresh": 0.18, "weight": 1.5, "msg": "Warning: cat near laptop — risk to device or cables."},
    "keyboard": {"thresh": 0.18, "weight": 1.0, "msg": "Caution: cat on/near keyboard."},
    "scissors": {"thresh": 0.10, "weight": 2.0, "msg": "Danger: scissors close to pet!"},
}


def assess_image(image_path: Path, label_path: Path, out_dir: Path, draw_flag: bool = True) -> Dict:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    diag = math.hypot(w, h)

    dets = read_yolo_labels(label_path)
    if not label_path.is_file():
        print(f"warning: missing labels for {label_path}", file=sys.stderr)
    for d in dets:
        d.bbox = to_pixels(d, w, h)

    cats = [d for d in dets if d.cid == 0]
    others = [d for d in dets if d.cid != 0]

    triggered: List[Dict] = []
    raw_score = 0.0

    for cat in cats:
        for other in others:
            cls_name = CLASS_NAMES[other.cid] if 0 <= other.cid < len(CLASS_NAMES) else str(other.cid)
            if cls_name not in RULES:
                continue
            rule = RULES[cls_name]
            box_cat = cat.bbox
            box_other = other.bbox
            pair_iou = iou(box_cat, box_other)
            cd = center_distance(box_cat, box_other)
            norm_cd = cd / diag
            triggered_flag = False
            # rule triggers if IoU>0 OR center dist < thresh
            if pair_iou > 0 or norm_cd < rule["thresh"]:
                triggered_flag = True
            if triggered_flag:
                raw_score += rule["weight"]
                triggered.append({
                    "cat_bbox": box_cat,
                    "other_class": cls_name,
                    "other_bbox": box_other,
                    "iou": pair_iou,
                    "center_dist_norm": norm_cd,
                    "weight": rule["weight"],
                    "message": rule["msg"],
                })

    total_possible = 2.0 * max(0, len(others))
    if total_possible <= 0:
        mischief_score = 0
    else:
        mischief_score = int(min(100, round((raw_score / total_possible) * 100)))

    top_warning = None
    if triggered:
        # pick highest-weight triggered rule
        top = max(triggered, key=lambda r: r["weight"])
        top_warning = top["message"]
    else:
        if len(cats) > 0 and len(others) == 0:
            top_warning = "All clear. Pet is alone or resting — low risk."
        else:
            top_warning = "No issues detected."

    result = {
        "image": str(image_path),
        "detections": [
            {"class": CLASS_NAMES[d.cid] if 0 <= d.cid < len(CLASS_NAMES) else str(d.cid), "bbox": d.bbox, "conf": d.conf}
            for d in dets
        ],
        "mischief_score": mischief_score,
        "top_warning": top_warning,
        "triggered": triggered,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / (image_path.stem + ".mischief.json")
    json_path.write_text(json.dumps(result, indent=2))

    if draw_flag:
        draw_img = img.copy()
        draw_ctx = ImageDraw.Draw(draw_img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        # colors per class
        colors = ["#FF0000", "#0000FF", "#00AA00", "#FF8800", "#8888FF", "#AA00AA", "#0066CC"]
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            col = colors[d.cid % len(colors)] if 0 <= d.cid < len(colors) else "#FFFFFF"
            draw_ctx.rectangle([x1, y1, x2, y2], outline=col, width=2)
            label = f"{CLASS_NAMES[d.cid] if 0<=d.cid<len(CLASS_NAMES) else d.cid}"
            if d.conf is not None:
                label += f" {d.conf:.2f}"
            bbox = draw_ctx.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            draw_ctx.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=col)
            draw_ctx.text((x1 + 2, y1 - text_h - 2), label, fill="#fff", font=font)

        # overlay score and top warning
        summary = f"Mischief: {mischief_score}% - {top_warning}"
        bbox = draw_ctx.textbbox((0, 0), summary, font=font)
        sw = bbox[2] - bbox[0]
        sh = bbox[3] - bbox[1]

        draw_ctx.rectangle([5, 5, 10 + sw, 10 + sh], fill="#222")
        draw_ctx.text((8, 8), summary, fill="#fff", font=font)

        out_img_path = out_dir / (image_path.stem + ".annotated.jpg")
        draw_img.save(out_img_path)

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, help="Single image path to analyze")
    ap.add_argument("--labels", type=Path, help="YOLO labels file for single image")
    ap.add_argument("--images-dir", type=Path, help="Directory with images (paired with labels-dir)")
    ap.add_argument("--labels-dir", type=Path, help="Directory with labels matching images-dir")
    ap.add_argument("--out", type=Path, default=Path("mischief_out"), help="Output directory")
    ap.add_argument("--draw", action="store_true", help="Write annotated images")
    args = ap.parse_args()

    results = []
    if args.image:
        lbl = args.labels or (args.image.with_suffix(".txt"))
        r = assess_image(args.image, lbl, args.out, draw_flag=args.draw)
        results.append(r)
    elif args.images_dir and args.labels_dir:
        for img in sorted(args.images_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            lbl = args.labels_dir / (img.stem + ".txt")
            r = assess_image(img, lbl, args.out, draw_flag=args.draw)
            results.append(r)
    else:
        ap.error("Provide --image and --labels, or --images-dir and --labels-dir")

    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
