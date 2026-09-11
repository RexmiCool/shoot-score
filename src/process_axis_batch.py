"""Batch helper for the axis-marker stage of the incremental loop.

When a valid axis model exists, the script runs auto-detection on all accepted
images and writes YOLO-style predictions next to the copied files. When no model
exists yet, it copies the raw images into the pending queue and launches the
manual axis labeler so the user can annotate them visually.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from detect_axis_markers import detect_axis_markers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(path: Path) -> list[Path]:
    """Return supported image files from a directory."""
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def filter_new_images(src: Path, pending_dir: Path, accepted_dir: Path, rejected_dir: Path) -> list[Path]:
    """Keep only images not already present in the pending or final queues."""
    processed = set()
    for history_dir in (pending_dir, accepted_dir, rejected_dir):
        if history_dir.exists():
            processed |= {p.name for p in history_dir.iterdir() if p.is_file()}
    return [p for p in iter_images(src) if p.name not in processed]


def write_axis_prediction_json(image_path: Path, output_path: Path, detections: dict) -> None:
    """Write an axis JSON file compatible with the existing label format."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image {image_path}")
    h, w = img.shape[:2]

    markers: dict[str, dict[str, float] | None] = {}
    for name, detection in detections.items():
        if detection is None:
            markers[name] = None
            continue
        markers[name] = {"x": float(detection.point.x), "y": float(detection.point.y)}

    payload = {
        "image_width": int(w),
        "image_height": int(h),
        "markers": markers,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_axis_batch(
    src: Path,
    pending_dir: Path,
    accepted_dir: Path,
    rejected_dir: Path | None = None,
    weights: Path | None = None,
    conf_thr: float = 0.25,
    iou_thr: float = 0.45,
    imgsz: int = 1056,
) -> int:
    """Process the accepted axis images into the pending queue.

    If a model exists, auto-detect markers and create a JSON label next to the
    pending copy. Otherwise, simply copy the images to pending so a human can
    label them manually.
    """
    src = Path(src)
    pending_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = Path(rejected_dir) if rejected_dir is not None else accepted_dir.parent / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    images = filter_new_images(src, pending_dir, accepted_dir, rejected_dir)
    if not images:
        print(f"[INFO] No new image in {src} (already in pending/accepted/rejected).")
        return 0

    model_path = Path(weights) if weights else None
    used_model = model_path is not None and model_path.exists()

    count = 0
    for image in images:
        dest = pending_dir / image.name
        shutil.copy2(image, dest)
        if used_model:
            detections = detect_axis_markers(
                dest,
                model_path,
                conf_thr=conf_thr,
                iou_thr=iou_thr,
                imgsz=imgsz,
            )
            labels_path = dest.with_name(f"{dest.stem}_axis_labels.json")
            write_axis_prediction_json(dest, labels_path, detections)
            print(f"[AUTO] Axis detection: {dest.name} -> {labels_path.name}")
        else:
            print(f"[PENDING] No axis model: {dest.name} copied to pending for manual labelling.")
        count += 1

    # if not used_model and images:
    labeler = Path(__file__).with_name("label_axis_markers.py")
    print(f"\n[MANUAL] Launching axis labeler on {pending_dir}")
    subprocess.run(
        [
            sys.executable,
            str(labeler),
            str(pending_dir),
            "--accepted-dir",
            str(accepted_dir),
            "--rejected-dir",
            str(Path(__file__).resolve().parent.parent / "data" / "axis_labelled" / "rejected"),
        ],
        check=False,
    )

    return count


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/original_photos/accepted"))
    parser.add_argument("--pending", type=Path, default=Path("data/axis_labelled/pending"))
    parser.add_argument("--accepted", type=Path, default=Path("data/axis_labelled/accepted"))
    parser.add_argument("--rejected", type=Path, default=Path("data/axis_labelled/rejected"))
    parser.add_argument("--weights", type=Path, default=Path("models/axis_yolo/weights/best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1056)
    args = parser.parse_args()

    process_axis_batch(
        src=args.src,
        pending_dir=args.pending,
        accepted_dir=args.accepted,
        rejected_dir=args.rejected,
        weights=args.weights,
        conf_thr=args.conf,
        iou_thr=args.iou,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
