"""Prepare four ROI datasets from manually labelled target photos.

Input labels are written by ``label_axis_markers.py``. Each output sample is
centered on the expected viewfinder position, not on the labelled point, so
the model must learn both presence and the point offset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from axis_marker_common import AXIS_NAMES, PATCH_SIZE, crop_patch, expected_axis_points, load_axis_labels

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and "_axis_labels" not in p.stem)


def prepare_one(
    image_path: Path,
    output_root: Path,
    screen_aspect: float,
    frame_ratio: float,
    val_ratio: float,
) -> int:
    label_path = image_path.with_name(f"{image_path.stem}_axis_labels.json")
    if not label_path.exists():
        print(f"[SKIP] {image_path.name}: labels absents")
        return 0
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[SKIP] {image_path.name}: image illisible")
        return 0
    labels = load_axis_labels(label_path)
    expected = expected_axis_points(image.shape[1], image.shape[0], screen_aspect, frame_ratio)
    split = "val" if random.random() < val_ratio else "train"
    sample_count = 0

    for position in AXIS_NAMES:
        position_dir = output_root / position / split
        position_dir.mkdir(parents=True, exist_ok=True)
        expected_point = expected[position]
        patch = crop_patch(image, expected_point.x, expected_point.y, PATCH_SIZE)
        label = labels[position]
        if label is None:
            present = 0
            dx = 0.0
            dy = 0.0
        else:
            present = 1
            dx = (label.x - expected_point.x) / (PATCH_SIZE / 2)
            dy = (label.y - expected_point.y) / (PATCH_SIZE / 2)
            distance = np.hypot(
                label.x - expected_point.x,
                label.y - expected_point.y
            )

            print(position, round(distance))
            if distance > 150:
                print(f"[OUTLIER] {image_path.name} {position} {distance:.1f}")

        
        stem = f"{image_path.stem}_{position}"
        image_out = position_dir / f"{stem}.png"
        label_out = position_dir / f"{stem}.json"
        cv2.imwrite(str(image_out), patch)
        label_out.write_text(
            json.dumps({"present": present, "dx": float(np.clip(dx, -1, 1)), "dy": float(np.clip(dy, -1, 1))}),
            encoding="utf-8",
        )
        sample_count += 1
    return sample_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Photo ou dossier contenant les JSON *_axis_labels.json")
    parser.add_argument("--out", type=Path, default=Path("data/axis_markers"))
    parser.add_argument("--screen-aspect", type=float, default=9 / 16)
    parser.add_argument("--frame-ratio", type=float, default=0.82)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    images = collect_images(args.path)
    total = sum(prepare_one(p, args.out, args.screen_aspect, args.frame_ratio, args.val_ratio) for p in images)
    print(f"[OK] {total} patches crees dans {args.out}")


if __name__ == "__main__":
    main()
