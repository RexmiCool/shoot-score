"""Batch flattening stage for the incremental pipeline.

Images validated in the axis stage are flattened to produce the flat images used
in the impact stage. This script loads the retained axis labels and writes the
flattened images into the homographied pending queue.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from axis_marker_common import load_axis_labels
from flatten_markers import flatten_with_markers, save_flatten

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(path: Path) -> list[Path]:
    """List image files in a directory."""
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def filter_new_images(src: Path, pending_dir: Path, accepted_dir: Path, rejected_dir: Path) -> list[Path]:
    """Keep only images not already flattened or already reviewed in downstream queues."""
    processed = set()
    for history_dir in (pending_dir, accepted_dir, rejected_dir):
        if history_dir.exists():
            processed |= {
                p.stem.removesuffix("_flat")
                for p in history_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            }
    return [p for p in iter_images(src) if p.stem not in processed]


def process_homography_batch(
    src: Path,
    pending_dir: Path,
    accepted_dir: Path | None = None,
    rejected_dir: Path | None = None,
    debug: bool = False,
) -> int:
    """Flatten accepted axis images and send the results into the homography review queue."""
    src = Path(src)
    pending_dir = Path(pending_dir)
    accepted_dir = Path(accepted_dir) if accepted_dir is not None else pending_dir.parent / "accepted"
    rejected_dir = Path(rejected_dir) if rejected_dir is not None else pending_dir.parent / "rejected"

    pending_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    images = filter_new_images(src, pending_dir, accepted_dir, rejected_dir)
    if not images:
        print(f"[INFO] No new image in {src} (already in pending/accepted/rejected).")
        return 0

    count = 0
    for image_path in images:
        label_path = image_path.with_name(f"{image_path.stem}_axis_labels.json")
        if not label_path.exists():
            print(f"[SKIP] Missing axis labels for {image_path.name}")
            continue

        labels = load_axis_labels(label_path)
        points = {name: value for name, value in labels.items() if value is not None}
        if len(points) < 4:
            print(f"[SKIP] Incomplete markers for {image_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[SKIP] Cannot read {image_path.name}")
            continue

        result = flatten_with_markers(image, points)
        flat_path = save_flatten(result, pending_dir, image_path.stem, debug=debug)
        print(f"[OK] Flattened {image_path.name} -> {flat_path.name}")
        count += 1

    if count:
        review_script = Path(__file__).with_name("review_pending.py")
        print(f"\n[MANUAL] Launching homography review on {pending_dir}")
        subprocess.run(
            [
                sys.executable,
                str(review_script),
                "--pending",
                str(pending_dir),
                "--accepted",
                str(accepted_dir),
                "--rejected",
                str(rejected_dir),
            ],
            check=False,
        )

    return count


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/axis_labelled/accepted"))
    parser.add_argument("--pending", type=Path, default=Path("data/homographied/pending"))
    parser.add_argument("--accepted", type=Path, default=Path("data/homographied/accepted"))
    parser.add_argument("--rejected", type=Path, default=Path("data/homographied/rejected"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    process_homography_batch(
        args.src,
        args.pending,
        accepted_dir=args.accepted,
        rejected_dir=args.rejected,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
