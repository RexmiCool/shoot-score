"""Manually sort incoming photos into accepted/rejected.

This script is the first gate of the incremental data loop. New photos arrive in
``incoming`` and must be reviewed by a human before they can be processed by the
axis pipeline.

Usage:
    python src/tri_incoming.py --incoming data/original_photos/incoming \
        --accepted data/original_photos/accepted --rejected data/original_photos/rejected
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(path: Path) -> list[Path]:
    """Return supported images from a directory."""
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def move_file(src: Path, dst_dir: Path) -> None:
    """Move a file to the destination directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if target.exists():
        target.unlink()
    shutil.move(str(src), str(target))


def _display_image(image_path: Path) -> str:
    """Open a window for visual review and wait for a keyboard decision."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Impossible de lire {image_path.name}; image ignorée.")
        return "skip"

    h, w = image.shape[:2]
    scale = min(1.0, 1400 / max(w, h))
    window_name = "Incoming review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(w * scale), int(h * scale))

    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - 52), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cv2.putText(
        image,
        "A=ACCEPT  R=REJECT  S=SKIP  Q=QUITTER",
        (12, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("a"), ord("A"), ord("y"), ord("Y")):
            cv2.destroyAllWindows()
            return "accept"
        if key in (ord("r"), ord("R"), ord("n"), ord("N")):
            cv2.destroyAllWindows()
            return "reject"
        if key in (ord("s"), ord("S")):
            cv2.destroyAllWindows()
            return "skip"
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            return "quit"


def sort_incoming(incoming_dir: Path, accepted_dir: Path, rejected_dir: Path) -> tuple[int, int, int]:
    """Visually review each image in the incoming directory.

    The user can accept, reject, or skip each one with a windowed review like the
    other manual labeling scripts in the project.
    """
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(incoming_dir)
    if not images:
        print(f"[INFO] No image to review in {incoming_dir}.")
        return 0, 0, 0

    accepted_count = 0
    rejected_count = 0
    skipped_count = 0

    index = 0
    while index < len(images):
        image = images[index]
        print(f"\n--- {index + 1}/{len(images)} {image.name} ---")
        decision = _display_image(image)
        if decision == "accept":
            move_file(image, accepted_dir)
            accepted_count += 1
            index += 1
        elif decision == "reject":
            move_file(image, rejected_dir)
            rejected_count += 1
            index += 1
        elif decision == "skip":
            skipped_count += 1
            index += 1
        else:
            print("[QUIT] Review interrupted.")
            break

    print(f"\n[OK] accepted={accepted_count} rejected={rejected_count} skipped={skipped_count}")
    return accepted_count, rejected_count, skipped_count


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", type=Path, default=Path("data/original_photos/incoming"))
    parser.add_argument("--accepted", type=Path, default=Path("data/original_photos/accepted"))
    parser.add_argument("--rejected", type=Path, default=Path("data/original_photos/rejected"))
    args = parser.parse_args()

    sort_incoming(args.incoming, args.accepted, args.rejected)


if __name__ == "__main__":
    main()
