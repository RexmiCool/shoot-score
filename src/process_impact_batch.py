"""Batch helper for the impact stage of the incremental loop.

Accepted flattened images are either auto-processed by the impact model or copied
into the impact pending queue and sent to the manual impact labeler.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from detect_impacts_yolo import detect_impacts_yolo

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(path: Path) -> list[Path]:
    """Return image files in a directory."""
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def filter_new_images(src: Path, accepted_dir: Path, rejected_dir: Path) -> list[Path]:
    """Keep only flattened images not already processed by the impact review queues."""
    processed = set()
    for history_dir in (accepted_dir, rejected_dir):
        if history_dir.exists():
            processed |= {p.name for p in history_dir.iterdir() if p.is_file()}
    return [p for p in iter_images(src) if p.name not in processed]


def process_impact_batch(
    src: Path,
    pending_dir: Path,
    accepted_dir: Path,
    weights: Path | None = None,
    conf_thr: float = 0.25,
    iou_thr: float = 0.4,
    device: str | None = None,
) -> int:
    """Process new flattened images into the impact pending queue."""
    src = Path(src)
    pending_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)

    images = filter_new_images(src, accepted_dir, Path(__file__).resolve().parent.parent / "data" / "impact_labelled" / "rejected")
    if not images:
        print(f"[INFO] No new image in {src} (already in accepted/rejected).")
        return 0

    model_path = Path(weights) if weights else None
    used_model = model_path is not None and model_path.exists()

    count = 0
    for image in images:
        dest = pending_dir / image.name
        if used_model:
            detect_impacts_yolo(
                flat_img_path=image,
                weights=str(model_path),
                conf_thr=conf_thr,
                iou_thr=iou_thr,
                output_dir=str(pending_dir),
                device=device or "cpu",
            )
            shader_copy = pending_dir / image.name
            if shader_copy.exists():
                shutil.copy2(image, shader_copy)
            print(f"[AUTO] Impact detection: {image.name} -> {pending_dir}.")
        else:
            shutil.copy2(image, dest)
            print(f"[PENDING] No impact model: {image.name} copied to pending for manual labelling.")
        count += 1

    if not used_model and images:
        labeler = Path(__file__).with_name("label_impacts.py")
        print(f"\n[MANUAL] Launching impact labeler on {pending_dir}")
        subprocess.run(
            [
                sys.executable,
                str(labeler),
                str(pending_dir),
                "--accepted-dir",
                str(accepted_dir),
                "--rejected-dir",
                str(Path(__file__).resolve().parent.parent / "data" / "impact_labelled" / "rejected"),
                "--skip-done",
            ],
            check=False,
        )

    return count


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/homographied/accepted"))
    parser.add_argument("--pending", type=Path, default=Path("data/impact_labelled/pending"))
    parser.add_argument("--accepted", type=Path, default=Path("data/impact_labelled/accepted"))
    parser.add_argument("--weights", type=Path, default=Path("models/yolo_impacts/weights/best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    process_impact_batch(
        src=args.src,
        pending_dir=args.pending,
        accepted_dir=args.accepted,
        weights=args.weights,
        conf_thr=args.conf,
        iou_thr=args.iou,
        device=args.device,
    )


if __name__ == "__main__":
    main()
