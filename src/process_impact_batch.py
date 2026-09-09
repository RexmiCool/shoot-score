"""Batch helper for the impact stage of the incremental loop.

Accepted flattened images are either auto-processed by the impact model or copied
into the impact pending queue and sent to the manual impact labeler.
"""

from __future__ import annotations

import argparse
import json
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


def filter_new_images(src: Path, pending_dir: Path, accepted_dir: Path, rejected_dir: Path) -> list[Path]:
    """Keep only flattened images not already present in the pending or final review queues."""
    processed = set()
    for history_dir in (pending_dir, accepted_dir, rejected_dir):
        if history_dir.exists():
            processed |= {p.name for p in history_dir.iterdir() if p.is_file()}
    return [p for p in iter_images(src) if p.name not in processed]


def write_impact_prediction_json(image_path: Path, model_json_path: Path, output_path: Path) -> None:
    """Convert the YOLO model output to the editable impact label format used by the review tool."""
    with model_json_path.open("r", encoding="utf-8") as fh:
        model_data = json.load(fh)

    impacts = []
    for impact in model_data.get("impacts", []):
        if isinstance(impact, dict) and "cx_px" in impact and "cy_px" in impact:
            impacts.append({"cx_px": float(impact["cx_px"]), "cy_px": float(impact["cy_px"])})

    payload = {"source": image_path.name, "impacts": impacts}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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

    rejected_dir = Path(__file__).resolve().parent.parent / "data" / "impact_labelled" / "rejected"
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
            auto_json_path = detect_impacts_yolo(
                flat_img_path=image,
                weights=str(model_path),
                conf_thr=conf_thr,
                iou_thr=iou_thr,
                output_dir=str(pending_dir),
                device=device or "cpu",
            )
            stem = image.stem.removesuffix("_flat")
            model_json = auto_json_path.with_suffix(".json") if auto_json_path is not None else None
            labels_json = pending_dir / f"{stem}_labels.json"
            if model_json is not None and model_json.exists():
                write_impact_prediction_json(image, model_json, labels_json)
                print(f"[AUTO] Impact detection: {image.name} -> {labels_json.name}")
            else:
                print(f"[AUTO] Impact detection: {image.name} -> {pending_dir}.")
        else:
            print(f"[PENDING] No impact model: {image.name} copied to pending for manual labelling.")
        count += 1

    if images:
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
                str(rejected_dir),
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
