"""Small orchestrator for the incremental data loop.

This script is intentionally lightweight: it centralizes the order of the steps
and delegates the real work to the specialized batch scripts.

Usage examples:
    python src/data_loop.py --step axis --axis-model models/axis_yolo/weights/best.pt
    python src/data_loop.py --step homography
    python src/data_loop.py --step impact --impact-model models/yolo_impacts/weights/best.pt
    python src/data_loop.py --step all --axis-model ... --impact-model ...
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from process_axis_batch import process_axis_batch
from process_homography_batch import process_homography_batch
from process_impact_batch import process_impact_batch
from tri_incoming import sort_incoming


def ensure_default_layout(root: Path) -> None:
    """Create the default workflow directories if they do not exist."""
    for directory in [
        root / "original_photos" / "incoming",
        root / "original_photos" / "accepted",
        root / "original_photos" / "rejected",
        root / "axis_labelled" / "pending",
        root / "axis_labelled" / "accepted",
        root / "axis_labelled" / "rejected",
        root / "homographied" / "pending",
        root / "homographied" / "accepted",
        root / "homographied" / "rejected",
        root / "impact_labelled" / "pending",
        root / "impact_labelled" / "accepted",
        root / "impact_labelled" / "rejected",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def iter_images(path: Path) -> list[Path]:
    """Return supported image files from a directory."""
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def maybe_train_axis_model(root: Path) -> bool:
    """Prepare and train the axis YOLO model from the accepted axis dataset."""
    accepted_dir = root / "axis_labelled" / "accepted"
    if not accepted_dir.exists():
        return False
    images = iter_images(accepted_dir)
    if not images:
        return False

    script_dir = Path(__file__).resolve().parent
    dataset_dir = root / "axis_yolo"
    model_out = root.parent / "models" / "axis_yolo"
    prepare_script = script_dir / "prepare_axis_yolo_dataset.py"
    train_script = script_dir / "train_yolo.py"

    print("\n[TRAIN] Axis model: preparing dataset and retraining.")
    subprocess.run(
        [
            sys.executable,
            str(prepare_script),
            str(accepted_dir),
            "--out",
            str(dataset_dir),
            "--val-ratio",
            "0.2",
        ],
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--data",
            str(dataset_dir / "dataset.yaml"),
            "--model",
            "yolo11n.pt",
            "--epochs",
            "80",
            "--imgsz",
            "1056",
            "--batch",
            "8",
            "--out",
            str(model_out),
            "--aug-preset",
            "axis_markers",
        ],
        check=False,
    )

    source_weights = model_out / "yolo_impacts" / "weights" / "best.pt"
    target_weights = model_out / "weights" / "best.pt"
    if source_weights.exists():
        target_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_weights, target_weights)
        print(f"[OK] Axis weights synced to {target_weights}")

    return True


def maybe_train_impact_model(root: Path) -> bool:
    """Prepare and train the impact YOLO model from the accepted impact dataset."""
    accepted_dir = root / "impact_labelled" / "accepted"
    if not accepted_dir.exists():
        return False
    images = iter_images(accepted_dir)
    if not images:
        return False

    script_dir = Path(__file__).resolve().parent
    dataset_dir = root / "yolo"
    model_out = root.parent / "models"
    prepare_script = script_dir / "prepare_yolo_dataset.py"
    train_script = script_dir / "train_yolo.py"

    print("\n[TRAIN] Impact model: preparing dataset and retraining.")
    subprocess.run(
        [
            sys.executable,
            str(prepare_script),
            str(accepted_dir),
            "--out",
            str(dataset_dir),
            "--val-ratio",
            "0.2",
        ],
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--data",
            str(dataset_dir / "dataset.yaml"),
            "--model",
            "yolo11n.pt",
            "--epochs",
            "200",
            "--imgsz",
            "1056",
            "--batch",
            "8",
            "--out",
            str(model_out),
            "--aug-preset",
            "robust",
        ],
        check=False,
    )
    return True


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Root directory containing the incremental workflow.",
    )
    parser.add_argument(
        "--step",
        choices=["axis", "homography", "impact", "all"],
        required=True,
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--tri-incoming",
        action="store_true",
        help="Review each image in original_photos/incoming before processing the accepted queue.",
    )
    parser.add_argument("--axis-model", type=Path, default=Path("models/axis_yolo/weights/best.pt"))
    parser.add_argument("--impact-model", type=Path, default=Path("models/yolo_impacts/weights/best.pt"))
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Do not retrain YOLO models after accepted datasets are updated.",
    )
    args = parser.parse_args()

    ensure_default_layout(args.root)

    if args.tri_incoming:
        incoming_dir = args.root / "original_photos" / "incoming"
        accepted_dir = args.root / "original_photos" / "accepted"
        rejected_dir = args.root / "original_photos" / "rejected"
        print("\n[STEP] Review incoming photos")
        sort_incoming(incoming_dir, accepted_dir, rejected_dir)

    incoming_dir = args.root / "original_photos" / "incoming"
    accepted_dir = args.root / "original_photos" / "accepted"
    incoming_images = [p for p in incoming_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    accepted_images = [p for p in accepted_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    if not accepted_images and incoming_images:
        print("\n[INFO] Photos are waiting in original_photos/incoming.")
        print("[INFO] The loop only processes files in original_photos/accepted.")
        print("[INFO] Run: .\\.venv\\Scripts\\python src/data_loop.py --step all --tri-incoming")
        print("[INFO] Or: .\\.venv\\Scripts\\python src/tri_incoming.py")
        raise SystemExit(0)

    if args.step in {"axis", "all"}:
        print("\n[STEP] Axis batch")
        process_axis_batch(
            src=args.root / "original_photos" / "accepted",
            pending_dir=args.root / "axis_labelled" / "pending",
            accepted_dir=args.root / "axis_labelled" / "accepted",
            rejected_dir=args.root / "axis_labelled" / "rejected",
            weights=args.axis_model,
        )
        if not args.skip_train:
            maybe_train_axis_model(args.root)

    if args.step in {"homography", "all"}:
        print("\n[STEP] Homography batch")
        process_homography_batch(
            src=args.root / "axis_labelled" / "accepted",
            pending_dir=args.root / "homographied" / "pending",
            accepted_dir=args.root / "homographied" / "accepted",
            rejected_dir=args.root / "homographied" / "rejected",
        )

    if args.step in {"impact", "all"}:
        print("\n[STEP] Impact batch")
        process_impact_batch(
            src=args.root / "homographied" / "accepted",
            pending_dir=args.root / "impact_labelled" / "pending",
            accepted_dir=args.root / "impact_labelled" / "accepted",
            weights=args.impact_model,
        )
        if not args.skip_train:
            maybe_train_impact_model(args.root)

    print("\n[OK] Data loop step(s) finished.")
    print("Next: review pending folders and move validated images to accepted.")


if __name__ == "__main__":
    main()
