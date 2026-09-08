"""New end-to-end pipeline driven by the four ``1`` markers.

Flow (no black disk, no ring detection, no centre heuristic):

    photo brute
      -> YOLO axis markers (detect_axis_markers)
      -> homographie (flatten_markers)
      -> image flat 1056x1056
      -> YOLO impacts (detect_impacts_yolo)
      -> score géométrique (centre + rayons théoriques)

The impacts detector already falls back to purely geometric scoring around the
model centre ``(528, 528)`` when no ``*_rings.json`` is present, which this
pipeline never produces — so scoring stays 100% geometric.

Usage:
    python src/pipeline_markers.py photo.jpg \
        --axis-weights models/axis_yolo/weights/best.pt \
        --impact-weights models/yolo_impacts/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from axis_marker_common import AXIS_NAMES
from detect_axis_markers import detect_axis_markers, markers_to_points
from detect_impacts_yolo import detect_impacts_yolo, _auto_device
from flatten_markers import flatten_with_markers, save_flatten

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_AXIS_WEIGHTS = "models/axis_yolo/weights/best.pt"
DEFAULT_IMPACT_WEIGHTS = "models/yolo_impacts/weights/best.pt"
DEFAULT_OUT = "outputs"


def run_pipeline(
    image_path: Path,
    out_root: Path,
    axis_weights: str,
    impact_weights: str,
    conf: float,
    iou: float,
    device: str,
    debug: bool,
    show: bool,
) -> Path | None:
    """Process one raw photo end-to-end with the marker-based pipeline.

    Args:
        image_path: Raw photo path.
        out_root: Root output directory (a ``<stem>`` subfolder is created).
        axis_weights: YOLO weights for the four markers.
        impact_weights: YOLO weights for the impacts.
        conf: Confidence threshold for the impacts detector.
        iou: NMS IoU threshold for the impacts detector.
        device: Torch device string for ultralytics engines.
        debug: When ``True``, write intermediate debug images.
        show: When ``True``, open the final annotated result.

    Returns:
        Path to the final impacts result, or ``None`` if a marker was missing
        or the image could not be read.
    """
    stem = image_path.stem
    out_dir = out_root / stem
    print(f"\n{'#' * 62}\n# {image_path.name}\n{'#' * 62}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[SKIP] Image illisible : {image_path}")
        return None

    # ── Step 1: detect the four markers ───────────────────────────────────────
    detections = detect_axis_markers(image_path, axis_weights, device=device)
    points = markers_to_points(detections)
    missing = [name for name in AXIS_NAMES if points[name] is None]
    for name in AXIS_NAMES:
        det = detections.get(name)
        status = f"conf={det.confidence:.3f}" if det else "MANQUANT"
        print(f"[AXIS] {name:>6}: {status}")
    if missing:
        print(f"[SKIP] Marqueurs manquants : {missing}. Correction manuelle requise (mobile).")
        return None

    # ── Step 2: homography-based flattening ───────────────────────────────────
    result = flatten_with_markers(image, points)  # type: ignore[arg-type]
    flat_path = save_flatten(result, out_dir, stem, debug=debug)
    print(f"[FLAT] {flat_path.name}")

    # ── Step 3: impacts + geometric scoring on the flat image ─────────────────
    return detect_impacts_yolo(
        flat_path,
        weights=impact_weights,
        conf_thr=conf,
        iou_thr=iou,
        device=device,
        show=show,
    )


def _collect_images(path: Path) -> list[Path]:
    """Return the raw images to process, skipping already-flattened files."""
    if path.is_file():
        return [path]
    return sorted(
        f
        for f in path.glob("*.*")
        if f.suffix.lower() in IMAGE_EXTENSIONS
        and not f.stem.startswith("_")
        and "_flat" not in f.stem
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline marqueurs : photo -> YOLO axis -> homographie -> YOLO impacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", help="Photo brute (JPG) ou dossier de photos")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--axis-weights", default=DEFAULT_AXIS_WEIGHTS)
    parser.add_argument("--impact-weights", default=DEFAULT_IMPACT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    device = args.device or _auto_device()
    src = Path(args.image)
    if not src.exists():
        print(f"[ERREUR] Chemin introuvable : {src}")
        sys.exit(1)

    images = _collect_images(src)
    if not images:
        print(f"[ERREUR] Aucune image trouvée dans : {src}")
        sys.exit(1)

    out_root = Path(args.out)
    ok = fail = 0
    for image_path in images:
        if run_pipeline(
            image_path, out_root, args.axis_weights, args.impact_weights,
            args.conf, args.iou, device, args.debug, args.show,
        ):
            ok += 1
        else:
            fail += 1

    print(f"\n{'=' * 62}\nTerminé : {ok} OK / {fail} échec(s)\nRésultats : {out_root.resolve()}")


if __name__ == "__main__":
    _main()
