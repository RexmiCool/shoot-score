"""
Prepare and curate impact labels for YOLO training.

Workflow:
1. Prepare phase:
   - Apply legacy target crop on each raw image.
   - Save crop as <stem>_flat.jpg under one folder per sample.
   - Run YOLO inference to pre-fill impacts.
   - Save editable labels in <stem>_labels.json.
2. Review phase (interactive OpenCV UI):
   - Browse each prepared sample.
   - Keep or reject sample.
   - Add/remove impacts manually.

Output layout (under --out):
  <stem>/
    <stem>_flat.jpg
    <stem>_impacts_yolo.jpg
    <stem>_impacts_yolo.json
    <stem>_labels.json           (kept samples)
    <stem>_labels_rejected.json  (rejected samples)

You can then build a YOLO dataset with:
  python src/prepare_yolo_dataset.py <out_dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from detect_impacts_yolo import DEFAULT_WEIGHTS, _auto_device, detect_impacts_yolo
from localize_target import detect_black_disk

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

BLACK_DISK_RADIUS_MM = 100.0
OUTER_CIRCLE_RADIUS_MM = 250.0

MARK_RADIUS = 10
MARK_COLOR = (0, 255, 255)
DELETE_RADIUS = 28
DISPLAY_MAX = 1200
FONT = cv2.FONT_HERSHEY_SIMPLEX

MANIFEST_NAME = "_curation_manifest.json"


def collect_input_images(path: Path) -> list[Path]:
    """Collect supported images from file or directory.

    Args:
        path: Source file or directory.

    Returns:
        Sorted list of image paths.
    """
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def make_sample_stem(root: Path, image_path: Path) -> str:
    """Build a deterministic sample id from relative path.

    Args:
        root: Root directory used for traversal.
        image_path: Input image path.

    Returns:
        Safe stem string usable as folder and file prefix.
    """
    rel = image_path.relative_to(root) if root.is_dir() else Path(image_path.name)
    no_suffix = rel.with_suffix("")
    raw = "_".join(no_suffix.parts)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return cleaned or image_path.stem


def _find_black_disk_in_image(img: np.ndarray) -> tuple[int, int, int, float] | None:
    """Detect the central black disk using legacy thresholding heuristics."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = 0.005 * h * w
    cx_img, cy_img = w / 2, h / 2
    tol_x, tol_y = w * 0.25, h * 0.25

    candidates: list[tuple[float, np.ndarray, float, float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (peri**2) if peri > 0 else 0.0
        if circ < 0.5:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if abs(cx - cx_img) > tol_x or abs(cy - cy_img) > tol_y:
            continue
        candidates.append((area, cnt, cx, cy, r))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, _, cx, cy, r = candidates[0]
    return int(cx), int(cy), int(r), BLACK_DISK_RADIUS_MM / r


def crop_target_legacy(img: np.ndarray) -> np.ndarray | None:
    """Crop target area using the old processing strategy.

    Args:
        img: Raw BGR image.

    Returns:
        Cropped square image, or None when target is not detected.
    """
    disk = _find_black_disk_in_image(img)
    if disk is None:
        return None

    cx0, cy0, _, mm_per_px0 = disk
    r_outer0 = int(OUTER_CIRCLE_RADIUS_MM / mm_per_px0)
    margin_px0 = int(10.0 / mm_per_px0)
    half = r_outer0 + margin_px0

    x1 = max(0, cx0 - half)
    y1 = max(0, cy0 - half)
    x2 = min(img.shape[1], cx0 + half)
    y2 = min(img.shape[0], cy0 + half)

    crop = img[y1:y2, x1:x2].copy()
    side = min(crop.shape[:2])
    return crop[:side, :side]


def crop_target_robust(img: np.ndarray) -> np.ndarray | None:
    """Crop target area using the robust OpenCV detector from localize_target."""
    disk = detect_black_disk(img)
    if disk is None:
        return None

    cx0, cy0, _, mm_per_px0 = disk
    r_outer0 = int(OUTER_CIRCLE_RADIUS_MM / mm_per_px0)
    margin_px0 = int(10.0 / mm_per_px0)
    half = r_outer0 + margin_px0

    x1 = max(0, cx0 - half)
    y1 = max(0, cy0 - half)
    x2 = min(img.shape[1], cx0 + half)
    y2 = min(img.shape[0], cy0 + half)

    crop = img[y1:y2, x1:x2].copy()
    side = min(crop.shape[:2])
    return crop[:side, :side]


def crop_target(img: np.ndarray, crop_mode: str) -> tuple[np.ndarray | None, str]:
    """Select crop strategy.

    Returns (crop, mode_used) where mode_used is one of:
    - legacy
    - robust
    - robust->legacy
    """
    if crop_mode == "legacy":
        return crop_target_legacy(img), "legacy"
    if crop_mode == "robust":
        return crop_target_robust(img), "robust"

    crop = crop_target_robust(img)
    if crop is not None:
        return crop, "robust"
    return crop_target_legacy(img), "robust->legacy"


def _load_json(path: Path, default: Any) -> Any:
    """Load JSON file and fallback to default on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data: Any) -> None:
    """Write JSON content with stable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_prepare(
    source: Path,
    out_dir: Path,
    weights: str,
    conf: float,
    iou: float,
    device: str,
    overwrite: bool,
    crop_mode: str,
) -> Path:
    """Run batch preparation with OpenCV crop and YOLO pre-labeling.

    Args:
        source: Input image file or directory.
        out_dir: Output session root.
        weights: YOLO weights path.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        device: Inference device identifier.
        overwrite: Recompute even if sample already exists.
        crop_mode: Crop strategy ('auto', 'robust', or 'legacy').

    Returns:
        Path to the manifest JSON.
    """
    images = collect_input_images(source)
    if not images:
        raise FileNotFoundError(f"No images found in: {source}")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / MANIFEST_NAME
    manifest = _load_json(manifest_path, {"source": str(source), "samples": []})
    existing_by_stem = {entry.get("sample_stem"): entry for entry in manifest.get("samples", [])}

    root_for_stem = source if source.is_dir() else source.parent
    kept_entries: list[dict[str, Any]] = []

    print(f"[PREP] {len(images)} image(s) to process")
    for idx, img_path in enumerate(images, 1):
        sample_stem = make_sample_stem(root_for_stem, img_path)
        sample_dir = out_dir / sample_stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        flat_path = sample_dir / f"{sample_stem}_flat.jpg"
        labels_path = sample_dir / f"{sample_stem}_labels.json"
        rejected_path = sample_dir / f"{sample_stem}_labels_rejected.json"

        print(f"\n[{idx}/{len(images)}] {img_path.name}")

        do_compute = overwrite or not flat_path.exists()
        if do_compute:
            raw = cv2.imread(str(img_path))
            if raw is None:
                print(f"  [SKIP] unreadable image: {img_path}")
                continue

            crop, mode_used = crop_target(raw, crop_mode)
            if crop is None:
                print(f"  [SKIP] target not detected (mode={mode_used})")
                continue

            cv2.imwrite(str(flat_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  [OK] crop ({mode_used}) -> {flat_path.name}")

        infer_ok = overwrite or not labels_path.exists()
        if infer_ok:
            annot_path = detect_impacts_yolo(
                flat_path,
                weights=weights,
                conf_thr=conf,
                iou_thr=iou,
                output_dir=str(out_dir),
                show=False,
                device=device,
            )
            if annot_path is None:
                print("  [SKIP] YOLO inference failed")
                continue

            impacts_json = sample_dir / f"{sample_stem}_impacts_yolo.json"
            impacts_data = _load_json(impacts_json, {"impacts": []})
            impacts = [
                {"cx_px": float(imp["cx_px"]), "cy_px": float(imp["cy_px"])}
                for imp in impacts_data.get("impacts", [])
            ]

            labels_data = {
                "source": flat_path.name,
                "raw_source": str(img_path.resolve()),
                "prelabel": "yolo",
                "keep": True,
                "impacts": impacts,
            }
            _save_json(labels_path, labels_data)
            if rejected_path.exists():
                rejected_path.unlink()
            print(f"  [OK] prelabels -> {labels_path.name} ({len(impacts)} impacts)")

        existing = existing_by_stem.get(sample_stem, {})
        keep = (
            bool(_load_json(labels_path, {"keep": True}).get("keep", True))
            if labels_path.exists()
            else False
        )
        if rejected_path.exists():
            keep = False

        kept_entries.append(
            {
                "sample_stem": sample_stem,
                "sample_dir": str(sample_dir.resolve()),
                "raw_image": str(img_path.resolve()),
                "flat_image": str(flat_path.resolve()),
                "labels_json": str(labels_path.resolve()),
                "rejected_json": str(rejected_path.resolve()),
                "keep": keep,
                "status": existing.get("status", "prepared"),
            }
        )

    manifest["source"] = str(source.resolve())
    manifest["samples"] = kept_entries
    _save_json(manifest_path, manifest)
    print(f"\n[OK] Manifest updated: {manifest_path}")
    return manifest_path


class CurationReviewer:
    """Interactive reviewer for prepared samples."""

    def __init__(self, manifest_path: Path):
        """Initialize reviewer state and load manifest.

        Args:
            manifest_path: Path to session manifest JSON.
        """
        self.manifest_path = manifest_path
        self.manifest = _load_json(manifest_path, {"samples": []})
        self.samples: list[dict[str, Any]] = self.manifest.get("samples", [])
        self.index = 0
        self.window_name = "YOLO Curation [L:add R:del K:keep X:reject S:save N/P:nav Q:quit]"

    def _load_current_sample(self) -> tuple[np.ndarray, list[tuple[int, int]], bool, str]:
        """Load current sample image and editable labels.

        Returns:
            Tuple of base image, impacts list, keep flag, and stem.

        Raises:
            FileNotFoundError: If current flat image cannot be loaded.
        """
        sample = self.samples[self.index]
        flat_path = Path(sample["flat_image"])
        labels_path = Path(sample["labels_json"])
        rejected_path = Path(sample["rejected_json"])

        img = cv2.imread(str(flat_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {flat_path}")

        if labels_path.exists():
            data = _load_json(labels_path, {"impacts": [], "keep": True})
            keep = bool(data.get("keep", True))
        elif rejected_path.exists():
            data = _load_json(rejected_path, {"impacts": [], "keep": False})
            keep = False
        else:
            data = {"impacts": [], "keep": True}
            keep = True

        impacts = [
            (int(round(p["cx_px"])), int(round(p["cy_px"]))) for p in data.get("impacts", [])
        ]
        stem = sample["sample_stem"]
        return img, impacts, keep, stem

    def _save_current(self, impacts: list[tuple[int, int]], keep: bool) -> None:
        """Persist current sample state to JSON and manifest.

        Args:
            impacts: Edited impact center points.
            keep: Whether sample is accepted.
        """
        sample = self.samples[self.index]
        flat_path = Path(sample["flat_image"])
        labels_path = Path(sample["labels_json"])
        rejected_path = Path(sample["rejected_json"])

        payload = {
            "source": flat_path.name,
            "raw_source": sample["raw_image"],
            "prelabel": "yolo",
            "keep": keep,
            "impacts": [{"cx_px": int(cx), "cy_px": int(cy)} for cx, cy in impacts],
        }

        if keep:
            _save_json(labels_path, payload)
            if rejected_path.exists():
                rejected_path.unlink()
            sample["status"] = "kept"
        else:
            _save_json(rejected_path, payload)
            if labels_path.exists():
                labels_path.unlink()
            sample["status"] = "rejected"

        sample["keep"] = keep
        self.manifest["samples"] = self.samples
        _save_json(self.manifest_path, self.manifest)

    def _render(
        self,
        base: np.ndarray,
        impacts: list[tuple[int, int]],
        keep: bool,
        stem: str,
    ) -> np.ndarray:
        """Render an annotated preview for current sample."""
        canvas = base.copy()
        h, w = canvas.shape[:2]

        for idx, (cx, cy) in enumerate(impacts, 1):
            cv2.circle(canvas, (cx, cy), MARK_RADIUS, MARK_COLOR, 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 3, MARK_COLOR, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(idx),
                (cx + MARK_RADIUS + 3, cy + 5),
                FONT,
                0.5,
                MARK_COLOR,
                1,
                cv2.LINE_AA,
            )

        state_txt = "KEEP" if keep else "REJECT"
        state_color = (0, 220, 120) if keep else (0, 0, 220)
        title = (
            f"{stem} | {self.index + 1}/{len(self.samples)} | impacts={len(impacts)} | {state_txt}"
        )

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - 32), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        cv2.putText(canvas, title, (8, h - 10), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, state_txt, (w - 120, 28), FONT, 0.8, state_color, 2, cv2.LINE_AA)

        return canvas

    def run(self, start_index: int = 0) -> None:
        """Start interactive review loop.

        Args:
            start_index: Zero-based index to begin review from.
        """
        if not self.samples:
            print("[INFO] No prepared samples found in manifest")
            return

        self.index = max(0, min(start_index, len(self.samples) - 1))
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while 0 <= self.index < len(self.samples):
            try:
                base, impacts, keep, stem = self._load_current_sample()
            except FileNotFoundError as exc:
                print(f"[SKIP] {exc}")
                self.index += 1
                continue

            h, w = base.shape[:2]
            scale = min(1.0, DISPLAY_MAX / max(h, w))
            disp_w = int(w * scale)
            disp_h = int(h * scale)
            cv2.resizeWindow(self.window_name, disp_w, disp_h)

            clicked = {"changed": False}

            def _on_mouse(event: int, xd: int, yd: int, _flags: int, _param: Any) -> None:
                xi = int(xd / scale)
                yi = int(yd / scale)
                if event == cv2.EVENT_LBUTTONDOWN:
                    impacts.append((xi, yi))
                    clicked["changed"] = True
                elif event == cv2.EVENT_RBUTTONDOWN and impacts:
                    dists = [np.hypot(xi - cx, yi - cy) for cx, cy in impacts]
                    nearest = int(np.argmin(dists))
                    if dists[nearest] <= DELETE_RADIUS / scale:
                        impacts.pop(nearest)
                        clicked["changed"] = True

            cv2.setMouseCallback(self.window_name, _on_mouse)

            while True:
                frame = self._render(base, impacts, keep, stem)
                if scale != 1.0:
                    frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(30) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    self._save_current(impacts, keep)
                    cv2.destroyAllWindows()
                    print("[DONE] Review stopped by user")
                    return
                if key in (ord("s"), ord("S")):
                    self._save_current(impacts, keep)
                    print("[SAVE] current sample")
                elif key in (ord("k"), ord("K")):
                    keep = True
                elif key in (ord("x"), ord("X")):
                    keep = False
                elif key in (ord("z"), ord("Z")) and impacts:
                    impacts.pop()
                elif key in (ord("n"), ord("N")):
                    self._save_current(impacts, keep)
                    self.index += 1
                    break
                elif key in (ord("p"), ord("P")):
                    self._save_current(impacts, keep)
                    self.index = max(0, self.index - 1)
                    break

        cv2.destroyAllWindows()
        print("[DONE] Review finished")


def summarize_manifest(manifest_path: Path) -> None:
    """Print a compact status summary for the curation session.

    Args:
        manifest_path: Path to manifest JSON.
    """
    manifest = _load_json(manifest_path, {"samples": []})
    samples = manifest.get("samples", [])
    kept = sum(1 for s in samples if s.get("status") == "kept")
    rejected = sum(1 for s in samples if s.get("status") == "rejected")
    prepared = len(samples) - kept - rejected
    print("\n[SUMMARY]")
    print(f"  total    : {len(samples)}")
    print(f"  kept     : {kept}")
    print(f"  rejected : {rejected}")
    print(f"  pending  : {prepared}")
    print(f"  manifest : {manifest_path}")


def main() -> None:
    """CLI entrypoint for curation workflow."""
    ap = argparse.ArgumentParser(
        description=("OpenCV crop + YOLO prelabels + interactive curation for training dataset.")
    )
    ap.add_argument(
        "source",
        help="Input image file or folder (example: data/all/cible)",
    )
    ap.add_argument(
        "--out",
        default="outputs/curation_yolo",
        help="Output folder for prepared samples and labels",
    )
    ap.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="YOLO weights path",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )
    ap.add_argument(
        "--iou",
        type=float,
        default=0.4,
        help="YOLO NMS IoU threshold",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Inference device ('0' for CUDA, 'cpu' otherwise)",
    )
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run only preparation (no interactive review)",
    )
    ap.add_argument(
        "--review-only",
        action="store_true",
        help="Run only interactive review (reuse existing manifest)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute crop/inference even if files already exist",
    )
    ap.add_argument(
        "--crop-mode",
        choices=("auto", "robust", "legacy"),
        default="auto",
        help="Crop strategy: robust detector, strict legacy detector, or auto fallback",
    )
    ap.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for review mode",
    )
    args = ap.parse_args()

    src = Path(args.source)
    out_dir = Path(args.out)
    if not src.exists() and not args.review_only:
        raise FileNotFoundError(f"Input path not found: {src}")

    device = args.device if args.device else _auto_device()
    manifest_path = out_dir / MANIFEST_NAME

    if not args.review_only:
        manifest_path = run_prepare(
            source=src,
            out_dir=out_dir,
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            device=device,
            overwrite=args.overwrite,
            crop_mode=args.crop_mode,
        )

    if not args.prepare_only:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found for review: {manifest_path}. Run prepare first."
            )
        reviewer = CurationReviewer(manifest_path)
        reviewer.run(start_index=args.start_index)

    summarize_manifest(manifest_path)
    print("\nNext step:")
    print(f"  python src/prepare_yolo_dataset.py {out_dir}")


if __name__ == "__main__":
    main()
