"""Detect the four printed ``1`` markers with a YOLO model.

This is the new entry point of the geometry pipeline: it replaces the black-disk
detection (``localize_target.py``) and the per-position patch CNN
(``axis_marker_cnn.py``). A single YOLO model detects four classes
(``top``/``left``/``bottom``/``right``); the highest-confidence box of each
class gives the marker centre.

Two inference engines are supported, mirroring ``detect_impacts_yolo.py``:
  * ``ultralytics`` (PyTorch ``.pt`` weights) — desktop / training machine;
  * ``onnxruntime`` (``.onnx`` weights) — mobile / Raspberry Pi, no PyTorch.

Usage:
    python src/detect_axis_markers.py photo.jpg --weights models/axis_yolo/best.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from axis_marker_common import AXIS_NAMES, Point

try:  # ultralytics is optional (absent on ARM targets)
    from ultralytics import YOLO as _YOLO

    _HAS_ULTRALYTICS = True
except Exception:  # pragma: no cover - depends on environment
    _YOLO = None  # type: ignore[assignment,misc]
    _HAS_ULTRALYTICS = False

CLASS_TO_NAME: dict[int, str] = {0: "top", 1: "left", 2: "bottom", 3: "right"}
DEFAULT_CONF: float = 0.25
DEFAULT_IOU: float = 0.45
DEFAULT_IMGSZ: int = 1056


@dataclass(frozen=True)
class MarkerDetection:
    """A single detected marker.

    Attributes:
        point: Marker centre in source-image pixels.
        confidence: Detection confidence in ``[0, 1]``.
    """

    point: Point
    confidence: float


def _keep_best_per_class(
    detections: list[tuple[int, float, float, float]],
) -> dict[str, MarkerDetection]:
    """Keep the highest-confidence detection for each of the four classes.

    Args:
        detections: List of ``(class_id, confidence, cx, cy)`` tuples.

    Returns:
        Mapping of axis name to its best :class:`MarkerDetection`. Missing
        classes are simply absent from the mapping.
    """
    best: dict[str, MarkerDetection] = {}
    for class_id, confidence, cx, cy in detections:
        name = CLASS_TO_NAME.get(class_id)
        if name is None:
            continue
        current = best.get(name)
        if current is None or confidence > current.confidence:
            best[name] = MarkerDetection(Point(cx, cy), confidence)
    return best


def _detect_ultralytics(
    image_path: Path,
    weights: Path,
    conf_thr: float,
    iou_thr: float,
    imgsz: int,
    device: str | None,
) -> dict[str, MarkerDetection]:
    """Run detection with ultralytics YOLO on ``.pt`` weights."""
    model = _YOLO(str(weights))
    results = model.predict(
        source=str(image_path),
        conf=conf_thr,
        iou=iou_thr,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    boxes = results[0].boxes
    detections: list[tuple[int, float, float, float]] = []
    if boxes is not None and len(boxes):
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                (int(box.cls[0]), float(box.conf[0]), (x1 + x2) / 2, (y1 + y2) / 2)
            )
    return _keep_best_per_class(detections)


def _detect_onnx(
    image_path: Path,
    onnx_path: Path,
    conf_thr: float,
    iou_thr: float,
    imgsz: int,
) -> dict[str, MarkerDetection]:
    """Run detection with onnxruntime on ``.onnx`` weights (no PyTorch)."""
    import onnxruntime as ort

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Impossible de lire : {image_path}")
    h0, w0 = img.shape[:2]

    # Letterbox into imgsz x imgsz (grey padding), identical to training layout.
    ratio = min(imgsz / h0, imgsz / w0)
    nh, nw = int(round(h0 * ratio)), int(round(w0 * ratio))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_top = (imgsz - nh) // 2
    pad_left = (imgsz - nw) // 2
    canvas[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    raw = session.run(None, {session.get_inputs()[0].name: blob})[0]

    # YOLOv8 head: (1, 4 + num_classes, N) → (N, 4 + num_classes).
    preds = raw[0].T
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    keep = confidences >= conf_thr
    boxes_xywh = boxes_xywh[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]
    if len(boxes_xywh) == 0:
        return {}

    cx, cy, bw, bh = boxes_xywh.T
    # Corner boxes in letterbox space for NMS.
    x = cx - bw / 2
    y = cy - bh / 2
    nms_boxes = np.stack([x, y, bw, bh], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_thr, iou_thr)
    kept = idxs.flatten() if hasattr(idxs, "flatten") else list(idxs)

    detections: list[tuple[int, float, float, float]] = []
    for i in kept:
        # Map centre back to the original image space (undo letterbox).
        src_cx = float((cx[i] - pad_left) / ratio)
        src_cy = float((cy[i] - pad_top) / ratio)
        detections.append((int(class_ids[i]), float(confidences[i]), src_cx, src_cy))
    return _keep_best_per_class(detections)


def detect_axis_markers(
    image_path: str | Path,
    weights: str | Path,
    conf_thr: float = DEFAULT_CONF,
    iou_thr: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    device: str | None = None,
) -> dict[str, MarkerDetection]:
    """Detect the four target markers in a raw photo.

    The engine is selected automatically: ultralytics when ``.pt`` weights are
    available and installed, otherwise onnxruntime on the sibling ``.onnx``.

    Args:
        image_path: Raw photo path.
        weights: Path to ``.pt`` (or ``.onnx``) YOLO weights.
        conf_thr: Minimum detection confidence.
        iou_thr: NMS IoU threshold.
        imgsz: Inference square size.
        device: Torch device for ultralytics (``"0"``/``"cpu"``); ignored by ONNX.

    Returns:
        Mapping of detected axis names to :class:`MarkerDetection`. Classes that
        were not detected are absent.

    Raises:
        FileNotFoundError: If neither ``.pt`` nor ``.onnx`` weights exist.
    """
    image_path = Path(image_path)
    w_path = Path(weights)
    onnx_path = w_path if w_path.suffix.lower() == ".onnx" else w_path.with_suffix(".onnx")

    # Force native onnxruntime path for .onnx weights to avoid ultralytics
    # selecting CUDAExecutionProvider on machines without full CUDA ORT deps.
    if w_path.suffix.lower() == ".onnx" and w_path.exists():
        return _detect_onnx(image_path, w_path, conf_thr, iou_thr, imgsz)
    if _HAS_ULTRALYTICS and w_path.exists():
        return _detect_ultralytics(image_path, w_path, conf_thr, iou_thr, imgsz, device)
    if onnx_path.exists():
        return _detect_onnx(image_path, onnx_path, conf_thr, iou_thr, imgsz)
    raise FileNotFoundError(
        f"Aucun poids YOLO axis trouvé : ni {w_path} ni {onnx_path}."
    )


def markers_to_points(
    detections: dict[str, MarkerDetection],
) -> dict[str, Point | None]:
    """Convert detections to a plain ``name → Point`` mapping (``None`` if absent).

    Args:
        detections: Output of :func:`detect_axis_markers`.

    Returns:
        Mapping with all four axis names; missing markers map to ``None``.
    """
    return {name: (detections[name].point if name in detections else None) for name in AXIS_NAMES}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Photo brute")
    parser.add_argument("--weights", type=Path, default=Path("models/axis_yolo/weights/best.pt"))
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    detections = detect_axis_markers(
        args.image, args.weights, args.conf, args.iou, args.imgsz, args.device
    )
    for name in AXIS_NAMES:
        det = detections.get(name)
        if det is None:
            print(f"{name:>6}: (non détecté)")
        else:
            print(f"{name:>6}: ({det.point.x:.1f}, {det.point.y:.1f})  conf={det.confidence:.3f}")


if __name__ == "__main__":
    _main()
