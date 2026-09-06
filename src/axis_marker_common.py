"""Shared geometry and label helpers for the four target ``1`` markers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

AXIS_NAMES = ("top", "left", "bottom", "right")
PATCH_SIZE = 128


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def viewfinder_crop(
    width: int,
    height: int,
    screen_aspect: float = 9 / 16,
    frame_ratio: float = 0.82,
) -> tuple[float, float, float, float]:
    """Return the source crop used by the portrait camera viewfinder.

    ``screen_aspect`` is width / height. The default matches a typical Android
    portrait screen; pass the real device ratio when preparing a dataset.
    """
    screen_w = 1.0
    screen_h = screen_w / screen_aspect
    frame_size = screen_w * frame_ratio
    scale = max(screen_w / width, screen_h / height)
    rendered_w = width * scale
    rendered_h = height * scale
    offset_x = (rendered_w - screen_w) / 2
    offset_y = (rendered_h - screen_h) / 2
    frame_x = (screen_w - frame_size) / 2
    frame_y = (screen_h - frame_size) / 2
    crop_w = frame_size / scale
    crop_h = crop_w
    origin_x = np.clip((frame_x + offset_x) / scale, 0, max(0, width - crop_w))
    origin_y = np.clip((frame_y + offset_y) / scale, 0, max(0, height - crop_h))
    return float(origin_x), float(origin_y), float(crop_w), float(crop_h)


def expected_axis_points(
    width: int,
    height: int,
    screen_aspect: float = 9 / 16,
    frame_ratio: float = 0.82,
    target_radius_ratio: float = 0.46,
) -> dict[str, Point]:
    """Return expected top/left/bottom/right marker coordinates in source pixels."""
    origin_x, origin_y, crop_w, crop_h = viewfinder_crop(
        width, height, screen_aspect, frame_ratio
    )
    center_x = origin_x + crop_w / 2
    center_y = origin_y + crop_h / 2
    radius = min(crop_w, crop_h) * target_radius_ratio
    return {
        "top": Point(center_x, center_y - radius),
        "left": Point(center_x - radius, center_y),
        "bottom": Point(center_x, center_y + radius),
        "right": Point(center_x + radius, center_y),
    }


def crop_patch(image: np.ndarray, cx: float, cy: float, size: int = PATCH_SIZE) -> np.ndarray:
    """Crop a fixed-size grayscale patch, padding at image boundaries."""
    half = size // 2
    h, w = image.shape[:2]
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE,
        )
        x0 += pad_left
        y0 += pad_top
    return image[y0:y0 + size, x0:x0 + size]


def load_axis_labels(path: Path) -> dict[str, Point | None]:
    """Load labels written by ``label_axis_markers.py``."""
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("markers", data)
    labels: dict[str, Point | None] = {}
    for name in AXIS_NAMES:
        value = raw.get(name)
        labels[name] = None if value is None else Point(float(value["x"]), float(value["y"]))
    return labels
