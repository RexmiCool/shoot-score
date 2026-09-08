"""Perspective rectification driven only by the four printed ``1`` markers.

This module replaces the black-disk / ellipse flattening (``flatten_target.py``)
in the new pipeline. The four markers ``top``, ``left``, ``bottom`` and
``right`` are the *single* geometric source of truth for the target. Their
positions in the raw photo are mapped, via a homography, onto four fixed
reference points of an ideal flattened C50 model (``OUTPUT_SIZE`` square).

Reference model
---------------
The flattened image is a square of ``OUTPUT_SIZE`` pixels. The target centre is
known *by construction* at ``(OUTPUT_CENTER, OUTPUT_CENTER)``. The physical
scale ``MM_PER_PX`` is fixed by the outer scoring ring (``250 mm`` radius), so
every score ring can be generated theoretically from the centre and this scale
(see :func:`ring_radii_px`). No ring or centre *detection* is ever performed.

Calibration note
----------------
``AXIS_MARKER_RADIUS_MM`` is the physical distance (in millimetres) between the
target centre and each printed ``1`` marker on the real C50 face. It is the
only value that must be measured once on a real target. Placing the four
reference points at ``AXIS_MARKER_RADIUS_MM / MM_PER_PX`` pixels from the centre
guarantees that the flattened image keeps the correct millimetre scale, so the
theoretical score rings line up with the printed rings.

Usage
-----
    from flatten_markers import flatten_with_markers, reference_axis_points
    flat = flatten_with_markers(image_bgr, detected_points)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from axis_marker_common import AXIS_NAMES, Point

# ── Reference model constants (ideal flattened C50) ───────────────────────────
OUTPUT_SIZE: int = 1056
OUTPUT_CENTER: float = OUTPUT_SIZE / 2.0  # 528.0 — target centre by construction

OUTER_RING_RADIUS_MM: float = 250.0  # physical radius of the outermost score ring
OUTPUT_OUTER_RADIUS_PX: float = OUTPUT_SIZE * 0.46  # keeps parity with legacy flatten
MM_PER_PX: float = OUTER_RING_RADIUS_MM / OUTPUT_OUTER_RADIUS_PX  # ≈ 0.5155 mm/px

# Physical centre→marker distance of the printed ``1`` digits on a C50 face.
# MUST be measured once on a real target and adjusted here if needed.
AXIS_MARKER_RADIUS_MM: float = 235.0
AXIS_MARKER_RADIUS_PX: float = AXIS_MARKER_RADIUS_MM / MM_PER_PX

# Score-ring radii of a C50 face, from the 10-ring inwards to the 1-ring.
# Index 0 → radius of the 10 zone, index 9 → radius of the 1 zone.
SCORE_RING_RADII_MM: tuple[float, ...] = (
    25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0,
)


@dataclass(frozen=True)
class FlattenResult:
    """Result of a marker-based rectification.

    Attributes:
        image: The flattened ``OUTPUT_SIZE`` square BGR image.
        homography: The ``3x3`` homography mapping source pixels to the model.
    """

    image: np.ndarray
    homography: np.ndarray


def reference_axis_points() -> dict[str, Point]:
    """Return the fixed destination points of the four markers in the model.

    The points lie on the horizontal and vertical axes of the flattened image,
    at ``AXIS_MARKER_RADIUS_PX`` from the centre.

    Returns:
        Mapping ``{"top", "left", "bottom", "right"}`` to their reference
        :class:`~axis_marker_common.Point` in the flattened model.
    """
    r = AXIS_MARKER_RADIUS_PX
    return {
        "top": Point(OUTPUT_CENTER, OUTPUT_CENTER - r),
        "left": Point(OUTPUT_CENTER - r, OUTPUT_CENTER),
        "bottom": Point(OUTPUT_CENTER, OUTPUT_CENTER + r),
        "right": Point(OUTPUT_CENTER + r, OUTPUT_CENTER),
    }


def ring_radii_px() -> list[tuple[float, int]]:
    """Return theoretical score-ring radii in pixels with their score.

    The rings are purely geometric: derived from the known centre and the fixed
    ``MM_PER_PX`` scale. No detection is performed.

    Returns:
        List of ``(radius_px, score)`` tuples, ordered from the 10 ring
        outwards to the 1 ring.
    """
    return [
        (radius_mm / MM_PER_PX, 10 - index)
        for index, radius_mm in enumerate(SCORE_RING_RADII_MM)
    ]


def _points_to_array(points: dict[str, Point]) -> np.ndarray:
    """Convert a marker mapping to an ordered ``(4, 2)`` float32 array.

    Args:
        points: Mapping of the four axis names to their :class:`Point`.

    Returns:
        Array of the four points ordered as :data:`AXIS_NAMES`.

    Raises:
        ValueError: If any of the four markers is missing.
    """
    ordered = []
    for name in AXIS_NAMES:
        point = points.get(name)
        if point is None:
            raise ValueError(f"Marqueur manquant pour l'homographie : {name!r}")
        ordered.append((point.x, point.y))
    return np.asarray(ordered, dtype=np.float32)


def compute_homography(source_points: dict[str, Point]) -> np.ndarray:
    """Compute the homography mapping source markers to the reference model.

    Args:
        source_points: Detected/validated positions of the four markers in the
            raw image, keyed by ``top``/``left``/``bottom``/``right``.

    Returns:
        The ``3x3`` homography matrix (``float64``).

    Raises:
        ValueError: If a marker is missing.
    """
    src = _points_to_array(source_points)
    dst = _points_to_array(reference_axis_points())
    # Exactly four correspondences → deterministic projective transform.
    return cv2.getPerspectiveTransform(src, dst)


def flatten_with_markers(
    image: np.ndarray,
    source_points: dict[str, Point],
) -> FlattenResult:
    """Rectify an image using only the four ``1`` markers.

    Args:
        image: Raw BGR image containing the target.
        source_points: Positions of the four markers in ``image``.

    Returns:
        A :class:`FlattenResult` with the flattened image and the homography.

    Raises:
        ValueError: If a marker is missing.
    """
    homography = compute_homography(source_points)
    flat = cv2.warpPerspective(
        image,
        homography,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return FlattenResult(image=flat, homography=homography)


def annotate_model(flat: np.ndarray) -> np.ndarray:
    """Draw the theoretical centre, marker points and score rings for debug.

    Args:
        flat: A flattened ``OUTPUT_SIZE`` image.

    Returns:
        A copy of ``flat`` with the reference geometry overlaid.
    """
    annot = flat.copy()
    center = (int(round(OUTPUT_CENTER)), int(round(OUTPUT_CENTER)))
    for radius_px, _ in ring_radii_px():
        cv2.circle(annot, center, int(round(radius_px)), (0, 180, 0), 1)
    for name, point in reference_axis_points().items():
        cv2.drawMarker(
            annot,
            (int(round(point.x)), int(round(point.y))),
            (0, 220, 255),
            cv2.MARKER_TILTED_CROSS,
            22,
            2,
        )
        cv2.putText(
            annot,
            name,
            (int(point.x) + 6, int(point.y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.drawMarker(annot, center, (255, 0, 0), cv2.MARKER_CROSS, 40, 2)
    return annot


def save_flatten(result: FlattenResult, out_dir: Path, stem: str, debug: bool = False) -> Path:
    """Persist a flattened image (and optional debug overlay) to disk.

    Args:
        result: The rectification result to save.
        out_dir: Destination directory (created if needed).
        stem: Base name of the source image (without suffix).
        debug: When ``True``, also write ``<stem>_flat_annot.jpg``.

    Returns:
        The path to the written ``<stem>_flat.jpg``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_path = out_dir / f"{stem}_flat.jpg"
    cv2.imwrite(str(flat_path), result.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if debug:
        annot_path = out_dir / f"{stem}_flat_annot.jpg"
        cv2.imwrite(str(annot_path), annotate_model(result.image), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return flat_path
