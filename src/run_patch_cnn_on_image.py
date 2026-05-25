"""Run patch CNN inference on a full image.

This script scans a grayscale image with a 64x64 sliding window, applies the
trained mini patch classifier, and saves:
- an annotated image with top detections
- a JSON file with detection details

Typical usage:
    uv run python src/run_patch_cnn_on_image.py outputs/.../sample_flat.jpg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn


PATCH_SIZE = 64


class MiniCNN(nn.Module):
    """Small binary classifier for 64x64 grayscale patches."""

    def __init__(self) -> None:
        """Initialize network layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass.

        Args:
            x: Tensor of shape (N, 1, 64, 64).

        Returns:
            Tensor of logits with shape (N, 2).
        """
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@dataclass(frozen=True)
class Detection:
    """Represents one detected impact candidate."""

    cx_px: int
    cy_px: int
    prob: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute stable softmax probabilities.

    Args:
        logits: 1D logits array.

    Returns:
        1D probability array.
    """
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _iter_centers(width: int, height: int, stride: int) -> Iterable[tuple[int, int]]:
    """Yield valid patch centers for sliding-window inference.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        stride: Sliding step in pixels.

    Yields:
        Tuples (cx, cy) of valid centers.
    """
    half = PATCH_SIZE // 2
    for cy in range(half, height - half + 1, stride):
        for cx in range(half, width - half + 1, stride):
            yield cx, cy


def _extract_patch(image_gray: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Extract a 64x64 patch centered at (cx, cy).

    Args:
        image_gray: Grayscale source image.
        cx: Center x in pixels.
        cy: Center y in pixels.

    Returns:
        Patch as float32 normalized array in [0, 1], shape (64, 64).
    """
    half = PATCH_SIZE // 2
    patch = image_gray[cy - half : cy + half, cx - half : cx + half]
    patch = patch.astype(np.float32) / 255.0
    return patch


def _load_model(weights_path: Path, device: torch.device) -> MiniCNN:
    """Load trained MiniCNN weights.

    Args:
        weights_path: Path to mini_impact_cnn.pt.
        device: Inference device.

    Returns:
        Loaded model in eval mode.
    """
    model = MiniCNN().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_on_image(
    image_path: Path,
    weights_path: Path,
    out_dir: Path,
    stride: int,
    threshold: float,
    nms_radius_px: int,
    top_k: int,
    batch_size: int,
) -> tuple[Path, Path]:
    """Run sliding-window patch classification on a full image.

    Args:
        image_path: Input image path.
        weights_path: Trained PT weights path.
        out_dir: Output folder for artifacts.
        stride: Sliding step in pixels.
        threshold: Minimum probability to keep a detection.
        nms_radius_px: NMS suppression radius in pixels.
        top_k: Max number of detections.
        batch_size: Inference batch size.

    Returns:
        Tuple (annotated_image_path, json_path).

    Raises:
        FileNotFoundError: If image or weights are missing.
        ValueError: If input image cannot be decoded.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise ValueError(f"Cannot decode image: {image_path}")

    height, width = image_gray.shape
    centers = list(_iter_centers(width, height, stride))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(weights_path, device)

    probs = np.zeros((len(centers),), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            batch_centers = centers[start : start + batch_size]
            batch = np.stack(
                [_extract_patch(image_gray, cx, cy) for cx, cy in batch_centers], axis=0
            )
            batch_t = torch.from_numpy(batch).unsqueeze(1).to(device)
            logits = model(batch_t).cpu().numpy()
            batch_probs = np.array([_softmax(row)[1] for row in logits], dtype=np.float32)
            probs[start : start + len(batch_centers)] = batch_probs

    nx = len(range(PATCH_SIZE // 2, width - PATCH_SIZE // 2 + 1, stride))
    ny = len(range(PATCH_SIZE // 2, height - PATCH_SIZE // 2 + 1, stride))
    prob_map = probs.reshape(ny, nx)

    detections: list[Detection] = []
    work = prob_map.copy()
    nms_cells = max(1, int(round(nms_radius_px / stride)))

    while len(detections) < top_k:
        y_idx, x_idx = np.unravel_index(np.argmax(work), work.shape)
        best = float(work[y_idx, x_idx])
        if best < threshold:
            break

        cx = PATCH_SIZE // 2 + x_idx * stride
        cy = PATCH_SIZE // 2 + y_idx * stride
        detections.append(Detection(cx_px=cx, cy_px=cy, prob=best))

        x0 = max(0, x_idx - nms_cells)
        x1 = min(work.shape[1], x_idx + nms_cells + 1)
        y0 = max(0, y_idx - nms_cells)
        y1 = min(work.shape[0], y_idx + nms_cells + 1)
        work[y0:y1, x0:x1] = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    annotated_path = out_dir / f"{stem}_patchcnn.jpg"
    json_path = out_dir / f"{stem}_patchcnn.json"

    annotated = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    for det in detections:
        color = (0, 255, 0) if det.prob >= 0.7 else (0, 165, 255)
        cv2.circle(annotated, (det.cx_px, det.cy_px), 11, color, 2)
        cv2.putText(
            annotated,
            f"{det.prob:.2f}",
            (det.cx_px + 6, det.cy_px - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    payload = {
        "source": str(image_path),
        "weights": str(weights_path),
        "device": str(device),
        "patch_size": PATCH_SIZE,
        "stride": int(stride),
        "threshold": float(threshold),
        "nms_radius_px": int(nms_radius_px),
        "top_k": int(top_k),
        "n_candidates": int(len(centers)),
        "n_detections": int(len(detections)),
        "detections": [
            {
                "cx_px": int(d.cx_px),
                "cy_px": int(d.cy_px),
                "prob": float(round(d.prob, 6)),
            }
            for d in detections
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] candidates={len(centers)} detections={len(detections)}")
    print(f"[OK] image -> {annotated_path}")
    print(f"[OK] json  -> {json_path}")

    return annotated_path, json_path


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run mini patch CNN on a full image")
    parser.add_argument("image", type=Path, help="Input image path (prefer *_flat.jpg)")
    parser.add_argument(
        "--weights", type=Path, default=Path("mini_impact_cnn.pt"), help="PT weights path"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/patch_cnn_eval"), help="Output folder"
    )
    parser.add_argument("--stride", type=int, default=8, help="Sliding step in pixels")
    parser.add_argument(
        "--threshold", type=float, default=0.55, help="Detection probability threshold"
    )
    parser.add_argument(
        "--nms-radius", type=int, default=14, help="NMS suppression radius in pixels"
    )
    parser.add_argument("--top-k", type=int, default=80, help="Max detections to keep")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_on_image(
        image_path=args.image,
        weights_path=args.weights,
        out_dir=args.out,
        stride=args.stride,
        threshold=args.threshold,
        nms_radius_px=args.nms_radius,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
