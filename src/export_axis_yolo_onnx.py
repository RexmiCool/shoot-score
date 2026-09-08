"""Export axis-marker YOLO weights to Android ONNX asset.

The output asset is consumed by the Android native module for automatic
pre-placement of top/left/bottom/right markers.

Usage:
    python src/export_axis_yolo_onnx.py
    python src/export_axis_yolo_onnx.py --weights models/axis_yolo/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("models/axis_yolo/weights/best.pt")
DEFAULT_IMGSZ = 1056
DEFAULT_MOBILE_ASSET = Path("mobile/android/app/src/main/assets/axis_markers_yolo.onnx")


def export_axis_yolo_onnx(weights: Path, imgsz: int) -> Path:
    """Export axis-marker YOLO weights to ONNX.

    Args:
        weights: Path to YOLO ``.pt`` weights.
        imgsz: Export image size.

    Returns:
        Path to the generated ``.onnx`` file.

    Raises:
        FileNotFoundError: If the input weights file does not exist.
    """
    if not weights.exists():
        raise FileNotFoundError(f"Poids introuvables: {weights}")

    model = YOLO(str(weights))
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        dynamic=False,
        opset=12,
    )
    onnx_path = Path(out)
    print(f"[OK] Export ONNX axis: {onnx_path}")
    return onnx_path


def copy_to_mobile_asset(onnx_path: Path, asset_path: Path) -> Path:
    """Copy exported ONNX model to Android assets.

    Args:
        onnx_path: Source ONNX path.
        asset_path: Target Android asset path.

    Returns:
        Final copied asset path.
    """
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, asset_path)
    print(f"[OK] Asset Android axis: {asset_path}")
    return asset_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--mobile-asset", type=Path, default=DEFAULT_MOBILE_ASSET)
    args = parser.parse_args()

    onnx_path = export_axis_yolo_onnx(args.weights, args.imgsz)
    copy_to_mobile_asset(onnx_path, args.mobile_asset)


if __name__ == "__main__":
    main()
