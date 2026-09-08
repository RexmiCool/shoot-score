"""Export the four axis-marker CNN checkpoints to Android ONNX assets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from models.axis_marker_cnn import AxisMarkerCNN

POSITIONS = ("top", "left", "bottom", "right")


def export_one(position: str, weights_dir: Path, assets_dir: Path, work_dir: Path, patch_size: int) -> Path:
    weights = weights_dir / f"axis_marker_{position}.pt"
    onnx_path = work_dir / f"axis_marker_{position}.onnx"
    asset_path = assets_dir / onnx_path.name
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint absent: {weights}")

    model = AxisMarkerCNN().eval()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    work_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 1, patch_size, patch_size),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes=None,
    )
    shutil.copy2(onnx_path, asset_path)
    print(f"[OK] {position}: {asset_path}")
    return asset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("models/axis_markers"))
    parser.add_argument("--work", type=Path, default=Path("outputs/axis_markers_export"))
    parser.add_argument("--assets", type=Path, default=Path("mobile/android/app/src/main/assets"))
    parser.add_argument("--patch-size", type=int, default=128)
    args = parser.parse_args()
    for position in POSITIONS:
        export_one(position, args.weights, args.assets, args.work, args.patch_size)


if __name__ == "__main__":
    main()
