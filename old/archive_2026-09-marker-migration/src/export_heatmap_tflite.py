"""Export dense heatmap CNN checkpoints to ONNX and TFLite.

Pipeline:
1. Load PyTorch `ImpactHeatmapCNN` checkpoint.
2. Export to ONNX.
3. Convert ONNX to TensorFlow using `onnx2tf`.
4. Convert SavedModel to float32 TFLite.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tensorflow as tf
import torch

from models.impact_heatmap_cnn import ImpactHeatmapCNN


def export_onnx(weights_path: Path, onnx_path: Path, input_size: int) -> None:
    """Export PyTorch model to ONNX.

    Args:
        weights_path: Path to `.pt` checkpoint.
        onnx_path: Output ONNX path.
        input_size: Square input size (e.g. 512).

    Raises:
        FileNotFoundError: If checkpoint does not exist.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    model = ImpactHeatmapCNN()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 1, input_size, input_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["heatmap"],
        dynamic_axes={"input": {0: "batch"}, "heatmap": {0: "batch"}},
        opset_version=13,
    )
    print(f"[OK] ONNX exported: {onnx_path}")


def onnx_to_saved_model(onnx_path: Path, tf_out_dir: Path) -> Path:
    """Convert ONNX to TensorFlow SavedModel via onnx2tf CLI.

    Args:
        onnx_path: Source ONNX file.
        tf_out_dir: Output folder for generated TensorFlow assets.

    Returns:
        Path to generated SavedModel directory.

    Raises:
        RuntimeError: If onnx2tf conversion fails.
    """
    tf_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["onnx2tf", "-i", str(onnx_path), "-o", str(tf_out_dir)]
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"onnx2tf conversion failed.\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    # onnx2tf outputs SavedModel directly to tf_out_dir
    if (tf_out_dir / "saved_model.pb").exists():
        saved_model_dir = tf_out_dir
    else:
        # Fall back to looking for nested saved_model directory
        candidates = [p for p in tf_out_dir.glob("**/saved_model.pb") if p.is_file()]
        if not candidates:
            raise RuntimeError("SavedModel not found after onnx2tf conversion.")
        saved_model_dir = candidates[0].parent

    print(f"[OK] SavedModel generated: {saved_model_dir}")
    return saved_model_dir


def saved_model_to_tflite(saved_model_dir: Path, tflite_path: Path) -> None:
    """Convert TensorFlow SavedModel to float32 TFLite.

    Args:
        saved_model_dir: Source SavedModel directory.
        tflite_path: Output TFLite file path.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = []
    tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    print(f"[OK] TFLite exported: {tflite_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Export heatmap model to TFLite")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/impacts_heatmap/model_best.pt"),
        help="Path to trained heatmap checkpoint",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/impacts_heatmap/impact_heatmap_cnn.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument(
        "--tf-out",
        type=Path,
        default=Path("tf_model/impact_heatmap"),
        help="Temporary ONNX->TF output folder",
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=Path("mobile/android/app/src/main/assets/impact_heatmap_cnn.tflite"),
        help="Output TFLite path",
    )
    parser.add_argument("--input-size", type=int, default=512)
    return parser


def main() -> None:
    """Script entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    export_onnx(args.weights, args.onnx, args.input_size)
    saved_model_dir = onnx_to_saved_model(args.onnx, args.tf_out)

    # Check if onnx2tf already generated TFLite file
    generated_tflite = args.tf_out / f"{args.tf_out.name}_float32.tflite"
    if generated_tflite.exists():
        print(f"[OK] Using pre-generated TFLite: {generated_tflite}")
        args.tflite.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(generated_tflite, args.tflite)
        print(f"[OK] TFLite copied to: {args.tflite}")
    else:
        # Fall back to manual SavedModel -> TFLite conversion
        saved_model_to_tflite(saved_model_dir, args.tflite)


if __name__ == "__main__":
    main()
