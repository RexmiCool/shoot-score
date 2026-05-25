"""Quick sanity test for dense heatmap TFLite inference on one flat image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for script CLI.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Test dense heatmap TFLite model")
    parser.add_argument("image", type=Path, help="Input *_flat.jpg")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("mobile/android/app/src/main/assets/impact_heatmap_cnn.tflite"),
        help="Path to TFLite model",
    )
    parser.add_argument("--size", type=int, default=512)
    return parser


def main() -> None:
    """Run one inference and print tensor diagnostics."""
    args = build_arg_parser().parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    resized = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_shape = input_details["shape"].tolist()
    out_shape = output_details["shape"].tolist()

    if len(in_shape) != 4:
        raise ValueError(f"Unexpected input shape: {in_shape}")

    if in_shape[1] == 1:
        tensor = normalized[np.newaxis, np.newaxis, :, :]
    elif in_shape[-1] == 1:
        tensor = normalized[np.newaxis, :, :, np.newaxis]
    else:
        raise ValueError(f"Unsupported input layout: {in_shape}")

    interpreter.set_tensor(input_details["index"], tensor.astype(np.float32))
    interpreter.invoke()

    output = interpreter.get_tensor(output_details["index"])
    values = output.astype(np.float32).reshape(-1)

    print("input_shape:", in_shape)
    print("output_shape:", out_shape)
    print("heatmap_min:", float(values.min()))
    print("heatmap_max:", float(values.max()))
    print("heatmap_mean:", float(values.mean()))


if __name__ == "__main__":
    main()
