from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

CLASS_TO_NAME = {
0: "top",
1: "left",
2: "bottom",
3: "right",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_image(image_root: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_one(
prediction_txt: Path,
image_root: Path,
) -> bool:

    stem = prediction_txt.stem

    image_path = find_image(image_root, stem)

    if image_path is None:
        print(f"[SKIP] image introuvable pour {stem}")
        return False

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[SKIP] image illisible: {image_path}")
        return False

    height, width = image.shape[:2]

    markers = {
        "top": None,
        "left": None,
        "bottom": None,
        "right": None,
    }

    lines = prediction_txt.read_text(
        encoding="utf-8"
        ).splitlines()

    best_conf = {
    "top": -1.0,
    "left": -1.0,
    "bottom": -1.0,
    "right": -1.0,
    }

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        class_id = int(parts[0])

        if class_id not in CLASS_TO_NAME:
            continue

        name = CLASS_TO_NAME[class_id]

        cx = float(parts[1]) * width
        cy = float(parts[2]) * height

        confidence = 1.0

        if len(parts) >= 6:
            confidence = float(parts[5])

        if confidence < best_conf[name]:
            continue

        best_conf[name] = confidence

        markers[name] = {
        "x": round(cx, 2),
        "y": round(cy, 2),
        }

    payload = {
    "source": image_path.name,
    "image_width": width,
    "image_height": height,
    "markers": markers,
    }

    output = image_path.with_name(
    f"{image_path.stem}_axis_labels.json"
    )

    output.write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
    )

    print(f"[OK] {output}")

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--predictions",
    type=Path,
    required=True,
    help="Dossier YOLO labels",
    )

    parser.add_argument(
    "--images",
    type=Path,
    required=True,
    help="Dossier des images originales",
    )

    args = parser.parse_args()

    files = sorted(
    args.predictions.glob("*.txt")
    )

    count = 0

    for prediction in files:
        count += int(
            convert_one(
            prediction,
            args.images,
            )
        )

    print(f"[OK] {count} fichiers convertis")


if __name__ == "__main__":
    main()