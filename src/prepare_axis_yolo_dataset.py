from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_MAP = {
    "top": 0,
    "left": 1,
    "bottom": 2,
    "right": 3,
}


def collect_images(path: Path) -> list[Path]:
is_file():
        return [path]

    return sorted(
        p
        for p in path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and "_axis_labels" not in p.stem
    )


def create_bbox(
    x: float,
    y: float,
    image_w: int,
    image_h: int,
    box_size: int,
):
    cx = x / image_w
    cy = y / image_h
    bw = box_size / image_w
    bh = box_size / image_h

    return cx, cy, bw, bh


def convert_image(
    image_path: Path,
    output_root: Path,
    val_ratio: float,
    box_size: int,
):
    label_path = image_path.with_name(
        f"{image_path.stem}_axis_labels.json"
    )

    if not label_path.exists():
        return 0

    data = json.loads(
        label_path.read_text(encoding="utf-8")
    )

    width = data["image_width"]
    height = data["image_height"]

    markers = data["markers"]

    split = "val" if random.random() < val_ratio else "train"

    image_out = (
        output_root
        / "images"
        / split
        / image_path.name
    )

    label_out = (
        output_root
        / "labels"
        / split
        / f"{image_path.stem}.txt"
    )

    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, image_out)

    lines = []

    for name, class_id in CLASS_MAP.items():

        point = markers.get(name)

        if point is None:
            continue

        cx, cy, bw, bh = create_bbox(
            point["x"],
            point["y"],
            width,
            height,
            box_size,
        )

        lines.append(
            f"{class_id} "
            f"{cx:.6f} "
            f"{cy:.6f} "
            f"{bw:.6f} "
            f"{bh:.6f}"
        )

    label_out.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return 1


def write_yaml(output_root: Path):
    yaml_text = """
path: .

train: images/train
val: images/val

names:
  0: top
  1: left
  2: bottom
  3: right
""".strip()

    (output_root / "dataset.yaml").write_text(
        yaml_text,
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        type=Path,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/axis_yolo"),
    )

    parser.add_argument(
        "--box-size",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)

    images = collect_images(args.path)

    total = 0

    for image in images:
        total += convert_image(
            image,
            args.out,
            args.val_ratio,
            args.box_size,
        )

    write_yaml(args.out)

    print(
        f"[OK] {total} images converties vers {args.out}"
    )


if __name__ == "__main__":
    main()