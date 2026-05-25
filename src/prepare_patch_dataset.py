import json
import random
from pathlib import Path

import cv2
import numpy as np

PATCH_SIZE = 64
JITTER_PX = 4
NEGATIVE_SAMPLES_PER_IMAGE = 30


def crop_patch(img, cx, cy, size=64):
    h, w = img.shape
    half = size // 2
    x0 = int(np.clip(cx - half, 0, w - size))
    y0 = int(np.clip(cy - half, 0, h - size))
    return img[y0 : y0 + size, x0 : x0 + size]


def prepare_dataset(outputs_dir, out_dir):
    outputs_dir = Path(outputs_dir)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    flat_images = list(outputs_dir.rglob("*_flat.jpg"))
    print(f"[INFO] {len(flat_images)} images _flat.jpg trouvées")

    idx = 0
    used = 0

    for flat_path in flat_images:
        label_path = flat_path.with_name(flat_path.name.replace("_flat.jpg", "_labels.json"))

        if not label_path.exists():
            print(f"[SKIP] {flat_path.parent.name} → pas de labels")
            continue

        used += 1
        print(f"[USE]  {flat_path.parent.name}")

        img = cv2.imread(str(flat_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        positives = [(i["cx_px"], i["cy_px"]) for i in data["impacts"]]

        # ✅ POSITIFS
        for cx, cy in positives:
            for _ in range(3):
                dx = random.randint(-JITTER_PX, JITTER_PX)
                dy = random.randint(-JITTER_PX, JITTER_PX)
                patch = crop_patch(img, cx + dx, cy + dy)
                cv2.imwrite(str(out_dir / "images" / f"{idx}.png"), patch)
                with open(out_dir / "labels" / f"{idx}.txt", "w") as f:
                    f.write("1\n")
                idx += 1

        # ✅ NÉGATIFS
        h, w = img.shape
        added = 0
        tries = 0
        while added < NEGATIVE_SAMPLES_PER_IMAGE and tries < 300:
            tries += 1
            cx = random.randint(PATCH_SIZE // 2, w - PATCH_SIZE // 2)
            cy = random.randint(PATCH_SIZE // 2, h - PATCH_SIZE // 2)

            if any(np.hypot(cx - px, cy - py) < 18 for px, py in positives):
                continue

            patch = crop_patch(img, cx, cy)
            cv2.imwrite(str(out_dir / "images" / f"{idx}.png"), patch)
            with open(out_dir / "labels" / f"{idx}.txt", "w") as f:
                f.write("0\n")
            idx += 1
            added += 1

    print(f"[✅ OK] {idx} patches générés à partir de {used} cibles")


if __name__ == "__main__":
    prepare_dataset("outputs", "data/patches")
