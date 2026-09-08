# src/prepare_heatmap_dataset.py
"""
Prépare un dataset pour la détection d'impacts par heatmap.

Entrées :
  - *_flat.jpg     images canonisées (1056×1056)
  - *_labels.json  centres des impacts en pixels (cx_px, cy_px)

Sorties :
  - images/   → images redimensionnées (512×512)
  - heatmaps/ → heatmaps gaussiennes correspondantes (512×512, float32)
  - meta.json → paramètres globaux du dataset

Usage :
  python src/prepare_heatmap_dataset.py outputs/flatten
  python src/prepare_heatmap_dataset.py outputs/flatten --out data/impacts_heatmap --val-ratio 0.2
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Paramètres globaux (FIGÉS par décision projet)
# ──────────────────────────────────────────────────────────────────────────────

INPUT_SIZE = 1056  # taille image flat d'entrée
TARGET_SIZE = 512  # taille IA
SCALE = TARGET_SIZE / INPUT_SIZE

SIGMA = 8.0  # écart-type de la gaussienne (en px, espace 512)
# NOTE: sigma=2.5 was too small (only ~626 px/impact, bg/pos ratio 419:1).
# sigma=8 gives ~6400 px/impact (ratio 41:1), compatible with w_pos=50.
VAL_RATIO = 0.20
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ──────────────────────────────────────────────────────────────────────────────


def make_gaussian_heatmap(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """
    Crée une heatmap gaussienne 2D normalisée centrée en (cx, cy).
    """
    x = np.arange(0, size, dtype=np.float32)
    y = np.arange(0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return heatmap


def prepare_dataset(
    root: Path,
    out_dir: Path,
    val_ratio: float,
):
    random.seed(SEED)

    label_files = sorted(root.rglob("*_labels.json"))
    if not label_files:
        print(f"[ERREUR] Aucun *_labels.json trouvé sous {root}")
        sys.exit(1)

    samples = []

    # ── Collecte des échantillons ──────────────────────────────────────────
    for lp in label_files:
        stem = lp.stem.removesuffix("_labels")

        flat_path = lp.parent / f"{stem}_flat.jpg"
        if not flat_path.exists():
            continue

        with open(lp, encoding="utf-8") as f:
            data = json.load(f)

        impacts = data.get("impacts", [])
        if not impacts:
            continue

        samples.append(
            {
                "flat_path": flat_path,
                "impacts": impacts,
            }
        )

    if not samples:
        print("[ERREUR] Aucun échantillon valide.")
        sys.exit(1)

    print(f"[LOAD] {len(samples)} image(s) chargée(s)")

    # ── Split train / val ──────────────────────────────────────────────────
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_ratio))
    splits = {
        "train": samples[:-n_val],
        "val": samples[-n_val:],
    }

    # ── Création des dossiers ──────────────────────────────────────────────
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "heatmaps" / split).mkdir(parents=True, exist_ok=True)

    # ── Génération images + heatmaps ───────────────────────────────────────
    for split, split_samples in splits.items():
        print(f"\n[{split.upper()}]")
        for s in split_samples:
            img = cv2.imread(str(s["flat_path"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize image
            img_rs = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

            heatmap = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

            for imp in s["impacts"]:
                cx = imp["cx_px"] * SCALE
                cy = imp["cy_px"] * SCALE

                g = make_gaussian_heatmap(TARGET_SIZE, cx, cy, SIGMA)
                heatmap += g

            # Clamp [0,1]
            # Accentuer les pics (très important)
            heatmap = np.clip(heatmap, 0.0, 1.0)
            heatmap = heatmap**0.5  # <-- CRITIQUE

            # Sauvegarde
            name = s["flat_path"].name.replace("_flat.jpg", "")

            img_out = out_dir / "images" / split / f"{name}.png"
            hm_out = out_dir / "heatmaps" / split / f"{name}.npy"

            cv2.imwrite(str(img_out), img_rs)
            np.save(hm_out, heatmap)

            print(f"  {name:30s}  impacts={len(s['impacts'])}")

    # ── Meta informations ──────────────────────────────────────────────────
    meta = {
        "input_size": INPUT_SIZE,
        "target_size": TARGET_SIZE,
        "scale": SCALE,
        "sigma": SIGMA,
        "val_ratio": val_ratio,
        "seed": SEED,
        "description": "Heatmap dataset for bullet impact detection",
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Dataset heatmap prêt → {out_dir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Prépare un dataset heatmap pour la détection d'impacts."
    )
    ap.add_argument("root", help="Dossier contenant les *_labels.json et *_flat.jpg")
    ap.add_argument("--out", default="data/impacts_heatmap", help="Dossier de sortie")
    ap.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Fraction validation")
    args = ap.parse_args()

    prepare_dataset(Path(args.root), Path(args.out), args.val_ratio)
