# src/detect_impacts_heatmap.py
"""
Détection des impacts par heatmap (remplace YOLO).

Entrée :
  - image *_flat.jpg (1056×1056)

Sorties :
  - *_impacts_heatmap.json
  - *_impacts_heatmap.jpg

Usage :
  uv run python src/detect_impacts_heatmap.py outputs/flatten/XXXXXXXX_flat.jpg --show
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from flatten_target import OUTPUT_CENTER, MM_PER_PX_OUT
from models.impact_heatmap_cnn import ImpactHeatmapCNN

# -----------------------------
# PARAMÈTRES
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "impacts_heatmap" / "model_best.pt"
TARGET_SIZE = 512
INPUT_SIZE = 1056
SCALE = INPUT_SIZE / TARGET_SIZE

CONF_THR = 0.25  # seuil heatmap (sigmoid peaks rarely exceed 0.6-0.8)
MIN_DIST_PX = 6  # séparation min (espace 512)

# Couronne de score (mm)
SCORE_TABLE = [
    (25, 10),
    (50, 9),
    (75, 8),
    (100, 7),
    (125, 6),
    (150, 5),
    (175, 4),
    (200, 3),
    (225, 2),
    (250, 1),
]


# -----------------------------
# UTILITAIRES
# -----------------------------
def score_from_mm(dist_mm: float) -> int:
    for r, s in SCORE_TABLE:
        if dist_mm <= r:
            return s
    return 0


def find_peaks(hm: np.ndarray, thr: float, min_dist: int):
    """Maxima locaux simples (pas de NMS lourde)."""
    peaks = []
    work = hm.copy()

    while True:
        y, x = np.unravel_index(np.argmax(work), work.shape)
        v = work[y, x]
        if v < thr:
            break
        peaks.append((x, y, float(v)))
        x0, x1 = max(0, x - min_dist), min(work.shape[1], x + min_dist + 1)
        y0, y1 = max(0, y - min_dist), min(work.shape[0], y + min_dist + 1)
        work[y0:y1, x0:x1] = 0.0

    return peaks


# -----------------------------
# PIPELINE
# -----------------------------
def run(flat_path: Path, show: bool = False):
    img = cv2.imread(str(flat_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(flat_path)

    # Resize pour l’IA
    img512 = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img512.astype(np.float32) / 255.0)[None, None, :, :]

    # Load modèle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImpactHeatmapCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    total = sum(p.abs().sum().item() for p in model.parameters())
    print("Model weight L1 sum:", f"{total:.6f}")
    model.eval()

    with torch.no_grad():
        hm = model(t.to(device))[0, 0].cpu().numpy()

    # Détection des pics
    peaks512 = find_peaks(hm, CONF_THR, MIN_DIST_PX)
    print("Heatmap max value:", hm.max())

    impacts = []
    annot = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i, (x512, y512, conf) in enumerate(peaks512, 1):
        cx_px = x512 * SCALE
        cy_px = y512 * SCALE

        # score géométrique
        dist_mm = np.hypot(cx_px - OUTPUT_CENTER, cy_px - OUTPUT_CENTER) * MM_PER_PX_OUT
        score = score_from_mm(dist_mm)

        impacts.append(
            {
                "cx_px": round(cx_px, 1),
                "cy_px": round(cy_px, 1),
                "conf": round(conf, 3),
                "score": score,
                "dist_centre_mm": round(dist_mm, 1),
            }
        )

        # annotation
        c = (0, 255, 0) if score >= 7 else (0, 165, 255)
        cv2.circle(annot, (int(cx_px), int(cy_px)), 10, c, 2)
        cv2.putText(
            annot, f"{score}", (int(cx_px) + 6, int(cy_px) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2
        )

    # Sauvegarde
    stem = flat_path.stem.replace("_flat", "")
    out_dir = flat_path.parent
    json_path = out_dir / f"{stem}_impacts_heatmap.json"
    img_path = out_dir / f"{stem}_impacts_heatmap.jpg"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": flat_path.name,
                "engine": "heatmap",
                "n_impacts": len(impacts),
                "impacts": impacts,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    cv2.imwrite(str(img_path), annot, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"[OK] {len(impacts)} impacts → {img_path.name}")
    if show:
        cv2.imshow("impacts", annot)
        cv2.waitKey(0)

    return img_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Détection d'impacts par heatmap")
    ap.add_argument("image", help="Image *_flat.jpg")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    run(Path(args.image), show=args.show)
