"""
Utilitaires partagés pour l'analyse de cibles de tir.
Importé par find_black_disk.py, crop_only.py et diff_impacts.py.
"""
import cv2
import numpy as np
import os
import sys
import subprocess
from pathlib import Path

# ── Constantes géométriques de la cible ───────────────────────────────────────
BLACK_DISK_RADIUS_MM   = 100.0   # rayon du disque noir central (mm)
OUTER_CIRCLE_RADIUS_MM = 250.0   # rayon du cercle extérieur (mm)
N_INTERMEDIATE         = 5       # nombre d'anneaux entre disque noir et cercle ext.
BULLET_DIAM_MM         = 9.0     # diamètre du projectile (mm)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Utilitaires généraux ──────────────────────────────────────────────────────

def open_file(path: Path) -> None:
    """Ouvre un fichier avec l'application par défaut du système."""
    if sys.platform == "win32":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def collect_images(path_str: str) -> list[Path]:
    """
    Retourne la liste des images à traiter depuis un chemin (fichier ou dossier).
    Lève SystemExit si le chemin n'existe pas.
    """
    p = Path(path_str)
    if p.is_dir():
        return sorted(f for f in p.glob("*.*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    if p.is_file():
        return [p]
    print(f"Fichier introuvable : {path_str}")
    sys.exit(1)


# ── Détection du disque noir ──────────────────────────────────────────────────

def find_black_disk_in_image(img: np.ndarray):
    """
    Détecte le disque noir central dans une image BGR.

    Utilise un seuil adaptatif basé sur l'histogramme pour être robuste
    aux variations d'exposition (flash, contre-jour, etc.).

    Retourne (cx, cy, r_px, mm_per_px) ou None si non trouvé.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Seuil adaptatif : percentile 25 × 1.3  →  fonctionne même avec flash
    thresh = int(np.percentile(blur, 25) * 1.3)
    thresh = max(30, min(thresh, 180))
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    min_area       = 0.005 * h * w
    cx_img, cy_img = w / 2, h / 2
    tol_x, tol_y   = w * 0.25, h * 0.25

    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
        if circ < 0.5:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if abs(cx - cx_img) > tol_x or abs(cy - cy_img) > tol_y:
            continue
        candidates.append((area, cnt, cx, cy, r))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, _, cx, cy, r = candidates[0]
    return int(cx), int(cy), int(r), BLACK_DISK_RADIUS_MM / r


# ── Crop de la cible ──────────────────────────────────────────────────────────

def crop_target(
    img_orig: np.ndarray,
    margin_mm: float = 0.0,
    remove_bottom_mm: float = 0.0,
) -> tuple[np.ndarray, float] | None:
    """
    Crop l'image originale autour du disque noir détecté.

    - margin_mm        : marge ajoutée de chaque côté autour du cercle extérieur.
    - remove_bottom_mm : épaisseur de la bande retirée en bas après crop.

    Retourne (img_cropped, mm_per_px) ou None si le disque noir n'est pas trouvé.
    """
    result = find_black_disk_in_image(img_orig)
    if result is None:
        return None

    cx, cy, r_disk, mm_per_px = result
    r_outer   = int(OUTER_CIRCLE_RADIUS_MM / mm_per_px)
    margin_px = int(margin_mm / mm_per_px)
    half      = r_outer + margin_px
    h0, w0    = img_orig.shape[:2]

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w0, cx + half)
    y2 = min(h0, cy + half)

    img_crop = img_orig[y1:y2, x1:x2].copy()
    side     = min(img_crop.shape[:2])
    img_crop = img_crop[:side, :side]

    if remove_bottom_mm > 0:
        remove_px = int(remove_bottom_mm / mm_per_px)
        if img_crop.shape[0] > remove_px:
            img_crop = img_crop[: img_crop.shape[0] - remove_px, :]

    return img_crop, mm_per_px
