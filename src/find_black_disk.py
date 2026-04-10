"""
Detecte tous les cercles d'une cible de tir et les impacts.

Phase 1 (image originale) : localise le disque noir central -> crop.
Phase 2 (image croppee)   : annote les cercles (interieurs, intermediaires,
                             exterieur) et detecte les impacts de 9mm.

Usage : uv run python src/find_black_disk.py <image> [--out outputs] [--debug] [--show]
"""
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

from target_utils import (
    BLACK_DISK_RADIUS_MM, OUTER_CIRCLE_RADIUS_MM, N_INTERMEDIATE, BULLET_DIAM_MM,
    find_black_disk_in_image, crop_target, collect_images, open_file,
)


# -- Helpers geometriques -----------------------------------------------------

def _inner_circle_radii_px(mm_per_px: float, n: int = 3) -> list:
    """Rayons (px) des cercles blancs a l'interieur du disque noir.
    Equidistants, pas = step_mm depuis le bord du disque vers le centre.
    """
    step_mm = (OUTER_CIRCLE_RADIUS_MM - BLACK_DISK_RADIUS_MM) / (N_INTERMEDIATE + 1)
    return [int((BLACK_DISK_RADIUS_MM - (i + 1) * step_mm) / mm_per_px) for i in range(n)]


# -- Detection des impacts ----------------------------------------------------

def _detect_impacts(img, cx, cy, r_disk_px, r_outer_px, mm_per_px,
                    debug_dir=None, stem=""):
    """
    Detecte les impacts de projectile sur l'image croppee.

    - Zone blanche (anneau) : trous sombres -> seuillage adaptatif inverse.
    - Disque noir           : anomalies claires -> top-hat morphologique.

    Retourne une liste de dicts {cx, cy, r_px, r_mm, zone}.
    """
    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    km    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    r_nom_px = (BULLET_DIAM_MM / 2.0) / mm_per_px
    r_min_px = int(r_nom_px * 0.4)
    r_max_px = int(r_nom_px * 2.0)
    min_area = np.pi * r_min_px ** 2
    max_area = np.pi * r_max_px ** 2

    def _circ(cnt):
        a = cv2.contourArea(cnt)
        p = cv2.arcLength(cnt, True)
        return 4 * np.pi * a / p ** 2 if p > 0 else 0

    impacts = []

    # -- Zone blanche ----------------------------------------------------------
    mask_w = np.zeros((h, w), np.uint8)
    cv2.circle(mask_w, (cx, cy), int(r_outer_px * 1.05), 255, -1)
    cv2.circle(mask_w, (cx, cy), int(r_disk_px  * 1.05),   0, -1)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{stem}_W1_mask.jpg"), mask_w)

    th_w = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 8)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{stem}_W2_adapt_thresh.jpg"), th_w)

    th_w = cv2.bitwise_and(th_w, th_w, mask=mask_w)
    th_w = cv2.morphologyEx(th_w, cv2.MORPH_OPEN,  km, iterations=1)
    th_w = cv2.morphologyEx(th_w, cv2.MORPH_CLOSE, km, iterations=2)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{stem}_W3_morph.jpg"), th_w)

    dbg_w = img.copy() if debug_dir else None
    for cnt in cv2.findContours(th_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if not (min_area <= cv2.contourArea(cnt) <= max_area):
            continue
        if _circ(cnt) < 0.15:
            continue
        (icx, icy), ir = cv2.minEnclosingCircle(cnt)
        impacts.append(dict(cx=int(icx), cy=int(icy), r_px=int(ir),
                            r_mm=ir * mm_per_px, zone="white"))
        if debug_dir:
            cv2.circle(dbg_w, (int(icx), int(icy)), int(ir), (0, 255, 255), 2)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{stem}_W4_detections.jpg"), dbg_w)

    # -- Disque noir (top-hat) -------------------------------------------------
    mask_b = np.zeros((h, w), np.uint8)
    cv2.circle(mask_b, (cx, cy), int(r_disk_px * 0.95), 255, -1)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{stem}_B1_mask.jpg"), mask_b)

    ksize  = max(int(r_nom_px * 2.5) | 1, 11)
    k_th   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, k_th)
    tophat = cv2.bitwise_and(tophat, tophat, mask=mask_b)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{stem}_B2_tophat.jpg"), tophat)

    _, th_b = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_b    = cv2.morphologyEx(th_b, cv2.MORPH_OPEN,  km, iterations=1)
    th_b    = cv2.morphologyEx(th_b, cv2.MORPH_CLOSE, km, iterations=2)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{stem}_B3_morph.jpg"), th_b)

    dbg_b = img.copy() if debug_dir else None
    for cnt in cv2.findContours(th_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if not (min_area <= cv2.contourArea(cnt) <= max_area):
            continue
        if _circ(cnt) < 0.15:
            continue
        (icx, icy), ir = cv2.minEnclosingCircle(cnt)
        impacts.append(dict(cx=int(icx), cy=int(icy), r_px=int(ir),
                            r_mm=ir * mm_per_px, zone="black"))
        if debug_dir:
            cv2.circle(dbg_b, (int(icx), int(icy)), int(ir), (180, 105, 255), 2)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{stem}_B4_detections.jpg"), dbg_b)

    return impacts


# -- Fonction principale -------------------------------------------------------

def find_black_disk(image_path, output_dir="outputs", debug=False, show=False):
    stem    = Path(image_path).stem
    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 : crop
    img_orig = cv2.imread(str(image_path))
    if img_orig is None:
        raise FileNotFoundError(image_path)
    h0, w0 = img_orig.shape[:2]
    print(f"[P1] Image originale : {w0}x{h0}px")

    result = crop_target(img_orig)
    if result is None:
        print("[P1] Disque noir non trouve dans l'image originale.")
        return
    img_crop, _ = result

    if debug:
        crop_path = out_dir / f"{stem}_cropped.jpg"
        cv2.imwrite(str(crop_path), img_crop)
        print(f"[P1] Crop sauvegarde : {crop_path.name}  ({img_crop.shape[1]}x{img_crop.shape[0]}px)")

    # Phase 2 : analyse du crop
    img  = img_crop
    h, w = img.shape[:2]
    print(f"\n[P2] Analyse du crop : {w}x{h}px")

    disk2 = find_black_disk_in_image(img)
    if disk2 is None:
        print("[P2] Disque noir non trouve dans le crop.")
        return
    cx, cy, r_disk, mm_per_px = disk2
    r_outer = int(OUTER_CIRCLE_RADIUS_MM / mm_per_px)
    print(f"[P2] Disque noir : centre=({cx},{cy})  r={r_disk}px  mm/px={mm_per_px:.4f}")

    inner_radii_px = _inner_circle_radii_px(mm_per_px)
    step_mm        = (OUTER_CIRCLE_RADIUS_MM - BLACK_DISK_RADIUS_MM) / (N_INTERMEDIATE + 1)

    # Annotation des cercles
    annot_c    = img.copy()
    thick      = max(2, h // 300)
    font_scale = max(0.5, h / 1500)
    font_thick = max(1, h // 600)

    def label(canvas, txt, x, y, color):
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.75, color, font_thick)

    cv2.drawMarker(annot_c, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.circle(annot_c, (cx, cy), r_disk, (0, 0, 255), thick)
    label(annot_c, f"noir {int(r_disk*mm_per_px*2)}mm", cx + r_disk + 5, cy - 10, (0, 0, 255))

    for r_px in inner_radii_px:
        cv2.circle(annot_c, (cx, cy), r_px, (255, 255, 0), thick)
        label(annot_c, f"{int(r_px*mm_per_px*2)}mm", cx + r_px + 5, cy, (255, 255, 0))

    for i in range(1, N_INTERMEDIATE + 1):
        r_mm = BLACK_DISK_RADIUS_MM + i * step_mm
        r_px = int(r_mm / mm_per_px)
        cv2.circle(annot_c, (cx, cy), r_px, (0, 165, 255), thick)
        label(annot_c, f"{int(r_mm*2)}mm", cx + r_px + 5,
              cy + i * int(font_scale * 25), (0, 165, 255))
        print(f"  intermediaire {i} : r={r_px}px  diam={int(r_mm*2)}mm")

    cv2.circle(annot_c, (cx, cy), r_outer, (0, 255, 0), thick)
    label(annot_c, f"ext. {int(r_outer*mm_per_px*2)}mm",
          cx + r_outer + 5, cy + 30, (0, 255, 0))

    cercles_path = out_dir / f"{stem}_cercles.jpg"
    cv2.imwrite(str(cercles_path), annot_c)
    print(f"\n[P2] Cercles -> {cercles_path.name}")

    # Detection et annotation des impacts
    impacts = _detect_impacts(img, cx, cy, r_disk, r_outer, mm_per_px,
                              debug_dir=out_dir if debug else None, stem=stem)
    print(f"[IMPACTS] {len(impacts)} impact(s) detecte(s) :")

    annot_i = img.copy()
    for i, imp in enumerate(impacts):
        dist_mm = np.hypot(imp["cx"] - cx, imp["cy"] - cy) * mm_per_px
        print(f"  #{i+1} centre=({imp['cx']},{imp['cy']})  "
              f"diam={imp['r_mm']*2:.1f}mm  dist_centre={dist_mm:.1f}mm  zone={imp['zone']}")
        color = (0, 255, 255) if imp["zone"] == "white" else (180, 105, 255)
        cv2.circle(annot_i, (imp["cx"], imp["cy"]), imp["r_px"], color, thick)
        cv2.circle(annot_i, (imp["cx"], imp["cy"]), 4, color, -1)
        label(annot_i, f"#{i+1}", imp["cx"] + imp["r_px"] + 4, imp["cy"], color)

    impacts_path = out_dir / f"{stem}_impacts.jpg"
    cv2.imwrite(str(impacts_path), annot_i)
    print(f"[P2] Impacts -> {impacts_path.name}")

    if show:
        open_file(cercles_path)
        open_file(impacts_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out",   default="outputs")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--show",  action="store_true")
    args = ap.parse_args()

    for img_path in collect_images(args.image):
        print(f"\n{'='*60}\n{img_path.name}\n{'='*60}")
        find_black_disk(img_path, args.out, debug=args.debug, show=args.show)
