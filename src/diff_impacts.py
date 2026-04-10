"""
Détecte les impacts en comparant une cible avec impacts à une cible de référence sans impacts.

Stratégie :
  1. Détecter le disque noir dans les deux images → centre (cx,cy) et rayon r_disk.
  2. Transformer chaque image via une transformation affine (scale + translation) pour
     ramener les deux dans un espace canonique où le cercle extérieur a exactement
     CANONICAL_OUTER_RADIUS pixels de rayon centré en CANONICAL_CENTER.
     → Tous les anneaux sont parfaitement superposés par construction.
  3. Masquer tout ce qui est hors du cercle extérieur.
  4. Normaliser la luminosité (correspondance d'histogramme) pour absorber les
     différences d'éclairage.
  5. Calculer la différence absolue en niveaux de gris.
  6. Seuiller et filtrer les contours par taille compatible avec un impact de 9mm.

Usage :
  uv run python src/diff_impacts.py <image_ou_dossier> --ref data/ressources/cible.jpg --out outputs [--debug]
"""
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# ── Constantes géométriques ───────────────────────────────────────────────────
BLACK_DISK_RADIUS_MM   = 100.0
OUTER_CIRCLE_RADIUS_MM = 250.0
BULLET_DIAM_MM         = 9.0

# Espace canonique
CANONICAL_SIZE         = 1200          # taille de l'image de sortie (px)
CANONICAL_CENTER       = CANONICAL_SIZE // 2
CANONICAL_OUTER_RADIUS = int(CANONICAL_SIZE * 0.46)   # r du cercle ext. en canonique


def _find_black_disk_in_image(img):
    """Détecte le disque noir central. Retourne (cx, cy, r_px, mm_per_px) ou None.
    Seuil adaptatif basé sur l'histogramme — robuste au flash."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = int(np.percentile(blur, 25) * 1.3)
    thresh = max(30, min(thresh, 180))
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    min_area  = 0.005 * h * w
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


def _to_canonical(img):
    """
    Transforme l'image vers l'espace canonique en utilisant une transformation
    affine (scale + translation) calculée à partir du disque noir détecté.
    Tous les anneaux seront parfaitement superposés avec n'importe quelle autre
    image transformée de la même façon.

    Retourne (canonical_img, mm_per_px_canonical) ou None.
    """
    result = _find_black_disk_in_image(img)
    if result is None:
        return None
    cx, cy, r_disk, mm_per_px = result

    # Rayon du cercle extérieur dans l'image originale (px)
    r_outer_orig = OUTER_CIRCLE_RADIUS_MM / mm_per_px

    # Facteur d'échelle : amener r_outer_orig → CANONICAL_OUTER_RADIUS
    scale = CANONICAL_OUTER_RADIUS / r_outer_orig

    # Translation pour centrer
    tx = CANONICAL_CENTER - cx * scale
    ty = CANONICAL_CENTER - cy * scale

    M = np.float32([[scale, 0, tx],
                    [0, scale, ty]])

    canonical = cv2.warpAffine(
        img, M, (CANONICAL_SIZE, CANONICAL_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(200, 200, 200),
    )

    mm_per_px_c = OUTER_CIRCLE_RADIUS_MM / CANONICAL_OUTER_RADIUS
    return canonical, mm_per_px_c


def _normalize_brightness(ref_gray, tgt_gray, mask):
    """
    Ajuste la luminosité de tgt_gray pour qu'elle corresponde à ref_gray
    dans la zone masquée (correspondance de moyenne et d'écart-type).
    """
    ref_vals = ref_gray[mask > 0].astype(np.float32)
    tgt_vals = tgt_gray[mask > 0].astype(np.float32)
    if ref_vals.std() < 1 or tgt_vals.std() < 1:
        return tgt_gray
    alpha = ref_vals.std() / tgt_vals.std()
    beta  = ref_vals.mean() - alpha * tgt_vals.mean()
    tgt_norm = np.clip(tgt_gray.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return tgt_norm


def detect_impacts_by_diff(img_ref_path, img_target_path, out_dir, debug=False):
    """Compare une image cible avec la référence et détecte les impacts."""
    stem    = Path(img_target_path).stem
    img_dir = Path(out_dir) / stem
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement ─────────────────────────────────────────────────────────────
    img_ref_orig    = cv2.imread(str(img_ref_path))
    img_target_orig = cv2.imread(str(img_target_path))
    if img_ref_orig is None:
        print(f"[ERREUR] Référence introuvable : {img_ref_path}"); return
    if img_target_orig is None:
        print(f"[ERREUR] Image introuvable : {img_target_path}"); return

    # ── Transformation vers l'espace canonique ─────────────────────────────────
    res_ref = _to_canonical(img_ref_orig)
    if res_ref is None:
        print("[ERREUR] Disque noir non trouvé dans la référence."); return
    ref_can, mm_per_px_c = res_ref

    res_tgt = _to_canonical(img_target_orig)
    if res_tgt is None:
        print(f"[ERREUR] Disque noir non trouvé dans {img_target_path}."); return
    tgt_can, _ = res_tgt

    if debug:
        cv2.imwrite(str(img_dir / f"{stem}_01_ref_canonical.jpg"),    ref_can)
        cv2.imwrite(str(img_dir / f"{stem}_02_target_canonical.jpg"), tgt_can)

        # Overlay 50/50 pour vérifier la superposition
        overlay = cv2.addWeighted(ref_can, 0.5, tgt_can, 0.5, 0)
        cv2.imwrite(str(img_dir / f"{stem}_03_overlay.jpg"), overlay)

    # ── Masque circulaire (cercle extérieur uniquement) ────────────────────────
    mask_circle = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8)
    cv2.circle(mask_circle, (CANONICAL_CENTER, CANONICAL_CENTER),
               CANONICAL_OUTER_RADIUS, 255, -1)

    # ── Conversion en niveaux de gris ──────────────────────────────────────────
    ref_gray = cv2.cvtColor(ref_can, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(tgt_can, cv2.COLOR_BGR2GRAY)

    # ── Normalisation de luminosité (dans la zone circulaire) ──────────────────
    tgt_gray_norm = _normalize_brightness(ref_gray, tgt_gray, mask_circle)
    if debug:
        cv2.imwrite(str(img_dir / f"{stem}_04_ref_gray.jpg"),      ref_gray)
        cv2.imwrite(str(img_dir / f"{stem}_05_target_gray_norm.jpg"), tgt_gray_norm)

    # ── Différence absolue masquée ─────────────────────────────────────────────
    diff = cv2.absdiff(ref_gray, tgt_gray_norm)
    diff = cv2.bitwise_and(diff, diff, mask=mask_circle)
    if debug:
        cv2.imwrite(str(img_dir / f"{stem}_06_diff.jpg"), diff)

        # Version amplifiée pour faciliter l'inspection visuelle
        diff_amp = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(str(img_dir / f"{stem}_07_diff_amplified.jpg"), diff_amp)

    # ── Seuillage ─────────────────────────────────────────────────────────────
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    # Seuil Otsu sur la zone masquée uniquement
    diff_masked_vals = diff[mask_circle > 0]
    otsu_val, _ = cv2.threshold(diff_masked_vals, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # On prend max(otsu, 20) pour éviter un seuil trop bas sur image propre
    thresh_val = max(int(otsu_val * 1.5), 20)
    _, th = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
    th = cv2.bitwise_and(th, th, mask=mask_circle)
    if debug:
        cv2.imwrite(str(img_dir / f"{stem}_08_thresh_{thresh_val}.jpg"), th)

    # ── Morphologie ───────────────────────────────────────────────────────────
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k_open,  iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close, iterations=1)
    if debug:
        cv2.imwrite(str(img_dir / f"{stem}_09_morph.jpg"), th)

    # ── Exclusion des zones de bords forts de la référence (chiffres, texte) ──
    # Un léger désalignement fait apparaître les contours des chiffres dans la diff.
    # On dilate les bords de la référence et on les exclut.
    ref_edges = cv2.Canny(ref_gray, 25, 75)
    excl_radius_px = int(10.0 / mm_per_px_c)   # exclure 10mm autour de chaque bord
    k_excl = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (excl_radius_px * 2 + 1, excl_radius_px * 2 + 1))
    ref_edges_dilated = cv2.dilate(ref_edges, k_excl)
    mask_no_text = cv2.bitwise_not(ref_edges_dilated)
    th = cv2.bitwise_and(th, th, mask=mask_no_text)
    if debug:
        cv2.imwrite(str(img_dir / f"{stem}_10_ref_edges_excl.jpg"), ref_edges_dilated)
        cv2.imwrite(str(img_dir / f"{stem}_11_th_no_text.jpg"),     th)

    # ── Filtrage des contours : taille + rayon enclos + ratio d'aspect ─────────
    r_nom_px = (BULLET_DIAM_MM / 2.0) / mm_per_px_c
    # Impact de 9mm : on tolère 40% à 200% du rayon nominal
    r_min_px = int(r_nom_px * 0.4)
    r_max_px = int(r_nom_px * 2.0)   # max ~18mm de diamètre
    min_area = np.pi * r_min_px ** 2
    max_area = np.pi * r_max_px ** 2

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    impacts = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        (icx, icy), ir = cv2.minEnclosingCircle(cnt)
        # Filtre strict sur le rayon réel du cercle englobant
        if ir > r_max_px or ir < r_min_px:
            continue
        # Filtre de forme : ratio d'aspect du bounding rect (évite les lignes/croix)
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        if aspect < 0.35:
            continue
        dist_mm = np.hypot(icx - CANONICAL_CENTER, icy - CANONICAL_CENTER) * mm_per_px_c
        impacts.append(dict(cx=int(icx), cy=int(icy), r_px=max(int(ir), 6),
                            diam_mm=ir * mm_per_px_c * 2,
                            dist_centre_mm=dist_mm))

    # ── Annotation ────────────────────────────────────────────────────────────
    annot = tgt_can.copy()
    thick      = max(2, CANONICAL_SIZE // 400)
    font_scale = 0.55
    for i, imp in enumerate(impacts):
        cv2.circle(annot, (imp['cx'], imp['cy']), imp['r_px'], (0, 255, 255), thick)
        cv2.circle(annot, (imp['cx'], imp['cy']), 4, (0, 255, 255), -1)
        cv2.putText(annot, f"#{i+1}",
                    (imp['cx'] + imp['r_px'] + 4, imp['cy']),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
        print(f"  #{i+1}  centre=({imp['cx']},{imp['cy']})  "
              f"diam≈{imp['diam_mm']:.1f}mm  dist_centre={imp['dist_centre_mm']:.1f}mm")

    # Dessiner aussi le cercle extérieur de référence
    cv2.circle(annot, (CANONICAL_CENTER, CANONICAL_CENTER),
               CANONICAL_OUTER_RADIUS, (0, 200, 0), 1)

    annot_path = img_dir / f"{stem}_impacts.jpg"
    cv2.imwrite(str(annot_path), annot)
    print(f"[OK] {len(impacts)} impact(s) → {annot_path}")
    return impacts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Détecte les impacts par comparaison avec une cible de référence.")
    ap.add_argument("image",  help="Image cible (avec impacts) ou dossier")
    ap.add_argument("--ref",  default=r"data\ressources\cible.jpg",
                    help="Cible de référence sans impacts")
    ap.add_argument("--out",  default="outputs")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    images = []
    p = Path(args.image)
    if p.is_dir():
        for f in p.glob("*.*"):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                images.append(f)
    elif p.is_file():
        images.append(p)
    else:
        print(f"Fichier introuvable : {args.image}")
        sys.exit(1)

    for img_path in images:
        print(f"\n{'='*60}\n{img_path.name}\n{'='*60}")
        detect_impacts_by_diff(args.ref, img_path, args.out, debug=args.debug)
