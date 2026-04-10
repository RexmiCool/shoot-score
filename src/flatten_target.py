"""
Étape 2 du pipeline : correction de perspective + crop circulaire de la zone de score.

Principe :
  Le disque noir central est un cercle parfait (Ø200mm). S'il apparaît comme
  une ellipse dans l'image, c'est uniquement dû à la perspective (caméra non
  perpendiculaire à la cible). En fittant une ellipse sur son contour et en
  calculant la transformation affine qui ramène cette ellipse à un cercle
  parfait, on corrige la perspective de toute l'image.

  Sortie : image carrée OUTPUT_SIZE×OUTPUT_SIZE centrée sur la cible.
  Seule la zone de score (anneau extérieur Ø500mm) est conservée ; le fond
  hors du cercle est mis en blanc.

Usage :
    python src/flatten_target.py <image_ou_dossier> [--out outputs] [--debug] [--show]
"""

import argparse
import sys
from math import pi
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from localize_target import detect_black_disk, _open_file, _collect_images

# ── Constantes cible ───────────────────────────────────────────────────────────
BLACK_DISK_RADIUS_MM   = 100.0   # rayon disque noir (mm)
OUTER_CIRCLE_RADIUS_MM = 250.0   # rayon anneau extérieur / zone de score (mm)

# ── Paramètres de sortie ───────────────────────────────────────────────────────
OUTPUT_SIZE         = 1040                       # côté de l'image de sortie (px)
OUTPUT_CENTER       = OUTPUT_SIZE // 2           # 520
OUTPUT_OUTER_RADIUS = int(OUTPUT_SIZE * 0.46)    # ≈ 478 px pour le cercle Ø500mm
# → mm/px en sortie = 250 / 478 ≈ 0.523 mm/px
# → rayon disque en sortie = 100 / 0.523 ≈ 191 px  (soit 40% du rayon extérieur ✓)
OUTPUT_DISK_RADIUS  = int(OUTPUT_OUTER_RADIUS * BLACK_DISK_RADIUS_MM / OUTER_CIRCLE_RADIUS_MM)
MM_PER_PX_OUT       = OUTER_CIRCLE_RADIUS_MM / OUTPUT_OUTER_RADIUS  # ≈ 0.523 mm/px


# ── Étape 2-A : extraction du contour du disque noir ──────────────────────────

def extract_disk_contour(
    img: np.ndarray,
    cx: int, cy: int, r_disk: int,
    debug_dir: Path | None = None,
    stem: str = "",
) -> np.ndarray | None:
    """
    Extrait le contour précis du disque noir dans une ROI serrée autour
    du centre détecté. Les impacts (trous clairs dans le disque) sont
    rebouchés par fermeture morphologique avant extraction du contour.

    Retourne le contour en coordonnées image complète (shape: Nx1x2), ou None.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ROI serrée : disque + 20% de marge
    margin = int(r_disk * 0.20)
    x1 = max(0, cx - r_disk - margin)
    y1 = max(0, cy - r_disk - margin)
    x2 = min(w, cx + r_disk + margin)
    y2 = min(h, cy + r_disk + margin)

    roi = cv2.GaussianBlur(gray[y1:y2, x1:x2], (7, 7), 0)
    roi_cx, roi_cy = cx - x1, cy - y1

    best_cnt = None
    for thresh in [40, 55, 70, 85, 100, 120]:
        _, mask = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY_INV)

        # Fermeture pour reboucher les impacts (trous clairs dans le disque noir)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        candidates = []
        for cnt in cnts:
            if len(cnt) < 20:
                continue
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            circ = 4 * pi * area / peri ** 2 if peri > 0 else 0
            if circ < 0.50:
                continue
            (ccx, ccy), r = cv2.minEnclosingCircle(cnt)
            if (abs(ccx - roi_cx) < r_disk * 0.25
                    and abs(ccy - roi_cy) < r_disk * 0.25
                    and 0.65 * r_disk < r < 1.35 * r_disk):
                candidates.append((circ, cnt))

        if candidates:
            best_cnt = max(candidates, key=lambda x: x[0])[1]
            print(f"[CONTOUR] seuil={thresh}  pts={len(best_cnt)}"
                  f"  circ={max(candidates, key=lambda x: x[0])[0]:.3f}")
            break

    if best_cnt is None:
        return None

    # Remise en coordonnées image complète
    full_cnt = best_cnt + np.array([[[x1, y1]]], dtype=np.int32)

    if debug_dir:
        dbg = img.copy()
        thick = max(2, r_disk // 60)
        cv2.drawContours(dbg, [full_cnt], -1, (0, 255, 100), thick)
        cv2.imwrite(str(debug_dir / f"{stem}_2a_disk_contour.jpg"), dbg)

    return full_cnt


# ── Étape 2-B : fit d'ellipse + matrice de correction ─────────────────────────

def compute_flatten_transform(
    ellipse,
    debug_dir: Path | None = None,
    stem: str = "",
) -> np.ndarray:
    """
    Calcule la matrice affine 2×3 qui :
      1. Transforme l'ellipse du disque noir en cercle parfait.
      2. Centre le résultat en (OUTPUT_CENTER, OUTPUT_CENTER).
      3. Dimensionne l'image pour que l'anneau extérieur (Ø500mm) ait
         rayon OUTPUT_OUTER_RADIUS dans l'image de sortie.

    Formule :
      A = R(θ) @ diag(sx, sy) @ R(-θ)
      où sx = OUTPUT_DISK_RADIUS / a   (grand axe → rayon cible)
         sy = OUTPUT_DISK_RADIUS / b   (petit axe → même rayon, corrige foreshortening)
      avec a = demi-grand axe, b = demi-petit axe, θ = orientation.

    Retourne la matrice affine M (2×3, float64).
    """
    (ex, ey), (w_e, h_e), angle = ellipse

    # Identifier grand axe / petit axe indépendamment de l'ordre OpenCV
    if w_e >= h_e:
        a     = w_e / 2.0    # demi-grand axe
        b     = h_e / 2.0    # demi-petit axe (foreshortened)
        theta = np.deg2rad(angle)
    else:
        a     = h_e / 2.0
        b     = w_e / 2.0
        theta = np.deg2rad(angle + 90.0)

    ecc = 1.0 - b / a if a > 0 else 0.0
    print(f"[ELLIPSE] a={a*2:.1f}px  b={b*2:.1f}px  "
          f"angle={angle:.1f}°  excentricité={ecc:.3f}")

    # sx : grand axe (non distordu) → rayon de sortie
    # sy : petit axe (comprimé par la perspective) → même rayon (sy > sx)
    sx = OUTPUT_DISK_RADIUS / a
    sy = OUTPUT_DISK_RADIUS / b

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R_p = np.array([[ cos_t, -sin_t], [ sin_t, cos_t]])   # R(+θ)
    R_n = np.array([[ cos_t,  sin_t], [-sin_t, cos_t]])   # R(-θ)

    # A = R(θ) @ diag(sx, sy) @ R(-θ)  — ramène l'ellipse au cercle, préserve l'orientation
    A = R_p @ np.diag([sx, sy]) @ R_n

    # Translation : centre ellipse → OUTPUT_CENTER
    tx = OUTPUT_CENTER - (A[0, 0] * ex + A[0, 1] * ey)
    ty = OUTPUT_CENTER - (A[1, 0] * ex + A[1, 1] * ey)

    M = np.array([[A[0, 0], A[0, 1], tx],
                  [A[1, 0], A[1, 1], ty]], dtype=np.float64)
    return M


# ── Étape 2-C : application + crop circulaire ─────────────────────────────────

def apply_flatten(
    img: np.ndarray,
    M: np.ndarray,
    debug_dir: Path | None = None,
    stem: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique la correction affine de perspective.

    Retourne (flat, flat_annot) :
      - flat      : image corrigée brute (OUTPUT_SIZE × OUTPUT_SIZE)
      - flat_annot: idem + disque noir (rouge) et centre (croix bleue)
    """
    flat = cv2.warpAffine(
        img, M, (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(220, 220, 220),
    )

    # Annotation minimale : disque noir + centre
    annot = flat.copy()
    cv2.circle(annot, (OUTPUT_CENTER, OUTPUT_CENTER),
               OUTPUT_DISK_RADIUS, (0, 0, 255), 2)              # disque noir (rouge)
    cv2.drawMarker(annot, (OUTPUT_CENTER, OUTPUT_CENTER),
                   (255, 0, 0), cv2.MARKER_CROSS, 40, 2)        # centre (croix bleue)

    return flat, annot


# ── Pipeline complète ─────────────────────────────────────────────────────────

def process(
    image_path,
    output_dir: str = "outputs",
    debug: bool = False,
    show: bool = False,
    hint_cx_norm: float | None = None,
    hint_cy_norm: float | None = None,
) -> Path | None:
    """
    Photo brute → zone de score corrigée en perspective (1040×1040px, fond blanc).

    hint_cx_norm, hint_cy_norm : coordonnées normalisées (0-1) du centre de la
    croix de visée dans l'image, issues de l'application mobile. Quand fournis,
    la détection du disque noir cherche en priorité dans cette zone.
    """
    stem    = Path(image_path).stem
    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERREUR] Impossible de lire : {image_path}")
        return None
    h, w = img.shape[:2]
    print(f"\n{'='*60}\n[IMAGE] {Path(image_path).name}  {w}×{h}px")

    # ── Étape 1 : disque noir ──────────────────────────────────────────────────
    disk = detect_black_disk(
        img,
        debug_dir=out_dir if debug else None,
        stem=stem,
        hint_cx_norm=hint_cx_norm,
        hint_cy_norm=hint_cy_norm,
    )
    if disk is None:
        print("[ERREUR] Disque noir non trouvé.")
        return None
    cx, cy, r_disk, mm_per_px = disk

    if debug:
        dbg = img.copy()
        r_outer = int(OUTER_CIRCLE_RADIUS_MM / mm_per_px)
        cv2.circle(dbg, (cx, cy), r_disk,  (0,   0, 255), max(3, h // 300))
        cv2.circle(dbg, (cx, cy), r_outer, (0, 220,   0), max(2, h // 400))
        cv2.drawMarker(dbg, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 60, 3)
        cv2.imwrite(str(out_dir / f"{stem}_1_disk_detected.jpg"), dbg)

    # ── Étape 2-A : contour précis du disque ──────────────────────────────────
    contour = extract_disk_contour(
        img, cx, cy, r_disk,
        debug_dir=out_dir if debug else None,
        stem=stem,
    )

    if contour is None or len(contour) < 5:
        print("[WARN] Contour non trouvé — fallback ellipse circulaire (pas de correction).")
        ellipse = ((float(cx), float(cy)),
                   (float(r_disk * 2), float(r_disk * 2)), 0.0)
    else:
        ellipse = cv2.fitEllipse(contour)
        if debug:
            dbg = img.copy()
            cv2.ellipse(dbg, ellipse, (0, 200, 255), max(2, r_disk // 60))
            cv2.imwrite(str(out_dir / f"{stem}_2b_ellipse_fit.jpg"), dbg)

    # ── Étape 2-B : matrice affine de correction ───────────────────────────────
    M = compute_flatten_transform(
        ellipse,
        debug_dir=out_dir if debug else None,
        stem=stem,
    )

    # ── Étape 2-C : correction perspective ────────────────────────────────────
    flat, annot = apply_flatten(
        img, M,
        debug_dir=out_dir if debug else None,
        stem=stem,
    )

    # ── Sauvegarde ─────────────────────────────────────────────────────────────
    flat_path  = out_dir / f"{stem}_flat.jpg"
    annot_path = out_dir / f"{stem}_flat_annot.jpg"
    cv2.imwrite(str(flat_path),  flat,  [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(annot_path), annot, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[OK] → {flat_path.name}")
    print(f"[OK] → {annot_path.name}")

    if show:
        _open_file(annot_path)

    return annot_path


# ── Résumé visuel ─────────────────────────────────────────────────────────────

def make_summary(annot_paths: list[Path], out_dir: Path) -> Path:
    """
    Génère une planche de contact avec toutes les images flat_annot.
    Chaque vignette affiche le nom du fichier source.
    Code couleur : vert = trouvé, rouge = échec.
    """
    COLS   = 4
    CARD_W = 280
    CARD_H = 300

    # Séparer trouvés / échecs (annot_path=None si échec)
    total = len(annot_paths)
    rows  = (total + COLS - 1) // COLS
    canvas = np.full((rows * CARD_H, COLS * CARD_W, 3), 25, dtype=np.uint8)

    for idx, entry in enumerate(annot_paths):
        col_i = idx % COLS
        row_i = idx // COLS
        x0 = col_i * CARD_W
        y0 = row_i * CARD_H

        stem   = entry["stem"]
        path   = entry["path"]    # None si échec
        ok     = path is not None and path.exists()
        border = (0, 180, 0) if ok else (0, 0, 200)

        # Vignette
        if ok:
            img_th = cv2.imread(str(path))
            if img_th is not None:
                th_h = CARD_H - 44
                scale = th_h / img_th.shape[0]
                th_w  = max(1, int(img_th.shape[1] * scale))
                thumb = cv2.resize(img_th, (th_w, th_h), interpolation=cv2.INTER_AREA)
                # Centrer dans la carte
                tx = x0 + (CARD_W - th_w) // 2
                ty = y0 + 4
                if tx >= 0 and tx + th_w <= canvas.shape[1]:
                    canvas[ty:ty + th_h, tx:tx + th_w] = thumb

        # Cadre coloré
        cv2.rectangle(canvas, (x0 + 1, y0 + 1),
                      (x0 + CARD_W - 2, y0 + CARD_H - 2), border, 3)

        # Nom (tronqué) + statut
        label = (stem[-20:] + "  OK") if ok else (stem[-20:] + "  ECHEC")
        cv2.putText(canvas, label,
                    (x0 + 6, y0 + CARD_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)

    out_path = out_dir / "_summary_flat.jpg"
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n[RÉSUMÉ] {sum(1 for e in annot_paths if e['path'])} / {total} → {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Corrige la perspective via l'ellipse du disque noir "
                    "et crop la zone de score (Ø500mm) en 1040×1040px.")
    ap.add_argument("image", help="Image ou dossier d'images")
    ap.add_argument("--out",   default="outputs",
                    help="Dossier de sortie (défaut : outputs/)")
    ap.add_argument("--debug", action="store_true",
                    help="Sauvegarde les images intermédiaires")
    ap.add_argument("--show",  action="store_true",
                    help="Ouvre l'image annotée finale")
    args = ap.parse_args()

    images      = _collect_images(args.image)
    out_dir     = Path(args.out)
    annot_paths = []

    for img_path in images:
        result = process(img_path, args.out, debug=args.debug, show=args.show)
        annot_paths.append({"stem": img_path.stem, "path": result})

    if len(images) > 1:
        summary_path = make_summary(annot_paths, out_dir)
        if args.show:
            _open_file(summary_path)
