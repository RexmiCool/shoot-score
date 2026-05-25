"""
Script de debug pour la détection du disque noir central.

Pour chaque image, génère une planche de contact montrant :
  - Col 1 : image originale (réduite) + cercle détecté tracé dessus
  - Col 2-6 : masque binaire pour chaque seuil testé (40, 55, 70, 85, 100)
              avec les contours candidats colorés et leurs métriques

En bas de chaque planche : tableau récap (seuil retenu, r_px, circularité, etc.)

Génère aussi outputs/debug_disk_summary.jpg : toutes les images côte à côte
avec un code couleur vert/orange/rouge (trouvé / fausse détection / pas trouvé).

Usage :
    python src/debug_disk.py <image_ou_dossier> [--out outputs]
"""

import argparse
import sys
from math import pi
from pathlib import Path

import cv2
import numpy as np

# ── Constantes identiques à localize_target.py ────────────────────────────────
BLACK_DISK_RADIUS_MM = 100.0
OUTER_CIRCLE_RADIUS_MM = 250.0
THRESHOLDS = [40, 55, 70, 85, 100]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Taille des vignettes dans la planche de contact
THUMB_H = 480  # hauteur d'une vignette (px)


# ── Helpers visuels ────────────────────────────────────────────────────────────


def _thumb(img: np.ndarray, h: int = THUMB_H) -> np.ndarray:
    """Redimensionne une image en gardant le ratio, hauteur cible = h."""
    oh, ow = img.shape[:2]
    scale = h / oh
    return cv2.resize(img, (max(1, int(ow * scale)), h), interpolation=cv2.INTER_AREA)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Convertit une image niveaux de gris en BGR si nécessaire."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _put_lines(
    canvas: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    scale: float = 0.45,
    color=(220, 220, 220),
    thickness: int = 1,
) -> int:
    """Écrit plusieurs lignes de texte, retourne la position y finale."""
    lh = int(scale * 30 + 8)
    for line in lines:
        cv2.putText(
            canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
        )
        y += lh
    return y


def _banner(text: str, w: int, color_bgr: tuple, h: int = 28) -> np.ndarray:
    """Crée une banderole colorée avec du texte centré."""
    banner = np.full((h, w, 3), color_bgr, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    x = max(4, (w - tw) // 2)
    y = (h + th) // 2
    cv2.putText(
        banner, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA
    )
    return banner


# ── Analyse d'une image : un seuil ────────────────────────────────────────────


def _analyze_threshold(
    img_orig: np.ndarray,
    blur: np.ndarray,
    thresh: int,
    kernel: np.ndarray,
    min_area: float,
    max_area: float,
    cx_img: float,
    cy_img: float,
    tol_x: float,
    tol_y: float,
):
    """
    Applique le seuil, la morphologie et le filtrage des contours.
    Retourne (vis_photo, vis_mask, candidates) :
      - vis_photo : photo originale annotée avec tous les candidats localisés
      - vis_mask  : masque binaire annoté (contexte morphologique)
      - candidates : liste de dicts pour les contours qui passent tous les filtres
    """
    h, w = blur.shape[:2]
    thick = max(2, h // 300)  # épaisseur des tracés proportionnelle à la résolution

    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Deux vues : photo originale + masque binaire
    vis_photo = img_orig.copy()
    vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    candidates = []
    rejected = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        peri = cv2.arcLength(cnt, True)
        circ = 4 * pi * area / peri**2 if peri > 0 else 0
        (cx, cy), r = cv2.minEnclosingCircle(cnt)

        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

        in_area = min_area <= area <= max_area
        in_circ = circ >= 0.60
        in_aspect = aspect >= 0.80  # un cercle a un bounding rect quasi carré
        in_pos = abs(cx - cx_img) <= tol_x and abs(cy - cy_img) <= tol_y
        r_outer = OUTER_CIRCLE_RADIUS_MM / (BLACK_DISK_RADIUS_MM / r) if r > 0 else 0
        ok_outer = r_outer <= min(h, w) * 0.95

        passes = in_area and in_circ and in_aspect and in_pos and ok_outer

        # Raison du rejet (pour la couleur)
        reason_color = (
            (0, 0, 220)
            if not in_circ
            else (0, 200, 220)
            if not in_aspect
            else (0, 130, 255)
            if not in_area
            else (200, 0, 220)
            if not in_pos
            else (40, 40, 160)
            if not ok_outer
            else (0, 220, 0)
        )

        info = dict(
            area=area,
            circ=circ,
            cx=float(cx),
            cy=float(cy),
            r=float(r),
            aspect=aspect,
            in_area=in_area,
            in_circ=in_circ,
            in_aspect=in_aspect,
            in_pos=in_pos,
            ok_outer=ok_outer,
            passes=passes,
        )

        if passes:
            candidates.append(info)

            # ── Sur la PHOTO : cercle épais vert + croix + métriques ──────────
            cv2.circle(vis_photo, (int(cx), int(cy)), int(r), (0, 220, 0), thick)
            cv2.drawMarker(
                vis_photo,
                (int(cx), int(cy)),
                (0, 220, 0),
                cv2.MARKER_CROSS,
                max(30, int(r // 4)),
                thick,
            )
            # Cercle extérieur attendu en pointillé (dessiné en vert clair)
            r_ext = int(r * OUTER_CIRCLE_RADIUS_MM / BLACK_DISK_RADIUS_MM)
            cv2.circle(vis_photo, (int(cx), int(cy)), r_ext, (0, 200, 120), max(1, thick - 1))
            label_photo = f"circ={circ:.2f}  r={int(r)}px  asp={aspect:.2f}"
            cv2.putText(
                vis_photo,
                label_photo,
                (int(cx) - int(r), max(30, int(cy) - int(r) - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, h / 5000),
                (0, 255, 0),
                thick,
            )

            # ── Sur le MASQUE : contour + cercle ─────────────────────────────
            cv2.drawContours(vis_mask, [cnt], -1, (0, 200, 0), 2)
            cv2.circle(vis_mask, (int(cx), int(cy)), int(r), (0, 255, 0), 2)

        else:
            rejected.append(info)

            # ── Sur la PHOTO : cercle fin coloré + raison du rejet ───────────
            cv2.circle(vis_photo, (int(cx), int(cy)), int(r), reason_color, max(1, thick - 1))
            # Petite étiquette raison
            reason_txt = (
                "circ"
                if not in_circ
                else "aire"
                if not in_area
                else "pos"
                if not in_pos
                else "outer"
            )
            cv2.putText(
                vis_photo,
                reason_txt,
                (int(cx) - int(r), max(20, int(cy) - int(r) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.40, h / 7000),
                reason_color,
                max(1, thick - 1),
            )

            # ── Sur le MASQUE : contour coloré fin ───────────────────────────
            cv2.drawContours(vis_mask, [cnt], -1, reason_color, 1)

    # Légende sur la photo (coin bas-gauche)
    legend_lines = [
        "Vert epais  : candidat valide",
        "Bleu/rouge  : circ < 0.60",
        "Cyan        : aspect < 0.80 (non circulaire)",
        "Orange      : hors zone aire",
        "Violet      : hors position",
    ]
    legend_y = h - int(len(legend_lines) * max(24, h // 120)) - 10
    _put_lines(
        vis_photo,
        legend_lines,
        10,
        legend_y,
        scale=max(0.45, h / 7000),
        color=(220, 220, 220),
        thickness=max(1, thick - 1),
    )
    # Même légende sur le masque
    _put_lines(
        vis_mask, legend_lines, 4, vis_mask.shape[0] - 100, scale=0.35, color=(180, 180, 180)
    )

    return vis_photo, vis_mask, candidates


# ── Planche de contact pour une image ─────────────────────────────────────────


def make_debug_sheet(image_path: Path, out_dir: Path) -> dict:
    """
    Crée la planche de contact de debug pour une image.
    Retourne un dict avec le statut et les infos de détection.
    """
    stem = image_path.stem
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERREUR] Impossible de lire : {image_path}")
        return {"stem": stem, "status": "error", "result": None}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    min_area = 0.002 * h * w
    max_area = 0.35 * h * w
    cx_img, cy_img = w / 2.0, h / 2.0
    tol_x, tol_y = w * 0.45, h * 0.45

    # ── 1. Analyser chaque seuil ───────────────────────────────────────────────
    threshold_results = []
    final_result = None
    winning_thresh = None

    for thresh in THRESHOLDS:
        vis_photo, vis_mask, candidates = _analyze_threshold(
            img, blur, thresh, kernel, min_area, max_area, cx_img, cy_img, tol_x, tol_y
        )
        threshold_results.append((thresh, vis_photo, vis_mask, candidates))

        if candidates and final_result is None:
            # Même logique que localize_target : prendre le meilleur (circ × area)
            best = max(candidates, key=lambda c: c["circ"] * c["area"])
            final_result = best
            winning_thresh = thresh

    # ── 2. Construire la colonne "image originale" annotée ─────────────────────
    orig_annot = img.copy()
    status_color_ok = (0, 180, 0)  # vert par défaut  # noqa: F841

    if final_result:
        cx_f = int(final_result["cx"])
        cy_f = int(final_result["cy"])
        r_f = int(final_result["r"])
        mm_per_px = BLACK_DISK_RADIUS_MM / r_f
        r_outer = int(OUTER_CIRCLE_RADIUS_MM / mm_per_px)

        # Disque noir en rouge
        cv2.circle(orig_annot, (cx_f, cy_f), r_f, (0, 0, 255), max(3, h // 300))
        # Cercle extérieur attendu en vert
        cv2.circle(orig_annot, (cx_f, cy_f), r_outer, (0, 255, 0), max(2, h // 400))
        # Centre en croix blanche
        cv2.drawMarker(orig_annot, (cx_f, cy_f), (255, 255, 255), cv2.MARKER_CROSS, 60, 3)

        info_lines = [
            f"TROUVE  seuil={winning_thresh}",
            f"centre ({cx_f}, {cy_f})",
            f"r_disk  = {r_f} px",
            f"mm/px   = {mm_per_px:.4f}",
            f"r_outer = {r_outer} px",
            f"circul. = {final_result['circ']:.3f}",
        ]
        status = "found"
    else:
        info_lines = [
            "NON TROUVE",
            "",
            f"min_area = {int(min_area)}",
            f"max_area = {int(max_area)}",
            f"image    = {w}x{h}",
            "Aucun candidat valide",
        ]
        status = "not_found"

    # Écrire les infos en bas à gauche de l'originale
    _put_lines(
        orig_annot,
        info_lines,
        10,
        h - 200,
        scale=0.7,
        color=(0, 220, 255) if final_result else (0, 80, 255),
        thickness=2,
    )

    # Sauvegarde pleine résolution de l'image annotée (disque détecté)
    annot_path = out_dir / f"{stem}_disk_annot.jpg"
    cv2.imwrite(str(annot_path), orig_annot, [cv2.IMWRITE_JPEG_QUALITY, 92])

    thumb_orig = _thumb(orig_annot, THUMB_H)

    # ── 3. Vignettes par seuil : photo annotée (2/3) + masque (1/3) empilés ──
    PHOTO_H = int(THUMB_H * 0.68)  # hauteur de la vignette photo
    MASK_H = THUMB_H - PHOTO_H  # hauteur de la vignette masque

    thresh_thumbs = []
    for thresh, vis_photo, vis_mask, candidates in threshold_results:
        t_photo = _thumb(vis_photo, PHOTO_H)
        t_mask = _thumb(vis_mask, MASK_H)

        # Harmoniser la largeur (la photo et le masque ont la même largeur d'image
        # source, donc les thumbs ont la même largeur — on prend le max au cas où)
        col_w = max(t_photo.shape[1], t_mask.shape[1])

        def _pad_w(img_col, target_w):
            dw = target_w - img_col.shape[1]
            if dw <= 0:
                return img_col
            pad = np.zeros((img_col.shape[0], dw, 3), dtype=np.uint8)
            return np.hstack([img_col, pad])

        t_photo = _pad_w(t_photo, col_w)
        t_mask = _pad_w(t_mask, col_w)

        # Séparateur horizontal entre photo et masque
        sep = np.full((3, col_w, 3), 80, dtype=np.uint8)

        # Banderole de titre
        n_ok = len(candidates)
        color = (0, 130, 0) if n_ok > 0 else (60, 60, 60)
        title = f"Seuil {thresh}  |  {n_ok} candidat(s)"
        if thresh == winning_thresh:
            color = (0, 160, 0)
            title += "  <- RETENU"
        banner = _banner(title, col_w, color)

        # Étiquette masque
        mask_label = _banner("masque morpho", col_w, (40, 40, 40), h=18)

        col = np.vstack([banner, t_photo, sep, mask_label, t_mask])
        thresh_thumbs.append(col)

    # Banderole pour l'originale
    orig_banner_color = (0, 120, 0) if status == "found" else (0, 0, 160)
    orig_banner_text = f"{stem}  —  {'✓ Disque trouvé' if status == 'found' else '✗ Non trouvé'}"
    orig_banner = _banner(orig_banner_text, thumb_orig.shape[1], orig_banner_color)
    col_orig = np.vstack([orig_banner, thumb_orig])

    # Aligner toutes les colonnes à la même hauteur
    target_h = max(col_orig.shape[0], *(t.shape[0] for t in thresh_thumbs))

    def pad_h(col, th):
        dh = th - col.shape[0]
        if dh <= 0:
            return col
        pad = np.zeros((dh, col.shape[1], 3), dtype=np.uint8)
        return np.vstack([col, pad])

    cols = [pad_h(col_orig, target_h)] + [pad_h(t, target_h) for t in thresh_thumbs]

    # ── 4. Histogramme de gris (bande en bas) ─────────────────────────────────
    hist_w = sum(c.shape[1] for c in cols)
    hist_h = 100
    hist_canvas = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = (hist / hist.max() * (hist_h - 10)).astype(int)
    for i in range(256):
        x = int(i * hist_w / 256)
        bar_h = hist_norm[i]
        cv2.line(hist_canvas, (x, hist_h), (x, hist_h - bar_h), (180, 180, 180), 1)
    # Lignes verticales pour les seuils testés
    for thresh, _, _, candidates in threshold_results:
        xv = int(thresh * hist_w / 256)
        color = (0, 255, 0) if thresh == winning_thresh else (100, 100, 255)
        cv2.line(hist_canvas, (xv, 0), (xv, hist_h), color, 1)
        cv2.putText(
            hist_canvas, str(thresh), (xv + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
        )
    cv2.putText(
        hist_canvas,
        "Histogramme de luminance (gris)  |  lignes = seuils testés",
        (8, hist_h - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 160, 160),
        1,
    )

    # ── 5. Assemblage final ────────────────────────────────────────────────────
    row = np.hstack(cols)
    sheet = np.vstack([row, hist_canvas])

    out_path = out_dir / f"{stem}_disk_debug.jpg"
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  → {out_path.name}  [{status}]")
    print(f"  → {annot_path.name}  [pleine résolution]")

    return {
        "stem": stem,
        "status": status,
        "result": final_result,
        "winning_thresh": winning_thresh,
        "sheet_path": out_path,
        "annot_path": annot_path,
    }


# ── Résumé : toutes les images en une seule image ─────────────────────────────


def make_summary(results: list[dict], out_dir: Path):
    """
    Génère une image de synthèse : une vignette par image avec code couleur.
    """
    CARD_W, CARD_H = 300, 340
    COLS = 4

    rows_needed = (len(results) + COLS - 1) // COLS
    summary = np.full((rows_needed * CARD_H, COLS * CARD_W, 3), 30, dtype=np.uint8)

    for idx, res in enumerate(results):
        col_i = idx % COLS
        row_i = idx // COLS
        x0 = col_i * CARD_W
        y0 = row_i * CARD_H

        # Lire l'image annotée (avec le disque détecté tracé dessus)
        img_path = res.get("annot_path") or res.get("orig_path")
        if img_path and img_path.exists():
            img = cv2.imread(str(img_path))
            thumb = _thumb(img, CARD_H - 60)
            # Centrer la vignette dans la carte
            th, tw = thumb.shape[:2]
            tx = x0 + (CARD_W - tw) // 2
            ty = y0 + 30
            if tx >= 0 and ty >= 0 and tx + tw <= summary.shape[1] and ty + th <= summary.shape[0]:
                summary[ty : ty + th, tx : tx + tw] = thumb

        # Cadre coloré selon le statut
        color = (
            (0, 180, 0)
            if res["status"] == "found"
            else (0, 100, 220)
            if res["status"] == "not_found"
            else (60, 60, 60)
        )
        cv2.rectangle(summary, (x0, y0), (x0 + CARD_W - 1, y0 + CARD_H - 1), color, 3)

        # Texte
        stem_short = res["stem"][-18:]  # tronquer si trop long
        lines = [stem_short]
        if res["status"] == "found" and res["result"]:
            r = res["result"]
            lines += [
                f"✓ seuil={res['winning_thresh']}",
                f"r={int(r['r'])}px  c={r['circ']:.2f}",
            ]
        else:
            lines.append("✗ non trouve")

        _put_lines(summary, lines, x0 + 6, y0 + 16, scale=0.40, color=(255, 255, 255))

    out_path = out_dir / "_summary_disk.jpg"
    cv2.imwrite(str(out_path), summary, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\n[RÉSUMÉ] → {out_path}")
    return out_path


# ── Point d'entrée ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Debug visuel de la détection du disque noir sur une ou plusieurs images."
    )
    ap.add_argument("image", nargs="+", help="Image(s) ou dossier(s) d'images")
    ap.add_argument(
        "--out",
        default="outputs/debug_disk",
        help="Dossier de sortie (défaut : outputs/debug_disk/)",
    )
    ap.add_argument("--show", action="store_true", help="Ouvre le résumé à la fin")
    args = ap.parse_args()

    # Collecter les images (accepte mix de fichiers et dossiers)
    images = []
    for path_str in args.image:
        p = Path(path_str)
        if p.is_dir():
            images.extend(sorted(f for f in p.glob("*.*") if f.suffix.lower() in IMAGE_EXTENSIONS))
        elif p.is_file():
            images.append(p)
        else:
            print(f"[ERREUR] Chemin introuvable : {path_str}")
            sys.exit(1)

    if not images:
        print("[ERREUR] Aucune image trouvée.")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Traitement de {len(images)} image(s) → {out_dir}\n")

    results = []
    n_found = 0
    for img_path in images:
        print(f"[{img_path.name}]")
        res = make_debug_sheet(img_path, out_dir)
        res["orig_path"] = img_path
        results.append(res)
        if res["status"] == "found":
            n_found += 1

    print(f"\n{'=' * 50}")
    print(f"Résultat : {n_found}/{len(images)} disques trouvés")
    print(f"{'=' * 50}")

    for res in results:
        icon = "✓" if res["status"] == "found" else "✗"
        detail = ""
        if res["status"] == "found" and res["result"]:
            r = res["result"]
            detail = f"  seuil={res['winning_thresh']}  r={int(r['r'])}px  circ={r['circ']:.2f}"
        print(f"  {icon}  {res['stem']}{detail}")

    summary_path = make_summary(results, out_dir)

    if args.show:
        import os
        import subprocess

        if sys.platform == "win32":
            os.startfile(str(summary_path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(summary_path)])
        else:
            subprocess.Popen(["xdg-open", str(summary_path)])


if __name__ == "__main__":
    main()
