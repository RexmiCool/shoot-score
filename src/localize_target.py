"""
Pipeline de localisation et de mise à plat d'une cible de tir.

Étape 1 – Localisation par le disque noir central
    → Détecte le grand disque noir (Ø200mm) qui est l'ancre la plus robuste
      à l'environnement (neige, arbres, mur, personnes…).

Étape 2a – Mise à plat par ellipses (sans marqueurs)
    → Sous perspective, les anneaux circulaires de la cible apparaissent
      comme des ellipses. On fit une ellipse sur l'anneau extérieur et on
      calcule l'homographie pour le ramener à un cercle parfait.

Étape 2b – Mise à plat par marqueurs ArUco (optionnel, plus robuste)
    → Si 4 marqueurs ArUco sont collés aux coins du carton, on calcule
      directement l'homographie à partir de leurs coins.

Résultat : image carrée 1200×1200px, cible centrée, corrigée en perspective.

Usage :
    python src/localize_target.py <image_ou_dossier> [--out outputs] [--debug] [--show]
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
from math import pi

import cv2
import numpy as np

# ── Constantes géométriques de la cible ───────────────────────────────────────
BLACK_DISK_RADIUS_MM = 100.0  # rayon du disque noir central (mm)
OUTER_CIRCLE_RADIUS_MM = 250.0  # rayon du cercle extérieur (mm)

# Paramètres de l'image de sortie
OUTPUT_SIZE = 1200  # côté de l'image de sortie (px)
OUTPUT_CENTER = OUTPUT_SIZE // 2
OUTPUT_OUTER_RADIUS = int(OUTPUT_SIZE * 0.46)  # rayon cible dans l'image de sortie

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Utilitaires système ────────────────────────────────────────────────────────


def _open_file(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _collect_images(path_str: str) -> list[Path]:
    p = Path(path_str)
    if p.is_dir():
        return sorted(f for f in p.glob("*.*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    if p.is_file():
        return [p]
    print(f"[ERREUR] Chemin introuvable : {path_str}")
    sys.exit(1)


# ── Étape 1 : Détection du disque noir ────────────────────────────────────────


def detect_black_disk(
    img: np.ndarray,
    debug_dir: Path | None = None,
    stem: str = "",
    hint_cx_norm: float | None = None,
    hint_cy_norm: float | None = None,
):
    """
    Détecte le disque noir central de la cible dans une image BGR pleine résolution.

    Le disque noir est l'ancre la plus fiable : grand (Ø200mm), très sombre,
    circulaire, présent sur toutes les cibles standard. Il reste détectable
    même avec un fond difficile (neige, mur blanc, végétation).

    Stratégie :
      1. Si hint_cx_norm / hint_cy_norm sont fournis (coordonnées normalisées 0-1
         indiquant où se trouvait la croix de visée lors de la prise de vue), on
         cherche en priorité dans une zone serrée (±25 % de la largeur) autour de
         ce point. Cela évite de confondre le disque noir de la cible avec une
         forme sombre sur le bord de l'image (pied de support, ombre, etc.).
      2. Si aucun candidat n'est trouvé dans cette zone (ou si aucun hint n'est
         fourni), on effectue la recherche globale habituelle sur toute l'image.

    Retourne (cx, cy, r_px, mm_per_px) ou None si non trouvé.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    min_area = 0.002 * h * w  # le disque fait au moins 0.2% de l'image
    max_area = 0.35 * h * w  # pas plus de 35%

    def _search(tol_x: float, tol_y: float, ref_cx: float, ref_cy: float):
        """Recherche le meilleur candidat dans la tolérance donnée autour de (ref_cx, ref_cy)."""
        best = None
        for thresh in [40, 55, 70, 85, 100]:
            _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if not (min_area <= area <= max_area):
                    continue
                peri = cv2.arcLength(cnt, True)
                circ = 4 * pi * area / peri**2 if peri > 0 else 0
                if circ < 0.60:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
                if aspect < 0.80:
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                if abs(cx - ref_cx) > tol_x or abs(cy - ref_cy) > tol_y:
                    continue
                candidates.append((circ * area, cnt, float(cx), float(cy), float(r), thresh))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best = candidates[0]
                if debug_dir:
                    cv2.imwrite(str(debug_dir / f"{stem}_01_disk_mask_t{thresh}.jpg"), mask)
                break
        return best

    best_candidate = None

    # ── Passe 1 : zone serrée autour du hint (si fourni) ──────────────────────
    if hint_cx_norm is not None and hint_cy_norm is not None:
        hint_x = hint_cx_norm * w
        hint_y = hint_cy_norm * h
        # Tolérance serrée : ±25 % de la plus petite dimension
        tight = min(w, h) * 0.25
        best_candidate = _search(tight, tight, hint_x, hint_y)
        if best_candidate is not None:
            print(f"[DISQUE] Trouvé dans la zone hint ({hint_cx_norm:.2f}, {hint_cy_norm:.2f})")

    # ── Passe 2 : recherche globale (fallback ou pas de hint) ─────────────────
    if best_candidate is None:
        cx_img, cy_img = w / 2, h / 2
        best_candidate = _search(w * 0.45, h * 0.45, cx_img, cy_img)

    if best_candidate is None:
        return None

    _, _, cx, cy, r, thresh_used = best_candidate
    mm_per_px = BLACK_DISK_RADIUS_MM / r

    # Garde : si r_outer attendu dépasse la taille de l'image, la détection est fausse
    r_outer_expected = OUTER_CIRCLE_RADIUS_MM / mm_per_px
    if r_outer_expected > min(h, w) * 0.95:
        print(
            f"[DISQUE] Fausse détection (r_outer={r_outer_expected:.0f}px > image {min(h, w)}px)."
        )
        return None

    print(
        f"[DISQUE] centre=({int(cx)},{int(cy)})  r={int(r)}px  "
        f"mm/px={mm_per_px:.4f}  diam≈{int(r * 2 * mm_per_px)}mm  (seuil={thresh_used})"
    )
    return int(cx), int(cy), int(r), mm_per_px


# ── Étape 2a : Mise à plat par fit d'ellipse ──────────────────────────────────


def _fit_outer_ellipse(
    img: np.ndarray,
    cx: int,
    cy: int,
    r_disk: int,
    mm_per_px: float,
    debug_dir: Path | None = None,
    stem: str = "",
):
    """
    Tente de fitter une ellipse sur l'anneau extérieur de la cible (Ø500mm).

    Stratégie robuste au fond complexe :
      - On connaît le centre (cx, cy) et mm_per_px via le disque noir.
      - On lance N rayons depuis le centre et on cherche la transition
        blanc→fond (ou lumière→sombre) sur chaque rayon dans la plage de
        distance attendue [0.75 × r_outer, 1.25 × r_outer].
      - On fitte une ellipse sur les points de bord collectés.

    Retourne l'ellipse (center, axes, angle) en convention OpenCV, ou None.
    """
    h, w = img.shape[:2]
    r_outer_approx = OUTER_CIRCLE_RADIUS_MM / mm_per_px
    r_min = r_outer_approx * 0.75
    r_max = r_outer_approx * 1.30

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Lancer N rayons depuis le centre, collecter le point de bord de chacun
    N_RAYS = 72  # tous les 5°
    border_pts = []

    for i in range(N_RAYS):
        angle_rad = 2 * pi * i / N_RAYS
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Échantillonner le rayon entre r_min et r_max
        prev_val = None
        for r in np.linspace(r_min, r_max, 80):
            px = int(cx + r * cos_a)
            py = int(cy + r * sin_a)
            if not (0 <= px < w and 0 <= py < h):
                break
            val = int(blur[py, px])
            if prev_val is not None:
                # Transition sombre → clair ou clair → sombre : bord de la cible
                if abs(val - prev_val) > 25:
                    border_pts.append([px, py])
                    break
            prev_val = val

    if len(border_pts) < 12:
        print(f"[ELLIPSE] Seulement {len(border_pts)} points de bord — skip.")
        return None

    pts_arr = np.array(border_pts, dtype=np.float32)

    if debug_dir:
        dbg = img.copy()
        for p in border_pts:
            cv2.circle(dbg, (p[0], p[1]), 8, (0, 255, 0), -1)
        cv2.imwrite(str(debug_dir / f"{stem}_02_border_pts.jpg"), dbg)

    ellipse = cv2.fitEllipse(pts_arr.reshape(-1, 1, 2).astype(np.float32))
    (ex, ey), (ma, Mi), angle = ellipse
    ecc = 1 - min(ma, Mi) / max(ma, Mi) if max(ma, Mi) > 0 else 0

    print(
        f"[ELLIPSE] centre=({ex:.1f},{ey:.1f})  axes=({ma:.1f},{Mi:.1f})  "
        f"angle={angle:.1f}°  excentricité={ecc:.3f}  ({len(border_pts)} pts)"
    )
    return ellipse


def _homography_from_ellipse(ellipse, output_size: int, output_outer_radius: int):
    """
    Calcule l'homographie qui transforme l'ellipse détectée en un cercle parfait
    centré dans l'image de sortie.

    L'ellipse représente le bord extérieur vu en perspective. La correction
    revient à trouver la transformation affine qui ramène cette ellipse
    à un cercle de rayon `output_outer_radius` centré en (output_size/2, output_size/2).

    Pour une ellipse (centre, (a,b), θ) :
      - On construit la matrice de l'ellipse dans un repère centré.
      - On en déduit la transformation affine qui la ramène au cercle.
    """
    (ex, ey), (ma, Mi), angle = ellipse
    a = max(ma, Mi) / 2.0  # demi grand axe (px)
    b = min(ma, Mi) / 2.0  # demi petit axe (px)
    theta = np.deg2rad(angle)

    # Matrice de rotation de l'ellipse
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Transformation qui ramène l'ellipse au cercle unité :
    # T_circle = R^T · diag(1/a, 1/b) · R
    S = np.diag([1.0 / a, 1.0 / b])
    T_norm = R.T @ S @ R

    # On veut que le cercle résultant ait rayon output_outer_radius
    scale = float(output_outer_radius)
    T_scale = T_norm * scale

    # Construire la matrice affine 3x3
    cx_out = output_size / 2.0
    cy_out = output_size / 2.0

    # Point source → destination : p_dst = T_scale · (p_src - ellipse_center) + output_center
    M_affine = np.eye(3, dtype=np.float64)
    M_affine[:2, :2] = T_scale
    # Translation : d = output_center - T_scale · ellipse_center
    tx = cx_out - (T_scale[0, 0] * ex + T_scale[0, 1] * ey)
    ty = cy_out - (T_scale[1, 0] * ex + T_scale[1, 1] * ey)
    M_affine[0, 2] = tx
    M_affine[1, 2] = ty

    return M_affine[:2]  # retourne la matrice affine 2×3 pour warpAffine


# ── Étape 2b : Mise à plat par ArUco ──────────────────────────────────────────


def _homography_from_aruco(
    img: np.ndarray, aruco_length_mm: float = 40.0, debug_dir: Path | None = None, stem: str = ""
):
    """
    Détecte 4 marqueurs ArUco (DICT_4X4_50) supposés placés aux 4 coins
    du carton de la cible et calcule l'homographie pour la mise à plat.

    Retourne (M_3x3, dst_size) ou None si moins de 4 marqueurs trouvés.

    Disposition attendue des marqueurs :
        ID 0 = coin haut-gauche
        ID 1 = coin haut-droit
        ID 2 = coin bas-droit
        ID 3 = coin bas-gauche
    """
    if not hasattr(cv2, "aruco"):
        return None

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)

    if ids is None or len(ids) < 4:
        n = 0 if ids is None else len(ids)
        print(f"[ARUCO] {n}/4 marqueurs détectés — fallback ellipse.")
        return None

    # Construire un dict id → coin intérieur du marqueur
    id_to_corner = {}
    for i, mid in enumerate(ids.flatten()):
        if mid in (0, 1, 2, 3):
            c = corners[i][0]  # 4 coins du marqueur (sens horaire depuis haut-gauche)
            id_to_corner[mid] = c

    if len(id_to_corner) < 4:
        print(f"[ARUCO] IDs manquants parmi 0-3 : {set(id_to_corner.keys())} — fallback ellipse.")
        return None

    # Coin intérieur de chaque marqueur (le plus proche du centre de la cible)
    src_pts = np.array(
        [
            id_to_corner[0][2],  # HG : coin bas-droit du marqueur 0
            id_to_corner[1][3],  # HD : coin bas-gauche du marqueur 1
            id_to_corner[2][0],  # BD : coin haut-gauche du marqueur 2
            id_to_corner[3][1],  # BG : coin haut-droit du marqueur 3
        ],
        dtype=np.float32,
    )

    side_px = float(OUTPUT_SIZE)
    dst_pts = np.array(
        [
            [0, 0],
            [side_px, 0],
            [side_px, side_px],
            [0, side_px],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(src_pts, dst_pts)

    if debug_dir:
        dbg = img.copy()
        cv2.aruco.drawDetectedMarkers(dbg, corners, ids)
        cv2.imwrite(str(debug_dir / f"{stem}_02_aruco.jpg"), dbg)

    print("[ARUCO] 4 marqueurs trouvés → homographie calculée.")
    return H


# ── Pipeline principale ────────────────────────────────────────────────────────


def localize_and_flatten(
    image_path,
    output_dir: str = "outputs",
    debug: bool = False,
    show: bool = False,
) -> Path | None:
    """
    Pipeline complète : photo brute → image cible mise à plat.

    1. Détecte le disque noir → centre et échelle approx.
    2a. Essaie les marqueurs ArUco (si présents).
    2b. Sinon, fitte une ellipse sur l'anneau extérieur.
    3. Applique la transformation → image OUTPUT_SIZE × OUTPUT_SIZE.

    Sauvegarde l'image mise à plat dans output_dir/<stem>/<stem>_flat.jpg.
    Retourne le chemin de l'image résultante, ou None en cas d'échec.
    """
    stem = Path(image_path).stem
    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement ─────────────────────────────────────────────────────────────
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERREUR] Impossible de lire : {image_path}")
        return None
    h0, w0 = img.shape[:2]
    print(f"\n[IMAGE] {Path(image_path).name}  {w0}×{h0}px")

    # ── Étape 1 : Disque noir ──────────────────────────────────────────────────
    disk = detect_black_disk(img, debug_dir=out_dir if debug else None, stem=stem)
    if disk is None:
        print("[ERREUR] Disque noir non trouvé. Vérifiez que la cible est visible.")
        return None
    cx, cy, r_disk, mm_per_px = disk

    # Image de debug : annoter le disque détecté sur l'originale
    if debug:
        dbg_disk = img.copy()
        cv2.circle(dbg_disk, (cx, cy), r_disk, (0, 0, 255), max(3, h0 // 300))
        cv2.drawMarker(dbg_disk, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 60, 3)
        r_outer_approx = int(OUTER_CIRCLE_RADIUS_MM / mm_per_px)
        cv2.circle(dbg_disk, (cx, cy), r_outer_approx, (0, 255, 0), max(2, h0 // 400))
        cv2.imwrite(str(out_dir / f"{stem}_01_disk_detected.jpg"), dbg_disk)

    # ── Étape 2 : Transformation vers image à plat ─────────────────────────────
    flat = None
    method_used = "?"

    # 2a — ArUco
    H_aruco = _homography_from_aruco(img, debug_dir=out_dir if debug else None, stem=stem)
    if H_aruco is not None:
        flat = cv2.warpPerspective(
            img,
            H_aruco,
            (OUTPUT_SIZE, OUTPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(200, 200, 200),
        )
        method_used = "ArUco"

    # 2b — Ellipse
    if flat is None:
        ellipse = _fit_outer_ellipse(
            img,
            cx,
            cy,
            r_disk,
            mm_per_px,
            debug_dir=out_dir if debug else None,
            stem=stem,
        )
        if ellipse is not None:
            M_affine = _homography_from_ellipse(ellipse, OUTPUT_SIZE, OUTPUT_OUTER_RADIUS)
            flat = cv2.warpAffine(
                img,
                M_affine,
                (OUTPUT_SIZE, OUTPUT_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(200, 200, 200),
            )
            method_used = "ellipse"

    # 2c — Fallback : recadrage simple centré sur le disque noir (sans correction perspective)
    if flat is None:
        print("[WARN] Fallback : pas de correction de perspective, simple recadrage.")
        r_outer = int(OUTER_CIRCLE_RADIUS_MM / mm_per_px)
        margin = int(r_outer * 0.10)
        half = r_outer + margin
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w0, cx + half)
        y2 = min(h0, cy + half)
        crop = img[y1:y2, x1:x2]
        flat = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        method_used = "fallback_crop"

    print(f"[FLAT]  méthode={method_used}  taille={flat.shape[1]}×{flat.shape[0]}px")

    # ── Annotation de contrôle : superposer la grille de la cible ─────────────
    mm_per_px_out = OUTER_CIRCLE_RADIUS_MM / OUTPUT_OUTER_RADIUS
    r_disk_out = int(BLACK_DISK_RADIUS_MM / mm_per_px_out)
    r_outer_out = OUTPUT_OUTER_RADIUS

    annot = flat.copy()
    thick = max(2, OUTPUT_SIZE // 400)
    # Disque noir
    cv2.circle(annot, (OUTPUT_CENTER, OUTPUT_CENTER), r_disk_out, (0, 0, 255), thick)
    # Anneaux intermédiaires (5 anneaux entre Ø200mm et Ø500mm)
    n_rings = 5
    step_mm = (OUTER_CIRCLE_RADIUS_MM - BLACK_DISK_RADIUS_MM) / (n_rings + 1)
    for i in range(1, n_rings + 1):
        r_mm = BLACK_DISK_RADIUS_MM + i * step_mm
        cv2.circle(
            annot, (OUTPUT_CENTER, OUTPUT_CENTER), int(r_mm / mm_per_px_out), (0, 165, 255), thick
        )
    # Cercle extérieur
    cv2.circle(annot, (OUTPUT_CENTER, OUTPUT_CENTER), r_outer_out, (0, 255, 0), thick)
    # Centre (croix bleue)
    cv2.drawMarker(annot, (OUTPUT_CENTER, OUTPUT_CENTER), (255, 0, 0), cv2.MARKER_CROSS, 40, 2)

    # ── Sauvegarde ─────────────────────────────────────────────────────────────
    flat_path = out_dir / f"{stem}_flat.jpg"
    annot_path = out_dir / f"{stem}_flat_annot.jpg"
    cv2.imwrite(str(flat_path), flat)
    cv2.imwrite(str(annot_path), annot)
    print(f"[OK]    → {flat_path.name}")
    print(f"[OK]    → {annot_path.name}")

    if show:
        _open_file(annot_path)

    return flat_path


# ── Point d'entrée CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Localise et met à plat une cible de tir sportif.")
    ap.add_argument("image", help="Chemin vers une image (ou un dossier d'images)")
    ap.add_argument("--out", default="outputs", help="Dossier de sortie (défaut : outputs/)")
    ap.add_argument(
        "--debug", action="store_true", help="Sauvegarde les images intermédiaires de chaque étape"
    )
    ap.add_argument(
        "--show", action="store_true", help="Ouvre l'image annotée avec l'application par défaut"
    )
    args = ap.parse_args()

    for img_path in _collect_images(args.image):
        localize_and_flatten(img_path, args.out, debug=args.debug, show=args.show)
