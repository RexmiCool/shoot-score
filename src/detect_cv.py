import argparse
import json
import os
import subprocess
import sys
from math import pi
from pathlib import Path

import cv2
import numpy as np


def crop_to_target(
    image_path, output_dir="outputs", sheet_size_mm=520.0, outer_ring_diam_mm=455.0, debug=False
):
    """
    Recadre l'image pour ne montrer que la cible.
    Exploite le contraste fort entre la feuille blanche et le fond sombre.
    Effectue une correction de perspective pour obtenir un carré.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 1) Isoler la feuille blanche sur fond sombre ────────────────────────────
    # Flou fort pour lisser les détails internes (anneaux, chiffres, disque noir)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # Seuil adaptatif selon luminosité du fond (bord haut de l'image)
    bg_lum = float(np.median(blur[:min(100, h//10), :]))
    threshold = max(bg_lum + 30, 80)
    _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    # Fermeture morphologique pour boucher les trous (disque noir central, anneaux)
    k_size = max(3, int(min(h, w) * 0.03) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=6)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

    if debug:
        cv2.imwrite(str(out_dir / (Path(image_path).stem + "_dbg_mask.jpg")), mask)
        print(f"[CROP] bg_lum={bg_lum:.1f} → seuil={threshold:.1f}")

    # ── 2) Trouver le plus grand contour → la feuille ───────────────────────────
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("[CROP] Aucun contour trouvé — image originale conservée.")
        out_img = out_dir / (Path(image_path).stem + "_cropped.jpg")
        cv2.imwrite(str(out_img), img)
        return str(out_img)

    sheet_cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(sheet_cnt, True)

    # Approximer en quadrilatère
    quad = None
    for eps in [0.01, 0.02, 0.03, 0.05, 0.08]:
        approx = cv2.approxPolyDP(sheet_cnt, eps * peri, True)
        if len(approx) == 4:
            quad = approx
            print(f"[CROP] Quadrilatère trouvé (eps={eps}) : {approx.reshape(4,2).tolist()}")
            break

    ordered = None
    if quad is not None:
        # ── 3) Correction de perspective ───────────────────────────────────────
        pts = quad.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1); d = np.diff(pts, axis=1)
        ordered = np.array([
            pts[np.argmin(s)],   # haut-gauche
            pts[np.argmin(d)],   # haut-droit
            pts[np.argmax(s)],   # bas-droit
            pts[np.argmax(d)],   # bas-gauche
        ], dtype=np.float32)
        side = 1040
        dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        cropped = cv2.warpPerspective(img, M, (side, side))
    else:
        print("[CROP] Quadrilatère non trouvé, utilisation du bounding rect.")
        x, y, bw, bh = cv2.boundingRect(sheet_cnt)
        cropped = img[y:y+bh, x:x+bw]
        cropped = cv2.resize(cropped, (1040, 1040))

    if debug:
        dbg = img.copy()
        if quad is not None:
            cv2.drawContours(dbg, [quad], -1, (0, 0, 255), max(4, h // 500))
            for p in ordered.astype(int):
                cv2.circle(dbg, tuple(p), 20, (0, 255, 0), -1)
        cv2.imwrite(str(out_dir / (Path(image_path).stem + "_dbg_contour.jpg")), dbg)

    out_img = out_dir / (Path(image_path).stem + "_cropped.jpg")
    cv2.imwrite(str(out_img), cropped)
    print(f"[CROP] Image recadrée sauvegardée -> {out_img.name}")
    return str(out_img)


def detect_target_circles(
    bgr,
    dp=1.2,
    min_dist_frac=0.03,
    param1=100,
    param2=40,
    min_radius_frac=0.03,
    max_radius_frac=0.55,
    center_tol_frac=0.05,
):
    """Détecte les cercles concentriques de la cible dans une image BGR déjà chargée.
    - Seuls les cercles dont le centre est proche du centre dominant sont conservés.
    - Retourne une liste de dicts {x, y, r} triée par rayon décroissant.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    min_dist = int(min(h, w) * min_dist_frac)
    min_radius = int(min(h, w) * min_radius_frac)
    max_radius = int(min(h, w) * max_radius_frac)
    center_tol = min(h, w) * center_tol_frac

    raw = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if raw is None:
        print("[CERCLES] Aucun cercle détecté")
        return []

    raw = np.around(raw[0]).astype(int)  # shape (N, 3) : x, y, r

    # Trouver le centre dominant par clustering des centres
    centers = raw[:, :2].astype(float)
    # Utiliser la médiane pondérée par le rayon comme estimateur du centre dominant
    weights = raw[:, 2].astype(float)
    cx_dom = float(np.average(centers[:, 0], weights=weights))
    cy_dom = float(np.average(centers[:, 1], weights=weights))

    # Filtrer : garder uniquement les cercles dont le centre est proche du centre dominant
    circles_list = []
    for x, y, r in raw:
        if np.hypot(x - cx_dom, y - cy_dom) <= center_tol:
            circles_list.append({"x": int(x), "y": int(y), "r": int(r)})

    circles_list.sort(key=lambda c: c["r"], reverse=True)
    print(
        f"[CERCLES] {len(circles_list)} cercles concentriques détectés "
        f"(centre dominant : {int(cx_dom)}, {int(cy_dom)})"
    )
    return circles_list


def circularity(contour):
    area = cv2.contourArea(contour)
    per = cv2.arcLength(contour, True)
    return 0 if per == 0 else 4 * pi * area / (per * per)


def solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    ha = cv2.contourArea(hull)
    return 0 if ha == 0 else area / ha


def estimate_mm_per_px_with_aruco(bgr, aruco_length_mm=40.0):
    if not hasattr(cv2, "aruco"):
        return None
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(bgr)
    if ids is None or len(corners) == 0:
        return None
    side_px = []
    for c in corners:
        pts = c[0]
        d = 0
        for i in range(4):
            d += np.linalg.norm(pts[i] - pts[(i + 1) % 4])
        side_px.append(d / 4.0)
    mean_side_px = float(np.mean(side_px))
    return aruco_length_mm / mean_side_px  # -> mm/px


def area_px_for_diam_mm(d_mm, mm_per_px):
    r_px = (d_mm / 2.0) / mm_per_px
    return pi * (r_px**2)


def detect_impacts(
    image_path,
    output_dir="outputs",
    target_width_mm=500.0,
    block_size=31,
    c_val=7,
    open_iter=1,
    close_iter=2,
    min_d_mm=8.0,
    max_d_mm=14.0,
    min_circ=0.55,
    min_solid=0.80,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    orig = img.copy()

    # 0) Détection des cercles de la cible
    target_circles = detect_target_circles(img)

    # 1) mm/px via ArUco (sinon via le plus grand cercle détecté, sinon fallback largeur cible)
    mm_per_px = estimate_mm_per_px_with_aruco(img)
    if mm_per_px is None:
        if target_circles:
            # Le plus grand cercle correspond au bord extérieur de la cible (diam=520mm)
            biggest_r_px = target_circles[0]["r"]
            mm_per_px = 520.0 / (2.0 * biggest_r_px)
            print(f"[SCALE] mm/px estimé via cercle cible : {mm_per_px:.4f}")
        else:
            mm_per_px = target_width_mm / w
            print(f"[SCALE] mm/px estimé via largeur image : {mm_per_px:.4f}")

    # 2) Gris + normalisation + lissage
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Seuillage adaptatif (inversé : trous en blanc)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size | 1, c_val
    )

    # 4) Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if open_iter > 0:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    # 5) Contours & filtres géométriques
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = area_px_for_diam_mm(min_d_mm, mm_per_px)
    max_area = area_px_for_diam_mm(max_d_mm, mm_per_px)

    impacts = []
    annot = orig.copy()
    idx = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        if circularity(cnt) < min_circ:
            continue
        if solidity(cnt) < min_solid:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        cv2.circle(annot, (int(cx), int(cy)), int(max(3, radius)), (0, 255, 0), 2)
        cv2.putText(
            annot,
            str(idx),
            (int(cx) + 6, int(cy) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        impacts.append(
            {
                "id": idx,
                "cx_px": float(cx),
                "cy_px": float(cy),
                "radius_px": float(radius),
                "area_px": float(area),
            }
        )
        idx += 1

    # Dessiner les cercles de la cible sur l'image annotée (rouge, épaisseur 2)
    for c in target_circles:
        cv2.circle(annot, (c["x"], c["y"]), c["r"], (0, 0, 255), 2)
    # Dessiner le centre de la cible (croix bleue) si des cercles ont été détectés
    if target_circles:
        cx0, cy0 = target_circles[0]["x"], target_circles[0]["y"]
        cv2.drawMarker(annot, (cx0, cy0), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)

    out_img = out_dir / (Path(image_path).stem + "_annot.jpg")
    out_json = out_dir / (Path(image_path).stem + "_impacts.json")
    cv2.imwrite(str(out_img), annot)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"mm_per_px": mm_per_px, "impacts": impacts}, f, ensure_ascii=False, indent=2)

    print(f"[OK] {len(impacts)} impacts détectés | mm/px={mm_per_px:.4f} | -> {out_img.name}")
    return str(out_img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str)
    ap.add_argument("--out", type=str, default="outputs")
    ap.add_argument(
        "--sheet_size_mm",
        type=float,
        default=520.0,
        help="Taille de la feuille cible (mm), défaut 520",
    )
    ap.add_argument(
        "--outer_ring_diam_mm",
        type=float,
        default=455.0,
        help="Diamètre de l'anneau extérieur de la cible (mm), défaut 455",
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out_img = crop_to_target(
        args.image,
        args.out,
        sheet_size_mm=args.sheet_size_mm,
        outer_ring_diam_mm=args.outer_ring_diam_mm,
        debug=args.debug,
    )
    if out_img:
        if sys.platform == "win32":
            os.startfile(out_img)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", out_img])
        else:
            subprocess.Popen(["xdg-open", out_img])
