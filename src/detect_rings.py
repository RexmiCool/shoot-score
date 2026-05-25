"""
Étape 3 : détection et identification des anneaux concentriques de la cible.

Part de l'image mise à plat par flatten_target.py.
Le disque noir est centré en (OUTPUT_CENTER, OUTPUT_CENTER).

Algorithme :
    1. Profil radial de gradient : pour chaque rayon r depuis le centre,
         on accumule l'énergie de bord (Canny) sur les pixels à distance r.
         Les pics du profil = bords des anneaux.
    2. Matching : chaque pic est assigné à l'anneau théorique le plus proche.
    3. Étalonnage linéaire px→mm par moindres carrés sur les paires matchées
         (sans ordonnée à l'origine car le centre est connu).
    4. Inférence : les anneaux non détectés sont calculés depuis l'étalonnage.

Géométrie de la cible (KNOWN_RINGS_MM) :
    100mm = bord du disque noir
    250mm = bord extérieur de la zone de score (1pt)
    Anneaux intermédiaires : 125, 150, 175, 200, 225mm

Usage :
        python src/detect_rings.py <flat.jpg ou dossier> [--out outputs] [--debug] [--show]

Le script accepte indifféremment :
    - un fichier *_flat.jpg
    - un dossier contenant des fichiers *_flat.jpg
Étape 3 : détection et identification des anneaux concentriques de la cible.

Part de l'image mise à plat par flatten_target.py.
Le disque noir est centré en (OUTPUT_CENTER, OUTPUT_CENTER).

Algorithme :
  1. Profil radial de gradient : pour chaque rayon r depuis le centre,
     on accumule l'énergie de bord (Canny) sur les pixels à distance r.
     Les pics du profil = bords des anneaux.
  2. Matching : chaque pic est assigné à l'anneau théorique le plus proche.
  3. Étalonnage linéaire px→mm par moindres carrés sur les paires matchées
     (sans ordonnée à l'origine car le centre est connu).
  4. Inférence : les anneaux non détectés sont calculés depuis l'étalonnage.

Géométrie de la cible (KNOWN_RINGS_MM) :
  100mm = bord du disque noir
  250mm = bord extérieur de la zone de score (1pt)
  Anneaux intermédiaires : 125, 150, 175, 200, 225mm

Usage :
    python src/detect_rings.py <flat.jpg ou dossier> [--out outputs] [--debug] [--show]

Le script accepte indifféremment :
  - un fichier *_flat.jpg
  - un dossier contenant des fichiers *_flat.jpg
"""

import argparse
import json
import sys
from math import pi
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from localize_target import _open_file
from flatten_target import OUTPUT_CENTER, MM_PER_PX_OUT


# ── Géométrie connue de la cible ───────────────────────────────────────────────
# Rayons théoriques des anneaux (bords de zones) en mm depuis le centre.
# Modifie cette liste si ta cible a une géométrie différente.
KNOWN_RINGS_MM: list[float] = [100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0]

# Anneaux à l'intérieur du disque noir (zones 7→10), pas détectables par Canny.
# 100mm = bord du disque (limite 6/7) ; 75mm = 7/8 ; 50mm = 8/9 ; 25mm = 9/10.
INNER_RINGS_MM: list[float] = [75.0, 50.0, 25.0]
# Score de la zone intérieure à chaque anneau (la zone la plus proche du centre).
INNER_RINGS_SCORE: dict[float, int] = {75.0: 8, 50.0: 9, 25.0: 10}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Helpers visuels ────────────────────────────────────────────────────────────


def _draw_dashed_circle(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple,
    thickness: int = 2,
    n_dashes: int = 40,
) -> None:
    """Trace un cercle en pointillés (pour les anneaux inférés)."""
    for i in range(0, n_dashes, 2):
        a1 = 2 * pi * i / n_dashes
        a2 = 2 * pi * (i + 1) / n_dashes
        p1 = (int(center[0] + radius * np.cos(a1)), int(center[1] + radius * np.sin(a1)))
        p2 = (int(center[0] + radius * np.cos(a2)), int(center[1] + radius * np.sin(a2)))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_dashed_ellipse(
    img: np.ndarray,
    ellipse: tuple,
    color: tuple,
    thickness: int = 2,
    n_dashes: int = 48,
) -> None:
    """Trace une ellipse en pointillés (pour les anneaux inférés).

    ellipse : format OpenCV → ((cx, cy), (ea, eb), angle_deg)
    Les traits et les espaces alternent par paires d'incréments angulaires.
    """
    (ecx, ecy), (ea, eb), eangle = ellipse
    a = ea / 2  # demi-axe 1
    b = eb / 2  # demi-axe 2
    theta = np.deg2rad(eangle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(0, n_dashes, 2):
        t1 = 2 * pi * i / n_dashes
        t2 = 2 * pi * (i + 1) / n_dashes
        x1 = ecx + a * np.cos(t1) * cos_t - b * np.sin(t1) * sin_t
        y1 = ecy + a * np.cos(t1) * sin_t + b * np.sin(t1) * cos_t
        x2 = ecx + a * np.cos(t2) * cos_t - b * np.sin(t2) * sin_t
        y2 = ecy + a * np.cos(t2) * sin_t + b * np.sin(t2) * cos_t
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)


def _save_profile_image(
    profile: np.ndarray,
    peaks_rel: list[int],
    r_min: int,
    mm_per_px: float,
    matches: dict,
    out_path: Path,
    pw: int = 900,
    ph: int = 220,
) -> None:
    """Sauvegarde une visualisation du profil radial avec pics et anneaux connus."""
    canvas = np.full((ph, pw, 3), 18, dtype=np.uint8)
    n = len(profile)
    if n == 0:
        return

    pmax = float(profile.max()) or 1.0
    sig = (profile / pmax * (ph - 50)).astype(int)

    # Courbe du profil (gris)
    for i in range(n - 1):
        x1 = int(i * pw / n)
        x2 = int((i + 1) * pw / n)
        cv2.line(canvas, (x1, ph - sig[i] - 1), (x2, ph - sig[i + 1] - 1), (160, 160, 160), 1)

    # Anneaux connus (bleu vertical)
    for r_mm in KNOWN_RINGS_MM:
        r_px_rel = r_mm / mm_per_px - r_min
        if 0 <= r_px_rel < n:
            xk = int(r_px_rel * pw / n)
            cv2.line(canvas, (xk, ph - 20), (xk, ph - 5), (200, 120, 0), 1)
            cv2.putText(
                canvas,
                f"{int(r_mm)}",
                (xk - 8, ph - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (200, 120, 0),
                1,
            )

    # Pics détectés (vert si matché, rouge sinon)
    matched_px_set = set(int(round(v)) for v in matches.values())
    for p_rel in peaks_rel:
        r_px_abs = p_rel + r_min
        xp = int(p_rel * pw / n)
        r_mm_est = r_px_abs * mm_per_px
        matched = r_px_abs in matched_px_set
        color = (0, 220, 0) if matched else (0, 80, 220)
        cv2.line(canvas, (xp, 0), (xp, ph - 30), color, 1)
        cv2.putText(
            canvas, f"{r_mm_est:.0f}mm", (xp + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1
        )

    cv2.putText(
        canvas,
        "Profil radial  |  Vert=detecte matche  Bleu=detecte non matche  Orange=anneau connu",
        (8, ph - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (120, 120, 120),
        1,
    )

    cv2.imwrite(str(out_path), canvas)


# ── Étape 3-A : profil radial de gradient ─────────────────────────────────────


def radial_edge_profile(
    gray: np.ndarray,
    cx: int,
    cy: int,
    r_min: int,
    r_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule pour chaque rayon r ∈ [r_min, r_max] la densité moyenne de bord
    (Canny) sur le cercle de rayon r centré en (cx, cy).

    Retourne :
      - profile[0..r_max-r_min] : densité normalisée de bord par rayon
      - edges                   : image Canny utilisée (pour debug)
    """
    h, w = gray.shape

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 60)

    # Distance entière de chaque pixel au centre
    y_g, x_g = np.mgrid[0:h, 0:w]
    dist_f = np.sqrt((x_g.astype(np.float32) - cx) ** 2 + (y_g.astype(np.float32) - cy) ** 2)
    dist_int = np.round(dist_f).astype(np.int32)

    r_total = int(dist_int.max()) + 1
    edge_sum = np.zeros(r_total, dtype=np.float64)
    pixel_cnt = np.zeros(r_total, dtype=np.int64)

    # Accumulation vectorisée par rayon
    np.add.at(edge_sum, dist_int.ravel(), edges.ravel().astype(np.float64))
    np.add.at(pixel_cnt, dist_int.ravel(), 1)

    # Normaliser par le nombre de pixels à chaque rayon (≈ 2πr)
    profile = np.zeros(r_total, dtype=np.float64)
    nonzero = pixel_cnt > 0
    profile[nonzero] = edge_sum[nonzero] / pixel_cnt[nonzero]

    r_max_clamp = min(r_max, r_total - 1)
    return profile[r_min : r_max_clamp + 1].copy(), edges


# ── Étape 3-B : détection des pics ────────────────────────────────────────────


def find_peaks(
    signal: np.ndarray,
    min_separation: int = 8,
    rel_threshold: float = 0.07,
) -> list[int]:
    """
    Trouve les maxima locaux dans un signal 1D.

    - min_separation : distance minimale entre deux pics (indices).
    - rel_threshold  : seuil relatif au maximum global.

    Retourne la liste des indices dans le référentiel du signal (0 = r_min).
    """
    if signal.size == 0 or signal.max() == 0:
        return []

    thresh = float(signal.max()) * rel_threshold
    peaks = []
    n = len(signal)
    i = 0

    while i < n:
        if signal[i] > thresh:
            j = i
            while j < n and signal[j] > thresh:
                j += 1
            peak_i = int(np.argmax(signal[i:j])) + i
            if not peaks or peak_i - peaks[-1] >= min_separation:
                peaks.append(peak_i)
            i = j
        else:
            i += 1

    return peaks


# ── Étape 3-C : matching et étalonnage ────────────────────────────────────────


def match_and_calibrate(
    detected_px: list[float],
    known_mm: list[float],
    mm_per_px_init: float,
    max_error_mm: float = 15.0,
) -> tuple[dict[float, float], float]:
    """
    Assigne chaque pic détecté (px) à l'anneau théorique le plus proche (mm).
    Étalonnage : mm = k × px  (k = mm_per_px calibré, sans ordonnée car
    le centre est connu et placé à l'origine).

    Retourne :
      - matches        : {r_mm_connu: r_px_detecte} pour les paires matchées
      - mm_per_px_cal  : facteur d'étalonnage calibré
    """
    if not detected_px:
        return {}, mm_per_px_init

    matches = {}
    used_idx = set()

    for r_mm in known_mm:
        r_px_expected = r_mm / mm_per_px_init
        best_err = float("inf")
        best_item = None

        for i, r_px in enumerate(detected_px):
            if i in used_idx:
                continue
            err_mm = abs(r_px * mm_per_px_init - r_mm)
            if err_mm < best_err and err_mm <= max_error_mm:
                best_err = err_mm
                best_item = (i, r_px)

        if best_item is not None:
            matches[r_mm] = best_item[1]
            used_idx.add(best_item[0])

    # Étalonnage : k = argmin Σ(mm_i - k·px_i)²  → k = Σ(mm·px) / Σ(px²)
    if len(matches) >= 2:
        px_arr = np.array(list(matches.values()), dtype=np.float64)
        mm_arr = np.array(list(matches.keys()), dtype=np.float64)
        mm_per_px_cal = float(np.dot(mm_arr, px_arr) / np.dot(px_arr, px_arr))
    else:
        mm_per_px_cal = mm_per_px_init

    return matches, mm_per_px_cal


# ── Étape 3-D : ajustement d'ellipse sur un anneau ────────────────────────────


def fit_ring_ellipse(
    edges: np.ndarray,
    cx: float,
    cy: float,
    r_px: float,
    band_px: int = 12,
) -> tuple | None:
    """
    Extrait les pixels de bord (Canny) dans la bande annulaire [r_px ± band_px]
    et ajuste une ellipse OpenCV sur ces points.

    Retourne ((cx, cy), (ea, eb), angle) ou None si l'ajustement échoue /
    est incohérent.

    Critères de validité :
      - Au moins 20 points dans la bande.
      - Le centre de l'ellipse reste proche du centre attendu (±r_px/2).
      - Le demi-grand axe est dans [0.60·r_px, 1.40·r_px].
    """
    h, w = edges.shape
    ys_g, xs_g = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs_g - cx) ** 2 + (ys_g - cy) ** 2)

    mask = (dist >= r_px - band_px) & (dist <= r_px + band_px) & (edges > 0)
    ys_pts, xs_pts = np.where(mask)

    if len(xs_pts) < 20:
        return None

    pts = np.column_stack([xs_pts, ys_pts]).reshape(-1, 1, 2).astype(np.float32)

    try:
        ellipse = cv2.fitEllipse(pts)
    except cv2.error:
        return None

    (ecx, ecy), (ea, eb), _ = ellipse
    # Rejeter si le centre dévie trop
    if abs(ecx - cx) > r_px * 0.5 or abs(ecy - cy) > r_px * 0.5:
        return None
    # Rejeter si le grand axe est incohérent
    half_major = max(ea, eb) / 2
    if not (r_px * 0.60 < half_major < r_px * 1.40):
        return None

    return ellipse


# ── Pipeline principale ────────────────────────────────────────────────────────


def detect_rings(
    flat_img_path,
    output_dir: str | None = None,
    debug: bool = False,
    show: bool = False,
) -> Path | None:
    """
    Détecte les anneaux concentriques sur une image mise à plat.

    output_dir : si None, les résultats sont déposés dans le même dossier
                 que le fichier flat.jpg (comportement par défaut).
    """
    flat_path = Path(flat_img_path)
    if not flat_path.exists():
        print(f"[ERREUR] Fichier introuvable : {flat_path}")
        return None

    # Déduire le stem de l'image originale
    stem_flat = flat_path.stem  # "20250209_110103_flat"
    stem = stem_flat.removesuffix("_flat")  # "20250209_110103"

    out_dir = Path(output_dir) / stem if output_dir else flat_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement ─────────────────────────────────────────────────────────────
    img = cv2.imread(str(flat_path))
    if img is None:
        print(f"[ERREUR] Impossible de lire : {flat_path}")
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx, cy = OUTPUT_CENTER, OUTPUT_CENTER
    mm_per_px_0 = MM_PER_PX_OUT

    print(f"\n{'=' * 60}")
    print(f"[IMAGE] {flat_path.name}  {w}x{h}px")
    print(f"[INIT ] centre=({cx},{cy})  mm/px={mm_per_px_0:.4f}")

    # ── Plage de recherche ─────────────────────────────────────────────────────
    r_min_mm = KNOWN_RINGS_MM[0] * 0.72  # 72mm  (un peu en dessous du disque)
    r_max_mm = KNOWN_RINGS_MM[-1] * 1.15  # 287mm (un peu au-delà de l'anneau ext.)
    r_min_px = max(1, int(r_min_mm / mm_per_px_0))
    r_max_px = min(int(r_max_mm / mm_per_px_0), min(cx, cy, w - cx, h - cy) - 5)

    print(f"[RANGE] r_min={r_min_px}px ({r_min_mm:.0f}mm)  r_max={r_max_px}px ({r_max_mm:.0f}mm)")

    # ── Profil radial ──────────────────────────────────────────────────────────
    profile, edges = radial_edge_profile(gray, cx, cy, r_min_px, r_max_px)

    min_sep_px = max(6, int(8.0 / mm_per_px_0))  # ≈ 8mm de séparation min
    peaks_rel = find_peaks(profile, min_separation=min_sep_px, rel_threshold=0.07)
    peaks_px = [float(p + r_min_px) for p in peaks_rel]

    print(
        f"[PICS ] {len(peaks_px)} pics : "
        f"{[int(p) for p in peaks_px]} px  ~  "
        f"{[round(p * mm_per_px_0) for p in peaks_px]} mm"
    )

    if debug:
        cv2.imwrite(str(out_dir / f"{stem}_dbg_edges.jpg"), edges)

    # ── Matching + étalonnage ──────────────────────────────────────────────────
    matches, mm_per_px_cal = match_and_calibrate(peaks_px, KNOWN_RINGS_MM, mm_per_px_0)

    n_matched = len(matches)
    print(
        f"[CALIB] {n_matched}/{len(KNOWN_RINGS_MM)} matchés  "
        f"mm/px={mm_per_px_cal:.5f}  (nominal={mm_per_px_0:.5f}  "
        f"D={abs(mm_per_px_cal - mm_per_px_0) / mm_per_px_0 * 100:.2f}%)"
    )

    # ── Résultats complets ─────────────────────────────────────────────────────
    rings = []
    for r_mm in KNOWN_RINGS_MM:
        if r_mm in matches:
            r_px = matches[r_mm]
            detected = True
        else:
            r_px = r_mm / mm_per_px_cal
            detected = False

        rings.append(
            {
                "radius_mm": float(r_mm),
                "radius_px": round(float(r_px), 1),
                "detected": detected,
            }
        )
        tag = "OK DETECTE" if detected else "   infere  "
        print(f"  {tag}  {r_mm:5.0f}mm -> {int(r_px):4d}px")

    # ── Ajustement d'ellipses ──────────────────────────────────────────────────
    # Seuils pour la détection des outliers.
    THRESH_RATIO = 0.05  # écart max toléré sur les ratios demi-axe / rayon (5 %)
    THRESH_CENTER = 10.0  # écart max toléré sur la position du centre (px)
    THRESH_ANGLE = 20.0  # écart angulaire max toléré (deg, modulo 180)
    MIN_ECCEN_FOR_ANGLE = 0.03  # n'active le filtre d'angle que si |r0-r1| > 3 %

    # ── Passe 1 : ajustement brut ──────────────────────────────────────────────
    ring_ellipses: dict[float, tuple | None] = {}
    for ring in rings:
        r_mm = ring["radius_mm"]
        if ring["detected"]:
            r_px = ring["radius_px"]
            band = max(8, int(r_px * 0.06))  # ≈ 6 % du rayon, min 8 px
            ring_ellipses[r_mm] = fit_ring_ellipse(edges, cx, cy, r_px, band)
        else:
            ring_ellipses[r_mm] = None

    # Paramètres normalisés de chaque ellipse ajustée avec succès.
    # ax0, ax1 = demi-axe / rayon attendu (idéalement ≈ 1)
    # eangle   = angle d'orientation (deg OpenCV)
    # ecx, ecy = centre de l'ellipse
    ring_lookup = {r["radius_mm"]: r for r in rings}
    fitted_params: dict[float, tuple[float, float, float, float, float]] = {}
    for r_mm, el in ring_ellipses.items():
        if el is None:
            continue
        r_px = ring_lookup[r_mm]["radius_px"]
        (ecx_el, ecy_el), (ea, eb), eangle = el
        fitted_params[r_mm] = (ea / (2 * r_px), eb / (2 * r_px), eangle, ecx_el, ecy_el)

    def _median_consensus(params: dict) -> tuple[float, float, float, float, float]:
        """Retourne (med_r0, med_r1, med_ang, med_ecx, med_ecy) ou des valeurs neutres."""
        if not params:
            return 1.0, 1.0, 0.0, float(cx), float(cy)
        vals = list(params.values())
        return (
            float(np.median([v[0] for v in vals])),
            float(np.median([v[1] for v in vals])),
            float(np.median([v[2] for v in vals])),
            float(np.median([v[3] for v in vals])),
            float(np.median([v[4] for v in vals])),
        )

    # ── Passe 2 : détection des outliers ──────────────────────────────────────
    med_r0, med_r1, med_ang, med_ecx, med_ecy = _median_consensus(fitted_params)

    outlier_keys: set[float] = set()
    for r_mm, (ax0, ax1, ang, ecx_el, ecy_el) in fitted_params.items():
        reasons: list[str] = []

        if abs(ax0 - med_r0) > THRESH_RATIO:
            reasons.append(f"ax0={ax0:.3f} vs {med_r0:.3f}")
        if abs(ax1 - med_r1) > THRESH_RATIO:
            reasons.append(f"ax1={ax1:.3f} vs {med_r1:.3f}")

        # Filtre angulaire uniquement si l'ellipse est suffisamment excentrique
        if abs(med_r0 - med_r1) > MIN_ECCEN_FOR_ANGLE:
            d_ang = abs(ang - med_ang) % 180.0
            d_ang = min(d_ang, 180.0 - d_ang)
            if d_ang > THRESH_ANGLE:
                reasons.append(f"angle={ang:.1f} vs {med_ang:.1f}deg")

        dist_c = float(np.hypot(ecx_el - med_ecx, ecy_el - med_ecy))
        if dist_c > THRESH_CENTER:
            reasons.append(f"centre={dist_c:.1f}px")

        if reasons:
            outlier_keys.add(r_mm)
            ring_ellipses[r_mm] = None
            print(f"  [ELLI ] outlier {r_mm:.0f}mm : {' | '.join(reasons)}")

    # ── Passe 3 : consensus propre (sans outliers) ─────────────────────────────
    clean_params = {k: v for k, v in fitted_params.items() if k not in outlier_keys}
    avg_r0, avg_r1, avg_angle, avg_ecx, avg_ecy = _median_consensus(clean_params)

    n_fit = len(fitted_params)
    n_out = len(outlier_keys)
    n_ok = n_fit - n_out
    suffix = f"  [{n_out} outlier(s) remplaces]" if n_out else ""
    print(
        f"[ELLI ] {n_ok}/{n_fit} ellipses valides"
        f"  ratio=({avg_r0:.3f},{avg_r1:.3f})"
        f"  angle={avg_angle:.1f}deg"
        f"  centre=({avg_ecx:.1f},{avg_ecy:.1f}){suffix}"
    )

    # ── Synthèse des ellipses manquantes (inférées + outliers remplacés) ───────
    # Le statut distingue : "fitted" | "outlier" | "inferred"
    ellipse_status: dict[float, str] = {}
    for ring in rings:
        r_mm = ring["radius_mm"]
        if r_mm in fitted_params and r_mm not in outlier_keys:
            ellipse_status[r_mm] = "fitted"
        elif r_mm in outlier_keys:
            ellipse_status[r_mm] = "outlier"
        else:
            ellipse_status[r_mm] = "inferred"

        if ring_ellipses[r_mm] is None:
            r_px = ring["radius_px"]
            ring_ellipses[r_mm] = (
                (avg_ecx, avg_ecy),
                (2 * r_px * avg_r0, 2 * r_px * avg_r1),
                avg_angle,
            )

    # ── Anneaux intérieurs (zones 7→10, dans le disque noir) ──────────────────
    # Toujours synthétisés depuis le consensus (fond noir, Canny n'y voit rien).
    inner_ellipses: dict[float, tuple] = {}
    for _r_mm in INNER_RINGS_MM:
        _r_px = _r_mm / mm_per_px_cal
        inner_ellipses[_r_mm] = (
            (avg_ecx, avg_ecy),
            (2 * _r_px * avg_r0, 2 * _r_px * avg_r1),
            avg_angle,
        )

    # ── Enrichissement JSON avec les paramètres d'ellipse ─────────────────────
    for ring in rings:
        r_mm = ring["radius_mm"]
        el = ring_ellipses[r_mm]
        status = ellipse_status.get(r_mm, "inferred")
        if el is not None:
            (ecx_el, ecy_el), (ea, eb), eangle = el
            ring["ellipse"] = {
                "status": status,
                "cx": round(float(ecx_el), 1),
                "cy": round(float(ecy_el), 1),
                "axis_1_px": round(float(ea), 1),
                "axis_2_px": round(float(eb), 1),
                "angle_deg": round(float(eangle), 1),
            }
        else:
            ring["ellipse"] = None

    # ── Sauvegarde JSON ────────────────────────────────────────────────────────
    inner_rings_json = []
    for _r_mm in INNER_RINGS_MM:
        _el = inner_ellipses[_r_mm]
        (_ecx, _ecy), (_ea, _eb), _eangle = _el
        inner_rings_json.append(
            {
                "radius_mm": float(_r_mm),
                "radius_px": round(_r_mm / mm_per_px_cal, 1),
                "score": INNER_RINGS_SCORE[_r_mm],
                "ellipse": {
                    "status": "inferred",
                    "cx": round(float(_ecx), 1),
                    "cy": round(float(_ecy), 1),
                    "axis_1_px": round(float(_ea), 1),
                    "axis_2_px": round(float(_eb), 1),
                    "angle_deg": round(float(_eangle), 1),
                },
            }
        )

    result = {
        "cx": cx,
        "cy": cy,
        "mm_per_px_nominal": mm_per_px_0,
        "mm_per_px_calibre": mm_per_px_cal,
        "n_detected": n_matched,
        "n_total": len(KNOWN_RINGS_MM),
        "rings": rings,
        "inner_rings": inner_rings_json,
    }
    json_path = out_dir / f"{stem}_rings.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ── Annotation ────────────────────────────────────────────────────────────
    # Code couleur :
    #   vert plein         = ellipse ajustée et validée ("fitted")
    #   vert pointillé     = ellipse outlier remplacée par synthèse ("outlier")
    #   orange pointillé   = anneau non détecté, entièrement inféré ("inferred")
    annot = img.copy()
    thick = 2

    COLOR_FITTED = (0, 210, 0)  # vert (ajustement validé)
    COLOR_INFERRED = (0, 150, 255)  # orange (inféré ou outlier remplacé)

    for ring in rings:
        r_px = ring["radius_px"]
        r_mm = ring["radius_mm"]
        status = ellipse_status.get(r_mm, "inferred")
        el = ring_ellipses.get(r_mm)

        color = COLOR_INFERRED if status == "inferred" else COLOR_FITTED

        if el is not None:
            (ecx_el, ecy_el), (ea, eb), eangle = el
            ellipse_cv = (
                (int(round(ecx_el)), int(round(ecy_el))),
                (int(round(ea)), int(round(eb))),
                eangle,
            )
            if status == "fitted":
                cv2.ellipse(annot, ellipse_cv, color, thick, cv2.LINE_AA)
            else:
                # outlier ou inferred → pointillés
                _draw_dashed_ellipse(annot, el, color, thick)
        else:
            # Repli sur cercle (ne devrait pas arriver)
            r_int = int(round(r_px))
            cv2.circle(annot, (cx, cy), r_int, color, thick)

        # Étiquette à 45° depuis le centre
        lx = cx + int(r_px * np.cos(pi / 4)) + 4
        ly = cy - int(r_px * np.sin(pi / 4)) - 4
        if 0 < lx < w - 40 and 4 < ly < h:
            cv2.putText(
                annot,
                f"{int(r_mm)}mm",
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    # ── Anneaux intérieurs (zones 7–10) ──────────────────────────────────────
    COLOR_INNER = (220, 220, 220)  # blanc cassé, visible sur le disque noir
    for _r_mm in INNER_RINGS_MM:
        _el = inner_ellipses[_r_mm]
        _r_px = _r_mm / mm_per_px_cal
        _draw_dashed_ellipse(annot, _el, COLOR_INNER, thickness=1, n_dashes=48)
        # Étiquette : score + mm (à 45° vers le haut-droit)
        _score = INNER_RINGS_SCORE[_r_mm]
        _lx = cx + int(_r_px * np.cos(pi / 4)) + 4
        _ly = cy - int(_r_px * np.sin(pi / 4)) - 4
        if 0 < _lx < w - 50 and 4 < _ly < h:
            cv2.putText(
                annot,
                f"{_score}  ({int(_r_mm)}mm)",
                (_lx, _ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                COLOR_INNER,
                1,
                cv2.LINE_AA,
            )

    # Centre
    cv2.drawMarker(annot, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 40, 2)

    # Légende
    lh = h - 1
    cv2.putText(
        annot,
        f"Detecte : {n_matched}/{len(KNOWN_RINGS_MM)}  mm/px={mm_per_px_cal:.4f}  outliers={n_out}",
        (10, lh - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annot,
        "Vert plein=ajuste  Vert pointille=outlier->synthese  Orange=infere",
        (10, lh - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    annot_path = out_dir / f"{stem}_rings.jpg"
    cv2.imwrite(str(annot_path), annot, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[OK]   -> {annot_path.name}")
    print(f"[OK]   -> {json_path.name}")

    if debug:
        _save_profile_image(
            profile,
            peaks_rel,
            r_min_px,
            mm_per_px_cal,
            matches,
            out_dir / f"{stem}_radial_profile.jpg",
        )

    if show:
        _open_file(annot_path)

    return annot_path


# ── CLI ────────────────────────────────────────────────────────────────────────


def _collect_flat_images(path_str: str) -> list[Path]:
    """Collecte les images *_flat.jpg dans un fichier ou un dossier."""
    p = Path(path_str)
    if p.is_file():
        return [p]
    if p.is_dir():
        results = sorted(
            f
            for f in p.rglob("*_flat.*")
            if f.suffix.lower() in IMAGE_EXTENSIONS
            and not f.stem.startswith("_")  # exclure _summary_flat etc.
        )
        if not results:
            # Fallback : toutes les images du dossier
            results = sorted(f for f in p.glob("*.*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        return results
    print(f"[ERREUR] Chemin introuvable : {path_str}")
    sys.exit(1)


def make_summary(annot_paths: list[Path | None], out_dir: Path) -> Path:
    """
    Génère une planche de contact avec toutes les images *_rings.jpg.

    - `annot_paths` : liste d'objets `Path` ou `None` (pour les échecs).
    - `out_dir`     : dossier de sortie.

    Retourne le Path du fichier résumé généré.
    """
    COLS = 4
    CARD_W = 280
    CARD_H = 300

    total = len(annot_paths)
    rows = (total + COLS - 1) // COLS
    canvas = np.full((rows * CARD_H, COLS * CARD_W, 3), 25, dtype=np.uint8)

    for idx, path in enumerate(annot_paths):
        col_i = idx % COLS
        row_i = idx // COLS
        x0 = col_i * CARD_W
        y0 = row_i * CARD_H

        ok = path is not None and Path(path).exists()
        border = (0, 180, 0) if ok else (0, 0, 200)

        if ok:
            img_th = cv2.imread(str(path))
            if img_th is not None:
                th_h = CARD_H - 44
                scale = th_h / img_th.shape[0]
                th_w = max(1, int(img_th.shape[1] * scale))
                thumb = cv2.resize(img_th, (th_w, th_h), interpolation=cv2.INTER_AREA)
                tx = x0 + (CARD_W - th_w) // 2
                ty = y0 + 4
                if tx >= 0 and tx + th_w <= canvas.shape[1]:
                    canvas[ty : ty + th_h, tx : tx + th_w] = thumb

        cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x0 + CARD_W - 2, y0 + CARD_H - 2), border, 3)

        stem = Path(path).stem if path is not None else "(missing)"
        label = (stem[-20:] + "  OK") if ok else (stem[-20:] + "  ECHEC")
        cv2.putText(
            canvas,
            label,
            (x0 + 6, y0 + CARD_H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path = out_dir / "_summary_rings.jpg"
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n[RÉSUMÉ] {sum(1 for p in annot_paths if p)} / {total} → {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Détecte les anneaux concentriques sur une image mise à plat."
    )
    ap.add_argument("image", help="Image *_flat.jpg ou dossier contenant des *_flat.jpg")
    ap.add_argument(
        "--out", default=None, help="Dossier de sortie (défaut : même dossier que l'image flat)"
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Sauvegarde les images intermédiaires (edges, profil radial)",
    )
    ap.add_argument("--show", action="store_true", help="Ouvre l'image annotée finale")
    args = ap.parse_args()

    images = _collect_flat_images(args.image)
    print(f"{len(images)} image(s) à traiter.")

    rings_paths = []
    for img_path in images:
        result = detect_rings(img_path, output_dir=args.out, debug=args.debug, show=args.show)
        rings_paths.append(result)

    # Génère la planche de contact si plusieurs images
    if len(images) > 1:
        if args.out:
            out_dir = Path(args.out)
        else:
            src_arg = Path(args.image)
            out_dir = src_arg if src_arg.is_dir() else Path(images[0]).parent
        summary_path = make_summary(rings_paths, out_dir)
        if args.show:
            _open_file(summary_path)
