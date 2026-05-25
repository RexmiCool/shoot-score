"""
Étape 4 : détection des impacts de balles sur l'image mise à plat.

Stratégie (sans image de référence) :
  ┌─ Zone blanche (papier, hors disque) ─────────────────────────────────────┐
  │  Transformée black-hat morphologique = closing(I) − I                    │
  │  Révèle les pixels anormalement SOMBRES (trous de balle dans le papier)  │
  └──────────────────────────────────────────────────────────────────────────┘
  ┌─ Disque noir central ─────────────────────────────────────────────────────┐
  │  Transformée top-hat morphologique = I − opening(I)                       │
  │  Révèle les pixels anormalement CLAIRS (papier visible sous le trou)      │
  └──────────────────────────────────────────────────────────────────────────┘
  Les deux cartes sont fusionnées, seuillées par zone (Otsu local), puis
  les contours sont filtrés par taille / circularité / éloignement des bords
  d'anneaux.  Chaque impact retenu est scoré via les ellipses calibrées du
  fichier *_rings.json associé.

Usage :
    python src/detect_impacts.py <flat.jpg ou dossier> [--out outputs] [--debug] [--show]
"""

import torch
from torchvision.transforms import functional as TF
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


PATCH_SIZE = 64
CNN_THRESHOLD = 0.60  # ajustable


# ── Paramètres de détection ────────────────────────────────────────────────────
BULLET_DIAM_MM = 9.0  # diamètre nominal d'un impact (9mm pistol)
MIN_DIAM_MM = 1.0  # ↓ plus permissif
MAX_DIAM_MM = 32.0  # ↑ plus permissif
MIN_CIRCULARITY = 0.20  # ↓ on accepte les formes imparfaites
EDGE_MARGIN_MM = 1.0  # ↓ moins agressif

# Noyau de fond morphologique  [tuned]
BG_KERNEL_PX = 21  # taille fixe du noyau de fond (pixels, impair)
OTSU_SCALE = 1.8  # multiplicateur du seuil Otsu  [tuned]
OTSU_FLOOR = 5  # seuil absolu minimum  [tuned]

# Score de la zone intérieure à chaque anneau extérieur calibré
OUTER_RING_SCORE: dict[float, int] = {
    100.0: 7,
    125.0: 6,
    150.0: 5,
    175.0: 4,
    200.0: 3,
    225.0: 2,
    250.0: 1,
}

# Fallback si rings.json absent : (rayon_mm_max, score) du plus petit au plus grand
SCORE_FALLBACK: list[tuple[float, int]] = [
    (25.0, 10),
    (50.0, 9),
    (75.0, 8),
    (100.0, 7),
    (125.0, 6),
    (150.0, 5),
    (175.0, 4),
    (200.0, 3),
    (225.0, 2),
    (250.0, 1),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Palette de couleurs par score (BGR)
SCORE_COLORS: dict[int, tuple] = {
    10: (0, 220, 255),  # jaune vif
    9: (0, 200, 220),
    8: (0, 170, 200),
    7: (180, 180, 180),  # gris (dans le disque)
    6: (0, 230, 100),
    5: (0, 210, 60),
    4: (80, 210, 0),
    3: (140, 180, 0),
    2: (160, 100, 0),
    1: (140, 60, 0),
    0: (40, 40, 180),  # rouge sombre (miss)
}


_CNN_MODEL = None


def _load_cnn_model():
    global _CNN_MODEL
    if _CNN_MODEL is not None:
        return _CNN_MODEL

    class MiniCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
            )
            self.fc = torch.nn.Linear(64, 2)

        def forward(self, x):
            x = self.net(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = MiniCNN()
    model.load_state_dict(torch.load("mini_impact_cnn.pt", map_location="cpu"))
    model.eval()
    _CNN_MODEL = model
    print("[CNN] mini_impact_cnn chargé")
    return model


def _classify_impact_patch(gray_img, cx, cy):
    model = _load_cnn_model()

    half = PATCH_SIZE // 2
    h, w = gray_img.shape
    x0 = int(np.clip(cx - half, 0, w - PATCH_SIZE))
    y0 = int(np.clip(cy - half, 0, h - PATCH_SIZE))

    patch = gray_img[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
    tensor = torch.tensor(patch / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    return {
        "impact_prob": float(probs[1]),
        "is_impact": float(probs[1]) >= CNN_THRESHOLD,
    }


# ── Chargement des données ─────────────────────────────────────────────────────


def _load_rings_json(flat_path: Path) -> dict | None:
    """Charge le *_rings.json associé à l'image flat, ou None si absent."""
    stem = flat_path.stem.removesuffix("_flat")
    json_path = flat_path.parent / f"{stem}_rings.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    print(f"  [WARN] rings.json introuvable pour {flat_path.name} (détection sans étalonnage)")
    return None


# ── Masques de zones ───────────────────────────────────────────────────────────


def _ellipse_mask(rings_data: dict, r_mm: float, shape: tuple, filled: bool = True) -> np.ndarray:
    """Masque (rempli ou contour) de l'ellipse calibrée au rayon r_mm."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    source = rings_data.get("inner_rings", []) + rings_data.get("rings", [])
    for ring in source:
        if ring["radius_mm"] == r_mm:
            el = ring.get("ellipse")
            if el is None:
                break
            center = (int(round(el["cx"])), int(round(el["cy"])))
            axes = (int(round(el["axis_1_px"] / 2)), int(round(el["axis_2_px"] / 2)))
            if axes[0] <= 0 or axes[1] <= 0:
                break
            thick = -1 if filled else 1
            cv2.ellipse(mask, center, axes, el["angle_deg"], 0, 360, 255, thick)
            return mask
    # Fallback : cercle simple
    cx, cy = OUTPUT_CENTER, OUTPUT_CENTER
    r_px = int(round(r_mm / MM_PER_PX_OUT))
    thick = -1 if filled else 1
    cv2.circle(mask, (cx, cy), r_px, 255, thick)
    return mask


def _ring_edge_mask(rings_data: dict, shape: tuple, mm_per_px: float) -> np.ndarray:
    """
    Masque binaire des zones proches des bords d'anneaux calibrés.
    Utilisé pour exclure les faux positifs sur les traits d'anneaux et les chiffres.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    margin = max(4, int(EDGE_MARGIN_MM / mm_per_px))
    all_rings = rings_data.get("inner_rings", []) + rings_data.get("rings", [])
    for ring in all_rings:
        el = ring.get("ellipse")
        if el is None:
            continue
        center = (int(round(el["cx"])), int(round(el["cy"])))
        axes = (int(round(el["axis_1_px"] / 2)), int(round(el["axis_2_px"] / 2)))
        if axes[0] <= 0 or axes[1] <= 0:
            continue
        cv2.ellipse(mask, center, axes, el["angle_deg"], 0, 360, 255, margin * 2)
    return mask


# ── Scoring ────────────────────────────────────────────────────────────────────


def _score_impact(cx_px: float, cy_px: float, rings_data: dict | None, mm_per_px: float) -> int:
    """
    Détermine le score d'un impact par appartenance aux ellipses calibrées.
    Itère du plus petit anneau au plus grand ; retourne le score du premier
    anneau qui contient le centre de l'impact.
    """
    if rings_data is None:
        # Fallback : distance euclidienne simple
        dist_mm = np.hypot(cx_px - OUTPUT_CENTER, cy_px - OUTPUT_CENTER) * mm_per_px
        return next((s for r, s in SCORE_FALLBACK if dist_mm <= r), 0)

    # Construire la liste complète triée par rayon croissant
    all_rings: list[tuple[float, int, dict | None]] = []
    for r in rings_data.get("inner_rings", []):
        all_rings.append((r["radius_mm"], r.get("score", 0), r.get("ellipse")))
    for r in rings_data.get("rings", []):
        sc = OUTER_RING_SCORE.get(r["radius_mm"], 0)
        all_rings.append((r["radius_mm"], sc, r.get("ellipse")))
    all_rings.sort(key=lambda x: x[0])

    for _, score, el in all_rings:
        if el is None:
            continue
        ea2 = el["axis_1_px"] / 2
        eb2 = el["axis_2_px"] / 2
        if ea2 <= 0 or eb2 <= 0:
            continue
        theta = np.deg2rad(el["angle_deg"])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx, dy = cx_px - el["cx"], cy_px - el["cy"]
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        if (xr / ea2) ** 2 + (yr / eb2) ** 2 <= 1.0:
            return score
    return 0  # miss


# ── Carte d'anomalies ─────────────────────────────────────────────────────────


def _anomaly_map(gray: np.ndarray) -> np.ndarray:
    """
    Calcule la carte d'anomalies morphologique :
      black-hat  = closing(I) − I  (trous sombres dans zones claires)
      top-hat    = I − opening(I)  (taches claires dans zones sombres)
    Retourne max(black-hat, top-hat).
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_KERNEL_PX, BG_KERNEL_PX))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    return np.maximum(blackhat, tophat)


def _otsu_threshold_zone(
    amap: np.ndarray, zone_mask: np.ndarray, scale: float = 1.15, floor: int = 10
) -> np.ndarray:
    """
    Seuillage Otsu calculé sur les pixels de la zone uniquement.
    `scale` rehausse légèrement le seuil Otsu pour réduire les faux positifs.
    `floor` garantit un seuil minimum absolu.
    """
    vals = amap[zone_mask > 0].reshape(-1).astype(np.uint8)
    if vals.size < 100 or int(vals.max()) == 0:
        return np.zeros_like(amap)

    thresh_val, _ = cv2.threshold(vals.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = max(int(thresh_val * scale), floor)

    binary = np.zeros_like(amap)
    binary[zone_mask > 0] = np.where(amap[zone_mask > 0] >= thresh_val, 255, 0).astype(np.uint8)
    return binary


# ── Filtrage des contours ──────────────────────────────────────────────────────


def _filter_contours(binary: np.ndarray, mm_per_px: float, edge_mask: np.ndarray) -> list[dict]:
    """
    Extrait et filtre les contours de la carte binaire.

    Critères de rejet :
      - Rayon enclosant hors de [MIN_DIAM_MM/2, MAX_DIAM_MM/2]
      - Circularité < MIN_CIRCULARITY
      - Centre tombant dans le masque de bords d'anneaux
    """
    r_min = (MIN_DIAM_MM / 2) / mm_per_px
    r_max = (MAX_DIAM_MM / 2) / mm_per_px

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < pi * r_min**2:
            continue
        (icx, icy), ir = cv2.minEnclosingCircle(cnt)
        if ir < r_min or ir > r_max:
            continue
        circ = area / (pi * ir**2) if ir > 0 else 0
        if circ < MIN_CIRCULARITY:  # noqa: use global constant
            continue
        # Exclure si le centre est sur un bord d'anneau
        ix_i, iy_i = int(round(icx)), int(round(icy))
        if (
            0 <= iy_i < edge_mask.shape[0]
            and 0 <= ix_i < edge_mask.shape[1]
            and edge_mask[iy_i, ix_i] > 0
        ):
            continue
        results.append(
            {
                "cx_px": float(icx),
                "cy_px": float(icy),
                "r_px": float(ir),
                "area_px": float(area),
                "circularity": round(float(circ), 3),
            }
        )
    return results


def _deduplicate(impacts: list[dict], mm_per_px: float) -> list[dict]:
    """Fusionne les détections multiples du même impact (critère : distance < r_min)."""
    min_dist = (MIN_DIAM_MM / 2) / mm_per_px
    kept: list[dict] = []
    for imp in sorted(impacts, key=lambda x: -x["area_px"]):
        cx, cy = imp["cx_px"], imp["cy_px"]
        if all(np.hypot(cx - k["cx_px"], cy - k["cy_px"]) > min_dist for k in kept):
            kept.append(imp)
    return kept


# ── Pipeline principale ────────────────────────────────────────────────────────


def detect_impacts(
    flat_img_path,
    output_dir: str | None = None,
    debug: bool = False,
    show: bool = False,
) -> Path | None:
    """Détecte les impacts sur une image mise à plat."""
    flat_path = Path(flat_img_path)
    if not flat_path.exists():
        print(f"[ERREUR] Fichier introuvable : {flat_path}")
        return None

    stem = flat_path.stem.removesuffix("_flat")
    out_dir = Path(output_dir) / stem if output_dir else flat_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement ─────────────────────────────────────────────────────────────
    img = cv2.imread(str(flat_path))
    if img is None:
        print(f"[ERREUR] Impossible de lire : {flat_path}")
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"\n{'=' * 60}")
    print(f"[IMAGE] {flat_path.name}  {w}×{h}px")

    # ── Données d'étalonnage ────────────────────────────────────────────────────
    rings_data = _load_rings_json(flat_path)
    mm_per_px = float(rings_data["mm_per_px_calibre"]) if rings_data else MM_PER_PX_OUT

    print(f"[CALIB] mm/px={mm_per_px:.5f}  {'(rings.json)' if rings_data else '(nominal)'}")

    # ── Masques de zones ────────────────────────────────────────────────────────
    if rings_data:
        mask_disk = _ellipse_mask(rings_data, 100.0, gray.shape, filled=True)
        mask_target = _ellipse_mask(rings_data, 250.0, gray.shape, filled=True)
        edge_mask = _ring_edge_mask(rings_data, gray.shape, mm_per_px)
    else:
        cx, cy = OUTPUT_CENTER, OUTPUT_CENTER
        mask_disk = np.zeros_like(gray)
        mask_target = np.zeros_like(gray)
        edge_mask = np.zeros_like(gray)
        cv2.circle(mask_disk, (cx, cy), int(100.0 / mm_per_px), 255, -1)
        cv2.circle(mask_target, (cx, cy), int(250.0 / mm_per_px), 255, -1)

    # Zone papier = intérieur de la cible MOINS le disque noir
    mask_paper = cv2.bitwise_and(mask_target, cv2.bitwise_not(mask_disk))

    # ── Carte d'anomalies ───────────────────────────────────────────────────────
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    amap = _anomaly_map(gray_blur)

    if debug:
        cv2.imwrite(
            str(out_dir / f"{stem}_dbg_amap.jpg"),
            cv2.normalize(amap, None, 0, 255, cv2.NORM_MINMAX),
        )
        cv2.imwrite(str(out_dir / f"{stem}_dbg_mask_disk.jpg"), mask_disk)
        cv2.imwrite(str(out_dir / f"{stem}_dbg_mask_paper.jpg"), mask_paper)
        cv2.imwrite(str(out_dir / f"{stem}_dbg_edge_mask.jpg"), edge_mask)

    # ── Seuillage par zone ──────────────────────────────────────────────────────
    binary_disk = _otsu_threshold_zone(amap, mask_disk, scale=OTSU_SCALE, floor=OTSU_FLOOR)
    binary_paper = _otsu_threshold_zone(amap, mask_paper, scale=OTSU_SCALE, floor=OTSU_FLOOR)
    binary_all = cv2.bitwise_or(binary_disk, binary_paper)

    if debug:
        cv2.imwrite(str(out_dir / f"{stem}_dbg_binary.jpg"), binary_all)

    # ── Morphologie de nettoyage ────────────────────────────────────────────────
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_clean = cv2.morphologyEx(binary_all, cv2.MORPH_OPEN, k_open, iterations=2)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, k_close, iterations=1)

    if debug:
        cv2.imwrite(str(out_dir / f"{stem}_dbg_binary_clean.jpg"), binary_clean)

    # ── Détection + filtrage + déduplication ────────────────────────────────────
    impacts_raw = _filter_contours(binary_clean, mm_per_px, edge_mask)

    impacts = []
    for imp in impacts_raw:
        cx, cy = imp["cx_px"], imp["cy_px"]
        cnn = _classify_impact_patch(gray, cx, cy)

        imp["impact_prob"] = round(cnn["impact_prob"], 3)
        imp["cnn_valid"] = cnn["is_impact"]
        imp["uncertain"] = 0.45 <= cnn["impact_prob"] <= 0.65

        if cnn["is_impact"]:
            impacts.append(imp)

    print(f"[CNN] {len(impacts_raw)} candidats → {len(impacts)} impacts validés")
    # ── Scoring ─────────────────────────────────────────────────────────────────
    cx0, cy0 = OUTPUT_CENTER, OUTPUT_CENTER
    for imp in impacts:
        imp["score"] = _score_impact(imp["cx_px"], imp["cy_px"], rings_data, mm_per_px)
        imp["cx_mm"] = round((imp["cx_px"] - cx0) * mm_per_px, 1)
        imp["cy_mm"] = round((imp["cy_px"] - cy0) * mm_per_px, 1)
        imp["dist_centre_mm"] = round(np.hypot(imp["cx_mm"], imp["cy_mm"]), 1)
        imp["diam_mm"] = round(imp["r_px"] * 2 * mm_per_px, 1)

    score_total = sum(i["score"] for i in impacts)
    print(f"[SCORE] total = {score_total} pt(s)")
    for i, imp in enumerate(impacts, 1):
        print(
            f"  #{i:02d}  score={imp['score']}  "
            f"dist={imp['dist_centre_mm']:.1f}mm  "
            f"diam={imp['diam_mm']:.1f}mm  "
            f"circ={imp['circularity']:.2f}"
        )

    # ── Sauvegarde JSON ─────────────────────────────────────────────────────────
    result = {
        "source": flat_path.name,
        "mm_per_px": mm_per_px,
        "n_impacts": len(impacts),
        "score_total": score_total,
        "impacts": impacts,
    }
    json_path = out_dir / f"{stem}_impacts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ── Annotation ──────────────────────────────────────────────────────────────
    annot = img.copy()

    for i, imp in enumerate(impacts, 1):
        cx_i = int(round(imp["cx_px"]))
        cy_i = int(round(imp["cy_px"]))
        r_i = max(5, int(round(imp["r_px"])))
        color = SCORE_COLORS.get(imp["score"], (200, 200, 200))

        # Cercle englobant + point central
        cv2.circle(annot, (cx_i, cy_i), r_i, color, 2, cv2.LINE_AA)
        cv2.circle(annot, (cx_i, cy_i), 3, color, -1, cv2.LINE_AA)

        # Score en grand + numéro d'impact en petit
        label_score = str(imp["score"])
        label_num = f"#{i}"
        lx = cx_i + r_i + 5
        ly = cy_i + 6
        if lx + 30 > w:
            lx = cx_i - r_i - 36
        cv2.putText(
            annot, label_score, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            annot, label_num, (lx, ly + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA
        )

    # Légende
    lh = h - 1
    cv2.putText(
        annot,
        f"Impacts : {len(impacts)}    Score total : {score_total} pt(s)",
        (10, lh - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    annot_path = out_dir / f"{stem}_impacts.jpg"
    cv2.imwrite(str(annot_path), annot, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[OK]   → {annot_path.name}")
    print(f"[OK]   → {json_path.name}")

    if show:
        _open_file(annot_path)

    return annot_path


# ── CLI ────────────────────────────────────────────────────────────────────────


def _collect_flat_images(path_str: str) -> list[Path]:
    p = Path(path_str)
    if p.is_file():
        return [p]
    if p.is_dir():
        results = sorted(
            f
            for f in p.rglob("*_flat.*")
            if f.suffix.lower() in IMAGE_EXTENSIONS and not f.stem.startswith("_")
        )
        if not results:
            results = sorted(f for f in p.glob("*.*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        return results
    print(f"[ERREUR] Chemin introuvable : {path_str}")
    sys.exit(1)


def make_summary(annot_paths: list[Path | None], scores: list[int | None], out_dir: Path) -> Path:
    """
    Génère une planche de contact avec toutes les images *_impacts.jpg.

    - `annot_paths` : liste de Path (ou None si traitement échoué).
    - `scores`      : score total correspondant à chaque image (ou None).
    - `out_dir`     : dossier de sortie du fichier résumé.
    """
    COLS = 4
    CARD_W = 280
    CARD_H = 320  # un peu plus haut que rings pour laisser place au score

    total = len(annot_paths)
    rows = (total + COLS - 1) // COLS
    canvas = np.full((rows * CARD_H, COLS * CARD_W, 3), 25, dtype=np.uint8)

    for idx, (path, score) in enumerate(zip(annot_paths, scores)):
        col_i = idx % COLS
        row_i = idx // COLS
        x0 = col_i * CARD_W
        y0 = row_i * CARD_H

        ok = path is not None and Path(path).exists()
        border = (0, 180, 0) if ok else (0, 0, 200)

        if ok:
            img_th = cv2.imread(str(path))
            if img_th is not None:
                th_h = CARD_H - 56
                scale = th_h / img_th.shape[0]
                th_w = max(1, int(img_th.shape[1] * scale))
                thumb = cv2.resize(img_th, (th_w, th_h), interpolation=cv2.INTER_AREA)
                tx = x0 + (CARD_W - th_w) // 2
                ty = y0 + 4
                if tx >= 0 and tx + th_w <= canvas.shape[1]:
                    canvas[ty : ty + th_h, tx : tx + th_w] = thumb

        cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x0 + CARD_W - 2, y0 + CARD_H - 2), border, 3)

        stem = Path(path).stem if path is not None else "(missing)"
        label = stem[-22:] if ok else stem[-22:] + "  ECHEC"
        cv2.putText(
            canvas,
            label,
            (x0 + 6, y0 + CARD_H - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        score_str = f"Score : {score} pt(s)" if score is not None else "—"
        score_color = (0, 220, 100) if ok else (80, 80, 80)
        cv2.putText(
            canvas,
            score_str,
            (x0 + 6, y0 + CARD_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            score_color,
            1,
            cv2.LINE_AA,
        )

    out_path = out_dir / "_summary_impacts.jpg"
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    ok_count = sum(1 for p in annot_paths if p is not None)
    print(f"\n[RÉSUMÉ] {ok_count} / {total} → {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Détecte les impacts de balles sur une image aplatie.")
    ap.add_argument("image", help="Image *_flat.jpg ou dossier contenant des *_flat.jpg")
    ap.add_argument(
        "--out", default=None, help="Dossier de sortie (défaut : même dossier que l'image)"
    )
    ap.add_argument(
        "--debug", action="store_true", help="Sauvegarde les images intermédiaires de debug"
    )
    ap.add_argument("--show", action="store_true", help="Ouvre l'image annotée finale")
    args = ap.parse_args()

    images = _collect_flat_images(args.image)
    print(f"{len(images)} image(s) à traiter.")

    impact_paths: list[Path | None] = []
    impact_scores: list[int | None] = []
    for img_path in images:
        result = detect_impacts(img_path, output_dir=args.out, debug=args.debug, show=args.show)
        impact_paths.append(result)
        # Lire le score total depuis le JSON généré
        if result is not None:
            stem_r = result.stem.removesuffix("_impacts")
            json_p = result.parent / f"{stem_r}_impacts.json"
            try:
                with open(json_p, encoding="utf-8") as _f:
                    impact_scores.append(json.load(_f)["score_total"])
            except Exception:
                impact_scores.append(None)
        else:
            impact_scores.append(None)

    if len(images) > 1:
        src_arg = Path(args.image)
        out_dir = (
            Path(args.out)
            if args.out
            else (src_arg if src_arg.is_dir() else impact_paths[0].parent)
        )
        summary_path = make_summary(impact_paths, impact_scores, out_dir)
        if args.show:
            _open_file(summary_path)
