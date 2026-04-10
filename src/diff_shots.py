"""
Détecte les NOUVEAUX impacts entre deux photos successives de la même cible.

Principe :
  1. Les deux photos passent par le pipeline complet (flatten → rings → YOLO).
     Si une image *_flat.jpg existe déjà, l'étape flatten est sautée.
  2. Les impacts de la photo "avant" sont mis en correspondance avec ceux de
     la photo "après" (distance < MATCH_TOL_MM).
  3. Les impacts non appariés dans la photo "après" sont les NOUVEAUX impacts.

Sortie dans <out>/<stem_after>/ :
  <stem_after>_diff.jpg   image annotée :
                            - cercles gris pointillés = anciens impacts
                            - cercles colorés pleins  = nouveaux impacts + score
  <stem_after>_diff.json  données structurées

Usage :
    python src/diff_shots.py <avant.jpg> <apres.jpg> [options]
    python src/diff_shots.py data/raw/shot1.jpg data/raw/shot2.jpg --show
    python src/diff_shots.py outputs/flatten/.../shot1_flat.jpg outputs/flatten/.../shot2_flat.jpg --show
"""

import argparse
import json
import sys
from math import pi
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from flatten_target import process as flatten
from detect_rings import detect_rings
from detect_impacts_yolo import detect_impacts_yolo, _auto_device
from localize_target import _open_file
from flatten_target import OUTPUT_CENTER

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_WEIGHTS   = "runs/detect/models/yolo_impacts/weights/best.pt"
DEFAULT_OUT       = "outputs"
DEFAULT_CONF      = 0.25
DEFAULT_IOU       = 0.4
MATCH_TOL_MM      = 8.0   # deux impacts à < 8mm → considérés identiques

SCORE_COLORS: dict[int, tuple] = {
    10: (0, 220, 255), 9: (0, 200, 220), 8: (0, 170, 200),
    7: (180, 180, 180), 6: (0, 230, 100), 5: (0, 210, 60),
    4: (80, 210, 0), 3: (140, 180, 0), 2: (160, 100, 0),
    1: (140, 60, 0), 0: (40, 40, 180),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_dashed_circle(img, cx, cy, r, color, thickness=2, n=36):
    for i in range(0, n, 2):
        a1, a2 = 2 * pi * i / n, 2 * pi * (i + 1) / n
        p1 = (int(cx + r * np.cos(a1)), int(cy + r * np.sin(a1)))
        p2 = (int(cx + r * np.cos(a2)), int(cy + r * np.sin(a2)))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _ensure_flat(
    image_path: Path,
    out_root: Path,
    debug: bool,
    hint_cx_norm: float | None = None,
    hint_cy_norm: float | None = None,
) -> Path | None:
    """Retourne le path du *_flat.jpg, en lançant flatten si nécessaire."""
    if "_flat" in image_path.stem:
        return image_path   # déjà aplati

    stem    = image_path.stem
    flat    = out_root / stem / f"{stem}_flat.jpg"
    if flat.exists():
        print(f"  [CACHE] flat existant : {flat.name}")
        return flat

    print(f"  [FLATTEN] {image_path.name} ...")
    annot = flatten(
        image_path,
        output_dir=str(out_root),
        debug=debug,
        hint_cx_norm=hint_cx_norm,
        hint_cy_norm=hint_cy_norm,
    )
    if annot is None:
        return None
    return flat if flat.exists() else None


def _ensure_impacts(flat_path: Path, weights: str, conf: float,
                    iou: float, device: str, debug: bool) -> dict | None:
    """Retourne le dict impacts_yolo, en lançant le pipeline si nécessaire."""
    stem      = flat_path.stem.removesuffix("_flat")
    json_path = flat_path.parent / f"{stem}_impacts_yolo.json"

    # Rings calibration (peut déjà exister)
    rings_json = flat_path.parent / f"{stem}_rings.json"
    if not rings_json.exists():
        print(f"  [RINGS ] {flat_path.name} ...")
        detect_rings(flat_path, debug=debug)

    # Impacts YOLO
    if not json_path.exists():
        print(f"  [YOLO  ] {flat_path.name} ...")
        detect_impacts_yolo(flat_path, weights=weights, conf_thr=conf,
                            iou_thr=iou, device=device)

    if not json_path.exists():
        print(f"  [ERREUR] impacts_yolo.json introuvable pour {flat_path.name}")
        return None

    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# ── Matching ──────────────────────────────────────────────────────────────────

def _match_impacts(before: list[dict], after: list[dict],
                   mm_per_px: float) -> tuple[list[dict], list[dict]]:
    """
    Apparie les impacts avant/après.
    Retourne (new_impacts, matched_before).
    Un impact "après" est nouveau s'il n'est à < MATCH_TOL_MM d'aucun impact "avant".
    """
    tol_px = MATCH_TOL_MM / mm_per_px
    matched_before_idx = set()
    new_impacts = []

    for imp_a in after:
        cx_a, cy_a = imp_a["cx_px"], imp_a["cy_px"]
        is_new = True
        for i, imp_b in enumerate(before):
            if i in matched_before_idx:
                continue
            dist = np.hypot(cx_a - imp_b["cx_px"], cy_a - imp_b["cy_px"])
            if dist <= tol_px:
                matched_before_idx.add(i)
                is_new = False
                break
        if is_new:
            new_impacts.append(imp_a)

    matched_before = [before[i] for i in matched_before_idx]
    return new_impacts, matched_before


# ── Pipeline diff ─────────────────────────────────────────────────────────────

def diff_shots(
    before_path: Path,
    after_path:  Path,
    out_root:    Path,
    weights:     str,
    conf:        float,
    iou:         float,
    device:      str,
    debug:       bool,
    show:        bool,
    hint_cx:     float | None = None,
    hint_cy:     float | None = None,
) -> Path | None:

    print(f"\n{'='*62}")
    print(f"  AVANT : {before_path.name}")
    print(f"  APRES : {after_path.name}")
    print(f"{'='*62}")

    # ── Étape 1 : mise à plat ─────────────────────────────────────────────────
    # Le hint s'applique sur les deux images (même cadrage)
    flat_before = _ensure_flat(before_path, out_root, debug, hint_cx, hint_cy)
    flat_after  = _ensure_flat(after_path,  out_root, debug, hint_cx, hint_cy)
    if flat_before is None or flat_after is None:
        print("[ERREUR] Mise à plat échouée.")
        return None

    # ── Étape 2 : impacts (rings + YOLO) ──────────────────────────────────────
    data_before = _ensure_impacts(flat_before, weights, conf, iou, device, debug)
    data_after  = _ensure_impacts(flat_after,  weights, conf, iou, device, debug)
    if data_before is None or data_after is None:
        print("[ERREUR] Détection d'impacts échouée.")
        return None

    mm_per_px     = float(data_after.get("mm_per_px", 0.523))
    impacts_before = data_before.get("impacts", [])
    impacts_after  = data_after.get("impacts",  [])

    print(f"\n[MATCH ] avant={len(impacts_before)}  apres={len(impacts_after)}"
          f"  tolerance={MATCH_TOL_MM}mm")

    new_impacts, matched_before = _match_impacts(
        impacts_before, impacts_after, mm_per_px)

    score_new   = sum(i["score"] for i in new_impacts)
    score_total = sum(i["score"] for i in impacts_after)

    print(f"[RESULT] {len(new_impacts)} nouveau(x) impact(s)  "
          f"score session = {score_new} pt(s)  "
          f"score total cible = {score_total} pt(s)")
    for i, imp in enumerate(new_impacts, 1):
        print(f"  NOUVEAU #{i:02d}  score={imp['score']}  "
              f"dist={imp['dist_centre_mm']:.1f}mm  "
              f"diam={imp['diam_mm']:.1f}mm")

    # ── Annotation ────────────────────────────────────────────────────────────
    img = cv2.imread(str(flat_after))
    if img is None:
        print(f"[ERREUR] Impossible de lire : {flat_after}")
        return None
    h, w = img.shape[:2]
    annot = img.copy()

    # Anciens impacts → cercles pointillés gris clair
    for imp in impacts_before:
        cx_i = int(round(imp["cx_px"]))
        cy_i = int(round(imp["cy_px"]))
        r_i  = max(5, int(round(imp["r_px"])))
        _draw_dashed_circle(annot, cx_i, cy_i, r_i, (120, 120, 120), thickness=1)

    # Nouveaux impacts → cercles colorés + score
    for i, imp in enumerate(new_impacts, 1):
        cx_i  = int(round(imp["cx_px"]))
        cy_i  = int(round(imp["cy_px"]))
        r_i   = max(5, int(round(imp["r_px"])))
        color = SCORE_COLORS.get(imp["score"], (200, 200, 200))

        cv2.circle(annot, (cx_i, cy_i), r_i + 2, color, 3, cv2.LINE_AA)
        cv2.circle(annot, (cx_i, cy_i), 4, color, -1, cv2.LINE_AA)

        lx = cx_i + r_i + 6
        ly = cy_i + 7
        if lx + 40 > w:
            lx = cx_i - r_i - 52
        cv2.putText(annot, str(imp["score"]),
                    (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2, cv2.LINE_AA)
        cv2.putText(annot, f"#{i}",
                    (lx, ly + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # Centre
    cv2.drawMarker(annot, (OUTPUT_CENTER, OUTPUT_CENTER),
                   (255, 80, 80), cv2.MARKER_CROSS, 30, 1)

    # Légende
    cv2.putText(annot,
                f"Anciens: {len(impacts_before)}  "
                f"Nouveaux: {len(new_impacts)}  "
                f"Score session: {score_new}pt  "
                f"Score total: {score_total}pt",
                (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annot,
                "Gris pointe = ancien  |  Couleur = nouveau",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    stem_after = flat_after.stem.removesuffix("_flat")
    out_dir    = flat_after.parent
    annot_path = out_dir / f"{stem_after}_diff.jpg"
    json_path  = out_dir / f"{stem_after}_diff.json"

    cv2.imwrite(str(annot_path), annot, [cv2.IMWRITE_JPEG_QUALITY, 92])

    result = {
        "before":        before_path.name,
        "after":         after_path.name,
        "mm_per_px":     mm_per_px,
        "match_tol_mm":  MATCH_TOL_MM,
        "n_before":      len(impacts_before),
        "n_after":       len(impacts_after),
        "n_new":         len(new_impacts),
        "score_session": score_new,
        "score_total":   score_total,
        "new_impacts":   new_impacts,
        "old_impacts":   impacts_before,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] -> {annot_path.name}")
    print(f"[OK] -> {json_path.name}")

    if show:
        _open_file(annot_path)

    return annot_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Detecte les nouveaux impacts entre deux series de tir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("before",
                    help="Photo (ou *_flat.jpg) de la cible AVANT la serie")
    ap.add_argument("after",
                    help="Photo (ou *_flat.jpg) de la cible APRES la serie")
    ap.add_argument("--out",     default=DEFAULT_OUT,
                    help="Dossier de sortie racine")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="Poids YOLO")
    ap.add_argument("--conf",    type=float, default=DEFAULT_CONF,
                    help="Seuil de confiance YOLO")
    ap.add_argument("--iou",     type=float, default=DEFAULT_IOU,
                    help="Seuil IoU NMS")
    ap.add_argument("--tol",     type=float, default=MATCH_TOL_MM,
                    help="Tolerance d'appariement avant/apres (mm)")
    ap.add_argument("--device",  default=None,
                    help="'0'=GPU, 'cpu'=CPU (auto si absent)")
    ap.add_argument("--debug",   action="store_true")
    ap.add_argument("--show",    action="store_true",
                    help="Ouvre l'image de resultat")
    args = ap.parse_args()

    MATCH_TOL_MM = args.tol   # override global si --tol fourni
    device = args.device if args.device else _auto_device()

    before = Path(args.before)
    after  = Path(args.after)
    for p in (before, after):
        if not p.exists():
            print(f"[ERREUR] Introuvable : {p}")
            sys.exit(1)

    diff_shots(
        before_path = before,
        after_path  = after,
        out_root    = Path(args.out),
        weights     = args.weights,
        conf        = args.conf,
        iou         = args.iou,
        device      = device,
        debug       = args.debug,
        show        = args.show,
    )
