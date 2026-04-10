"""
Optimisation des paramètres de détection d'impacts par grid search.

Lit les fichiers *_labels.json (produits par label_impacts.py) et pour chaque
combinaison de paramètres évalue Precision / Recall / F1 sur l'ensemble des
images labélisées.  Affiche un classement des 20 meilleures combinaisons et
sauvegarde le résultat complet dans tune_results.json.

Usage :
    python src/tune_impacts.py outputs/flatten
    python src/tune_impacts.py outputs/flatten --top 30
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from flatten_target import OUTPUT_CENTER, MM_PER_PX_OUT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Grille de paramètres à explorer ───────────────────────────────────────────
PARAM_GRID = {
    # Noyau de fond morphologique (doit être impair)
    "bg_kernel_px": [21, 31, 43, 55, 71],
    # Multiplicateur Otsu (> 1 = plus strict)
    "otsu_scale":   [1.0, 1.15, 1.3, 1.5, 1.8],
    # Seuil absolu minimal (floor)
    "otsu_floor":   [5, 10, 18, 28],
    # Circularité minimale
    "min_circ":     [0.15, 0.22, 0.35, 0.50],
    # Marge exclusion bords d'anneaux (mm)
    "edge_margin_mm": [2.0, 4.0, 6.0],
}

# Distance de tolérance pour considérer un impact "trouvé" (en mm)
MATCH_TOL_MM = 8.0   # un impact est "détecté" si on est à < 8mm du label

# Diamètre balle (mm) — pour les filtres de taille
BULLET_DIAM_MM = 9.0
MIN_DIAM_MM    = 3.5
MAX_DIAM_MM    = 18.0


# ── Utilitaires morphologiques (copiés/simplifiés depuis detect_impacts) ──────

def _anomaly_map(gray: np.ndarray, kernel_px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,   k)
    return np.maximum(bh, th)


def _threshold_zone(amap: np.ndarray, mask: np.ndarray,
                    scale: float, floor: int) -> np.ndarray:
    vals = amap[mask > 0].reshape(-1).astype(np.uint8)
    if vals.size < 50 or int(vals.max()) == 0:
        return np.zeros_like(amap)
    tv, _ = cv2.threshold(vals.reshape(1, -1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tv = max(int(tv * scale), floor)
    out = np.zeros_like(amap)
    out[mask > 0] = np.where(amap[mask > 0] >= tv, 255, 0).astype(np.uint8)
    return out


def _ring_edge_mask(rings_data: dict, shape: tuple, margin_px: int) -> np.ndarray:
    h, w   = shape[:2]
    mask   = np.zeros((h, w), dtype=np.uint8)
    all_rings = rings_data.get("inner_rings", []) + rings_data.get("rings", [])
    for ring in all_rings:
        el = ring.get("ellipse")
        if el is None:
            continue
        center = (int(round(el["cx"])),         int(round(el["cy"])))
        axes   = (int(round(el["axis_1_px"] / 2)),
                  int(round(el["axis_2_px"] / 2)))
        if axes[0] <= 0 or axes[1] <= 0:
            continue
        cv2.ellipse(mask, center, axes, el["angle_deg"], 0, 360, 255,
                    margin_px * 2)
    return mask


def _detect_in_zone(gray: np.ndarray, mask: np.ndarray,
                    edge_mask: np.ndarray, mm_per_px: float,
                    p: dict) -> list[tuple[float, float]]:
    """Retourne une liste de (cx_px, cy_px) pour les impacts détectés."""
    from math import pi

    amap = _anomaly_map(gray, p["bg_kernel_px"])
    binary = _threshold_zone(amap, mask, p["otsu_scale"], p["otsu_floor"])

    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_o, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_c, iterations=1)

    r_min = (MIN_DIAM_MM / 2) / mm_per_px
    r_max = (MAX_DIAM_MM / 2) / mm_per_px

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < pi * r_min ** 2:
            continue
        (icx, icy), ir = cv2.minEnclosingCircle(cnt)
        if ir < r_min or ir > r_max:
            continue
        circ = area / (pi * ir ** 2) if ir > 0 else 0
        if circ < p["min_circ"]:
            continue
        ix_i, iy_i = int(round(icx)), int(round(icy))
        if (0 <= iy_i < edge_mask.shape[0] and 0 <= ix_i < edge_mask.shape[1]
                and edge_mask[iy_i, ix_i] > 0):
            continue
        results.append((float(icx), float(icy)))

    # Déduplication
    min_dist = (MIN_DIAM_MM / 2) / mm_per_px
    kept = []
    for cx, cy in sorted(results, key=lambda _: 0):  # ordre stable
        if all(np.hypot(cx - kx, cy - ky) > min_dist for kx, ky in kept):
            kept.append((cx, cy))
    return kept


# ── Évaluation d'une combinaison de paramètres ────────────────────────────────

def _evaluate(params: dict, samples: list[dict]) -> dict:
    """
    Calcule Precision, Recall, F1 sur l'ensemble des échantillons labélisés.

    Un impact détecté est un TP si son centre est à < MATCH_TOL_MM d'un label.
    """
    tp_total = fp_total = fn_total = 0

    for s in samples:
        preds   = s["preds_fn"](params)    # liste (cx, cy) détectés
        labels  = s["labels"]              # liste (cx, cy) ground truth
        mm_per_px = s["mm_per_px"]
        tol_px  = MATCH_TOL_MM / mm_per_px

        matched_labels = set()
        matched_preds  = set()

        for pi_i, (px, py) in enumerate(preds):
            for li, (lx, ly) in enumerate(labels):
                if li in matched_labels:
                    continue
                if np.hypot(px - lx, py - ly) <= tol_px:
                    matched_labels.add(li)
                    matched_preds.add(pi_i)
                    break

        tp = len(matched_labels)
        fp = len(preds)  - tp
        fn = len(labels) - tp
        tp_total += tp
        fp_total += fp
        fn_total += fn

    prec   = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1     = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
    return {
        "precision": round(prec,   3),
        "recall":    round(recall, 3),
        "f1":        round(f1,     3),
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
    }


# ── Chargement des données ─────────────────────────────────────────────────────

def _load_samples(root: Path) -> list[dict]:
    label_files = sorted(root.rglob("*_labels.json"))
    if not label_files:
        print(f"[ERREUR] Aucun fichier *_labels.json trouvé sous {root}")
        sys.exit(1)
    print(f"[LOAD] {len(label_files)} image(s) labélisée(s)")

    samples = []
    for lp in label_files:
        with open(lp, encoding="utf-8") as f:
            label_data = json.load(f)

        labels = [(p["cx_px"], p["cy_px"]) for p in label_data.get("impacts", [])]
        stem   = lp.stem.removesuffix("_labels")

        # Image flat
        flat_path = lp.parent / f"{stem}_flat.jpg"
        if not flat_path.exists():
            # Chercher dans le dossier parent
            for ext in IMAGE_EXTENSIONS:
                candidates = list(lp.parent.glob(f"{stem}_flat{ext}"))
                if candidates:
                    flat_path = candidates[0]
                    break
        if not flat_path.exists():
            print(f"  [SKIP] flat introuvable pour {lp.name}")
            continue

        img = cv2.imread(str(flat_path))
        if img is None:
            print(f"  [SKIP] impossible de lire {flat_path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Rings JSON → masques
        rings_path = lp.parent / f"{stem}_rings.json"
        if rings_path.exists():
            with open(rings_path, encoding="utf-8") as f:
                rings_data = json.load(f)
            mm_per_px = float(rings_data["mm_per_px_calibre"])
        else:
            rings_data = None
            mm_per_px  = MM_PER_PX_OUT

        # Masque zone papier (intérieur de la cible, hors disque noir)
        h, w = gray.shape
        cx0, cy0 = OUTPUT_CENTER, OUTPUT_CENTER

        def _ell_mask(r_mm, filled=True, rd=rings_data, mp=mm_per_px, sh=gray.shape):
            m = np.zeros(sh, dtype=np.uint8)
            if rd:
                src = rd.get("inner_rings", []) + rd.get("rings", [])
                for ring in src:
                    if ring["radius_mm"] == r_mm:
                        el = ring.get("ellipse")
                        if el:
                            c = (int(round(el["cx"])),  int(round(el["cy"])))
                            a = (int(round(el["axis_1_px"] / 2)),
                                 int(round(el["axis_2_px"] / 2)))
                            if a[0] > 0 and a[1] > 0:
                                thick = -1 if filled else 1
                                cv2.ellipse(m, c, a, el["angle_deg"],
                                            0, 360, 255, thick)
                                return m
            r_px = int(round(r_mm / mp))
            cv2.circle(m, (cx0, cy0), r_px, 255, -1 if filled else 1)
            return m

        mask_disk   = _ell_mask(100.0, filled=True)
        mask_target = _ell_mask(250.0, filled=True)
        mask_paper  = cv2.bitwise_and(mask_target, cv2.bitwise_not(mask_disk))

        def _preds_fn(p, _gray=gray, _mask=mask_paper, _rd=rings_data,
                      _mp=mm_per_px):
            margin_px = max(2, int(p["edge_margin_mm"] / _mp))
            emask = _ring_edge_mask(_rd, _gray.shape, margin_px) if _rd else np.zeros_like(_gray)
            return _detect_in_zone(_gray, _mask, emask, _mp, p)

        print(f"  {flat_path.name}  {len(labels)} label(s)")
        samples.append({
            "name":      flat_path.name,
            "labels":    labels,
            "mm_per_px": mm_per_px,
            "preds_fn":  _preds_fn,
        })

    return samples


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid_search(root: Path, top_n: int = 20) -> list[dict]:
    samples = _load_samples(root)
    if not samples:
        print("[ERREUR] Aucun échantillon valide.")
        sys.exit(1)

    n_labels_total = sum(len(s["labels"]) for s in samples)
    print(f"\n[GRID] {n_labels_total} impact(s) au total dans {len(samples)} image(s)")

    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    print(f"[GRID] {len(combos)} combinaison(s) à tester...\n")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        metrics = _evaluate(params, samples)
        results.append({"params": params, **metrics})
        if i % 100 == 0 or i == len(combos):
            print(f"  {i}/{len(combos)}  best F1 so far = "
                  f"{max(r['f1'] for r in results):.3f}", end="\r")

    results.sort(key=lambda r: (-r["f1"], -r["recall"]))
    print(f"\n\n{'='*70}")
    print(f"  TOP {min(top_n, len(results))} COMBINAISONS (sur {len(combos)})")
    print(f"{'='*70}")
    header = (f"{'#':>3}  {'F1':>5}  {'Prec':>5}  {'Rec':>5}  "
              f"{'TP':>3}  {'FP':>3}  {'FN':>3}  Paramètres")
    print(header)
    print("-" * 70)
    for i, r in enumerate(results[:top_n], 1):
        p = r["params"]
        print(f"{i:>3}  {r['f1']:.3f}  {r['precision']:.3f}  {r['recall']:.3f}  "
              f"{r['tp']:>3}  {r['fp']:>3}  {r['fn']:>3}  "
              f"ker={p['bg_kernel_px']:>2} "
              f"scale={p['otsu_scale']:.2f} "
              f"floor={p['otsu_floor']:>2} "
              f"circ={p['min_circ']:.2f} "
              f"marg={p['edge_margin_mm']:.1f}mm")

    # Sauvegarde complète
    out_path = root / "tune_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Résultats complets → {out_path}")

    # Afficher les meilleurs paramètres à copier dans detect_impacts.py
    best = results[0]
    print(f"\n{'='*70}")
    print("  MEILLEURE COMBINAISON (à copier dans detect_impacts.py) :")
    print(f"{'='*70}")
    for k, v in best["params"].items():
        print(f"  {k:20s} = {v}")
    print(f"  → F1={best['f1']:.3f}  Prec={best['precision']:.3f}  "
          f"Rec={best['recall']:.3f}  "
          f"TP={best['tp']}  FP={best['fp']}  FN={best['fn']}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Grid search sur les paramètres de détection d'impacts.")
    ap.add_argument("root",
                    help="Dossier contenant les *_labels.json (ex: outputs/flatten)")
    ap.add_argument("--top", type=int, default=20,
                    help="Nombre de combinaisons à afficher (défaut: 20)")
    args = ap.parse_args()

    run_grid_search(Path(args.root), top_n=args.top)
