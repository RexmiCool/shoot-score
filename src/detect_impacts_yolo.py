"""
Détecte les impacts de balles avec le modèle YOLOv8 entraîné.

Charge le modèle depuis models/yolo_impacts/weights/best.pt,
applique l'inférence sur les images *_flat.jpg, score chaque
impact via le *_rings.json calibré, et produit :
  - <stem>_impacts_yolo.jpg  (image annotée)
  - <stem>_impacts_yolo.json (résultats structurés)

Usage :
    python src/detect_impacts_yolo.py <flat.jpg ou dossier> [--show] [--conf 0.25]
    python src/detect_impacts_yolo.py outputs/flatten --show
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

# ── Détection ultralytics (facultatif : absent sur RPi3) ──────────────────────
try:
    from ultralytics import YOLO as _YOLO

    _HAS_ULTRALYTICS = True
except Exception:
    _YOLO = None  # type: ignore[assignment,misc]
    _HAS_ULTRALYTICS = False
# ── Paramètres ────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "runs/detect/models/yolo_impacts/weights/best.pt"
DEFAULT_CONF = 0.25  # seuil de confiance minimum
DEFAULT_IOU = 0.4  # seuil IoU pour NMS (évite les doublons)


def _auto_device() -> str:
    """Retourne '0' si un GPU CUDA est disponible, 'cpu' sinon."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_DEVICE = _auto_device()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Score par zone (même que detect_impacts.py)
OUTER_RING_SCORE: dict[float, int] = {
    100.0: 7,
    125.0: 6,
    150.0: 5,
    175.0: 4,
    200.0: 3,
    225.0: 2,
    250.0: 1,
}
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
SCORE_COLORS: dict[int, tuple] = {
    10: (0, 220, 255),
    9: (0, 200, 220),
    8: (0, 170, 200),
    7: (180, 180, 180),
    6: (0, 230, 100),
    5: (0, 210, 60),
    4: (80, 210, 0),
    3: (140, 180, 0),
    2: (160, 100, 0),
    1: (140, 60, 0),
    0: (40, 40, 180),
}


# ── Scoring (identique à detect_impacts.py) ───────────────────────────────────


def _score_impact(cx_px: float, cy_px: float, rings_data: dict | None, mm_per_px: float) -> int:
    if rings_data is None:
        dist_mm = np.hypot(cx_px - OUTPUT_CENTER, cy_px - OUTPUT_CENTER) * mm_per_px
        return next((s for r, s in SCORE_FALLBACK if dist_mm <= r), 0)

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
        ea2, eb2 = el["axis_1_px"] / 2, el["axis_2_px"] / 2
        if ea2 <= 0 or eb2 <= 0:
            continue
        theta = np.deg2rad(el["angle_deg"])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx, dy = cx_px - el["cx"], cy_px - el["cy"]
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        if (xr / ea2) ** 2 + (yr / eb2) ** 2 <= 1.0:
            return score
    return 0


# ── Inférence ONNX (onnxruntime, sans PyTorch) ──────────────────────────────────────


def _infer_onnx(
    flat_path: Path,
    onnx_path: Path,
    conf_thr: float,
    iou_thr: float,
) -> list[dict]:
    """
    Inférence ONNX via onnxruntime — pas besoin de PyTorch ni d'ultralytics.
    Compatible RPi3 (Cortex-A53, ARMv8.0).

    Retourne une liste de dicts :
        {"x1": float, "y1": float, "x2": float, "y2": float, "conf": float}
    """
    import onnxruntime as ort  # installé sur aarch64 uniquement

    img_bgr = cv2.imread(str(flat_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Impossible de lire : {flat_path}")
    h0, w0 = img_bgr.shape[:2]
    inp_sz = 1056

    # ── Letterbox : redimensionner + centrer dans 1056×1056 ───────────────
    r = min(inp_sz / h0, inp_sz / w0)
    nh, nw = int(h0 * r), int(w0 * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((inp_sz, inp_sz, 3), 114, dtype=np.uint8)
    pad_top = (inp_sz - nh) // 2
    pad_left = (inp_sz - nw) // 2
    canvas[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized

    # ── Prétraitement : BGR→RGB, /255, NCHW ───────────────────────────
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 1056, 1056)

    # ── Inference ─────────────────────────────────────────────────────────
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    raw = sess.run(None, {sess.get_inputs()[0].name: blob})[0]  # (1, 5, 8400)

    # ── Post-processing YOLOv8 ───────────────────────────────────────
    #   raw[0] shape : (5, 8400)  → transposé → (8400, 5)
    #   colonnes : [cx, cy, w, h, conf_classe0]  (coordonnées letterbox px)
    preds = raw[0].T  # (8400, 5)
    scores_all = preds[:, 4]
    keep = scores_all >= conf_thr
    preds = preds[keep]
    if len(preds) == 0:
        return []

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    scores = preds[:, 4]

    # Retour dans l'espace image originale
    x1 = np.clip((cx - bw / 2 - pad_left) / r, 0, w0)
    y1 = np.clip((cy - bh / 2 - pad_top) / r, 0, h0)
    x2 = np.clip((cx + bw / 2 - pad_left) / r, 0, w0)
    y2 = np.clip((cy + bh / 2 - pad_top) / r, 0, h0)

    # NMS via cv2.dnn (pas besoin de torchvision)
    # NMSBoxes attend le format [x, y, w, h] (coin supérieur-gauche + dimensions)
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_thr, iou_thr)
    flat_idxs = idxs.flatten() if hasattr(idxs, "flatten") else list(idxs)

    return [
        {
            "x1": float(x1[i]),
            "y1": float(y1[i]),
            "x2": float(x2[i]),
            "y2": float(y2[i]),
            "conf": float(scores[i]),
        }
        for i in flat_idxs
    ]


# ── Pipeline d'inférence ──────────────────────────────────────────────────────


def detect_impacts_yolo(
    flat_img_path,
    weights: str = DEFAULT_WEIGHTS,
    conf_thr: float = DEFAULT_CONF,
    iou_thr: float = DEFAULT_IOU,
    output_dir: str | None = None,
    show: bool = False,
    device: str = DEFAULT_DEVICE,
) -> Path | None:
    """Détecte les impacts avec YOLO ou ONNX sur une image mise à plat."""
    flat_path = Path(flat_img_path)
    if not flat_path.exists():
        print(f"[ERREUR] Fichier introuvable : {flat_path}")
        return None

    stem = flat_path.stem.removesuffix("_flat")
    out_dir = Path(output_dir) / stem if output_dir else flat_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Sélection du moteur d'inférence ────────────────────────────────────
    w_path = Path(weights)
    onnx_path = w_path.with_suffix(".onnx")

    if _HAS_ULTRALYTICS and w_path.exists():
        _engine = "yolo"
        model_label = str(w_path)
    elif onnx_path.exists():
        _engine = "onnx"
        model_label = str(onnx_path)
    elif _HAS_ULTRALYTICS:
        print(f"[ERREUR] Poids introuvables : {w_path}")
        print("  Lance d'abord : python src/train_yolo.py")
        return None
    else:
        print("[ERREUR] Aucun moteur disponible.")
        print(f"  ultralytics/PyTorch absent et {onnx_path} introuvable.")
        print("  Exporte le modèle (sur Windows) : python src/export_onnx.py")
        return None

    # ── Chargement de l'image et étalonnage ────────────────────────────────
    img = cv2.imread(str(flat_path))
    if img is None:
        print(f"[ERREUR] Impossible de lire : {flat_path}")
        return None
    h, w = img.shape[:2]

    rings_path = flat_path.parent / f"{stem}_rings.json"
    rings_data = None
    mm_per_px = MM_PER_PX_OUT
    if rings_path.exists():
        with open(rings_path, encoding="utf-8") as f:
            rings_data = json.load(f)
        mm_per_px = float(rings_data["mm_per_px_calibre"])

    print(f"\n{'=' * 60}")
    print(f"[IMAGE] {flat_path.name}  {w}x{h}px  moteur={_engine.upper()}")
    print(f"[CALIB] mm/px={mm_per_px:.5f}  {'(rings.json)' if rings_data else '(nominal)'}")

    # ── Inférence (YOLO ou ONNX) ───────────────────────────────────────────────
    if _engine == "yolo":
        _model = _YOLO(str(w_path))
        _results = _model.predict(
            source=str(flat_path),
            conf=conf_thr,
            iou=iou_thr,
            verbose=False,
            device=device,
        )
        _boxes = _results[0].boxes
        raw_dets: list[dict] = []
        if _boxes is not None and len(_boxes):
            for _box in _boxes:
                _x1, _y1, _x2, _y2 = _box.xyxy[0].tolist()
                raw_dets.append(
                    {"x1": _x1, "y1": _y1, "x2": _x2, "y2": _y2, "conf": float(_box.conf[0])}
                )
    else:
        raw_dets = _infer_onnx(flat_path, onnx_path, conf_thr, iou_thr)

    impacts = []
    for _det in raw_dets:
        x1, y1 = _det["x1"], _det["y1"]
        x2, y2 = _det["x2"], _det["y2"]
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2
        r_px = max((x2 - x1), (y2 - y1)) / 2
        conf = _det["conf"]
        score = _score_impact(cx_px, cy_px, rings_data, mm_per_px)
        cx_mm = round((cx_px - OUTPUT_CENTER) * mm_per_px, 1)
        cy_mm = round((cy_px - OUTPUT_CENTER) * mm_per_px, 1)

        impacts.append(
            {
                "cx_px": round(cx_px, 1),
                "cy_px": round(cy_px, 1),
                "r_px": round(r_px, 1),
                "conf": round(conf, 3),
                "score": score,
                "cx_mm": cx_mm,
                "cy_mm": cy_mm,
                "dist_centre_mm": round(np.hypot(cx_mm, cy_mm), 1),
                "diam_mm": round(r_px * 2 * mm_per_px, 1),
            }
        )

    score_total = sum(i["score"] for i in impacts)
    print(f"[DETECT] {len(impacts)} impact(s)  score total = {score_total} pt(s)")
    for i, imp in enumerate(impacts, 1):
        print(
            f"  #{i:02d}  score={imp['score']}  conf={imp['conf']:.2f}  "
            f"dist={imp['dist_centre_mm']:.1f}mm  diam={imp['diam_mm']:.1f}mm"
        )

    # ── Sauvegarde JSON ────────────────────────────────────────────────────
    result_data = {
        "source": flat_path.name,
        "model": model_label,
        "mm_per_px": mm_per_px,
        "conf_thr": conf_thr,
        "n_impacts": len(impacts),
        "score_total": score_total,
        "impacts": impacts,
    }
    json_path = out_dir / f"{stem}_impacts_yolo.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    # ── Annotation ─────────────────────────────────────────────────────────
    annot = img.copy()
    for i, imp in enumerate(impacts, 1):
        cx_i = int(round(imp["cx_px"]))
        cy_i = int(round(imp["cy_px"]))
        r_i = max(5, int(round(imp["r_px"])))
        color = SCORE_COLORS.get(imp["score"], (200, 200, 200))

        cv2.circle(annot, (cx_i, cy_i), r_i, color, 2, cv2.LINE_AA)
        cv2.circle(annot, (cx_i, cy_i), 3, color, -1, cv2.LINE_AA)

        # Score + numéro + confiance
        lx = cx_i + r_i + 5
        ly = cy_i + 6
        if lx + 40 > w:
            lx = cx_i - r_i - 50
        cv2.putText(
            annot,
            str(imp["score"]),
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annot,
            f"#{i} {imp['conf']:.0%}",
            (lx, ly + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    lh = h - 1
    cv2.putText(
        annot,
        f"{_engine.upper()}  Impacts : {len(impacts)}    Score : {score_total} pt(s)"
        f"    conf>={conf_thr:.0%}",
        (10, lh - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    annot_path = out_dir / f"{stem}_impacts_yolo.jpg"
    cv2.imwrite(str(annot_path), annot, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[OK]   -> {annot_path.name}")
    print(f"[OK]   -> {json_path.name}")

    if show:
        _open_file(annot_path)

    return annot_path


# ── CLI ───────────────────────────────────────────────────────────────────────


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
        return results or sorted(f for f in p.glob("*.*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"[ERREUR] Chemin introuvable : {path_str}")
    sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Détection d'impacts par YOLOv8.")
    ap.add_argument("image", help="Image *_flat.jpg ou dossier")
    ap.add_argument(
        "--weights", default=DEFAULT_WEIGHTS, help=f"Poids du modèle (défaut: {DEFAULT_WEIGHTS})"
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Seuil de confiance (défaut: {DEFAULT_CONF})",
    )
    ap.add_argument(
        "--iou", type=float, default=DEFAULT_IOU, help=f"Seuil IoU NMS (défaut: {DEFAULT_IOU})"
    )
    ap.add_argument(
        "--out", default=None, help="Dossier de sortie (défaut: même dossier que l'image)"
    )
    ap.add_argument(
        "--device", default=DEFAULT_DEVICE, help="Device : '0'=GPU, 'cpu'=CPU (défaut: 0)"
    )
    ap.add_argument("--show", action="store_true", help="Ouvre l'image annotée finale")
    args = ap.parse_args()

    images = _collect_flat_images(args.image)
    print(f"{len(images)} image(s) \u00e0 traiter.")
    for img_path in images:
        detect_impacts_yolo(
            img_path,
            weights=args.weights,
            conf_thr=args.conf,
            iou_thr=args.iou,
            output_dir=args.out,
            show=args.show,
            device=args.device,
        )
