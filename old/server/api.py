"""
API REST FastAPI — expose le pipeline shoot-score à l'application mobile.

Endpoints :
  GET  /health              vérification du serveur
  POST /process             une image → impacts détectés + image annotée (base64)
  POST /diff                deux images (avant/après) → nouveaux impacts + image annotée

Lancement :
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Détection automatique de la plateforme :
  - x86_64 / AMD64 avec poids YOLO disponibles → YOLOv8
  - aarch64 (Raspberry Pi) ou poids absents    → détection morphologique (sans PyTorch)
"""

import base64
import json
import platform
import shutil
import time
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from flatten_target import process as flatten
from detect_rings import detect_rings
from diff_shots import diff_shots, MATCH_TOL_MM

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "runs/detect/models/yolo_impacts/weights/best.pt"
DEFAULT_CONF = 0.2
DEFAULT_IOU = 0.4
API_OUT_ROOT = Path("outputs/api")

# ── Sélection du moteur de détection ─────────────────────────────────────────
# On tente toujours YOLO en premier (fonctionne sur x86 CUDA, x86 CPU, ARM CPU).
# Fallback morphologique seulement si PyTorch n'est pas installé ou si les
# poids sont absents.
_arch = platform.machine().lower()
_weights = Path(DEFAULT_WEIGHTS)

try:
    _weights_onnx = _weights.with_suffix(".onnx")
    if not _weights.exists() and not _weights_onnx.exists():
        raise FileNotFoundError(f"Aucun poids trouvé : {_weights} ou {_weights_onnx}")
    from detect_impacts_yolo import detect_impacts_yolo, _auto_device, _HAS_ULTRALYTICS

    device = _auto_device()
    # Moteur réel : YOLO si .pt présent + ultralytics dispo, sinon ONNX Runtime
    if _HAS_ULTRALYTICS and Path(DEFAULT_WEIGHTS).exists():
        engine_name = "yolo"
    else:
        engine_name = "onnx"
    _USE_YOLO = True
    print(f"[API] moteur={engine_name}  device={device}  arch={_arch}")
except FileNotFoundError as e:
    from detect_impacts import detect_impacts as _detect_morph

    device = "cpu"
    _USE_YOLO = False
    print(f"[API] moteur=MORPHO  ({e})")
except Exception as e:
    from detect_impacts import detect_impacts as _detect_morph

    device = "cpu"
    _USE_YOLO = False
    print(f"[API] moteur=MORPHO  (YOLO/ONNX indisponible : {e})")


app = FastAPI(
    title="shoot-score API",
    description="Détection automatique d'impacts de balles sur cibles de tir",
    version="1.0.0",
)

# CORS — nécessaire pour les requêtes depuis Expo (et dev web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modèles de réponse ────────────────────────────────────────────────────────


class Impact(BaseModel):
    cx_px: float
    cy_px: float
    r_px: float
    score: int
    dist_centre_mm: float
    diam_mm: float
    cx_mm: float
    cy_mm: float


class ProcessResponse(BaseModel):
    n_impacts: int
    total_score: int
    mm_per_px: float
    engine: str
    impacts: list[Impact]
    flat_b64: str
    img_width: int
    img_height: int


class DiffResponse(BaseModel):
    n_before: int
    n_after: int
    n_new: int
    score_session: int
    score_total: int
    match_tol_mm: float
    engine: str
    new_impacts: list[Impact]
    all_impacts: list[Impact]  # tous les impacts détectés sur l'image après
    flat_b64: str  # image après aplatie sans annotation
    img_width: int
    img_height: int


# ── Utilitaires ───────────────────────────────────────────────────────────────


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _img_wh(path: Path) -> tuple[int, int]:
    """Retourne (width, height) d'une image sans charger tous les pixels."""
    try:
        from PIL import Image as _PIL

        with _PIL.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        pass
    try:
        import cv2 as _cv2

        img = _cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass
    return 0, 0


def _normalize_impact(raw: dict) -> dict:
    """
    Normalise un dict d'impact pour qu'il soit compatible avec le modèle Impact,
    qu'il vienne du détecteur YOLO (_impacts_yolo.json) ou morphologique (_impacts.json).
    """
    return {
        "cx_px": float(raw.get("cx_px", 0)),
        "cy_px": float(raw.get("cy_px", 0)),
        "r_px": float(raw.get("r_px", 0)),
        "score": int(raw.get("score", 0)),
        "dist_centre_mm": float(raw.get("dist_centre_mm", 0)),
        "diam_mm": float(raw.get("diam_mm", 0)),
        "cx_mm": float(raw.get("cx_mm", 0)),
        "cy_mm": float(raw.get("cy_mm", 0)),
    }


def _run_pipeline(
    img_path: Path,
    out_root: Path,
    hint_cx: float | None = None,
    hint_cy: float | None = None,
) -> tuple[Path, dict, str]:
    """
    Exécute flatten → rings → détection (YOLO ou morphologique).
    Retourne (annot_path, impacts_data, engine_name).
    """
    stem = img_path.stem
    flat = out_root / stem / f"{stem}_flat.jpg"

    # Flatten
    if not flat.exists():
        annot = flatten(
            img_path,
            output_dir=str(out_root),
            debug=False,
            hint_cx_norm=hint_cx,
            hint_cy_norm=hint_cy,
        )
        if annot is None or not flat.exists():
            raise HTTPException(500, f"Mise à plat échouée pour {img_path.name}")

    # Rings
    rings_json = flat.parent / f"{stem}_rings.json"
    if not rings_json.exists():
        result = detect_rings(flat, debug=False)
        if result is None:
            raise HTTPException(500, f"Détection des anneaux échouée pour {img_path.name}")

    if _USE_YOLO:
        # ── YOLO ──────────────────────────────────────────────────────────────
        impacts_json = flat.parent / f"{stem}_impacts_yolo.json"
        annot_path = flat.parent / f"{stem}_impacts_yolo.jpg"
        if not impacts_json.exists():
            detect_impacts_yolo(
                flat,
                weights=DEFAULT_WEIGHTS,
                conf_thr=DEFAULT_CONF,
                iou_thr=DEFAULT_IOU,
                device=device,
            )
        engine = "yolo"
    else:
        # ── Morphologique ──────────────────────────────────────────────────────
        impacts_json = flat.parent / f"{stem}_impacts.json"
        annot_path = flat.parent / f"{stem}_impacts.jpg"
        if not impacts_json.exists():
            _detect_morph(flat, output_dir=str(flat.parent.parent), debug=False)
        engine = "morpho"

    if not impacts_json.exists():
        raise HTTPException(500, f"Détection échouée pour {img_path.name}")

    with open(impacts_json, encoding="utf-8") as f:
        data = json.load(f)

    return annot_path, data, engine


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    _engine = engine_name if _USE_YOLO else "morpho"
    return {"status": "ok", "device": device, "engine": _engine, "arch": _arch}


@app.post("/process", response_model=ProcessResponse)
async def process_image(
    image: UploadFile = File(...),
    hint_cx: float = Form(0.5),
    hint_cy: float = Form(0.5),
):
    """
    Traite une photo de cible et retourne les impacts détectés.

    Body : multipart/form-data avec le champ `image` (JPEG/PNG).
    Champs optionnels : `hint_cx`, `hint_cy` (float 0-1), coordonnées
    normalisées du centre de la croix de visée dans l'image.
    """
    ts = int(time.time() * 1000)
    out_root = API_OUT_ROOT / str(ts)

    # Sauvegarde du fichier uploadé
    suffix = Path(image.filename or "photo.jpg").suffix or ".jpg"
    img_path = out_root / f"photo{suffix}"
    _save_upload(image, img_path)

    try:
        annot_path, data, engine = _run_pipeline(
            img_path,
            out_root,
            hint_cx=hint_cx if 0 < hint_cx < 1 else None,
            hint_cy=hint_cy if 0 < hint_cy < 1 else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    raw_impacts = [_normalize_impact(i) for i in data.get("impacts", [])]
    impacts = [Impact(**i) for i in raw_impacts]
    total = sum(i.score for i in impacts)
    mm_per_px = float(data.get("mm_per_px", 0.523))
    flat_path = out_root / img_path.stem / f"{img_path.stem}_flat.jpg"
    flat_b64 = _img_to_b64(flat_path) if flat_path.exists() else ""
    img_w, img_h = _img_wh(flat_path) if flat_path.exists() else (0, 0)

    return ProcessResponse(
        n_impacts=len(impacts),
        total_score=total,
        mm_per_px=mm_per_px,
        engine=engine,
        impacts=impacts,
        flat_b64=flat_b64,
        img_width=img_w,
        img_height=img_h,
    )


@app.post("/diff", response_model=DiffResponse)
async def diff_images(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    hint_cx: float = Form(0.5),
    hint_cy: float = Form(0.5),
):
    """
    Compare deux photos de la même cible et retourne les NOUVEAUX impacts.

    Body : multipart/form-data avec les champs `before` et `after`.
    Champs optionnels : `hint_cx`, `hint_cy` (float 0-1).
    """
    ts = int(time.time() * 1000)
    out_root = API_OUT_ROOT / str(ts)

    suffix_b = Path(before.filename or "before.jpg").suffix or ".jpg"
    suffix_a = Path(after.filename or "after.jpg").suffix or ".jpg"
    before_path = out_root / f"before{suffix_b}"
    after_path = out_root / f"after{suffix_a}"
    _save_upload(before, before_path)
    _save_upload(after, after_path)

    try:
        diff_path = diff_shots(
            before_path=before_path,
            after_path=after_path,
            out_root=out_root,
            weights=DEFAULT_WEIGHTS,
            conf=DEFAULT_CONF,
            iou=DEFAULT_IOU,
            device=device,
            debug=False,
            show=False,
            hint_cx=hint_cx if 0 < hint_cx < 1 else None,
            hint_cy=hint_cy if 0 < hint_cy < 1 else None,
        )
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    if diff_path is None:
        raise HTTPException(500, "Calcul du différentiel échoué")

    # Lire le json diff
    after_stem = after_path.stem
    json_path = out_root / after_stem / f"{after_stem}_diff.json"
    if not json_path.exists():
        # chercher dans le sous-dossier créé par diff_shots
        candidates = list(out_root.rglob("*_diff.json"))
        json_path = candidates[0] if candidates else None

    if json_path is None or not json_path.exists():
        raise HTTPException(500, "Fichier diff.json introuvable")

    with open(json_path, encoding="utf-8") as f:
        diff_data = json.load(f)

    # Image aplatie SANS annotation (image de référence pour le rendu mobile)
    after_stem = after_path.stem
    flat_after_path = out_root / after_stem / f"{after_stem}_flat.jpg"
    flat_b64 = _img_to_b64(flat_after_path) if flat_after_path.exists() else ""
    img_w, img_h = _img_wh(flat_after_path) if flat_after_path.exists() else (0, 0)

    # Tous les impacts détectés sur l'image après (nouveaux + anciens matchés)
    if _USE_YOLO:
        after_all_json = out_root / after_stem / f"{after_stem}_impacts_yolo.json"
    else:
        after_all_json = out_root / after_stem / f"{after_stem}_impacts.json"
    all_impacts: list[Impact] = []
    if after_all_json.exists():
        with open(after_all_json, encoding="utf-8") as f:
            after_all_data = json.load(f)
        all_impacts = [Impact(**_normalize_impact(i)) for i in after_all_data.get("impacts", [])]

    new_impacts = [Impact(**_normalize_impact(i)) for i in diff_data.get("new_impacts", [])]

    return DiffResponse(
        n_before=diff_data.get("n_before", 0),
        n_after=diff_data.get("n_after", 0),
        n_new=diff_data.get("n_new", 0),
        score_session=diff_data.get("score_session", 0),
        score_total=diff_data.get("score_total", 0),
        match_tol_mm=diff_data.get("match_tol_mm", MATCH_TOL_MM),
        engine="yolo" if _USE_YOLO else "morpho",
        new_impacts=new_impacts,
        all_impacts=all_impacts,
        flat_b64=flat_b64,
        img_width=img_w,
        img_height=img_h,
    )
