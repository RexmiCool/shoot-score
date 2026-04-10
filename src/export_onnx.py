"""
Exporte le modèle YOLOv8 entraîné (best.pt) vers ONNX.

Le fichier ONNX est utilisé sur les plateformes sans PyTorch compatible
(ex. Raspberry Pi 3 — Cortex-A53, ARMv8.0) via onnxruntime.

Usage :
    python src/export_onnx.py
    python src/export_onnx.py --weights runs/detect/models/yolo_impacts/weights/best.pt
    python src/export_onnx.py --imgsz 640
"""

import argparse
from pathlib import Path

DEFAULT_WEIGHTS = "runs/detect/models/yolo_impacts/weights/best.pt"
DEFAULT_IMGSZ   = 640


def export_onnx(weights: str = DEFAULT_WEIGHTS, imgsz: int = DEFAULT_IMGSZ) -> Path:
    """
    Exporte *weights* (*.pt) en ONNX simplifié.

    Returns le chemin du fichier .onnx généré.
    """
    from ultralytics import YOLO

    pt_path = Path(weights)
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Poids introuvables : {pt_path}\n"
            "  Lance d'abord : python src/train_yolo.py"
        )

    print(f"[EXPORT] {pt_path}  →  ONNX  (imgsz={imgsz})")
    model = YOLO(str(pt_path))
    out   = model.export(
        format   = "onnx",
        imgsz    = imgsz,
        simplify = True,      # onnxsim — supprime les ops inutiles
        dynamic  = False,     # batch fixe = 1, plus simple sur RPi
        opset    = 12,        # compatible onnxruntime ≥ 1.10
    )
    onnx_path = Path(out)
    print(f"[OK]  → {onnx_path}  ({onnx_path.stat().st_size // 1024} Ko)")
    return onnx_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exporte YOLOv8 → ONNX.")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help=f"Chemin du .pt (défaut : {DEFAULT_WEIGHTS})")
    ap.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ,
                    help=f"Taille d'entrée en pixels (défaut : {DEFAULT_IMGSZ})")
    args = ap.parse_args()
    export_onnx(args.weights, args.imgsz)
