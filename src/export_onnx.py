"""
Exporte le modèle YOLOv8 entraîné (best.pt) vers ONNX.

Le fichier ONNX est utilisé sur les plateformes sans PyTorch compatible
(ex. Raspberry Pi 3 — Cortex-A53, ARMv8.0) via onnxruntime.

Usage :
    python src/export_onnx.py
    python src/export_onnx.py --weights runs/detect/models/yolo_impacts/weights/best.pt
    python src/export_onnx.py --imgsz 1056
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_WEIGHTS = "runs/detect/models/yolo_impacts/weights/best.pt"
DEFAULT_IMGSZ = 1056
DEFAULT_MOBILE_ASSET = Path("mobile/android/app/src/main/assets/impact_yolo.onnx")


def export_onnx(weights: str = DEFAULT_WEIGHTS, imgsz: int = DEFAULT_IMGSZ) -> Path:
    """
    Exporte *weights* (*.pt) en ONNX simplifié.

    Returns le chemin du fichier .onnx généré.
    """
    from ultralytics import YOLO

    pt_path = Path(weights)
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Poids introuvables : {pt_path}\n  Lance d'abord : python src/train_yolo.py"
        )

    print(f"[EXPORT] {pt_path}  →  ONNX  (imgsz={imgsz})")
    model = YOLO(str(pt_path))
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,  # onnxsim — supprime les ops inutiles
        dynamic=False,  # batch fixe = 1, plus simple sur RPi
        opset=12,  # compatible onnxruntime ≥ 1.10
    )
    onnx_path = Path(out)
    print(f"[OK]  → {onnx_path}  ({onnx_path.stat().st_size // 1024} Ko)")
    return onnx_path


def copy_to_mobile_assets(onnx_path: Path, mobile_asset: Path) -> Path:
    """Copy the exported ONNX model into Android assets.

    Args:
        onnx_path: Source ONNX file.
        mobile_asset: Destination asset path inside the Android app.

    Returns:
        The copied asset path.
    """
    mobile_asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, mobile_asset)
    print(f"[OK]  → asset Android : {mobile_asset}  ({mobile_asset.stat().st_size // 1024} Ko)")
    return mobile_asset


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exporte YOLOv8 → ONNX.")
    ap.add_argument(
        "--weights", default=DEFAULT_WEIGHTS, help=f"Chemin du .pt (défaut : {DEFAULT_WEIGHTS})"
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Taille d'entrée en pixels (défaut : {DEFAULT_IMGSZ})",
    )
    ap.add_argument(
        "--mobile-asset",
        type=Path,
        default=DEFAULT_MOBILE_ASSET,
        help=f"Asset Android de destination (défaut : {DEFAULT_MOBILE_ASSET})",
    )
    args = ap.parse_args()
    exported = export_onnx(args.weights, args.imgsz)
    copy_to_mobile_assets(exported, args.mobile_asset)
