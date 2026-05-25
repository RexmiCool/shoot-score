"""
Entraîne YOLOv8 sur le dataset d'impacts de balles.

Utilise le modèle yolov8n.pt (nano) pré-entraîné sur COCO comme point de départ.
Le fine-tuning prend ~10–30 min sur GPU pour 200 epochs avec ~30 images.

Augmentations clés activées :
  - Rotations 360° (degrees=180) : la cible est ronde, invariante en rotation
  - Flip horizontal + vertical
  - Jitter HSV, brightness, contrast
  - Mosaïque + mix-up

Usage :
    python src/train_yolo.py
    python src/train_yolo.py --data data/yolo/dataset.yaml --model yolov8s.pt --epochs 300
"""

import argparse
import sys
from pathlib import Path

# ── Paramètres d'entraînement ─────────────────────────────────────────────────
DEFAULT_DATA = "data/yolo/dataset.yaml"
DEFAULT_MODEL = "yolov8n.pt"  # nano = rapide, bon pour peu de données
DEFAULT_EPOCHS = 200
DEFAULT_IMGSZ = 1056  # résolution d'entrainement (1056 ou 1024)
DEFAULT_BATCH = 8  # à réduire si OOM GPU (4 si 8Go VRAM)
DEFAULT_OUT = "models"  # dossier de sortie des poids
DEFAULT_AUG_PRESET = "robust"


def _augmentation_config(preset: str) -> dict:
    """Retourne la configuration d'augmentation YOLO selon un preset.

    Args:
        preset: Nom du preset d'augmentation ("baseline" ou "robust").

    Returns:
        Dictionnaire prêt à être passé à ``yolo.train(**kwargs)``.

    Raises:
        ValueError: Si le preset est inconnu.
    """
    if preset == "baseline":
        return {
            "degrees": 180.0,
            "fliplr": 0.5,
            "flipud": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "translate": 0.1,
            "scale": 0.3,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.1,
        }

    if preset == "robust":
        return {
            # Géométrie: cas caméra inclinée, décentrée, bord cadre.
            "degrees": 180.0,
            "fliplr": 0.5,
            "flipud": 0.5,
            "translate": 0.20,
            "scale": 0.45,
            "shear": 7.5,
            "perspective": 0.0008,
            # Couleur/illumination: conditions réelles smartphone.
            "hsv_h": 0.02,
            "hsv_s": 0.85,
            "hsv_v": 0.55,
            # Compositions/occlusions partielles.
            "mosaic": 1.0,
            "close_mosaic": 12,
            "mixup": 0.15,
            "copy_paste": 0.2,
            "erasing": 0.25,
        }

    raise ValueError(f"Preset d'augmentation inconnu: {preset}")


def _auto_device() -> str:
    """Retourne '0' si un GPU CUDA est disponible, 'cpu' sinon."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_DEVICE = _auto_device()


def train(
    data: str = DEFAULT_DATA,
    model: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    out_dir: str = DEFAULT_OUT,
    device: str = DEFAULT_DEVICE,
    aug_preset: str = DEFAULT_AUG_PRESET,
) -> Path:
    """Lance l'entraînement et retourne le path du meilleur modèle.

    Args:
        data: Path vers ``dataset.yaml`` YOLO.
        model: Poids de départ (ex: ``yolov8n.pt``).
        epochs: Nombre d'epochs.
        imgsz: Résolution d'entraînement.
        batch: Batch size.
        out_dir: Dossier projet Ultralytics.
        device: Device Ultralytics (``0`` ou ``cpu``).
        aug_preset: Preset d'augmentation (``baseline`` ou ``robust``).

    Returns:
        Path vers ``best.pt``.
    """
    from ultralytics import YOLO

    data_path = Path(data)
    if not data_path.exists():
        print(f"[ERREUR] dataset.yaml introuvable : {data_path}")
        print("  Lance d'abord : python src/prepare_yolo_dataset.py outputs/flatten")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"[TRAIN] Modèle    : {model}")
    print(f"[TRAIN] Dataset   : {data_path}")
    print(f"[TRAIN] Epochs    : {epochs}")
    print(f"[TRAIN] Image sz  : {imgsz}px")
    print(f"[TRAIN] Batch     : {batch}")
    print(f"[TRAIN] Augment   : {aug_preset}")
    print(f"{'=' * 60}\n")

    yolo = YOLO(model)  # télécharge automatiquement si absent
    aug_cfg = _augmentation_config(aug_preset)

    yolo.train(
        data=str(data_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,  # "0"=GPU 0, "cpu"=CPU
        project=out_dir,
        name="yolo_impacts",
        exist_ok=True,
        # ── Optimiseur ──────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,  # lr finale = lr0 * lrf
        warmup_epochs=5,
        patience=50,  # early stopping
        # ── Augmentation ────────────────────────────────────────────────────
        **aug_cfg,
        # ── Seuils de détection ─────────────────────────────────────────────
        conf=0.25,  # seuil de confiance pour l'évaluation
        iou=0.5,  # IoU pour NMS
        # ── Misc ────────────────────────────────────────────────────────────
        plots=True,  # courbes loss/metrics
        save=True,
        verbose=True,
    )

    best_weights = Path(out_dir) / "yolo_impacts" / "weights" / "best.pt"
    print(f"\n[OK] Entraînement terminé.")
    print(f"[OK] Meilleurs poids → {best_weights}")
    print(f"\nPour détecter les impacts :")
    print(f"  python src/detect_impacts_yolo.py outputs/flatten --show")
    return best_weights


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entraîne YOLOv8 pour détecter les impacts de balles.")
    ap.add_argument(
        "--data", default=DEFAULT_DATA, help=f"Path vers dataset.yaml (défaut: {DEFAULT_DATA})"
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modèle de base (défaut: {DEFAULT_MODEL}). "
        "Essayer yolov8s.pt pour plus de précision.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Nombre d'epochs (défaut: {DEFAULT_EPOCHS})",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Résolution d'entrée (défaut: {DEFAULT_IMGSZ})",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Taille de batch (défaut: {DEFAULT_BATCH}, réduire à 4 si OOM)",
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"Dossier de sortie (défaut: {DEFAULT_OUT})")
    ap.add_argument(
        "--device", default=DEFAULT_DEVICE, help="Device : '0' pour GPU, 'cpu' pour CPU (défaut: 0)"
    )
    ap.add_argument(
        "--aug-preset",
        default=DEFAULT_AUG_PRESET,
        choices=["baseline", "robust"],
        help="Preset d'augmentation (défaut: robust)",
    )
    args = ap.parse_args()

    train(
        args.data,
        args.model,
        args.epochs,
        args.imgsz,
        args.batch,
        args.out,
        args.device,
        args.aug_preset,
    )
