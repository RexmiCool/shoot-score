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
DEFAULT_DATA   = "data/yolo/dataset.yaml"
DEFAULT_MODEL  = "yolov8n.pt"    # nano = rapide, bon pour peu de données
DEFAULT_EPOCHS = 200
DEFAULT_IMGSZ  = 640             # résolution d'entrainement (640 ou 1024)
DEFAULT_BATCH  = 8               # à réduire si OOM GPU (4 si 8Go VRAM)
DEFAULT_OUT    = "models"        # dossier de sortie des poids

def _auto_device() -> str:
    """Retourne '0' si un GPU CUDA est disponible, 'cpu' sinon."""
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEFAULT_DEVICE = _auto_device()


def train(
    data:    str = DEFAULT_DATA,
    model:   str = DEFAULT_MODEL,
    epochs:  int = DEFAULT_EPOCHS,
    imgsz:   int = DEFAULT_IMGSZ,
    batch:   int = DEFAULT_BATCH,
    out_dir: str = DEFAULT_OUT,
    device:  str = DEFAULT_DEVICE,
) -> Path:
    """Lance l'entraînement et retourne le path du meilleur modèle."""
    from ultralytics import YOLO

    data_path = Path(data)
    if not data_path.exists():
        print(f"[ERREUR] dataset.yaml introuvable : {data_path}")
        print("  Lance d'abord : python src/prepare_yolo_dataset.py outputs/flatten")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"[TRAIN] Modèle    : {model}")
    print(f"[TRAIN] Dataset   : {data_path}")
    print(f"[TRAIN] Epochs    : {epochs}")
    print(f"[TRAIN] Image sz  : {imgsz}px")
    print(f"[TRAIN] Batch     : {batch}")
    print(f"{'='*60}\n")

    yolo = YOLO(model)   # télécharge automatiquement si absent

    yolo.train(
        data    = str(data_path.resolve()),
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,     # "0"=GPU 0, "cpu"=CPU
        project = out_dir,
        name    = "yolo_impacts",
        exist_ok= True,

        # ── Optimiseur ──────────────────────────────────────────────────────
        optimizer = "AdamW",
        lr0       = 0.001,
        lrf       = 0.01,     # lr finale = lr0 * lrf
        warmup_epochs = 5,
        patience  = 50,       # early stopping

        # ── Augmentation ────────────────────────────────────────────────────
        # Les impacts sont invariants en rotation → on pousse à fond
        degrees   = 180.0,    # rotation aléatoire ±180°
        fliplr    = 0.5,      # flip horizontal
        flipud    = 0.5,      # flip vertical
        hsv_h     = 0.015,    # jitter teinte
        hsv_s     = 0.7,      # jitter saturation
        hsv_v     = 0.4,      # jitter luminosité
        translate = 0.1,      # translation
        scale     = 0.3,      # zoom
        mosaic    = 1.0,      # mosaïque 4 images (très utile avec peu de données)
        mixup     = 0.1,      # mix-up léger
        copy_paste= 0.1,      # copy-paste d'objets entre images

        # ── Seuils de détection ─────────────────────────────────────────────
        conf      = 0.25,     # seuil de confiance pour l'évaluation
        iou       = 0.5,      # IoU pour NMS

        # ── Misc ────────────────────────────────────────────────────────────
        plots     = True,     # courbes loss/metrics
        save      = True,
        verbose   = True,
    )

    best_weights = Path(out_dir) / "yolo_impacts" / "weights" / "best.pt"
    print(f"\n[OK] Entraînement terminé.")
    print(f"[OK] Meilleurs poids → {best_weights}")
    print(f"\nPour détecter les impacts :")
    print(f"  python src/detect_impacts_yolo.py outputs/flatten --show")
    return best_weights


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Entraîne YOLOv8 pour détecter les impacts de balles.")
    ap.add_argument("--data",   default=DEFAULT_DATA,
                    help=f"Path vers dataset.yaml (défaut: {DEFAULT_DATA})")
    ap.add_argument("--model",  default=DEFAULT_MODEL,
                    help=f"Modèle de base (défaut: {DEFAULT_MODEL}). "
                         "Essayer yolov8s.pt pour plus de précision.")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                    help=f"Nombre d'epochs (défaut: {DEFAULT_EPOCHS})")
    ap.add_argument("--imgsz",  type=int, default=DEFAULT_IMGSZ,
                    help=f"Résolution d'entrée (défaut: {DEFAULT_IMGSZ})")
    ap.add_argument("--batch",  type=int, default=DEFAULT_BATCH,
                    help=f"Taille de batch (défaut: {DEFAULT_BATCH}, "
                         "réduire à 4 si OOM)")
    ap.add_argument("--out",    default=DEFAULT_OUT,
                    help=f"Dossier de sortie (défaut: {DEFAULT_OUT})")
    ap.add_argument("--device", default=DEFAULT_DEVICE,
                    help="Device : '0' pour GPU, 'cpu' pour CPU (défaut: 0)")
    args = ap.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.out,
          args.device)
