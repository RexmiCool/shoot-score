"""
Entraîne YOLOv8 sur le dataset d'impacts de balles.

Utilise le modèle yolov8m.pt (nano) pré-entraîné sur COCO comme point de départ.
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
DEFAULT_MODEL = "yolo11m.pt"  # nano = rapide, bon pour peu de données
DEFAULT_EPOCHS = 150
DEFAULT_IMGSZ = 1280  # résolution d'entrainement (1280 ou 1024)
DEFAULT_BATCH = 4  # à réduire si OOM GPU (4 si 8Go VRAM)
DEFAULT_OUT = "models"  # dossier de sortie des poids
DEFAULT_AUG_PRESET = "auto"
DEFAULT_PATIENCE = 30
DEFAULT_TRAINING_PROFILE = "high_performance"


def _dataset_role(data_path: Path) -> str:
    """Infer a coarse dataset role from its path.

    Args:
        data_path: Path to the YOLO dataset YAML.

    Returns:
        ``"axis"`` for axis-marker datasets, ``"impacts"`` otherwise.
    """
    parts = [p.lower() for p in data_path.parts]
    joined = "/".join(parts)
    if "axis_yolo" in joined or "axis" in joined:
        return "axis"
    return "impacts"


def _augmentation_config(preset: str, data_role: str) -> tuple[str, dict]:
    """Retourne la configuration d'augmentation YOLO selon un preset.

    Args:
        preset: Nom du preset d'augmentation.
        data_role: Role du dataset (``"axis"`` ou ``"impacts"``).

    Returns:
        Tuple ``(preset_resolu, kwargs_aug)`` prêt pour ``yolo.train(**kwargs)``.

    Raises:
        ValueError: Si le preset est inconnu.
    """
    resolved = preset
    if preset == "auto":
        resolved = "axis_markers" if data_role == "axis" else "robust"

    if resolved == "baseline":
        return resolved, {
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

    if resolved == "robust":
        return resolved, {
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

    if resolved == "axis_markers":
        # Axis classes are orientation-dependent (top/left/bottom/right).
        # Do not rotate/flip, otherwise labels become semantically wrong.
        return resolved, {
            "degrees": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "translate": 0.08,
            "scale": 0.20,
            "shear": 2.0,
            "perspective": 0.0003,
            "hsv_h": 0.02,
            "hsv_s": 0.5,
            "hsv_v": 0.35,
            "mosaic": 0.3,
            "close_mosaic": 10,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.1,
        }

    raise ValueError(f"Preset d'augmentation inconnu: {preset}")


def _training_profile_config(
    profile: str,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    force_multi_scale: bool | None,
) -> tuple[str, dict]:
    """Resolve training profile into concrete training kwargs.

    Args:
        profile: Training profile name.
        epochs: Requested epoch count.
        imgsz: Requested image size.
        batch: Requested batch size.
        patience: Requested early-stopping patience.
        force_multi_scale: Optional override for ``multi_scale``.

    Returns:
        Tuple ``(resolved_profile, kwargs)`` with profile-specific YOLO
        training arguments.

    Raises:
        ValueError: If profile is unknown.
    """
    if profile == "standard":
        multi_scale = False if force_multi_scale is None else force_multi_scale
        return profile, {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "patience": patience,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "warmup_epochs": 5,
            "cos_lr": False,
            "cache": False,
            "multi_scale": multi_scale,
        }

    if profile == "high_performance":
        # Keep multi_scale disabled by default for stability. Some
        # ultralytics/torch combinations crash with ZeroDivisionError in
        # interpolate() when multi_scale is enabled.
        multi_scale = False if force_multi_scale is None else force_multi_scale
        return profile, {
            "epochs": max(epochs, 450),
            "imgsz": max(imgsz, 1280),
            "batch": batch,
            "patience": max(patience, 150),
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "warmup_epochs": 8,
            "cos_lr": True,
            "cache": "disk",
            "multi_scale": multi_scale,
            "close_mosaic": 20,
            "weight_decay": 0.0007,
            "save_period": 10,
        }

    raise ValueError(f"Profil d'entrainement inconnu: {profile}")


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
    patience: int = DEFAULT_PATIENCE,
    training_profile: str = DEFAULT_TRAINING_PROFILE,
    multi_scale: bool | None = None,
) -> Path:
    """Lance l'entraînement et retourne le path du meilleur modèle.

    Args:
        data: Path vers ``dataset.yaml`` YOLO.
        model: Poids de départ (ex: ``yolov8m.pt``).
        epochs: Nombre d'epochs.
        imgsz: Résolution d'entraînement.
        batch: Batch size.
        out_dir: Dossier projet Ultralytics.
        device: Device Ultralytics (``0`` ou ``cpu``).
        aug_preset: Preset d'augmentation (``baseline`` ou ``robust``).
        patience: Patience pour l'early stopping.
        training_profile: Profil global d'entrainement.
        multi_scale: Force multi-scale on/off. ``None`` keeps profile default.

    Returns:
        Path vers ``best.pt``.
    """
    from ultralytics import YOLO

    data_path = Path(data)
    if not data_path.exists():
        print(f"[ERREUR] dataset.yaml introuvable : {data_path}")
        print("  Lance d'abord : python src/prepare_yolo_dataset.py outputs/flatten")
        sys.exit(1)

    data_role = _dataset_role(data_path)
    resolved_preset, aug_cfg = _augmentation_config(aug_preset, data_role)
    resolved_profile, profile_cfg = _training_profile_config(
        training_profile,
        epochs,
        imgsz,
        batch,
        patience,
        multi_scale,
    )

    final_epochs = int(profile_cfg["epochs"])
    final_imgsz = int(profile_cfg["imgsz"])
    final_batch = int(profile_cfg["batch"])
    final_patience = int(profile_cfg["patience"])

    print(f"\n{'=' * 60}")
    print(f"[TRAIN] Modèle    : {model}")
    print(f"[TRAIN] Dataset   : {data_path}")
    print(f"[TRAIN] Rôle data : {data_role}")
    print(f"[TRAIN] Profil    : {resolved_profile} (arg={training_profile})")
    print(f"[TRAIN] Epochs    : {final_epochs}")
    print(f"[TRAIN] Image sz  : {final_imgsz}px")
    print(f"[TRAIN] Batch     : {final_batch}")
    print(f"[TRAIN] Patience  : {final_patience}")
    print(f"[TRAIN] MultiScale: {profile_cfg['multi_scale']}")
    print(f"[TRAIN] Augment   : {resolved_preset} (arg={aug_preset})")
    print(f"{'=' * 60}\n")

    if data_role == "axis" and resolved_preset in {"baseline", "robust"}:
        print("[WARN] Les presets baseline/robust appliquent rotations/flips.")
        print("[WARN] Pour top/left/bottom/right cela casse la sémantique des classes.")
        print("[WARN] Recommandé: --aug-preset axis_markers (ou auto).")

    if resolved_profile == "high_performance":
        print("[INFO] Profil high_performance actif: entrainement plus long et plus couteux.")
        print("[INFO] Recommande sur GPU CUDA avec VRAM confortable.")
        if profile_cfg["multi_scale"]:
            print("[WARN] multi_scale=True peut provoquer des crashes selon la version Torch.")

    yolo = YOLO(model)  # télécharge automatiquement si absent

    train_kwargs = {
        "data": str(data_path.resolve()),
        "device": device,  # "0"=GPU 0, "cpu"=CPU
        "project": out_dir,
        "name": "yolo_impacts",
        "exist_ok": True,
        # ── Seuils de détection ────────────────────────────────────────────
        "conf": 0.25,
        "iou": 0.5,
        # ── Misc ──────────────────────────────────────────────────────────
        "plots": True,
        "save": True,
        "verbose": True,
    }
    train_kwargs.update(aug_cfg)
    train_kwargs.update(profile_cfg)

    yolo.train(**train_kwargs)

    best_weights = Path(out_dir) / "yolo_impacts" / "weights" / "best.pt"
    print("\n[OK] Entraînement terminé.")
    print(f"[OK] Meilleurs poids → {best_weights}")
    print("\nPour détecter les impacts :")
    print("  python src/detect_impacts_yolo.py outputs/flatten --show")
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
    ap.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Patience early stopping (défaut: {DEFAULT_PATIENCE})",
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"Dossier de sortie (défaut: {DEFAULT_OUT})")
    ap.add_argument(
        "--device", default=DEFAULT_DEVICE, help="Device : '0' pour GPU, 'cpu' pour CPU (défaut: 0)"
    )
    ap.add_argument(
        "--aug-preset",
        default=DEFAULT_AUG_PRESET,
        choices=["auto", "axis_markers", "baseline", "robust"],
        help="Preset d'augmentation (défaut: auto)",
    )
    ap.add_argument(
        "--training-profile",
        default=DEFAULT_TRAINING_PROFILE,
        choices=["standard", "high_performance"],
        help="Profil d'entrainement global (défaut: standard)",
    )
    ap.add_argument(
        "--multi-scale",
        choices=["auto", "on", "off"],
        default="auto",
        help="Force le multi-scale (auto=profil, on/off=override explicite)",
    )
    args = ap.parse_args()

    multi_scale_override: bool | None
    if args.multi_scale == "on":
        multi_scale_override = True
    elif args.multi_scale == "off":
        multi_scale_override = False
    else:
        multi_scale_override = None

    train(
        args.data,
        args.model,
        args.epochs,
        args.imgsz,
        args.batch,
        args.out,
        args.device,
        args.aug_preset,
        args.patience,
        args.training_profile,
        multi_scale_override,
    )
