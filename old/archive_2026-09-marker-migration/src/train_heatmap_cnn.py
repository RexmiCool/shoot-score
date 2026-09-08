"""Train a lightweight dense heatmap CNN for impact detection.

The model predicts a full-resolution heatmap from a resized grayscale target
image.  This script expects datasets produced by ``prepare_heatmap_dataset.py``.

Key improvements over the original training script:
- Strong on-the-fly data augmentation (rotation, flip, brightness, noise) — critical
  for a small dataset (~65 images) and rotation-invariant circular targets.
- AdamW optimiser with cosine annealing LR and linear warm-up.
- Combined weighted MSE + weighted BCE loss for better peak/background separation.
- Gradient clipping to stabilise training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.impact_heatmap_cnn import ImpactHeatmapCNN


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for heatmap training.

    Attributes:
        epochs: Total training epochs.
        batch_size: Mini-batch size.
        learning_rate: Peak learning rate after warm-up.
        weight_decay: L2 regularisation for AdamW.
        warmup_epochs: Linear LR warm-up duration.
        positive_threshold: Target value above which a pixel is "positive".
        positive_weight: Loss weight multiplier on positive pixels.
        bce_weight: Weight of the BCE term in the combined loss (0 = pure MSE).
        clip_grad: Max gradient norm for gradient clipping.
    """

    epochs: int = 150
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    positive_threshold: float = 0.30
    positive_weight: float = 50.0
    bce_weight: float = 0.3
    clip_grad: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _apply_spatial_aug(
    image: np.ndarray,
    heatmap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply identical spatial transforms to image and heatmap.

    Applies random horizontal/vertical flips and a random 90°-multiple
    rotation.  These operations are safe for circular targets: the
    distribution of impacts is rotation-invariant.

    Args:
        image: Grayscale image array, shape ``(H, W)``.
        heatmap: Float32 heatmap array, shape ``(H, W)``.

    Returns:
        Augmented ``(image, heatmap)`` pair.
    """
    # Random flip along each axis
    if random.random() < 0.5:
        image = np.fliplr(image).copy()
        heatmap = np.fliplr(heatmap).copy()
    if random.random() < 0.5:
        image = np.flipud(image).copy()
        heatmap = np.flipud(heatmap).copy()

    # Random 90°-multiple rotation (0, 90, 180, 270)
    k = random.randint(0, 3)
    if k:
        image = np.rot90(image, k).copy()
        heatmap = np.rot90(heatmap, k).copy()

    return image, heatmap


def _apply_photometric_aug(image: np.ndarray) -> np.ndarray:
    """Apply random brightness, contrast, and Gaussian noise to image only.

    Args:
        image: Float32 grayscale image in ``[0, 1]``, shape ``(H, W)``.

    Returns:
        Augmented float32 image clamped to ``[0, 1]``.
    """
    # Brightness shift ± 0.15
    image = image + random.uniform(-0.15, 0.15)
    # Contrast scale 0.8 – 1.2
    mean = image.mean()
    image = mean + (image - mean) * random.uniform(0.8, 1.2)
    # Gaussian noise σ ≤ 0.04
    image = image + np.random.normal(0.0, random.uniform(0.0, 0.04), image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class HeatmapDataset(Dataset):
    """Dataset of grayscale images and dense heatmap targets.

    Args:
        root: Dataset root directory containing ``images/<split>/`` and
            ``heatmaps/<split>/`` sub-folders.
        split: Dataset split name (``train`` or ``val``).
        augment: Whether to apply random augmentation.
    """

    def __init__(self, root: Path, split: str, augment: bool = False) -> None:
        """Initialise a split dataset.

        Args:
            root: Dataset root directory.
            split: Dataset split name (``train`` or ``val``).
            augment: If ``True``, random augmentation is applied on each
                ``__getitem__`` call.
        """
        self.image_dir = root / "images" / split
        self.heatmap_dir = root / "heatmaps" / split
        self.sample_names = sorted(path.stem for path in self.image_dir.glob("*.png"))
        self.augment = augment

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sample_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and optionally augment one sample.

        Args:
            index: Sample index.

        Returns:
            Tuple ``(image, heatmap)`` each with shape ``(1, H, W)``.

        Raises:
            FileNotFoundError: If image or heatmap file cannot be read.
        """
        sample = self.sample_names[index]
        image_path = self.image_dir / f"{sample}.png"
        heatmap_path = self.heatmap_dir / f"{sample}.npy"

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        if not heatmap_path.exists():
            raise FileNotFoundError(f"Cannot read heatmap: {heatmap_path}")

        heatmap = np.load(heatmap_path)
        image_f = image.astype(np.float32) / 255.0

        if self.augment:
            image_f, heatmap = _apply_spatial_aug(image_f, heatmap)
            image_f = _apply_photometric_aug(image_f)

        return (
            torch.from_numpy(image_f[None, :, :]),
            torch.from_numpy(heatmap.astype(np.float32)[None, :, :]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────


def combined_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    positive_weight: float,
    positive_threshold: float,
    bce_weight: float,
) -> torch.Tensor:
    """Compute a combined weighted-MSE and weighted-BCE loss.

    The MSE term regresses exact Gaussian peak values; the BCE term
    provides a stronger gradient signal for positive/negative separation,
    which helps with the severe class imbalance (few impact pixels vs.
    background).

    Args:
        prediction: Predicted sigmoid heatmap, shape ``(B, 1, H, W)``.
        target: Ground-truth heatmap in ``[0, 1]``, same shape.
        positive_weight: Loss weight multiplier for positive pixels.
        positive_threshold: Target value threshold for the positive class.
        bce_weight: Contribution of the BCE term (0 = pure MSE).

    Returns:
        Scalar combined loss tensor.
    """
    pos_mask = target > positive_threshold
    weight = torch.ones_like(target)
    weight[pos_mask] = positive_weight

    mse = ((prediction - target) ** 2 * weight).mean()

    eps = 1e-6
    bce = (
        -(target * torch.log(prediction + eps) + (1.0 - target) * torch.log(1.0 - prediction + eps))
        * weight
    )
    bce = bce.mean()

    return (1.0 - bce_weight) * mse + bce_weight * bce


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a linear warm-up + cosine annealing LR schedule.

    Args:
        optimizer: The optimiser whose LR will be scheduled.
        warmup_epochs: Number of linear warm-up epochs.
        total_epochs: Total number of training epochs.

    Returns:
        A ``LambdaLR`` scheduler.
    """

    def lr_lambda(epoch: int) -> float:
        """Return LR multiplier for *epoch* (0-indexed).

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            LR multiplier in ``(0, 1]``.
        """
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    model: ImpactHeatmapCNN,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> float:
    """Evaluate average combined loss on a dataloader.

    Args:
        model: Heatmap model.
        loader: DataLoader to evaluate.
        device: Compute device.
        config: Training hyperparameters.

    Returns:
        Mean combined loss over all batches.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for image, target in loader:
            image = image.to(device)
            target = target.to(device)
            prediction = model(image)
            loss = combined_loss(
                prediction,
                target,
                positive_weight=config.positive_weight,
                positive_threshold=config.positive_threshold,
                bce_weight=config.bce_weight,
            )
            total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train(
    dataset_root: Path,
    output_root: Path,
    config: TrainingConfig,
    resume: Optional[Path] = None,
) -> Path:
    """Train dense heatmap model and save best checkpoint.

    Training uses AdamW with a linear warm-up + cosine annealing schedule
    and aggressive on-the-fly augmentation on the training split.

    Args:
        dataset_root: Root folder produced by ``prepare_heatmap_dataset.py``.
        output_root: Directory where checkpoints and metadata are written.
        config: Training hyperparameters.
        resume: Optional path to a ``.pt`` checkpoint to resume from.

    Returns:
        Path to the output directory.

    Raises:
        ValueError: If train or validation split is empty.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    train_dataset = HeatmapDataset(dataset_root, "train", augment=True)
    val_dataset = HeatmapDataset(dataset_root, "val", augment=False)

    if len(train_dataset) == 0:
        raise ValueError("Training split is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")

    print(f"[DATA] train={len(train_dataset)}  val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    model = ImpactHeatmapCNN().to(device)
    if resume is not None:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state)
        print(f"[RESUME] Loaded checkpoint: {resume}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] ImpactHeatmapCNN — {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, config.warmup_epochs, config.epochs)

    output_root.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0

        for image, target in tqdm(
            train_loader, desc=f"[TRAIN {epoch:03d}/{config.epochs}]", leave=False
        ):
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            prediction = model(image)
            loss = combined_loss(
                prediction,
                target,
                positive_weight=config.positive_weight,
                positive_threshold=config.positive_threshold,
                bce_weight=config.bce_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()

            train_loss += float(loss.item())

        scheduler.step()

        train_loss /= max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, device, config)
        lr_now = scheduler.get_last_lr()[0]

        improved = val_loss < best_val
        marker = "  ← best" if improved else ""
        print(
            f"[EPOCH {epoch:03d}]  train={train_loss:.5f}  val={val_loss:.5f}"
            f"  lr={lr_now:.2e}{marker}"
        )

        torch.save(model.state_dict(), output_root / "model_last.pt")
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), output_root / "model_best.pt")
        else:
            no_improve += 1

    metadata = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "positive_threshold": config.positive_threshold,
        "positive_weight": config.positive_weight,
        "bce_weight": config.bce_weight,
        "clip_grad": config.clip_grad,
        "loss": "combined_wmse_wbce",
        "model": "ImpactHeatmapCNN",
        "best_val_loss": best_val,
    }
    (output_root / "training_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Best val loss: {best_val:.5f}")
    print(f"[OK] Artifacts saved to: {output_root.resolve()}")
    return output_root


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Train dense impact heatmap CNN (optimised)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", type=Path, help="Path to data/impacts_heatmap")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--warmup", type=int, default=5, help="LR warm-up epochs")
    parser.add_argument("--pos-thr", type=float, default=0.30)
    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument(
        "--bce-weight", type=float, default=0.3, help="BCE term weight (0=pure MSE)"
    )
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/impacts_heatmap"),
        help="Output directory for checkpoints",
    )
    return parser


def main() -> None:
    """Script entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_epochs=args.warmup,
        positive_threshold=args.pos_thr,
        positive_weight=args.pos_weight,
        bce_weight=args.bce_weight,
        clip_grad=args.clip_grad,
    )
    train(dataset_root=args.dataset, output_root=args.out, config=config, resume=args.resume)


if __name__ == "__main__":
    main()
