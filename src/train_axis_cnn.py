"""Train one CNN per target-marker position.

Usage:
    python src/train_axis_cnn.py --data data/axis_markers --epochs 80

The command writes four checkpoints under ``models/axis_markers/``. Each
checkpoint predicts marker presence and a normalized position offset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.axis_marker_cnn import AxisMarkerCNN

POSITIONS = ("top", "left", "bottom", "right")


class AxisDataset(Dataset):
    def __init__(self, root: Path, position: str, split: str, augment: bool) -> None:
        self.root = root / position / split
        self.images = sorted(self.root.glob("*.png"))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        image = image.astype(np.float32) / 255.0
        label = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
            if random.random() < 0.35:
                image = np.clip(image * random.uniform(0.75, 1.25) + random.uniform(-0.08, 0.08), 0, 1)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0, 0.025, image.shape), 0, 1)
        image = image.astype(np.float32, copy=False)
        target = torch.tensor([label["present"], label["dx"], label["dy"]], dtype=torch.float32)
        return torch.from_numpy(image[None]), target


def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(train):
            output = model(images)
            presence_loss = bce(output[:, 0], targets[:, 0])
            positive = targets[:, 0] > 0.5
            offset_loss = smooth_l1(output[:, 1:][positive], targets[:, 1:][positive]) if positive.any() else output[:, 1:].sum() * 0
            loss = presence_loss + 2.0 * offset_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
        total_loss += float(loss.detach()) * len(images)
        predicted = (torch.sigmoid(output[:, 0]) >= 0.5).float()
        correct += int((predicted == targets[:, 0]).sum())
        count += len(images)
    return total_loss / max(1, count), correct / max(1, count)


def train_position(position: str, args: argparse.Namespace, device: torch.device) -> Path:
    train_set = AxisDataset(args.data, position, "train", augment=True)
    val_set = AxisDataset(args.data, position, "val", augment=False)
    if not train_set or not val_set:
        raise RuntimeError(f"Dataset incomplet pour {position}: train={len(train_set)} val={len(val_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = AxisMarkerCNN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best = float("inf")
    output = args.out / f"axis_marker_{position}.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, False)
        scheduler.step()
        if val_loss < best:
            best = val_loss
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[{position}] epoch={epoch:03d} train={train_loss:.4f}/{train_acc:.2%} val={val_loss:.4f}/{val_acc:.2%}")
    print(f"[OK] {position}: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/axis_markers"))
    parser.add_argument("--out", type=Path, default=Path("models/axis_markers"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    for position in POSITIONS:
        train_position(position, args, device)


if __name__ == "__main__":
    main()
