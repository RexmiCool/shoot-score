"""Test heatmap model on 3 raw images with full preprocessing pipeline."""

from pathlib import Path
import cv2
import numpy as np
import torch
import sys

import json

sys.path.insert(0, str(Path(__file__).parent))
from models.impact_heatmap_cnn import ImpactHeatmapCNN


def test_heatmap_on_images():
    """Load 3 images, preprocess, and run heatmap model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Load model
    model = ImpactHeatmapCNN(base_ch=28)
    weights_path = Path("models/impacts_heatmap/model_best.pt")
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"[MODEL] Loaded {weights_path}")

    # Find 3 images from curation dataset
    images_dir = Path("outputs/curation_cible")
    flat_images = sorted(images_dir.rglob("*_flat.jpg"))[:3]

    if not flat_images:
        print("[ERROR] No flat images found in outputs/curation_cible/")
        return

    print(f"[IMAGES] Found {len(flat_images)} image(s) to test\n")

    with torch.no_grad():
        for img_path in flat_images:
            print(f"{'=' * 70}")
            print(f"Image: {img_path.parent.name} / {img_path.name}")
            print(f"{'=' * 70}")

            # Load flat image (already perspective-corrected and 1056×1056)
            flat_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if flat_img is None:
                print("[SKIP] Could not load image")
                continue

            print(f"  Flat image shape: {flat_img.shape}")

            # Resize to 512×512
            img_512 = cv2.resize(flat_img, (512, 512), interpolation=cv2.INTER_AREA)
            img_norm = img_512.astype(np.float32) / 255.0

            # Convert to tensor: (1, 1, 512, 512)
            x = torch.from_numpy(img_norm[None, None, :, :]).to(device)

            # Forward pass
            heatmap_pred = model(x).squeeze().cpu().numpy()  # (512, 512)

            print(f"  Heatmap shape: {heatmap_pred.shape}")
            print(f"  Heatmap range: [{heatmap_pred.min():.4f}, {heatmap_pred.max():.4f}]")
            print(f"  Heatmap mean: {heatmap_pred.mean():.4f}")

            # Find peaks
            heatmap_8bit = (heatmap_pred * 255).astype(np.uint8)

            # Simple peak detection: find local maxima above 0.5
            peaks = np.where(heatmap_pred > 0.5)
            n_peaks = len(peaks[0])
            print(f"  Pixels > 0.5: {n_peaks}")

            if n_peaks > 0:
                peak_values = heatmap_pred[peaks]
                print(f"  Peak range: [{peak_values.min():.4f}, {peak_values.max():.4f}]")
                print(f"  Peak mean: {peak_values.mean():.4f}")

            # Load ground-truth labels
            labels_path = img_path.parent / f"{img_path.stem.replace('_flat', '')}_labels.json"
            if labels_path.exists():
                with open(labels_path) as f:
                    labels = json.load(f)
                n_impacts = len(labels.get("impacts", []))
                print(f"  Ground truth impacts: {n_impacts}")

            # Save visualization
            out_path = Path("outputs") / f"heatmap_test_{img_path.parent.name}.png"
            cv2.imwrite(str(out_path), heatmap_8bit)
            print(f"  Saved heatmap visualization: {out_path}")
            print()


if __name__ == "__main__":
    test_heatmap_on_images()
