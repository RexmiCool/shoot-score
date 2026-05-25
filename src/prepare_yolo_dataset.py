"""
Prépare le dataset YOLO depuis les fichiers *_labels.json produits par label_impacts.py.

Structure de sortie :
  data/yolo/
    images/
      train/   ← 80% des images
      val/     ← 20% des images
    labels/
      train/   ← fichiers .txt YOLO correspondants
      val/
    dataset.yaml

Format YOLO (1 ligne par impact) :
  <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
  class_id = 0 (une seule classe : "impact")
  Toutes les coordonnées normalisées par (largeur, hauteur) de l'image.

La bbox autour de chaque impact a une taille fixe de BBOX_SIZE_MM × BBOX_SIZE_MM
(convertie en pixels via le mm_per_px du *_rings.json associé).

Usage :
    python src/prepare_yolo_dataset.py outputs/flatten
    python src/prepare_yolo_dataset.py outputs/flatten --out data/yolo --val-ratio 0.2
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from flatten_target import MM_PER_PX_OUT

# ── Paramètres ────────────────────────────────────────────────────────────────
BBOX_SIZE_MM = 18.0  # taille de la bbox autour de chaque impact (mm)
# 18mm = 2× le calibre 9mm → halo de compression inclus
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def prepare_dataset(
    labels_root: Path,
    out_dir: Path,
    val_ratio: float = 0.20,
) -> Path:
    """
    Build a YOLO dataset from ``*_labels.json`` files.

    Args:
        labels_root: Root directory containing ``*_labels.json`` files and
            corresponding flattened images.
        out_dir: Output directory where YOLO ``images/``, ``labels/`` and
            ``dataset.yaml`` are written.
        val_ratio: Fraction of samples reserved for the validation split.

    Returns:
        Path to the generated ``dataset.yaml`` file.

    Raises:
        SystemExit: If no valid labels or images are found.
    """
    label_files = sorted(labels_root.rglob("*_labels.json"))
    if not label_files:
        print(f"[ERREUR] Aucun *_labels.json trouvé sous {labels_root}")
        sys.exit(1)
    print(f"[LOAD] {len(label_files)} fichier(s) labels trouvé(s)")

    # ── Collecte des échantillons valides ─────────────────────────────────────
    samples = []
    n_impacts_total = 0
    for lp in label_files:
        stem = lp.stem.removesuffix("_labels")
        flat_path = lp.parent / f"{stem}_flat.jpg"
        if not flat_path.exists():
            for ext in IMAGE_EXTENSIONS:
                cand = list(lp.parent.glob(f"{stem}_flat{ext}"))
                if cand:
                    flat_path = cand[0]
                    break
        if not flat_path.exists():
            print(f"  [SKIP] flat introuvable pour {lp.name}")
            continue

        with open(lp, encoding="utf-8") as f:
            data = json.load(f)
        impacts = data.get("impacts", [])

        # mm/px depuis rings.json, fallback nominal
        rings_path = lp.parent / f"{stem}_rings.json"
        mm_per_px = MM_PER_PX_OUT
        if rings_path.exists():
            with open(rings_path, encoding="utf-8") as f:
                mm_per_px = float(json.load(f)["mm_per_px_calibre"])

        samples.append(
            {
                "flat_path": flat_path,
                "impacts": impacts,
                "mm_per_px": mm_per_px,
            }
        )
        n_impacts_total += len(impacts)
        print(f"  {flat_path.name:40s}  {len(impacts):3d} impact(s)  mm/px={mm_per_px:.5f}")

    if not samples:
        print("[ERREUR] Aucun échantillon valide.")
        sys.exit(1)

    print(f"\n[INFO] {len(samples)} image(s)  {n_impacts_total} impact(s) au total")

    # ── Split train / val ─────────────────────────────────────────────────────
    random.seed(SEED)
    shuffled = samples[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    n_train = len(shuffled) - n_val
    splits = {"train": shuffled[:n_train], "val": shuffled[n_train:]}
    print(f"[SPLIT] train={n_train}  val={n_val}")

    # ── Création des dossiers ─────────────────────────────────────────────────
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Copie des images + écriture des labels YOLO ───────────────────────────
    for split, split_samples in splits.items():
        n_written = 0
        for s in split_samples:
            flat_path = s["flat_path"]
            mm_per_px = s["mm_per_px"]

            # Copie de l'image
            dst_img = out_dir / "images" / split / flat_path.name
            shutil.copy2(flat_path, dst_img)

            # Dimensions de l'image
            img = cv2.imread(str(flat_path))
            if img is None:
                print(f"  [SKIP] impossible de lire {flat_path.name}")
                continue
            h, w = img.shape[:2]

            # Taille bbox normalisée
            bbox_px = BBOX_SIZE_MM / mm_per_px
            w_norm = bbox_px / w
            h_norm = bbox_px / h

            # Lignes YOLO : "0 cx cy bw bh" (tout normalisé)
            lines = []
            for imp in s["impacts"]:
                cx_px = float(imp["cx_px"])
                cy_px = float(imp["cy_px"])

                # Important pour les impacts en bord: on conserve le centre
                # annoté et on clippe la bbox aux limites de l'image.
                half_bbox = bbox_px * 0.5
                x1 = max(0.0, cx_px - half_bbox)
                y1 = max(0.0, cy_px - half_bbox)
                x2 = min(float(w), cx_px + half_bbox)
                y2 = min(float(h), cy_px + half_bbox)

                bw_px = x2 - x1
                bh_px = y2 - y1
                if bw_px <= 1.0 or bh_px <= 1.0:
                    continue

                cx_norm = ((x1 + x2) * 0.5) / w
                cy_norm = ((y1 + y2) * 0.5) / h
                bw_norm = bw_px / w
                bh_norm = bh_px / h

                lines.append(
                    f"0 {cx_norm:.6f} {cy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}"
                )

            label_name = flat_path.stem + ".txt"
            with open(out_dir / "labels" / split / label_name, "w") as f:
                f.write("\n".join(lines))
            n_written += 1

        print(f"  [{split:5s}] {n_written} image(s) écrites")

    # ── dataset.yaml ─────────────────────────────────────────────────────────
    yaml_path = out_dir / "dataset.yaml"
    yaml_content = (
        f"# YOLO dataset - detection d'impacts de balles\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['impact']\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[OK] Dataset prêt → {yaml_path}")
    print(f"     Lance maintenant : python src/train_yolo.py {yaml_path}")
    return yaml_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convertit les *_labels.json en dataset YOLO.")
    ap.add_argument("root", help="Dossier contenant les *_labels.json (ex: outputs/flatten)")
    ap.add_argument(
        "--out", default="data/yolo", help="Dossier de sortie du dataset (défaut: data/yolo)"
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Fraction des images en validation (défaut: 0.20)",
    )
    args = ap.parse_args()

    prepare_dataset(Path(args.root), Path(args.out), args.val_ratio)
