"""
Pipeline complet : photo brute → détection des impacts (YOLO).

Enchaîne les 3 étapes :
  1. flatten_target   : correction de perspective + crop 1056×1056px
  2. detect_rings     : calibration des anneaux + étalonnage mm/px
  3. detect_impacts_yolo : détection des impacts + scoring

Produit dans <out>/<stem>/ :
  <stem>_flat.jpg            image mise à plat
  <stem>_rings.jpg           anneaux détectés (debug visuel)
  <stem>_rings.json          étalonnage
  <stem>_impacts_yolo.jpg    résultat final avec les impacts annotés  ← sorti principal
  <stem>_impacts_yolo.json   données structurées

Usage :
    python src/pipeline.py <photo.jpg>
    python src/pipeline.py data/raw/20260303_225933.jpg --show
    python src/pipeline.py data/raw/ --out outputs --show
    python src/pipeline.py data/raw/ --weights models/yolo_impacts/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from flatten_target import process as flatten
from detect_rings import detect_rings
from detect_impacts_yolo import detect_impacts_yolo, _auto_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_WEIGHTS = "runs/detect/models/yolo_impacts/weights/best.pt"
DEFAULT_OUT = "outputs"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.4


def run_pipeline(
    image_path: Path,
    out_root: Path,
    weights: str,
    conf: float,
    iou: float,
    device: str,
    debug: bool,
    show: bool,
) -> Path | None:
    """Traite une seule image de bout en bout. Retourne le Path du résultat final."""

    stem = image_path.stem
    out_dir = out_root / stem

    print(f"\n{'#' * 62}")
    print(f"# {image_path.name}")
    print(f"{'#' * 62}")

    # ── Étape 1 : mise à plat ─────────────────────────────────────────────────
    flat_annot = flatten(image_path, output_dir=str(out_root), debug=debug)
    if flat_annot is None:
        print(f"[SKIP] Mise à plat échouée pour {image_path.name}")
        return None

    # flatten() renvoie le _flat_annot.jpg ; on veut le _flat.jpg
    flat_path = out_dir / f"{stem}_flat.jpg"
    if not flat_path.exists():
        print(f"[SKIP] flat.jpg introuvable : {flat_path}")
        return None

    # ── Étape 2 : détection des anneaux ───────────────────────────────────────
    rings_result = detect_rings(flat_path, debug=debug)
    if rings_result is None:
        print(f"[WARN] Anneaux non détectés — scoring approximatif.")

    # ── Étape 3 : détection des impacts (YOLO) ────────────────────────────────
    result = detect_impacts_yolo(
        flat_path,
        weights=weights,
        conf_thr=conf,
        iou_thr=iou,
        device=device,
        show=show,
    )
    return result


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        f
        for f in path.glob("*.*")
        if f.suffix.lower() in IMAGE_EXTENSIONS
        and not f.stem.startswith("_")
        and "_flat" not in f.stem
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pipeline complet : photo brute -> impacts détectés (YOLO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("image", help="Photo brute (JPG) ou dossier contenant des photos")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Dossier de sortie racine")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Poids YOLO entraînés")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Seuil de confiance YOLO")
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU, help="Seuil IoU NMS")
    ap.add_argument("--device", default=None, help="Device : '0'=GPU, 'cpu'=CPU (auto si absent)")
    ap.add_argument("--debug", action="store_true", help="Sauvegarde les images intermédiaires")
    ap.add_argument("--show", action="store_true", help="Ouvre l'image de résultat final")
    args = ap.parse_args()

    device = args.device if args.device else _auto_device()
    print(f"[DEVICE] {device}  |  weights : {args.weights}")

    src = Path(args.image)
    if not src.exists():
        print(f"[ERREUR] Chemin introuvable : {src}")
        sys.exit(1)

    images = _collect_images(src)
    if not images:
        print(f"[ERREUR] Aucune image trouvée dans : {src}")
        sys.exit(1)

    out_root = Path(args.out)
    print(f"{len(images)} image(s) a traiter.\n")

    ok, fail = 0, 0
    for img_path in images:
        result = run_pipeline(
            image_path=img_path,
            out_root=out_root,
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            device=device,
            debug=args.debug,
            show=args.show,
        )
        if result:
            ok += 1
        else:
            fail += 1

    print(f"\n{'=' * 62}")
    print(f"Termine : {ok} OK  /  {fail} echec(s)")
    print(f"Resultats dans : {out_root.resolve()}")
