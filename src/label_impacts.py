"""
Outil de labélisation des impacts (ground truth).

Ouvre l'image *_flat.jpg (ou *_rings.jpg si disponible pour avoir les anneaux
dessinés en référence) dans une fenêtre interactive :

  Clic gauche   → ajouter un impact au point cliqué
  Clic droit    → supprimer le point le plus proche (dans un rayon de 30 px)
  R             → recharger depuis le fichier JSON existant
  Z             → annuler le dernier ajout
  S             → sauvegarder (aussi automatique à la fermeture)
  Q / Échap     → quitter (sauvegarde avant de quitter)

Le résultat est sauvegardé dans le même dossier que l'image flat :
  <stem>_labels.json

Usage :
    python src/label_impacts.py <flat.jpg>
    python src/label_impacts.py outputs/flatten/20260303_225933/20260303_225933_flat.jpg
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

MARK_RADIUS = 12  # rayon du cercle de marquage (px affiché)
MARK_COLOR = (0, 255, 255)  # jaune vif
DELETE_RADIUS = 30  # distance max pour la suppression au clic droit
FONT = cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_MAX = 1056  # taille max d'affichage (la flat est déjà 1056×1056)


class Labeler:
    def __init__(
        self,
        flat_path: Path,
        accepted_dir: Path | None = None,
        rejected_dir: Path | None = None,
    ):
        self.flat_path = flat_path
        self.stem = flat_path.stem.removesuffix("_flat")
        self.json_path = flat_path.parent / f"{self.stem}_labels.json"
        self.accepted_dir = accepted_dir
        self.rejected_dir = rejected_dir

        # Utiliser *_rings.jpg s'il existe (meilleur contexte visuel)
        rings_path = flat_path.parent / f"{self.stem}_rings.jpg"
        src_path = rings_path if rings_path.exists() else flat_path
        self.base_img = cv2.imread(str(src_path))
        if self.base_img is None:
            raise FileNotFoundError(f"Image introuvable : {src_path}")

        h, w = self.base_img.shape[:2]
        self.scale = min(1.0, DISPLAY_MAX / max(h, w))
        self.disp_w = int(w * self.scale)
        self.disp_h = int(h * self.scale)

        self.impacts: list[tuple[int, int]] = []
        self._load()

        self.win = "Labélisation des impacts  [Clic G=ajouter | Clic D=suppr | S=save | Z=annuler | Q=quitter]"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.win, self._on_mouse)

    # ── I/O ────────────────────────────────────────────────────────────────────

    def _load(self):
        if self.json_path.exists():
            with open(self.json_path, encoding="utf-8") as f:
                data = json.load(f)
            self.impacts = [(int(p["cx_px"]), int(p["cy_px"])) for p in data.get("impacts", [])]
            print(f"[LOAD] {len(self.impacts)} impact(s) chargé(s) depuis {self.json_path.name}")
        else:
            print(f"[NEW ] Pas de fichier existant — on repart de zéro.")

    def _save(self):
        data = {
            "source": self.flat_path.name,
            "impacts": [{"cx_px": cx, "cy_px": cy} for cx, cy in self.impacts],
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] {len(self.impacts)} impact(s) → {self.json_path.name}")

    # ── Rendu ──────────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        disp = self.base_img.copy()
        h, w = disp.shape[:2]

        for i, (cx, cy) in enumerate(self.impacts, 1):
            cv2.circle(disp, (cx, cy), MARK_RADIUS, MARK_COLOR, 2, cv2.LINE_AA)
            cv2.circle(disp, (cx, cy), 3, MARK_COLOR, -1)
            cv2.putText(
                disp, str(i), (cx + MARK_RADIUS + 3, cy + 5), FONT, 0.50, MARK_COLOR, 1, cv2.LINE_AA
            )

        # Bandeau info
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, h - 28), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, disp, 0.4, 0, disp)
        cv2.putText(
            disp,
            f"Impacts : {len(self.impacts)}  |  "
            "Clic G=ajouter  Clic D=suppr  S=save  Z=annuler  Q=quitter",
            (8, h - 8),
            FONT,
            0.40,
            (220, 220, 220),
            1,
        )

        if self.scale != 1.0:
            disp = cv2.resize(disp, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        return disp

    # ── Souris ─────────────────────────────────────────────────────────────────

    def _display_to_image(self, xd: int, yd: int) -> tuple[int, int]:
        return (int(xd / self.scale), int(yd / self.scale))

    def _on_mouse(self, event, xd, yd, flags, param):
        xi, yi = self._display_to_image(xd, yd)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.impacts.append((xi, yi))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.impacts:
                dists = [np.hypot(xi - cx, yi - cy) for cx, cy in self.impacts]
                nearest_i = int(np.argmin(dists))
                if dists[nearest_i] <= DELETE_RADIUS / self.scale:
                    removed = self.impacts.pop(nearest_i)
                    print(f"[DEL ] #{nearest_i + 1} supprimé → ({removed[0]},{removed[1]})")

    # ── Boucle principale ──────────────────────────────────────────────────────

    def move_to_status(self, status: str) -> str:
        """Move the flat image and JSON label to the accepted or rejected folder."""
        if status == "accept" and self.accepted_dir is not None:
            self.accepted_dir.mkdir(parents=True, exist_ok=True)
            image_dest = self.accepted_dir / self.flat_path.name
            json_dest = self.accepted_dir / self.json_path.name
            if image_dest.exists():
                image_dest.unlink()
            if self.json_path.exists():
                if json_dest.exists():
                    json_dest.unlink()
                self.json_path.replace(json_dest)
            self.flat_path.replace(image_dest)
            print(f"[MOVE] {self.flat_path.name} -> {image_dest}")
            print(f"[MOVE] {self.json_path.name} -> {json_dest}")
            return "next"
        if status == "reject" and self.rejected_dir is not None:
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
            image_dest = self.rejected_dir / self.flat_path.name
            json_dest = self.rejected_dir / self.json_path.name
            if image_dest.exists():
                image_dest.unlink()
            if self.json_path.exists():
                if json_dest.exists():
                    json_dest.unlink()
                self.json_path.replace(json_dest)
            self.flat_path.replace(image_dest)
            print(f"[REJECT] {self.flat_path.name} -> {image_dest}")
            print(f"[REJECT] {self.json_path.name} -> {json_dest}")
            return "next"
        return "stay"

    def run(self) -> str:
        """
        Lance la boucle interactive. Retourne 'next', 'prev' ou 'quit'
        selon la touche de sortie (pour navigation entre images).
        """
        print(
            f"\n[{self.flat_path.name}] Clic G=ajouter  Clic D=suppr  "
            "A=accepter  X=rejetter  S=save  Z=annuler  N=suivante  P=précédente  Q=quitter\n"
        )
        action = "next"
        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                self._save()
                action = "quit"
                break
            elif key in (ord("n"), ord("N")):
                self._save()
                action = "next"
                break
            elif key in (ord("p"), ord("P")):
                self._save()
                action = "prev"
                break
            elif key in (ord("a"), ord("A")):
                self._save()
                action = "accept"
                break
            elif key in (ord("x"), ord("X")):
                self._save()
                action = "reject"
                break
            elif key in (ord("s"), ord("S")):
                self._save()
            elif key in (ord("z"), ord("Z")):
                if self.impacts:
                    removed = self.impacts.pop()
                    print(f"[UNDO] Supprimé → ({removed[0]},{removed[1]})")
            elif key in (ord("r"), ord("R")):
                self._load()

        cv2.destroyAllWindows()
        print(f"[FIN] {len(self.impacts)} impact(s) sauvegardé(s) dans {self.json_path}\n")
        return action


# ── Collecte des images ────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images(path: Path) -> list[Path]:
    """Retourne la liste des *_flat.jpg à labéliser (fichier ou dossier)."""
    if path.is_file():
        return [path]
    results = sorted(
        f
        for f in path.rglob("*_flat.*")
        if f.suffix.lower() in IMAGE_EXTENSIONS and not f.stem.startswith("_")
    )
    if not results:
        results = sorted(f for f in path.glob("*.*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Labéliseur d'impacts. Accepte un fichier ou un dossier."
    )
    ap.add_argument("path", help="Image *_flat.jpg ou dossier")
    ap.add_argument(
        "--skip-done",
        action="store_true",
        help="Ignorer les images déjà labélisées (*_labels.json existant)",
    )
    ap.add_argument("--accepted-dir", type=Path, default=None, help="Dossier de destination après validation manuelle")
    ap.add_argument("--rejected-dir", type=Path, default=None, help="Dossier de destination si l'image est rejetée")
    args = ap.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"[ERREUR] Chemin introuvable : {root}")
        sys.exit(1)

    images = _collect_images(root)
    if not images:
        print(f"[ERREUR] Aucune image *_flat trouvée dans {root}")
        sys.exit(1)

    if args.skip_done:
        before = len(images)
        images = [
            p
            for p in images
            if not (p.parent / (p.stem.removesuffix("_flat") + "_labels.json")).exists()
        ]
        print(f"[FILTRE] {before - len(images)} déjà labélisée(s) ignorée(s)")

    print(f"{len(images)} image(s) à labéliser.")

    idx = 0
    while 0 <= idx < len(images):
        image_path = images[idx]
        print(f"\n--- Image {idx + 1}/{len(images)} ----------------------------")
        try:
            labeler = Labeler(
                image_path,
                accepted_dir=args.accepted_dir,
                rejected_dir=args.rejected_dir,
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            images.pop(idx)
            if idx >= len(images):
                idx = max(0, len(images) - 1)
            continue

        action = labeler.run()
        if action == "accept":
            labeler.move_to_status("accept")
            images.pop(idx)
            if idx >= len(images):
                idx = max(0, len(images) - 1)
            continue
        if action == "reject":
            labeler.move_to_status("reject")
            images.pop(idx)
            if idx >= len(images):
                idx = max(0, len(images) - 1)
            continue
        if action == "quit":
            print("Labélisation interrompue.")
            break
        elif action == "next":
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)

    print(f"\nLabélisation terminée ({min(idx, len(images))}/{len(images)} image(s) traitée(s)).")
