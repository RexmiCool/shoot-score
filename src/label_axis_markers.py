"""Manually label the four printed ``1`` markers on target photos.

Controls:
  Left click : place the next marker, or replace the selected marker
  Right click: remove the nearest marker
  1/2/3/4 : select top/left/bottom/right for the next click
  R : clear all markers
  S : save
  N/P : save and move next/previous image
  Q/Esc : save and quit

Labels are written next to each source image as ``<stem>_axis_labels.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from axis_marker_common import AXIS_NAMES, expected_axis_points

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
COLORS = [(0, 220, 255), (0, 255, 100), (255, 180, 0), (255, 80, 180)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


class AxisLabeler:
    def __init__(self, image_path: Path, screen_aspect: float, frame_ratio: float):
        self.image_path = image_path
        self.label_path = image_path.with_name(f"{image_path.stem}_axis_labels.json")
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise FileNotFoundError(image_path)
        h, w = self.image.shape[:2]
        self.scale = min(1.0, 1200 / max(w, h))
        self.points: list[dict[str, float] | None] = [None] * 4
        self.selected = 0
        self.expected = expected_axis_points(w, h, screen_aspect, frame_ratio)
        self._load()
        self.window = "Labelisation des quatre 1"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, int(w * self.scale), int(h * self.scale))
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _load(self) -> None:
        if not self.label_path.exists():
            return
        data = json.loads(self.label_path.read_text(encoding="utf-8"))
        markers = data.get("markers", {})
        self.points = [markers.get(name) for name in AXIS_NAMES]

    def save(self) -> None:
        h, w = self.image.shape[:2]
        payload = {
            "source": self.image_path.name,
            "image_width": w,
            "image_height": h,
            "markers": {name: self.points[i] for i, name in enumerate(AXIS_NAMES)},
        }
        self.label_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[SAVE] {self.label_path}")

    def _image_coords(self, x: int, y: int) -> tuple[float, float]:
        return x / self.scale, y / self.scale

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        ix, iy = self._image_coords(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[self.selected] = {"x": round(ix, 2), "y": round(iy, 2)}
            print(f"[SET] {AXIS_NAMES[self.selected]} = ({ix:.1f}, {iy:.1f})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            present = [
                (i, np.hypot(ix - p["x"], iy - p["y"]))
                for i, p in enumerate(self.points)
                if p is not None
            ]
            if present:
                index, distance = min(present, key=lambda item: item[1])
                if distance <= 70 / self.scale:
                    self.points[index] = None
                    self.selected = index

    def render(self) -> np.ndarray:
        image = self.image.copy()
        for i, name in enumerate(AXIS_NAMES):
            expected = self.expected[name]
            ex, ey = int(expected.x), int(expected.y)
            cv2.circle(image, (ex, ey), 38, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(image, f"attendu {name}", (ex - 42, ey - 44), FONT, 0.35, (150, 150, 150), 1)
            point = self.points[i]
            if point is None:
                continue
            x, y = int(point["x"]), int(point["y"])
            color = COLORS[i]
            cv2.circle(image, (x, y), 18, color, 2, cv2.LINE_AA)
            cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 30, 2)
            cv2.putText(image, f"{i + 1}:{name}", (x + 20, y + 5), FONT, 0.5, color, 1, cv2.LINE_AA)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, image.shape[0] - 34), (image.shape[1], image.shape[0]), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        text = f"selection={self.selected + 1}:{AXIS_NAMES[self.selected]} | clic=placer | 1-4=choisir | R=effacer | S=sauver | N/P | Q"
        cv2.putText(image, text, (8, image.shape[0] - 10), FONT, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
        if self.scale != 1:
            image = cv2.resize(image, (int(image.shape[1] * self.scale), int(image.shape[0] * self.scale)), interpolation=cv2.INTER_AREA)
        return image

    def run(self) -> str:
        while True:
            cv2.imshow(self.window, self.render())
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                self.save()
                cv2.destroyAllWindows()
                return "quit"
            if key in (ord("s"), ord("S")):
                self.save()
            elif key in (ord("r"), ord("R")):
                self.points = [None] * 4
            elif key in (ord("n"), ord("N")):
                self.save()
                cv2.destroyAllWindows()
                return "next"
            elif key in (ord("p"), ord("P")):
                self.save()
                cv2.destroyAllWindows()
                return "prev"
            elif ord("1") <= key <= ord("4"):
                self.selected = key - ord("1")


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and "_axis_labels" not in p.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Image ou dossier de photos")
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--screen-aspect", type=float, default=9 / 16, help="largeur/hauteur du telephone")
    parser.add_argument("--frame-ratio", type=float, default=0.82)
    args = parser.parse_args()
    images = collect_images(args.path)
    if args.skip_done:
        images = [p for p in images if not p.with_name(f"{p.stem}_axis_labels.json").exists()]
    index = 0
    while 0 <= index < len(images):
        print(f"[{index + 1}/{len(images)}] {images[index]}")
        action = AxisLabeler(images[index], args.screen_aspect, args.frame_ratio).run()
        if action == "quit":
            break
        index += 1 if action == "next" else -1
        index = max(0, index)


if __name__ == "__main__":
    main()
