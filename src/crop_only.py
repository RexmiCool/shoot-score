import cv2
import numpy as np
import argparse
from pathlib import Path

BLACK_DISK_RADIUS_MM   = 100.0
OUTER_CIRCLE_RADIUS_MM = 250.0
N_INTERMEDIATE = 5

def _find_black_disk_in_image(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    min_area = 0.005 * h * w
    cx_img, cy_img = w / 2, h / 2
    tol_x, tol_y = w * 0.25, h * 0.25
    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
        if circ < 0.5:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if abs(cx - cx_img) > tol_x or abs(cy - cy_img) > tol_y:
            continue
        candidates.append((area, cnt, cx, cy, r))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, _, cx, cy, r = candidates[0]
    return int(cx), int(cy), int(r), BLACK_DISK_RADIUS_MM / r

def crop_target(img_orig, debug_dir=None, stem=""):
    result = _find_black_disk_in_image(img_orig)
    if result is None:
        return None
    cx0, cy0, r_disk0, mm_per_px0 = result
    r_outer0   = int(OUTER_CIRCLE_RADIUS_MM / mm_per_px0)
    margin_px0 = int(10.0 / mm_per_px0)
    half = r_outer0 + margin_px0
    x1 = max(0, cx0 - half)
    y1 = max(0, cy0 - half)
    x2 = min(img_orig.shape[1], cx0 + half)
    y2 = min(img_orig.shape[0], cy0 + half)
    img_crop = img_orig[y1:y2, x1:x2].copy()
    side = min(img_crop.shape[:2])
    img_crop = img_crop[:side, :side]
    # Retirer 13 cm en bas
    # remove_bottom_px = int(130.0 / mm_per_px0)
    # if img_crop.shape[0] > remove_bottom_px:
    #     img_crop = img_crop[:img_crop.shape[0] - remove_bottom_px, :]
    return img_crop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    images = []
    if Path(args.image).is_dir():
        for p in Path(args.image).glob("*.*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                images.append(p)
    elif Path(args.image).is_file():
        images.append(Path(args.image))
    else:
        print(f"Fichier introuvable : {args.image}")
        return
    for img_path in images:
        stem = img_path.stem
        out_dir = Path(args.out) / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            print(f"Image non trouvée : {img_path}")
            continue
        img_crop = crop_target(img_orig, debug_dir=out_dir, stem=stem)
        if img_crop is not None:
            crop_path = out_dir / f"{stem}_cropped.jpg"
            cv2.imwrite(str(crop_path), img_crop)
            print(f"Crop sauvegardé : {crop_path}")
        else:
            print(f"Cible non trouvée dans {img_path}")

if __name__ == "__main__":
    main()
