"""Auto-labeling: jogadores em movimento geram blobs de eventos densos
-> caixas pseudo-labels pro treino do detector SNN. Zero anotacao manual.

Uso: python -m src.auto_label raw/m01.mp4 [mais videos]
"""
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .events import DVSSimulator

EV_W, EV_H = 320, 240
GRID = 16  # stride da heatmap (320x240 -> 20x15)
MAX_PER_VIDEO = 250


def detect_blobs(acc, min_area=18, dilate_k=5, min_density=0.10):
    raw = (acc != 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    mask = cv2.dilate(raw, kernel)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 4 or h < 5:
            continue
        if raw[labels == i].mean() < min_density:
            continue
        boxes.append((x, y, w, h))
    return boxes


def window_features(trail):
    """Ultimos 3 frames de eventos -> (2,240,320)."""
    acc = np.clip(sum(trail), -1, 1)
    return np.stack([np.clip(acc, 0, 1), np.clip(-acc, 0, 1)]).astype(np.float32)


def boxes_to_heatmap(boxes):
    """Centros -> heatmap Gaussiana (15x20) + size map (2x15x20)."""
    gh, gw = EV_H // GRID, EV_W // GRID
    hm = np.zeros((gh, gw), np.float32)
    size = np.zeros((2, gh, gw), np.float32)
    for (x, y, w, h) in boxes:
        cx, cy = (x + w / 2) / GRID, (y + h / 2) / GRID
        gx, gy = min(int(cx), gw - 1), min(int(cy), gh - 1)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                j, i = gy + dy, gx + dx
                if 0 <= j < gh and 0 <= i < gw:
                    hm[j, i] = max(hm[j, i],
                                   np.exp(-(dx * dx + dy * dy) / 2))
        size[0, gy, gx], size[1, gy, gx] = w / EV_W, h / EV_H
    return hm, size


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    n_saved = 0
    for spec in sys.argv[1:]:
        # vid ou vid:start_frac:end_frac (fracao do video, ex: 0.2:0.3)
        parts = spec.rsplit(":", 2)
        vid = parts[0]
        fr0, fr1 = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.0, 1.0)
        cap = cv2.VideoCapture(vid)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * fr0))
        dvs = DVSSimulator(size=(EV_W, EV_H))
        trail = deque(maxlen=3)
        cnt = fi = 0
        end = int(total * fr1)
        while cnt < MAX_PER_VIDEO and fi + total * fr0 < end:
            ok, f = cap.read()
            if not ok:
                break
            fi += 1
            trail.append(dvs.process(f))
            acc = np.clip(sum(trail), -1, 1)
            boxes = detect_blobs(acc)
            if not (2 <= len(boxes) <= 22):   # cena plausivel de jogo
                continue
            hm, size = boxes_to_heatmap(boxes)
            np.savez_compressed(out_dir / f"{Path(vid).stem}_{fi:05d}",
                                x=window_features(trail), hm=hm, size=size)
            cnt += 1
        cap.release()
        n_saved += cnt
        print(f"{spec}: {cnt} amostras")
    print(f"total: {n_saved}")


if __name__ == "__main__":
    main()
