"""Coleta blobs de eventos com auto-label por posicao:
dentro do campo = jogador (1); fora (arquibancada, placas, bordas) = ruido (0).

Uso: python -m src.harvest_players "raw/m02.mp4:0.3:0.4" ...
"""
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .events import DVSSimulator
from .auto_label import detect_blobs
from .player_snn import crop_events

EV_W, EV_H = 320, 240
MAX = 120


def label_of(box):
    x, y, w, h = box
    cy = y + h
    # campo: pe do blob abaixo da linha das arquibancadas, dentro das margens
    inside = cy > EV_H * 0.25 and EV_W * 0.03 < x + w / 2 < EV_W * 0.97
    return 1 if inside else 0


def main():
    out = Path("data_players")
    out.mkdir(exist_ok=True)
    X, Y = [], []
    for spec in sys.argv[1:]:
        parts = spec.rsplit(":", 2)
        vid = parts[0]
        fr0, fr1 = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.0, 1.0)
        cap = cv2.VideoCapture(vid)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * fr0))
        dvs = DVSSimulator(size=(EV_W, EV_H))
        trail = deque(maxlen=3)
        cnt = fi = 0
        while cnt < MAX and fi + total * fr0 < total * fr1:
            ok, f = cap.read()
            if not ok:
                break
            fi += 1
            trail.append(dvs.process(f))
            acc = np.clip(sum(trail), -1, 1)
            boxes = detect_blobs(acc)
            if not boxes:
                continue
            for b in boxes:
                X.append(crop_events(acc, b).numpy())
                Y.append(label_of(b))
            cnt += 1
        cap.release()
        print(f"{spec}: {len(Y)} blobs")
    np.savez_compressed(out / "players.npz",
                        x=np.stack(X), y=np.array(Y))
    u, c = np.unique(Y, return_counts=True)
    print(dict(zip(u.tolist(), c.tolist())), "total:", len(Y))


if __name__ == "__main__":
    main()
