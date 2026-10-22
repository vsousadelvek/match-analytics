"""Avaliacao honesta: SNN vs auto-labels (proxy de ground truth).
Precision/recall por frame com matching por centroide."""
from collections import deque

import cv2
import numpy as np
import torch

from src.events import DVSSimulator
from src.auto_label import detect_blobs, window_features
from src.snn_detector import EventDetectorSNN, decode

model = EventDetectorSNN()
model.load_state_dict(torch.load("checkpoints/snn_detector.pt",
                                 map_location="cpu", weights_only=False))
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cap = cv2.VideoCapture("raw/c3.mp4")          # video fora do clip de demo
dvs = DVSSimulator(size=(320, 240))
trail = deque(maxlen=3)
tp = fp = fn = 0
nfr = 0
while nfr < 600:
    ok, f = cap.read()
    if not ok:
        break
    nfr += 1
    trail.append(dvs.process(f))
    acc = np.clip(sum(trail), -1, 1)
    gt = detect_blobs(acc)
    x = torch.from_numpy(window_features(trail)).unsqueeze(0).to(device)
    with torch.no_grad():
        phm, psz = model(x)
    pred = decode(phm.cpu(), psz.cpu(), thr=0.4)

    used = set()
    for pb in pred:
        pc = np.array([pb[0] + pb[2] / 2, pb[1] + pb[3] / 2])
        best, bd = None, 30
        for i, gb in enumerate(gt):
            if i in used:
                continue
            gc = np.array([gb[0] + gb[2] / 2, gb[1] + gb[3] / 2])
            d = np.linalg.norm(pc - gc)
            if d < bd:
                best, bd = i, d
        if best is not None:
            tp += 1
            used.add(best)
        else:
            fp += 1
    fn += len(gt) - len(used)
cap.release()
print(f"TP={tp} FP={fp} FN={fn}")
print(f"precision={tp / max(tp + fp, 1):.2f} recall={tp / max(tp + fn, 1):.2f}")
