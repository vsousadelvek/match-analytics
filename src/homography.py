"""Homografia: pontos calibrados do campo -> mapa 2D do radar.

Calibra uma vez por video (pontos visuais no frame), depois projeta
o pe de cada jogador pro espaco do campo (unidades arbitrarias 100x60).
"""
import json
from pathlib import Path

import cv2
import numpy as np

PITCH_W, PITCH_H = 100, 60   # canvas do radar


class Homography:
    def __init__(self, calib_path=None):
        self.H = None
        if calib_path and Path(calib_path).exists():
            d = json.loads(Path(calib_path).read_text())
            src = np.array(d["src"], np.float32)
            dst = np.array(d["dst"], np.float32) * [PITCH_W / 105.0, PITCH_H / 68.0]
            self.H, _ = cv2.findHomography(src, dst)

    def map(self, x, y):
        if self.H is None:
            return None
        p = self.H @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])


def draw_pitch():
    """Radar 2D do campo num canvas 400x400 (campo 400x240 centralizado)."""
    sc = 4
    img = np.zeros((400, 400, 3), np.uint8)
    y0 = (400 - PITCH_H * sc) // 2
    pitch = np.zeros((PITCH_H * sc, PITCH_W * sc, 3), np.uint8)
    pitch[:, :] = (0, 90, 0)
    W, H = PITCH_W * sc, PITCH_H * sc
    white = (200, 255, 200)
    cv2.rectangle(pitch, (0, 0), (W - 1, H - 1), white, 1)
    cv2.line(pitch, (W // 2, 0), (W // 2, H), white, 1)
    cv2.circle(pitch, (W // 2, H // 2), int(9.15 * sc), white, 1)
    box_h = int(20 * sc)
    for x0 in (0, W - int(16.5 * sc)):
        cv2.rectangle(pitch, (x0, H // 2 - box_h), (x0 + int(16.5 * sc), H // 2 + box_h), white, 1)
    img[y0:y0 + pitch.shape[0]] = pitch
    cv2.putText(img, "radar 2D", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return img
