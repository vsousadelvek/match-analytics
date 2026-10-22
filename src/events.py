"""Simulador DVS: frames -> eventos ON/OFF por contraste temporal."""
import cv2
import numpy as np


class DVSSimulator:
    def __init__(self, size=(320, 240), threshold=0.15):
        self.size = size
        self.th = threshold
        self.ref = None

    def reset(self):
        self.ref = None

    def process(self, frame):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       self.size).astype(np.float32) / 255.0
        if self.ref is None:
            self.ref = g
            return np.zeros(self.size[::-1], np.float32)
        diff = np.log(g + 1e-3) - np.log(self.ref + 1e-3)
        self.ref = 0.95 * self.ref + 0.05 * g      # referencia adaptativa
        ev = np.clip(diff / self.th, -1, 1).astype(np.float32)
        ev[np.abs(ev) < 0.3] = 0
        return ev
