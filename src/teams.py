"""Classificacao de time por cor do uniforme — kmeans do zero."""
import numpy as np
import cv2


def kmeans(X, k=3, iters=20, seed=0):
    """kmeans em RGB. Retorna labels e centroides."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].astype(np.float32)
    for _ in range(iters):
        d = ((X[:, None] - C[None]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab, C


def jersey_color(frame, box):
    """Cor dominante do torso (regiao central superior do box),
    excluindo verde do campo."""
    x, y, w, h = [int(v) for v in box]
    x0, x1 = max(0, x + w // 4), x + 3 * w // 4
    y0, y1 = max(0, y + h // 8), y + h // 2
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    px = patch.reshape(-1, 3).astype(np.float32)   # BGR
    # remove pixels esverdeados (campo)
    non_green = ~((px[:, 1] > px[:, 0] + 20) & (px[:, 1] > px[:, 2] + 10))
    px = px[non_green]
    if len(px) < 4:
        return None
    return px.mean(0)


class TeamClassifier:
    """Fit numa colecao de cores de uniforme -> 2 clusters + ref."""

    def __init__(self):
        self.centers = None
        self.names = ["time A", "time B"]

    def fit(self, colors):
        colors = np.array([c for c in colors if c is not None])
        if len(colors) < 6:
            return False
        lab, C = kmeans(colors, k=min(3, len(colors)), iters=30)
        # descarta o cluster mais parecido com verde (campo)
        greenness = C[:, 1] - (C[:, 0] + C[:, 2]) / 2
        keep = np.argsort(greenness)[:2]
        self.centers = C[keep]
        return True

    def predict(self, color):
        if color is None or self.centers is None:
            return -1
        return int(((self.centers - color) ** 2).sum(1).argmin())
