"""Demo de partida estilo videogame:
video full-frame -> olhos SNN nos jogadores -> radar inset com Voronoi.

Uso: python -m src.demo_match raw/clip.mp4 --calib calib_m02.json -o demo/match.mp4
"""
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from .events import DVSSimulator
from .auto_label import detect_blobs, window_features
from .snn_detector import EventDetectorSNN, decode
from .player_snn import PlayerSNN, crop_events
from .tracker import Tracker
from .teams import TeamClassifier, jersey_color
from .homography import Homography, draw_pitch, PITCH_W, PITCH_H

EV_W, EV_H = 320, 240
OUT_W, OUT_H = 1280, 720
TEAM_COLORS = [(0, 165, 255), (255, 80, 200)]   # laranja / rosa
REF_COLOR = (200, 200, 200)
RADAR_W, RADAR_H = 420, 252


def voronoi_overlay(pitch_img, players):
    """Colore regiao de controle por time (voronoi por jogador)."""
    if not players:
        return
    h, w = pitch_img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    best_d = np.full((h, w), 1e18)
    best_t = np.zeros((h, w), np.int8)
    for (px, py, team) in players:
        d = (xx - px) ** 2 + (yy - py) ** 2
        upd = d < best_d
        best_d[upd] = d[upd]
        best_t[upd] = team
    ov = pitch_img.copy()
    for tm, col in enumerate(TEAM_COLORS):
        m = best_t == tm
        ov[m] = (ov[m] * 0.45 + np.array(col) * 0.55).astype(np.uint8)
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calib", default=None)
    ap.add_argument("-o", "--out", default="demo/match.mp4")
    ap.add_argument("--use-events", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = Path("checkpoints/player_snn.pt")
    use_snn = args.use_events and ck.exists()
    model = None
    if use_snn:
        model = PlayerSNN().to(device)
        model.load_state_dict(torch.load(ck, map_location="cpu",
                                       weights_only=False))
        model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (OUT_W, OUT_H))
    dvs = DVSSimulator(size=(EV_W, EV_H))
    tracker = Tracker(max_dist=30, max_misses=12, min_hits=6)
    trail = deque(maxlen=3)
    homo = Homography(args.calib)
    teams = TeamClassifier()

    dist = {}
    last_pos = {}
    team_assign = {}
    fit_frames = 250
    fit_colors = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        trail.append(dvs.process(frame))
        # janela ponderada: o frame recente domina (posicao atual, nao atrasada)
        w = [0.55, 0.30, 0.15][-len(trail):] if len(trail) < 3 else [0.55, 0.30, 0.15]
        acc = np.clip(sum(e * ww for e, ww in zip(list(trail)[::-1], w)), -1, 1)

        # clustering de eventos = posicoes exatas; SNN filtra jogador vs ruido
        boxes = detect_blobs(acc)
        if use_snn and boxes:
            xs = torch.stack([crop_events(acc, b) for b in boxes]).to(device)
            with torch.no_grad():
                keep = model(xs).argmax(1).cpu().numpy()
            boxes = [b for b, k in zip(boxes, keep) if k == 1]
        # so o campo (mascara poligonal por baixo das arquibancadas)
        boxes = [b for b in boxes
                 if b[1] + b[3] > EV_H * 0.25 and b[3] >= b[2] * 0.8
                 and b[2] <= 40 and b[3] <= 60
                 and EV_W * 0.02 < b[0] + b[2] / 2 < EV_W * 0.98]
        merged = []
        for b in sorted(boxes, key=lambda b: -(b[2] * b[3])):
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            if not any((cx - (m[0] + m[2] / 2)) ** 2 + (cy - (m[1] + m[3] / 2)) ** 2 < 12 ** 2
                       for m in merged):
                merged.append(b)
        tracks = tracker.step(merged)

        fw, fh = frame.shape[1], frame.shape[0]      # px do frame original
        fx, fy = fw / EV_W, fh / EV_H                # ev-space -> frame px
        view = cv2.resize(frame, (OUT_W, OUT_H))
        kx, ky = OUT_W / EV_W, OUT_H / EV_H    # ev-space -> canvas
        players_radar = []

        for t in tracks:
            x, y, w, h = t.box
            foot_x, foot_y = (x + w / 2) * kx, (y + h) * ky
            col = jersey_color(view, (x * kx, y * ky, w * kx, h * ky))
            if fit_frames > 0 and col is not None:
                fit_colors.append(col)
            if col is not None:
                vote = teams.predict(col)
                if vote >= 0:
                    t.team_votes = getattr(t, "team_votes", []) + [vote]
                    t.team_votes = t.team_votes[-15:]
            if getattr(t, "team_votes", None):
                team = max(set(t.team_votes), key=t.team_votes.count)
                team_assign[t.id] = team
            else:
                team = team_assign.get(t.id, -1)
            color = TEAM_COLORS[team] if team >= 0 else REF_COLOR

            if t.misses > 4:
                continue
            ex, ey = int((x + w / 2) * kx), int((y + h / 2) * ky)
            pts = [(int(c[0] * kx), int(c[1] * ky)) for c in t.trail[-12:]]
            for a, b in zip(pts, pts[1:]):
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 80:   # sem saltos
                    cv2.line(view, a, b, color, 2)
            cv2.ellipse(view, (ex, ey), (16, 9), 0, 0, 360, color, 3)
            cv2.circle(view, (ex, ey), 5, color, -1)
            cv2.ellipse(view, (ex, ey), (16, 9), 0, 0, 360, (255, 255, 255), 1)

            m = homo.map((x + w / 2) * fx, (y + h) * fy)
            if m:
                rx = int(m[0] / PITCH_W * RADAR_W)
                ry = int(m[1] / PITCH_H * RADAR_H)
                players_radar.append((rx, ry, team))
                if t.id in last_pos:
                    d = np.linalg.norm(np.array(m) - last_pos[t.id])
                    if d < 3:
                        dist[t.id] = dist.get(t.id, 0) + d * (105 / PITCH_W)
                last_pos[t.id] = np.array(m)

        fit_frames -= 1
        if fit_frames == 0 and fit_colors:
            teams.fit(fit_colors)
            print("times calibrados:", len(fit_colors), "amostras")

        # radar inset bottom-right com voronoi
        radar = cv2.resize(draw_pitch()[80:320], (RADAR_W, RADAR_H))
        radar = voronoi_overlay(radar, players_radar) if players_radar else radar
        for (rx, ry, team) in players_radar:
            c = TEAM_COLORS[team] if team >= 0 else REF_COLOR
            cv2.circle(radar, (rx, ry), 5, c, -1)
            cv2.circle(radar, (rx, ry), 6, (255, 255, 255), 1)
        view[OUT_H - RADAR_H - 14:OUT_H - 14, OUT_W - RADAR_W - 14:OUT_W - 14] = radar
        cv2.rectangle(view, (OUT_W - RADAR_W - 14, OUT_H - RADAR_H - 14),
                      (OUT_W - 14, OUT_H - 14), (255, 255, 255), 1)

        # header stats
        a_dist = sum(d for tid, d in dist.items() if team_assign.get(tid) == 0)
        b_dist = sum(d for tid, d in dist.items() if team_assign.get(tid) == 1)
        hud = np.zeros((64, OUT_W, 3), np.uint8)
        cv2.putText(hud, "MATCH ANALYTICS  |  SNN events + Kalman", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        cv2.putText(hud, f"Time A {a_dist:.0f}m", (OUT_W - 420, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEAM_COLORS[0], 2)
        cv2.putText(hud, f"Time B {b_dist:.0f}m", (OUT_W - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEAM_COLORS[1], 2)
        view[:64] = cv2.addWeighted(view[:64], 0.35, hud, 0.65, 0)

        out.write(view)

    cap.release()
    out.release()
    print(f"salvo: {args.out}")


if __name__ == "__main__":
    main()
