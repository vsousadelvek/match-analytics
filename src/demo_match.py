"""Demo de partida: eventos -> SNN detecta jogadores -> Kalman tracking
-> times por cor -> radar 2D + stats.

Uso: python -m src.demo_match raw/clip.mp4 --calib calib_m02.json -o demo/match.mp4
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch
from pathlib import Path

from .events import DVSSimulator
from .auto_label import detect_blobs, window_features
from .snn_detector import EventDetectorSNN, decode
from .tracker import Tracker
from .teams import TeamClassifier, jersey_color
from .homography import Homography, draw_pitch, PITCH_W, PITCH_H

EV_W, EV_H = 320, 240
TEAM_COLORS = [(0, 140, 255), (255, 60, 200)]   # laranja / rosa
REF_COLOR = (255, 255, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calib", default=None)
    ap.add_argument("-o", "--out", default="demo/match.mp4")
    ap.add_argument("--use-events", action="store_true",
                    help="usa detector SNN nos eventos em vez do auto-label")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventDetectorSNN().to(device)
    ck = Path("checkpoints/snn_detector.pt")
    use_snn = args.use_events and ck.exists()
    if use_snn:
        model.load_state_dict(torch.load(ck, map_location="cpu",
                                       weights_only=False))
        model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (1200, 400))
    dvs = DVSSimulator(size=(EV_W, EV_H))
    tracker = Tracker(max_dist=35, max_misses=15, min_hits=5)
    trail = deque(maxlen=3)
    homo = Homography(args.calib)
    teams = TeamClassifier()

    dist = {}          # track id -> distancia em metros do campo
    last_pos = {}
    team_assign = {}   # track id -> time

    # calibra times nos primeiros frames
    fit_frames = 250
    fit_colors = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        trail.append(dvs.process(frame))
        acc = np.clip(sum(trail), -1, 1)

        if use_snn:
            x = torch.from_numpy(window_features(trail)).unsqueeze(0).to(device)
            with torch.no_grad():
                phm, psz = model(x)
            boxes = decode(phm.cpu(), psz.cpu())
        else:
            boxes = detect_blobs(acc)
        # so o campo: ignora arquibancada/placas (topo e bordas)
        # e mantem blobs com proporcao de pessoa (h > w)
        boxes = [b for b in boxes
                 if b[1] + b[3] > EV_H * 0.22 and b[3] >= b[2] * 0.8
                 and b[2] <= 40 and b[3] <= 60]
        tracks = tracker.step(boxes)

        sx, sy = 400 / EV_W, 400 / EV_H
        view = cv2.resize(frame, (400, 400))
        pitch = draw_pitch()
        panel = np.zeros((400, 400, 3), np.uint8)

        for t in tracks:
            x, y, w, h = t.box
            foot_x, foot_y = (x + w / 2) * sx, (y + h) * sy
            # cor do uniforme
            col = jersey_color(view, (x * sx, y * sy, w * sx, h * sy))
            if fit_frames > 0 and col is not None:
                fit_colors.append(col)
            tid = t.id
            if col is not None:
                vote = teams.predict(col)
                if vote >= 0:
                    t.team_votes = getattr(t, "team_votes", []) + [vote]
                    t.team_votes = t.team_votes[-15:]
            if getattr(t, "team_votes", None):
                team = max(set(t.team_votes), key=t.team_votes.count)
                team_assign[tid] = team
            else:
                team = team_assign.get(tid, -1)
            color = TEAM_COLORS[team] if team >= 0 else REF_COLOR
            cv2.rectangle(view, (int(x * sx), int(y * sy)),
                          (int((x + w) * sx), int((y + h) * sy)), color, 2)

            # radar: pe do jogador projetado
            m = homo.map(foot_x, foot_y)
            if m:
                y0_off = (400 - PITCH_H * 4) // 2
                px = int(m[0] * 4)
                py = int(m[1] * 4) + y0_off
                cv2.circle(pitch, (px, py), 4, color, -1)
                cv2.circle(pitch, (px, py), 5, (255, 255, 255), 1)
                if tid in last_pos:
                    d = np.linalg.norm(np.array(m) - last_pos[tid])
                    if d < 3:                       # evita saltos
                        dist[tid] = dist.get(tid, 0) + d
                last_pos[tid] = np.array(m)

        fit_frames -= 1
        if fit_frames == 0 and fit_colors:
            teams.fit(fit_colors)
            print("times calibrados:", len(fit_colors), "amostras")

        # painel: stats por time
        stats = {0: [0, 0.0], 1: [0, 0.0], -1: [0, 0.0]}
        for tid, d in dist.items():
            team = team_assign.get(tid, -1)
            stats.setdefault(team, [0, 0.0])
            stats[team][0] += 1
            stats[team][1] += d
        cv2.putText(panel, "MATCH ANALYTICS", (60, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (120, 120, 120), 2)
        active_team = {}
        for t in tracks:
            tm = team_assign.get(t.id, -1)
            active_team[tm] = active_team.get(tm, 0) + 1
        cv2.putText(panel, f"em campo: {len(tracks)}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        yy = 200
        for tm, name in [(0, "Time A"), (1, "Time B")]:
            n = active_team.get(tm, 0)
            d = stats.get(tm, [0, 0])[1]
            cv2.putText(panel, f"{name}: {n} em campo | {d:.0f}m", (20, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAM_COLORS[tm], 2)
            yy += 50
        cv2.putText(panel, "pipeline: DVS -> SNN detector -> Kalman",
                    (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        cv2.putText(panel, "-> kmeans times -> homografia 2D", (20, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        cv2.putText(view, "camera", (10, 390), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        out.write(np.hstack([view, pitch, panel]))

    cap.release()
    out.release()
    print(f"salvo: {args.out}")


if __name__ == "__main__":
    main()
